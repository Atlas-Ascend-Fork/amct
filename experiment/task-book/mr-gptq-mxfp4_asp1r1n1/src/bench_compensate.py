# -*- coding: UTF-8 -*-
"""补偿循环的性能归因微基准 —— 验证「瓶颈是 per-column 的 kernel 启动开销」这一假设。

背景：报告 §6 记录「相对 stock GPTQ 提速 ~10×」，该数字为端到端墙钟实测，但由三项改动
打包取得（shared exponent 预计算 / 补偿移至 CPU / 改用标准 block-32 MXFP4），且
「瓶颈在 kernel 启动开销」是由差分实验推断的**机制假设**，未经直接测量。

本脚本不跑模型，只用合成张量复现补偿循环的内层，直接测四件事：

  P1  NPU 上单个小算子的启动/分发开销（tiny tensor，规模无关的那部分）
  P2  `_quant_col` 耗时随 out_features 的变化 —— **若在小规模上平坦则为 launch-bound**
      （这是判定 launch-bound 的关键判据：耗时不随计算量增长 ⇒ 时间没花在计算上）
  P3  单层补偿循环 CPU vs NPU 的实测耗时（in=4096 与 in=12288 两种规模）
  P4  主机-设备传输实测（报告中 64 GB / ~4.3s 的估算此前未实测）

  汇总：用 P1 的启动开销 × 由 P2 反推的每列 launch 数，预测 P3 的 NPU 耗时，与实测对比。
        两者同量级 ⇒ 支持 launch-bound 假设；相差悬殊 ⇒ 假设不成立，需另找解释。

用法：
    export ASCEND_RT_VISIBLE_DEVICES=0
    python3 -u bench_compensate.py 2>&1 | tee logs/bench_compensate.log
"""
import time
import math
import torch
import torch.nn.functional as F

from amct_pytorch.common.utils.quant_util import cal_shared_exponent
from amct_pytorch.common.utils.data_utils import float_to_fp4e2m1

try:
    import torch_npu  # noqa: F401
    DEV = "npu:0"
    def sync():
        torch.npu.synchronize()
except Exception:                                    # 无 NPU 时退化为纯 CPU 报告
    DEV = None
    def sync():
        pass


def _timeit(fn, n, warmup=5):
    """返回单次平均耗时（秒）。含设备同步，避免异步下测到假数。"""
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    sync()
    return (time.perf_counter() - t0) / n


def _quant_col(w, scale_col):
    """与 mr_gptq_module._quant_col 同构：单列 MXFP4 fake-quant。"""
    return float_to_fp4e2m1(w / scale_col) * scale_col


# ── P1：NPU 单算子启动/分发开销 ────────────────────────────────────────────
def p1_launch_overhead():
    print("\n=== P1  单算子启动/分发开销（tiny tensor，1 元素）===")
    if DEV is None:
        print("  无 NPU，跳过")
        return None
    x = torch.ones(1, device=DEV)
    for name, fn in [("加法 x+1", lambda: x + 1),
                     ("乘法 x*2", lambda: x * 2),
                     ("除法 x/3", lambda: x / 3)]:
        t = _timeit(fn, 2000)
        print(f"  {name:<10}  {t*1e6:8.2f} us/次")
    t_add = _timeit(lambda: x + 1, 2000)
    print(f"  → 取加法为启动开销估计：**{t_add*1e6:.2f} us/kernel**")
    return t_add


# ── P2：_quant_col 耗时 vs 规模（判定 launch-bound）────────────────────────
def p2_scaling(launch_us):
    print("\n=== P2  _quant_col 耗时随 out_features 变化（判定是否 launch-bound）===")
    print("  判据：耗时在小规模上**平坦**（不随元素数增长）⇒ 时间花在启动而非计算")
    results = {}
    for dev, tag in ([(DEV, "NPU"), ("cpu", "CPU")] if DEV else [("cpu", "CPU")]):
        print(f"  --- {tag} ---")
        print(f"  {'out_features':>12} {'耗时(us)':>12} {'相对最小规模':>14}")
        base = None
        for out in [64, 256, 1024, 4096, 16384, 65536]:
            w = torch.randn(out, device=dev, dtype=torch.float32)
            s = torch.full((out,), 0.5, device=dev, dtype=torch.float32)
            t = _timeit(lambda: _quant_col(w, s), 300)
            if base is None:
                base = t
            print(f"  {out:>12} {t*1e6:>12.2f} {t/base:>13.2f}x")
            results[(tag, out)] = t
        r = results[(tag, 65536)] / results[(tag, 64)]
        print(f"  → 规模放大 1024 倍，耗时变为 {r:.2f}x"
              f"（≈1 则完全 launch-bound；≈1024 则完全 compute-bound）")
    if DEV and launch_us:
        t64 = results[("NPU", 64)]
        print(f"  → NPU 上 out=64 时单列耗时 {t64*1e6:.2f} us，"
              f"除以启动开销 {launch_us*1e6:.2f} us ⇒ 约 **{t64/launch_us:.1f} 个 kernel/列**")
        return t64 / launch_us
    return None


# ── P3：单层补偿循环 CPU vs NPU ────────────────────────────────────────────
def _compensate_loop(weight, hessian_inv, blk_scale, group_size=32, block=128):
    """与 get_opt_weight_and_quant_factor 步骤 7 同构（去掉重排，只保留计时相关部分）。"""
    columns = weight.shape[1]
    for i1 in range(0, columns, block):
        i2 = min(i1 + block, columns)
        w_block = weight[:, i1:i2]
        err_block = torch.zeros_like(w_block)
        hinv_block = hessian_inv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            w = w_block[:, i]
            d = hinv_block[i, i]
            scale_col = blk_scale[:, (i1 + i) // group_size]
            w_q = _quant_col(w, scale_col)
            err = (w - w_q) / d
            w_block[:, i:] -= err.unsqueeze(1).matmul(hinv_block[i, i:].unsqueeze(0))
            err_block[:, i] = err
        weight[:, i2:] -= err_block.matmul(hessian_inv[i1:i2, i2:])
    return weight


def p3_layer(out_f, in_f, n_cols=256):
    """只跑前 n_cols 列以控制耗时，再按比例外推整层。"""
    print(f"\n=== P3  补偿循环 CPU vs NPU（out={out_f}, in={in_f}，实测前 {n_cols} 列）===")
    torch.manual_seed(0)
    out = {}
    for dev, tag in ([("cpu", "CPU"), (DEV, "NPU")] if DEV else [("cpu", "CPU")]):
        w = torch.randn(out_f, n_cols, device=dev, dtype=torch.float32)
        hinv = torch.eye(n_cols, device=dev, dtype=torch.float32) \
               + 0.01 * torch.randn(n_cols, n_cols, device=dev, dtype=torch.float32)
        hinv = torch.triu(hinv)
        blk = torch.full((out_f, max(1, n_cols // 32)), 0.5, device=dev, dtype=torch.float32)
        sync()
        t0 = time.perf_counter()
        _compensate_loop(w.clone(), hinv, blk)
        sync()
        t = time.perf_counter() - t0
        per_col = t / n_cols
        full = per_col * in_f
        out[tag] = (per_col, full)
        print(f"  {tag}:  {per_col*1e3:7.3f} ms/列   →  整层({in_f} 列) 外推 {full:7.2f} s")
    if "NPU" in out and "CPU" in out:
        sp = out["NPU"][0] / out["CPU"][0]
        print(f"  → **NPU / CPU = {sp:.1f}x**（>1 表示 CPU 更快）")
    return out


# ── P4：主机-设备传输实测 ──────────────────────────────────────────────────
def p4_transfer():
    print("\n=== P4  主机-设备传输实测（报告中 64GB/~4.3s 为估算，此处实测带宽）===")
    if DEV is None:
        print("  无 NPU，跳过")
        return
    for name, shape, dt in [("Hessian in=4096  [4096,4096] fp32", (4096, 4096), torch.float32),
                            ("Hessian in=12288 [12288,12288] fp32", (12288, 12288), torch.float32),
                            ("weight down_proj [4096,12288] fp16", (4096, 12288), torch.float16)]:
        nbytes = math.prod(shape) * (4 if dt == torch.float32 else 2)
        x = torch.randn(*shape, device=DEV, dtype=dt)
        t_d2h = _timeit(lambda: x.to("cpu"), 5, warmup=2)
        y = x.to("cpu")
        t_h2d = _timeit(lambda: y.to(DEV), 5, warmup=2)
        print(f"  {name}")
        print(f"    {nbytes/1e6:8.1f} MB   D2H {t_d2h*1e3:7.2f} ms ({nbytes/t_d2h/1e9:5.2f} GB/s)"
              f"   H2D {t_h2d*1e3:7.2f} ms ({nbytes/t_h2d/1e9:5.2f} GB/s)")
        del x, y


def main():
    print("=" * 76)
    print("补偿循环性能归因微基准    device =", DEV or "cpu only")
    print("=" * 76)
    launch = p1_launch_overhead()
    k_per_col = p2_scaling(launch)
    r4096 = p3_layer(4096, 4096)
    r12288 = p3_layer(4096, 12288)
    p4_transfer()

    print("\n" + "=" * 76)
    print("汇总：launch-bound 假设的检验")
    print("=" * 76)
    if launch and k_per_col and "NPU" in r4096:
        pred = launch * k_per_col
        meas = r4096["NPU"][0]
        print(f"  预测每列 NPU 耗时 = 启动开销 {launch*1e6:.2f} us × {k_per_col:.1f} kernel"
              f" = {pred*1e6:.1f} us")
        print(f"  实测每列 NPU 耗时 = {meas*1e6:.1f} us")
        print(f"  比值 = {meas/pred:.2f}x")
        print("  → 接近 1 ⇒ 补偿循环的 NPU 耗时确由 kernel 启动主导，假设成立；")
        print("    远大于 1 ⇒ 另有来源（计算本身 / 同步停顿 / 算子实现），假设不成立。")
    else:
        print("  无 NPU，无法检验。")
    print("\n注：本脚本用合成张量，不含 Hessian 求逆、旋转与 scale fitting；")
    print("    仅针对「逐列量化 + 误差补偿」这一内层循环，即提速改动的作用位置。")


if __name__ == "__main__":
    main()
