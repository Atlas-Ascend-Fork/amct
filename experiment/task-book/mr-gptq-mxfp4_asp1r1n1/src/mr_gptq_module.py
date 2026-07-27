# -*- coding: UTF-8 -*-
"""MR-GPTQ（Micro-Rotated GPTQ, arXiv:2509.23202）MXFP4 实现。

子类化 amct 的 GPTQuant，override 权重优化，做三件事：
  1. **标准 block-32 MXFP4 量化**（沿 in-features、每 32 元素共享 E8M0 指数），
     取代 stock 的 per-column `.repeat(1,2)` 非标准缩放；
  2. **shared exponent 预计算一次 + CPU 上做补偿**——消除 stock 每列重算 20+ 小算子、
     百万次 NPU kernel 启动的开销（2h → 分钟级）；
  3. MR-GPTQ 三增量（按开关渐进加入）：
       - block-wise Hadamard 微旋转
       - MXFP scale fitting（附录 H）
       - 静态激活重排
     本文件先落地 ①②（= 干净的 GPTQ-mxfp4 baseline），增量在 _INCREMENTS 开关下逐个加。

零改 amct 源码：import 本模块即运行时注册 'mr_gptq'。
"""
import math
import time

import torch
import torch.nn.functional as F

from amct_pytorch.classic.quantize_op.gptq_module import GPTQuant
from amct_pytorch.common.utils.quant_util import cal_shared_exponent
from amct_pytorch.common.utils.data_utils import float_to_fp4e2m1
from amct_pytorch.common.utils.vars import MXFP4_E2M1


# MR-GPTQ 增量开关（默认全关 = GPTQ-mxfp4 baseline；逐个打开做消融）
_INCREMENTS = {
    "hadamard": False,        # block-wise Hadamard 微旋转
    "scale_fitting": False,   # MXFP scale fitting（附录 H），作用于权重
    "act_reorder": False,     # 静态激活重排
    "act_quant": False,       # W4A4：激活也做 MXFP4（动态、逐32块）；须配 hadamard 压离群值
    "act_scale_fit": False,   # W4A4 优化①：激活量化也用 scale fitting（次幂次细网格，非幂次）
    "joint_comp": False,      # W4A4 优化②：用量化后的激活建 Hessian，权重补偿适配 W4A4 输入
    "act_clip": False,        # W4A4 优化③：激活 scale 裁剪，逐块搜 α 最小化 MSE（不加高精度层）
}

# 激活 scale 裁剪候选系数：scale ← α·scale。含 1.0（=不裁剪）→ 逐块取 MSE 最优者时，
# 结果在 MSE 意义上**不可能劣于**不裁剪。跨度取宽（0.6~1.0）而非细密，因最优 α 未知且逐块异质。
_ACT_CLIP_GRID = (1.0, 0.9, 0.8, 0.7, 0.6)

_HADAMARD_K = 128  # block-wise Hadamard 尺寸；论文取 128 是为【在线算子】的开销权衡。
                   # 本项目 fake-quant 只测精度、旋转无推理开销 → 可经 --hadamard_k 调大
                   # (512/1024/4096) 做消融：块越大离群值摊得越开，或能进一步压 W4A4 误差。
                   # 注意 in_features 须被 k 整除，否则该层跳过旋转（见 _do_hadamard）。

# W4A4 混合精度：层名含这些子串的层【不】量化激活（保 bf16/高精度），用于保护敏感层冲 0.4。
# 例：['down_proj'] 保护所有 down_proj 的激活；空 = 全量 W4A4。
_ACT_PROTECT = []

_PROGRESS = [0]  # 已补偿层计数（进度显示用；每进程从 0 开始）

_FP4_MAX_EXP = 2  # E2M1 最大指数（6 = 1.5·2²）；scale fitting 的 offset


def _sylvester_hadamard(k):
    """归一化 Sylvester-Hadamard H_k/√k（k 须为 2 的幂），正交且对称。"""
    H = torch.ones(1, 1)
    while H.shape[0] < k:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    return H / (k ** 0.5)


_HADAMARD_CACHE = {}


def _get_hadamard(k):
    """按 k 缓存 Hadamard 矩阵。k 大时矩阵不小（k=4096 → 64MB fp32），252 个量化模块
    各建一份会撑爆内存，故同一 k 全局只造一次、共享只读。"""
    H = _HADAMARD_CACHE.get(k)
    if H is None:
        H = _sylvester_hadamard(k)
        _HADAMARD_CACHE[k] = H
    return H


class MRGPTQuant(GPTQuant):
    """MR-GPTQ 量化算子。当前实现 = 高效标准 MXFP4 + GPTQ 误差补偿 + 可选增量。"""

    def __init__(self, ori_module, layer_name, quant_config):
        super().__init__(ori_module, layer_name, quant_config)
        self.increments = dict(_INCREMENTS)
        if self.wts_type != MXFP4_E2M1:
            raise ValueError(f"MR-GPTQ 目前仅支持 mxfp4_e2m1，收到 {self.wts_type}")
        self.hadamard_k = _HADAMARD_K
        self._H = _get_hadamard(self.hadamard_k)  # [k,k] CPU float32（全局缓存，同 k 共享）

    # ---- W4A4：本层是否量化激活（混合精度下敏感层跳过，保高精度） ----------
    def _do_act_quant(self):
        if not self.increments.get("act_quant"):
            return False
        return not any(p in self.layer_name for p in _ACT_PROTECT)

    # ---- block-wise Hadamard 旋转（增量①） --------------------------------
    def _do_hadamard(self):
        # in_features 需被 k 整除；否则该层跳过旋转（Qwen3 各层均可整除）
        return self.increments["hadamard"] and self.weight.shape[1] % self.hadamard_k == 0

    def _rotate_last_dim(self, x):
        """沿最后一维（in）做块对角 Hadamard：reshape [...,n_blk,k] @ H_k。
        matmul 在 fp32 下做再回原 dtype——k 大时（如 4096）fp16 累加 4096 项误差可达数个
        百分点，会污染消融；fp32 旋转对 k=128 影响可忽略、对大 k 是必需。"""
        k = self.hadamard_k
        shape = x.shape
        H = self._H.to(device=x.device, dtype=torch.float32)
        xr = x.reshape(*shape[:-1], shape[-1] // k, k).float().matmul(H)
        return xr.reshape(shape).to(x.dtype)

    def update_hessian(self, input_data):
        # 旋转后累积 Hessian，使 H_r = RᵀHR 与旋转权重一致（转输入比转 Hessian 便宜）
        if self._do_hadamard():
            input_data = self._rotate_last_dim(input_data)
        # W4A4 优化②（联合补偿）：用【量化后】的激活建 Hessian，使权重补偿适配 W4A4 推理输入
        if self.increments.get("joint_comp") and self._do_act_quant():
            input_data = self._quant_act_mxfp4(
                input_data, self.group_size, self.increments.get("act_scale_fit", False),
                self.increments.get("act_clip", False))
        super().update_hessian(input_data)

    # ---- MXFP scale fitting（附录 H 式2/3）：E8M0 网格重映射 ----------------
    @staticmethod
    def _fit_block_scale(weight, block_size=32):
        """把每块连续 log-scale 重映射到 [L_min,L_max] 的 256 级细网格，
        得到次幂次(sub-power-of-two) scale，减小 E8M0 粗网格的 scale 量化误差。
        L = log2(block_max) − FP4_MAX_EXP；q=clamp(round(255·(L−Lmin)/(Lmax−Lmin)),0,255)；
        L_fit = Lmin + q·(Lmax−Lmin)/255；返回每块浮点 scale [out, n_groups]。
        """
        out, in_features = weight.shape
        n_groups = (in_features + block_size - 1) // block_size
        pad = n_groups * block_size - in_features
        w = F.pad(weight, (0, pad)) if pad else weight
        w = w.reshape(out, n_groups, block_size)
        block_max = w.abs().amax(dim=-1)                             # [out, n_groups]
        # 连续理想 log2(scale)：block_max/scale = 6（用满 E2M1 范围，非 4）
        L = torch.log2(block_max) - math.log2(6.0)
        L = torch.where(torch.isfinite(L), L, torch.zeros_like(L))  # 空块→0
        Lmin, Lmax = L.min(), L.max()
        denom = (Lmax - Lmin).clamp_min(1e-6)
        q = torch.clamp(torch.round(255.0 * (L - Lmin) / denom), 0.0, 255.0)
        L_fit = Lmin + q * (Lmax - Lmin) / 255.0                    # 细网格
        return torch.pow(2.0, L_fit)                                # 每块浮点 scale

    # ---- block-32 MXFP4 单列量化（按每块浮点 scale） ----------------------
    @staticmethod
    def _quant_col(w, scale_col):
        """对单列 w[out] 用其组 scale scale_col[out] 做 MXFP4 fake-quant：
        w/scale → round 到 E2M1 网格 → ×scale。"""
        return float_to_fp4e2m1(w / scale_col) * scale_col

    # ---- 直接 MXFP4 量化（无补偿，RTN） -----------------------------------
    @staticmethod
    def _quant_direct(weight, blk_scale, group_size):
        """整块权重直接 MXFP4 量化：每列按其组 scale round 到 E2M1，不做 GPTQ 补偿。
        用于 MoE 冷/稀疏 expert（Hessian 缺失或秩亏，补偿不可用）——退化为 RTN 好过
        把权重列置零。weight[out,in]，blk_scale[out, n_groups]。"""
        in_features = weight.shape[1]
        scale_full = blk_scale.repeat_interleave(group_size, dim=1)[:, :in_features]
        return float_to_fp4e2m1(weight / scale_full) * scale_full

    # ---- 激活 scale 裁剪搜索（W4A4 优化③） ---------------------------------
    @classmethod
    def _quant_clip_search(cls, x2, blk_scale, group_size, grid=_ACT_CLIP_GRID):
        """逐块搜裁剪系数 α：scale ← α·scale，取块内量化 MSE 最小的 α。

        动机：默认 scale 让块内 |max| 恰好落到 E2M1 上界 6，即**最大的那个元素独占了全部
        动态范围**；E2M1 仅 16 级，其余 31 个元素被挤在粗网格上。适度裁剪（牺牲长尾的
        精确度）能显著提高块内多数元素的分辨率——这是不引入任何高精度层、纯 W4A4 的优化。

        grid 含 1.0 → 每块取 argmin MSE 时，**结果在 MSE 意义上不劣于不裁剪**（最坏退化为原行为）。
        x2[N, C]、blk_scale[N, n_groups]；返回 [N, C]。
        """
        n_rows, columns = x2.shape
        n_groups = blk_scale.shape[1]
        pad = n_groups * group_size - columns
        xg = (F.pad(x2, (0, pad)) if pad else x2).reshape(n_rows, n_groups, group_size)

        best_q = best_err = None
        for alpha in grid:
            s = (blk_scale * alpha).clamp_min(1e-20).unsqueeze(-1)   # [N, G, 1]
            q = float_to_fp4e2m1(xg / s) * s
            err = (q - xg).pow(2).sum(dim=-1, keepdim=True)          # [N, G, 1] 逐块 MSE
            if best_q is None:
                best_q, best_err = q, err
            else:
                take = err < best_err
                best_q = torch.where(take, q, best_q)
                best_err = torch.where(take, err, best_err)
        out = best_q.reshape(n_rows, n_groups * group_size)
        return out[:, :columns] if pad else out

    # ---- 激活 MXFP4 动态量化（W4A4，增量④） ---------------------------------
    @classmethod
    def _quant_act_mxfp4(cls, x, group_size=32, scale_fit=False, clip=False):
        """激活 MXFP4 动态伪量化：把 [*, in] 展平成 [N, in]，沿最后一维每 32 通道一块，
        **运行时**算 scale（激活是动态的，每次前向重算）→ 吸附到 E2M1。复用 _quant_direct。
        scale_fit=False：E8M0 幂次 scale；True：附录 H 次幂次细网格（W4A4 优化①，减小激活 scale 误差）。
        clip=True：在此 scale 上再逐块搜裁剪系数（W4A4 优化③）。
        须先做 Hadamard 旋转压激活离群值，否则 4-bit 激活会崩。"""
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).float()
        if scale_fit:
            blk = cls._fit_block_scale(x2, group_size)                   # 次幂次细网格
        else:
            blk = torch.pow(2.0, torch.nan_to_num(cal_shared_exponent(x2)))  # E8M0 幂次
        if clip:
            xq = cls._quant_clip_search(x2, blk, group_size)
        else:
            xq = cls._quant_direct(x2, blk, group_size)
        return xq.reshape(shape).to(x.dtype)

    # ---- Hessian 逆的上三角 Cholesky（override 父类，全程 CPU） -------------
    def cal_hessian_inverse(self, hessian, columns):
        # 父类用 self.weight.device(NPU) 造 index 张量，与 CPU hessian 冲突；
        # 这里全程 CPU：damping → cholesky → 逆 → 上三角 cholesky。
        damp = self.perc_damp * torch.mean(torch.diag(hessian))
        idx = torch.arange(columns)
        hessian[idx, idx] += damp
        L = torch.linalg.cholesky(hessian)
        hinv = torch.cholesky_inverse(L)
        return torch.linalg.cholesky(hinv, upper=True)

    # ---- GPTQ 误差补偿（CPU、标准 MXFP4、shared exponent 预计算） ----------
    @torch.no_grad()
    def get_opt_weight_and_quant_factor(self):
        columns = self.weight.shape[1]
        orig_device = self.weight.device
        orig_dtype = self.weight.dtype

        _PROGRESS[0] += 1
        _t0 = time.time()
        print(f"[MR-GPTQ] layer #{_PROGRESS[0]:>3}  {self.layer_name}  "
              f"(in={columns}, had={self._do_hadamard()}, "
              f"scale_fit={self.increments['scale_fitting']})", flush=True)

        # 全程 CPU float32：小算子无 NPU 启动开销，Cholesky 亦在 CPU（~0.7s）
        weight = self.weight.detach().to("cpu", torch.float32)
        if self._do_hadamard():
            weight = self._rotate_last_dim(weight)  # 旋转权重（Hessian 已在 update 时旋转）

        # 1) 预计算每块浮点 scale（沿 in-features，32 一组）——基于真实权重，置零前算。
        #    形状 [out, ceil(in/32)]；后续每列按 原始列号//32 取用。
        #    增量②：scale fitting → E8M0 网格重映射（附录 H 式2/3）
        if self.increments["scale_fitting"]:
            blk_scale = self._fit_block_scale(weight, self.group_size)
        else:
            blk_scale = torch.pow(2.0, torch.nan_to_num(cal_shared_exponent(weight)))

        # 2) MoE 冷 expert 兜底：校准前向从未路由到它 → 无 Hessian。
        #    （理论上 get_opt 只在 ≥1 次 update 后触发、hessian 不为 None；防御性保留）
        if self.hessian is None:
            wq = self._quant_direct(weight, blk_scale, self.group_size)
            print(f"[MR-GPTQ]   ↳ no Hessian (cold expert) → direct MXFP4 "
                  f"in {time.time() - _t0:.1f}s", flush=True)
            return wq.to(orig_device, orig_dtype), None, None

        hessian = self.hessian.to("cpu", torch.float32)
        diag = torch.diag(hessian)
        dead = diag == 0

        # 3) MoE 稀疏 expert 兜底：多数输入通道在校准集内零激活 → Hessian 秩亏，
        #    GPTQ 补偿不稳且 weight[:,dead]=0 会毁掉整块权重。退化为直接 MXFP4（RTN）。
        if bool(dead.all()) or dead.float().mean().item() > 0.5:
            wq = self._quant_direct(weight, blk_scale, self.group_size)
            print(f"[MR-GPTQ]   ↳ sparse expert (dead={dead.float().mean().item():.0%}) "
                  f"→ direct MXFP4 in {time.time() - _t0:.1f}s", flush=True)
            return wq.to(orig_device, orig_dtype), None, None

        # 4) 少量死列（dense 层常态）：置 1 稳定 Cholesky、对应权重列置零（stock GPTQ）
        hessian[dead, dead] = 1
        weight[:, dead] = 0

        # 5) 激活重排：按 Hessian 对角（激活能量）排序处理列。
        #    stock GPTQ 亦如此；MR-GPTQ 的“静态激活重排”增量将在此细化。
        perm = torch.argsort(diag, descending=True)
        invperm = torch.argsort(perm)
        weight = weight[:, perm]
        hessian = hessian[perm][:, perm]

        # 6) Hessian 逆的上三角 Cholesky（override 版，全程 CPU）
        #    极端秩亏时仍可能非正定 → 兜底直接 MXFP4（用旋转后原权重，未置零）。
        try:
            hessian_inv = self.cal_hessian_inverse(hessian, columns)
        except torch.linalg.LinAlgError:
            w0 = self.weight.detach().to("cpu", torch.float32)
            if self._do_hadamard():
                w0 = self._rotate_last_dim(w0)
            wq = self._quant_direct(w0, blk_scale, self.group_size)
            print(f"[MR-GPTQ]   ↳ Hessian not PD → direct MXFP4 "
                  f"in {time.time() - _t0:.1f}s", flush=True)
            return wq.to(orig_device, orig_dtype), None, None
        del hessian

        # 7) 分块 + 逐列量化 + 误差补偿（标准 GPTQ）
        block = self.block_size
        for i1 in range(0, columns, block):
            i2 = min(i1 + block, columns)
            w_block = weight[:, i1:i2]
            err_block = torch.zeros_like(w_block)
            hinv_block = hessian_inv[i1:i2, i1:i2]

            for i in range(i2 - i1):
                w = w_block[:, i]
                d = hinv_block[i, i]
                col_idx = perm[i1 + i]                      # 原始列号
                scale_col = blk_scale[:, col_idx // self.group_size]
                w_q = self._quant_col(w, scale_col)
                err = (w - w_q) / d
                w_block[:, i:] -= err.unsqueeze(1).matmul(hinv_block[i, i:].unsqueeze(0))
                err_block[:, i] = err

            weight[:, i2:] -= err_block.matmul(hessian_inv[i1:i2, i2:])

        # 8) 恢复列顺序；此时 weight 每列已是量化值 → 即 fake-quant 权重
        weight = weight[:, invperm].to(orig_device, orig_dtype)
        print(f"[MR-GPTQ]   ↳ done in {time.time() - _t0:.1f}s", flush=True)
        # mxfp4 无独立 scale/offset（含在共享指数里）
        return weight, None, None

    @torch.no_grad()
    def fake_quant_forward(self, inputs):
        # weight 已是（旋转+）量化后的权重；Hadamard 开时输入需同样旋转，
        # y = (x·R)·(W·R)ᵀ = x·Wᵀ（R 正交对称），weight-only 下 x·R 精确无损。
        if self._do_hadamard():
            inputs = self._rotate_last_dim(inputs)
        # W4A4：旋转后把激活也量化到 MXFP4（旋转已压离群值 → 激活可 4-bit）。
        # 注意：只在此 fake-quant 路径量化激活；校准前向走父类 F.linear、不量化激活。
        if self._do_act_quant():
            inputs = self._quant_act_mxfp4(
                inputs, self.group_size, self.increments.get("act_scale_fit", False),
                self.increments.get("act_clip", False))
        return F.linear(inputs, self.weight.to(inputs.dtype), self.bias)


# ---- 融合存储的 MoE 路由专家量化（amct 覆盖不到的大头） --------------------
@torch.no_grad()
def quantize_fused_experts(model, scale_fitting=True, min_experts_numel=1_000_000):
    """对融合存储的 MoE 路由专家做 MXFP4 fake-quant。

    背景：Qwen3.5-MoE 的路由专家不是一堆 nn.Linear，而是 `Qwen3_5MoeExperts` 模块里的
    3D 权重张量 `gate_up_proj`/`down_proj`（形状 [E, out, in]，占 35B 的大头）。amct 只
    量化 nn.Linear → 碰不到它们 → 不处理则「显存降低≥50%」必然不达标。

    做法：沿最后一维(in，收缩维)每 32 元素一组做标准 MXFP4（reshape [E*out, in] 后
    复用逐块量化）。weight-only RTN + 可选 scale fitting；**不做 Hadamard**（专家输入的
    旋转需与 router/shared_expert 同步，代价大；且 scale fitting 对 RTN 本就是加分项）。

    ⚠️显存：device_map 会把卡塞得很满(card 0 常仅剩 ~1GB)，故计算全程放 **CPU**
    （fp32 临时量 2GB+ 放卡上必 OOM）；回写时先把 CPU 结果赋给 W.data 释放卡上旧张量、
    empty_cache、再挪回卡，保证设备峰值不超基线。

    返回 (专家模块数, 已量化参数量)。对无此结构的模型(如 8B dense)是 no-op。
    """
    def _empty_cache():
        try:
            import torch_npu
            torch_npu.npu.empty_cache()
        except Exception:
            pass

    n_mod = 0
    quant_param = 0
    for name, mod in model.named_modules():
        if type(mod).__name__ != "Qwen3_5MoeExperts":
            continue
        n_mod += 1
        for pname in ("gate_up_proj", "down_proj"):
            W = getattr(mod, pname, None)
            if W is None or W.numel() < min_experts_numel:
                continue
            dev, dt = W.device, W.dtype
            E, O, I = W.shape
            w2 = W.data.reshape(E * O, I).to("cpu").float()  # 先 fp16→CPU 再转 fp32（避免卡上 fp32 临时量）
            if scale_fitting:
                blk = MRGPTQuant._fit_block_scale(w2, 32)          # 附录H 网格重映射
            else:
                blk = torch.pow(2.0, torch.nan_to_num(cal_shared_exponent(w2)))
            wq = MRGPTQuant._quant_direct(w2, blk, 32).reshape(E, O, I).to(dt)  # CPU fp16 结果
            del w2, blk
            W.data = wq                # 先赋 CPU：释放卡上旧权重存储
            _empty_cache()             # 回收，给挪回腾地方
            W.data = wq.to(dev)        # 再挪回原卡（此时峰值不超基线）
            del wq
            quant_param += W.numel()
        _empty_cache()
        print(f"[fused-experts] {name}  (scale_fit={scale_fitting}) 量化完成", flush=True)
    print(f"[fused-experts] 共 {n_mod} 个 Experts 模块，量化 {quant_param / 1e9:.2f}B 参数",
          flush=True)
    return n_mod, quant_param


# ---- 运行时注册 'mr_gptq'（零改 amct 源码） --------------------------------
def register():
    import amct_pytorch.algorithms as _algos
    import amct_pytorch.common.utils.vars as _vars
    from amct_pytorch.classic.deploy_op.weight_npu_quant_module import NpuWeightQuantizedLinear

    # 1) 注册算子 + 加入内置算法白名单
    _algos.AlgorithmRegistry.register("mr_gptq", "Linear", MRGPTQuant, NpuWeightQuantizedLinear)
    if "mr_gptq" not in _algos.BUILT_IN_ALGORITHM:
        _algos.BUILT_IN_ALGORITHM.append("mr_gptq")

    # 2) 加入 (算法, 量化类型组合) 支持表：weight-only MXFP4
    comb = "NOT_QUANTIZE mxfp4_e2m1"
    supported = _vars.ALGORITHM_SUPPORTED_QUANT_TYPE_COMB.get(comb)
    if supported is not None and "mr_gptq" not in supported:
        supported.append("mr_gptq")


register()
