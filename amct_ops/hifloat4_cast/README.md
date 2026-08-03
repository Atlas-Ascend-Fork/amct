# hifloat4_cast — HiFloat4 量化算子

HiFloat4（hifx4）是 4-bit 块缩放浮点格式：每元素 **S1P2**（1 符号 + 2 位尾数，8 档幅值），配**每 64 元素共享的 3 级 scale**（L1 `scale_factor` E6M2 覆盖 64 块、L2 覆盖每 8、L3 覆盖每 4）。

## 架构

```
┌─────────────────────────────────────────────────────────┐
│  CLI (amct_pytorch/cli/llm/)                            │
│  --quant_dtype hifp  →  auto-detect source format       │
│    ├─ PTQ:   fake-quant + unpack (HiF4 → float)         │
│    └─ Deploy: pack (float → HiF4 bytes)                 │
├─────────────────────────────────────────────────────────┤
│  Python (amct_pytorch/common/utils/hifloat4_quant/)      │
│  ├─ _hif4_encode_decode.py  共享编解码（fake-quant & pack）│
│  ├─ hif4_pack_unpack.py     float ↔ HiF4 uint8 打包/解包  │
│  └─ quant_dequant.py         fake-quant 入口             │
├─────────────────────────────────────────────────────────┤
│  NPU Kernel (op_kernel/hifloat4_cast_kernel.cpp)         │
│  └─ hifx_kernel / hifx_kernel_bf16    FP→HiF4→FP 往返   │
│     （仅 fake-quant；打包/解包在 Python 侧完成）            │
└─────────────────────────────────────────────────────────┘
```

- **1 个 NPU kernel**：`hifx_kernel`（fp32）/ `hifx_kernel_bf16`（bf16），做一次 FP→HiF4→FP 仿真往返
- **打包/解包**：纯 PyTorch 实现（`hif4_pack_unpack.py`），将 float 权重编码为 uint8 HiF4 字节，或从字节解码为 float
- **共享编解码**：fake-quant 和 pack 共用同一套 `_hif4_encode_decode`，保证评估路径和部署权重永远一致

## CLI 工作流

```bash
# PTQ（后训练量化）：fake-quant 仿真 or HiF4 解包
python -m amct_pytorch.cli.llm.ptq \
    --model /path/to/model --quant_dtype hifp --output_dir ./output

# Deploy（部署打包）：float → HiF4 uint8 字节，输出 safetensors
python -m amct_pytorch.cli.llm.deploy \
    --model /path/to/model --quant_dtype hifp --output_dir ./output
```

当 `--quant_dtype hifp` 时，CLI 会**自动检测**输入模型格式：
- **hif4 已打包** → PTQ 直接解包为 float checkpoint（无需构建模型）
- **bf16/fp16 原始权重** → PTQ 走 fake-quant 评估 or deploy 走真打包

## 接口

设备算子 `torch.ops.amct.hifloat4_fake_quant(Tensor input, int qdim=-1) -> Tensor`：输入 FP16/BF16，
输出同 shape/dtype，沿 `qdim` 每 64 元素构成一个共享 scale 块。`qdim` 是 schema 参数，permute/pad/切回
的布局处理在 **C++ host** 完成（Python 只是薄透传）：

```python
from amct_ops.hifloat4_cast import hifloat4_fake_quant
y = hifloat4_fake_quant(x)                     # 沿最后一维分块
y = hifloat4_fake_quant(w, qdim=1)             # 例：Linear 权重 [out, in] 沿 in
y = torch.ops.amct.hifloat4_fake_quant(x, -1)  # 等价
```

host 会把 `qdim` 移到最后一维、补零到 64 的整数倍（补零不影响块内 max）、调 kernel，再切回、还原维序。

## 正确性（验收红线）

kernel 逐步复刻纯 python 参考，逐元素对齐到 ~0 误差。在设备上对拍 AMCT 的 CPU 参考：

```python
import torch
from amct_ops.hifloat4_cast import hifloat4_fake_quant
from amct_pytorch.common.utils.hifloat4_quant import QType, quant_dequant_float

x = torch.randn(4, 512, dtype=torch.bfloat16)
y_ref = quant_dequant_float(x, QType('hifx4').dim(-1), force_py=True, force_fp32=True)  # CPU 参考
y_npu = hifloat4_fake_quant(x.npu()).cpu()                                              # AscendC kernel
assert (y_ref.float() - y_npu.float()).abs().max() < 1e-6
# 零值不能出 NaN
z = torch.zeros(1, 64, dtype=torch.bfloat16)
assert not torch.isnan(hifloat4_fake_quant(z.npu())).any()
```

`tests/amct_ops/test_hifloat4_cast.py` 在有 NPU 时对拍 kernel 与该 CPU 参考。

## 打包格式

```
scale : uint8 [M, N // 64, 4]     每 64-group 一条 4-byte 记录
    byte 0 : E6M2                  level-1 scale factor，E6M2 = 8 bits
    byte 1 : E1_8  (8 x 1 bit)    per-8-subgroup exponent bit
    byte 2 : E1_16 high 8 bits    per-4-subgroup exponent bit，块 0..7
    byte 3 : E1_16 low  8 bits    块 8..15
value : uint8 [M, N // 2]         相邻两个 4-bit code 打包为一个 byte
    每 nibble = (sign << 3) | code,  code = round(|in_grp| * 4) in 0..7
    低位优先：元素0放在低4位，元素1放在高4位
```

## 构建

```bash
# 先 source CANN set_env.sh（设置 ASCEND_HOME_PATH）
bash amct_ops/ops_build.sh hifloat4_cast                    # A2 (ascend910b)，仅此算子
bash amct_ops/ops_build.sh --soc ascend910_93 hifloat4_cast # A3
bash amct_ops/ops_build.sh                                  # 编译并打包全部算子为一个 wheel
pip install amct_ops/dist/amct_ops-*.whl
```

## 相对参考实现的改动

本实现基于 [HiFloat4](https://github.com/global-computing-consortium/HiFloat4)，在此基础上做了以下更新，其余算法部分保持一致：

**NPU 伪量化 kernel**（`hifloat4_cast_kernel.cpp`）：
- 增加 tail masking 修复：当 N 不是 BATCH=512 的整数倍时，从 BATCH-64 反向遍历清零所有尾元素（原本只重复清零最后一个 64-block，中间 stale 数据污染 block max）
- 其余与参考相同：3 级共享 scale（L1 E6M2 / L2 E1 8x / L3 E1 16x）+ S1P2 尾数，fp32 内部精度，bf16/E6M2 舍入用硬件 `vconv`

**Python 编解码**（`_hif4_encode_decode.py`）：
- **`keepdim` + 显式 `qdim`**：block max 归约使用 `amax(dim=qdim, keepdim=True)` 代替 `amax(dim=-1)` 链，匹配参考实现 —— 所有中间 tensor 保持同一 ndim，broadcast 自动处理，不需要手动 permute
- 在元素除掉块 scale、做完二三级缩放后，**量化成 S1P2 之前先做一次 `round_to_bf16`**
- 64-block 内出现 NaN / +Inf / -Inf 时：**整块置 NaN**（block-poison），打包 scale byte 0 写入 `0xFF`，解码时整块恢复为 NaN

**打包/解包**（`hif4_pack_unpack.py`）：
- 4-bit 打包采用**低位优先**：元素 0 放在低 4 bit，元素 1 放在高 4 bit
- scale 打包为 4 bytes：`[E6M2, E1_8, E1_16_hi, E1_16_lo]`
- 64-block 有非有限值时：scale byte 0 = `0xFF` → `hif4_bit_reconstruct` 将整块恢复为 NaN

## 状态 / 说明

- **NPU kernel**：fp32 下做块内归约与量化（bf16 / E6M2 舍入用硬件 `vconv`）。内核体由 `__DAV_C220_VEC__`
  守卫（A2/A3，dav-2201 = C220，覆盖 910B/910C）；其它架构为空实现，需在目标 NPU 上验证。
- **打包/解包**：Python 侧完成，与 NPU kernel 共享同一套编解码（`_hif4_encode_decode`），保证精度一致。
