# hifloat4_cast — HiFloat4 量化算子

HiFloat4（hifx4）是 4-bit 块缩放浮点格式：每元素 **S1P2**（1 符号 + 2 位尾数，8 档幅值），配**每 64 元素共享的 3 级 scale**（L1 `scale_factor` E6M2 覆盖 64 块、L2 覆盖每 8、L3 覆盖每 4）。

## 架构

```
┌─────────────────────────────────────────────────────────┐
│  CLI (amct_pytorch/cli/llm/)                            │
│  --quant_dtype hifp  →  bits=4 (HiF4) / bits=8 (HiF8)   │
│    ├─ PTQ:    fake-quant (注入 HiF4 误差)                │
│    └─ Deploy: blockwise → HiF4 打包字节                  │
│              tensorwise → fp32 解包                      │
├─────────────────────────────────────────────────────────┤
│  Python (amct_pytorch/quantization/dtypes/hifp_impl.py)  │
│  ├─ hif4_pack()              float → HiF4 uint8 打包     │
│  ├─ hif4_unpack()            HiF4 uint8 → fp32 解包      │
│  └─ hifloat4_fake_quant()    NPU kernel / CPU fallback   │
├─────────────────────────────────────────────────────────┤
│  NPU Kernel (op_kernel/hifloat4_cast_kernel.cpp)         │
│  └─ hifx_kernel / hifx_kernel_bf16    FP→HiF4→FP 往返   │
│     （仅 fake-quant；打包/解包在 Python 侧完成）            │
└─────────────────────────────────────────────────────────┘
```

- **1 个 NPU kernel**：`hifx_kernel`（fp32）/ `hifx_kernel_bf16`（bf16），做一次 FP→HiF4→FP 仿真往返
- **打包/解包**：纯 PyTorch 实现（`hifp_impl.py`），`hif4_pack` 编码为 uint8 HiF4 字节，`hif4_unpack` 解码回 fp32
- **共享编解码**：fake-quant 和 pack 共用同一套 `hif4_encode` / `hif4_decode`，保证评估路径和部署权重永远一致

## CLI 工作流

```bash
# PTQ: fake-quant 仿真（注入 HiF4 误差，不产出文件）
python -m amct_pytorch.cli.llm.ptq \
    --model /path/to/model --quant_dtype hifp --w_bits 4 --output_dir ./output

# Deploy blockwise: fp16/bf16 → HiF4 打包字节
python -m amct_pytorch.cli.llm.deploy \
    --model /path/to/model --quant_dtype hifp --granularity block --output_dir ./output

# Deploy tensorwise: HiF4 打包权重 → fp32 解包
python -m amct_pytorch.cli.llm.deploy \
    --model /path/to/model --quant_dtype hifp --granularity tensor --output_dir ./output
```

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

kernel 逐步复刻纯 python 参考，逐比特对齐到 ~0 误差。在设备上逐比特比较 AMCT 的 CPU 参考：

```python
import torch
from amct_ops.hifloat4_cast import hifloat4_fake_quant
from amct_pytorch.quantization.dtypes.hifp_impl import hif4_pack, hif4_unpack

# pack/unpack 往返
x = torch.randn(2, 128, dtype=torch.bfloat16)
scale, value = hif4_pack(x)
y = hif4_unpack(scale, value)
assert y.shape == x.shape

# NPU kernel vs CPU fallback 逐比特比较（~0 误差）
y_npu = hifloat4_fake_quant(x.npu()).cpu()
assert (y_npu.float() - y).abs().max() < 1e-6
```

`tests/amct_ops/test_hifloat4_cast.py` 在有 NPU 时逐比特比较 kernel 与该 CPU 参考。

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

## 状态 / 说明

- **NPU kernel**：fp32 下做块内归约与量化（bf16 / E6M2 舍入用硬件 `vconv`）。内核体由 `__DAV_C220_VEC__`
  守卫（A2/A3，dav-2201 = C220，覆盖 910B/910C）；其它架构为空实现，需在目标 NPU 上验证。
- **打包/解包**：Python 侧完成，与 NPU kernel 共享同一套编解码（`_hif4_encode_decode`），保证精度一致。
