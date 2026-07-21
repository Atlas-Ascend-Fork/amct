# hifloat4_cast — HiFloat4 FP→HiF4→FP 仿真算子

FP16/BF16 → HiFloat4 → FP16/BF16 仿真（fake-quant），面向昇腾 NPU。HiFloat4（hifx4）是 4-bit
块缩放浮点：每元素 **S1P2**（1 符号 + 2 位尾数，8 档幅值），再配**每 64 元素共享的 3 级 scale**
（L1 `scale_factor` E6M2 覆盖 64 块、L2 覆盖每 8、L3 覆盖每 4）。本算子做一次 FP→FP 往返
（**不真打包 4-bit**），是 `amct_pytorch` 中 `hifloat4_fake_quant` 的 NPU 后端，用于离线精度 / PPL 评估。

与 `hifloat8_cast` 不同，**没有查找表（LUT）**：scale 是块内归约（算术），S1P2 只有 8 档幅值，
LUT 无收益。

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

kernel 逐步复刻纯 python 参考，逐元素对齐到 ~0 误差。在设备上对拍 amct 的 CPU 参考：

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

## 构建

```bash
# 先 source CANN set_env.sh（设置 ASCEND_HOME_PATH）
bash amct_ops/ops_build.sh hifloat4_cast                    # A2 (ascend910b)，仅此算子
bash amct_ops/ops_build.sh --soc ascend910_93 hifloat4_cast # A3
bash amct_ops/ops_build.sh                                  # 编译并打包全部算子为一个 wheel
pip install amct_ops/dist/amct_ops-*.whl
```

## 状态 / 说明

- 仅仿真。真 4-bit 打包 codec + 低比特 matmul 为后续工作。
- kernel 在 fp32 下做块内归约与量化（bf16 / E6M2 舍入用硬件 `vconv`）。内核体由 `__DAV_C220_VEC__`
  守卫（A2/A3，dav-2201 = C220，覆盖 910B/910C）；其它架构为空实现，需在目标 NPU 上验证。
