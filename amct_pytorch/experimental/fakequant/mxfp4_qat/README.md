# MXFP4 QAT Linear（experimental）

一个**最小可用**的 MXFP4 量化感知训练（QAT）实现：`MXFP4QATLinear` 可直接替换 `torch.nn.Linear`，前向按 MXFP4 数值伪量化、反向通过 STE（straight-through estimator，直通估计器）把梯度回传到高精度主权重，从而让模型在训练阶段就适应 MXFP4 的量化误差。

本目录只提供**训练侧的量化算子与层**，不含训练框架。第 5 节说明如何把它接入你自己的框架（Megatron / DeepSpeed / HuggingFace Trainer 等）。

## 1. MXFP4 与 STE

**MXFP4** = 逐元素 FP4 E2M1 尾数 + 逐 block 的 E8M0（2 的幂）共享缩放：

- 最后一维每 `block_size`（默认 32）个相邻元素共享一个 scale；
- `scale = 2^round(log2(max_abs / scale_factor))`，`scale_factor` 默认 `6.0`；
- 元素码本为 `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}`（乘以 scale）。

量化是分段常量函数，几乎处处导数为 0，无法直接反传。**STE** 的做法是：前向用量化值、反向把量化算子当作恒等映射，因此梯度可以穿过量化点抵达高精度权重：

```
forward :  y = Q(x)
backward:  dL/dx = dL/dy            # 普通 STE
           dL/dx = dL/dy * (|x| <= 6*scale)   # clipped STE（clip_grad=True）
```

`clip_grad=True` 时会把**发生截断（saturation）**的元素梯度置零。因为 scale 被强制取整到 2 的幂（可能向下取整），block 内最大值有约一半概率超出 `6*scale` 而被截断，这些位置的梯度方向具有误导性，屏蔽后训练通常更稳。

> 实现上等价于 MindSpeed-LLM 的 `x + (x_q - x).detach()` 写法，本目录改用显式的
> `torch.autograd.Function`，便于在 `backward` 里做梯度屏蔽，也更直观。

## 2. 目录结构

```
mxfp4_qat/
├── fake_quant.py   # MXFP4 QDQ、STE autograd.Function、量化器模块、配置
├── linear.py       # MXFP4QATLinear + convert_to_mxfp4_qat
├── README.md
└── README_en.md
```

`fake_quant.py` 只依赖 `torch`，可以单文件拷进任意训练仓使用。

## 3. 快速开始

```python
import sys
sys.path.insert(0, ".../amct_pytorch/experimental/fakequant")

from mxfp4_qat import MXFP4QATConfig, convert_to_mxfp4_qat

model.load_state_dict(torch.load(ckpt))                       # 从 float 预训练权重出发
convert_to_mxfp4_qat(model, MXFP4QATConfig(quantize_input=True))
# 其余训练代码不变
```

## 4. API

```python
import sys
sys.path.insert(0, ".../amct_pytorch/experimental/fakequant")

from mxfp4_qat import MXFP4QATConfig, MXFP4QATLinear, convert_to_mxfp4_qat
```

### `MXFP4QATConfig`

| 字段 | 默认 | 说明 |
|------|------|------|
| `quantize_weight` | `True` | 是否伪量化权重 |
| `quantize_input` | `False` | 是否伪量化层输入。`False` → W4A16（推荐起点），`True` → W4A4 |
| `block_size` | `32` | 共享 scale 的元素数；Ascend-C 算子只支持 32 |
| `scale_factor` | `6.0` | 增大 → scale 变小，inlier 分辨率更高但截断更多；减小则相反 |
| `clip_grad` | `False` | `True` 使用 clipped STE |
| `backend` | `"auto"` | `"auto"`（NPU 上自动用 Ascend-C 算子，否则纯 PyTorch）/ `"torch"` / `"npu"` |

### `MXFP4QATLinear`

`nn.Linear` 的子类，前向为 `F.linear(Q(x), Q(W), b)`（bias 不量化）。

由于量化器无参数、无 buffer，**`state_dict` 与 float 层完全一致**：float 权重可以直接 load 进 QAT 模型，QAT 训练完的权重也可以 load 回 float 模型或交给 AMCT 的 deploy 流程导出。

```python
layer = MXFP4QATLinear(in_features, out_features, config=MXFP4QATConfig())
layer = MXFP4QATLinear.from_linear(existing_linear, config)   # 复用原 Parameter，不额外占显存
```

### `convert_to_mxfp4_qat(module, config=None, skip_names=())`

原地递归替换 `module` 下所有 `nn.Linear`。`skip_names` 按模块点分路径做**子串匹配**，命中则跳过该子树：

```python
convert_to_mxfp4_qat(
    model,
    MXFP4QATConfig(quantize_input=True, clip_grad=True),
    skip_names=("lm_head", "embed_tokens"),   # 敏感层保持 float
)
```

### 底层函数

```python
mxfp4_quant_dequant(x, block_size=32, scale_factor=6.0, backend="auto")  # 无梯度，与 mxfp4_ascendc 参考实现 bit-exact
mxfp4_fake_quant(x, block_size=32, scale_factor=6.0, clip_grad=False, backend="auto")  # 带 STE
mxfp4_saturation_mask(x, block_size=32, scale_factor=6.0)  # 截断位置掩码
MXFP4FakeQuantizer(block_size=32, scale_factor=6.0, clip_grad=False, backend="auto")  # nn.Module 形态
```

## 5. 接入自己的训练框架

### 方式一：模型里是标准 `nn.Linear`

建模完成、加载完预训练权重之后，`optimizer` 创建**之前**插入一行即可，其余训练代码不用改：

```python
model = build_model()
model.load_state_dict(torch.load(ckpt))          # 从 float 预训练权重出发

convert_to_mxfp4_qat(model, MXFP4QATConfig(quantize_input=True))

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
# ... 正常训练循环 ...
```

`from_linear` 复用原 `Parameter` 对象，所以在 optimizer 之后转换也不会失效；但放在 optimizer 之前更保险（避免 parameter group 引用悬空）。

### 方式二：框架有自定义 Linear（Megatron `ColumnParallelLinear` 等）

无法用继承替换时，把 `MXFP4FakeQuantizer` 作为量化器挂到层上，在 `forward` 里手动调用即可 —— 这正是 MindSpeed-LLM 的做法：

```python
from mxfp4_qat import MXFP4FakeQuantizer


class FakeQuantColumnParallelLinear(ColumnParallelLinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_quantizer = MXFP4FakeQuantizer()
        self.input_quantizer = MXFP4FakeQuantizer()

    def forward(self, input_, weight=None, **kwargs):
        input_ = self.input_quantizer(input_)
        # 父类 forward 读取 self.weight，因此临时替换 .data 后再恢复。
        # 前向已在量化值上完成，梯度经 STE 正确回传到原始高精度权重。
        original = self.weight.data
        self.weight.data = self.weight_quantizer(original)
        try:
            return super().forward(input_, weight=weight, **kwargs)
        finally:
            self.weight.data = original
```

MoE 的 GroupedMatmul 专家权重同理：在调用 GMM 之前对 `w1` / `w2` 和 permute 后的专家输入各过一次量化器。

### 方式三：只想复用量化算子

直接调用 `mxfp4_fake_quant(x)`，它就是一个可微的 `Tensor -> Tensor` 函数，放在任何位置（KV cache、logits、residual 等）都可以。

### 训练建议

- **从 float 预训练权重出发**做 QAT 微调，不要从随机初始化开始训。
- **先 W4A16 再 W4A4**：激活量化掉点通常明显大于权重量化，先确认 `quantize_input=False` 能收敛。
- **学习率取预训练的 1/10 左右**并配 cosine 衰减。
- **敏感层保持 float**：`lm_head`、embedding、第一/最后一层通常通过 `skip_names` 排除。
- **NPU 上务必用 Ascend-C 后端**：纯 PyTorch 路径每次 QDQ 有十几个 elementwise kernel，大模型训练开销不可忽略；`backend="auto"` 会在 NPU 张量上自动切换。
- 训练完成后用 `amct_pytorch` 的 deploy 流程导出真实低比特权重；QAT 只是让权重"适应"MXFP4，导出仍需常规量化链路。

### Ascend-C 算子的定位

`backend="auto"` 时按以下顺序查找 `mxfp4` 包：环境变量 `MXFP4_ASCENDC_PATH` → 同级 `../mxfp4_ascendc/python`。算子需先自行编译：

```bash
cd ../mxfp4_ascendc && bash build.sh
```

未编译或不在 NPU 上时自动退回纯 PyTorch 路径（结果 bit-exact 一致，仅速度不同）；显式指定 `backend="npu"` 而算子不可用时会抛出带修复指引的 `RuntimeError`。

## 6. 限制

- 属于试验特性（`experimental`），接口可能调整。
- 只覆盖 `nn.Linear`；卷积、Embedding、Attention 内部的 matmul 未处理。
- scale 与截断阈值均由数据静态推导，未实现可学习的 scale / clipping（LSQ、PACT 等）。
- `block_size != 32` 只有纯 PyTorch 路径支持。
- 伪量化仅复现 MXFP4 的数值行为，不代表目标硬件上真实低比特算子的性能。
