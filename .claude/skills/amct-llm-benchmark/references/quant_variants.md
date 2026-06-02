# 量化变体扩展指南

`scripts/ppl_eval.py --quant <name>` 用 `get_quant_wrapper()` 路由到具体包装器。新增量化方式只要写一个 `wrap_model_xxx(model, prefix="") -> int` 函数，原地替换 Linear，返回替换数量。本文档给出 4 种常见变体的接入模式。

## 通用模板

所有 wrapper 共享同一签名和递归结构。差别只在 Linear 替换类的实现：

```python
import torch
import torch.nn as nn

class FakeQuantLinearXXX(nn.Module):
    def __init__(self, ori: nn.Linear):
        super().__init__()
        # 1) 算 scale（per-tensor / per-channel / per-group）
        # 2) 一次性把权重 fake-quant 到目标格式
        # 3) self.register_buffer("fakequant_weight", ...)
        # 4) self.bias = ori.bias

    @torch.no_grad()
    def forward(self, x):
        # 激活 fake-quant（如有）+ F.linear
        ...

def wrap_model_xxx(model: nn.Module, prefix: str = "") -> int:
    n = 0
    for name, child in list(model.named_children()):
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear) and full.startswith("model.layers."):
            setattr(model, name, FakeQuantLinearXXX(child))
            n += 1
        else:
            n += wrap_model_xxx(child, full)
    return n
```

注册到 `ppl_eval.py:get_quant_wrapper`：

```python
if name == "xxx":
    from xxx_wrap import wrap_model_xxx
    return wrap_model_xxx
```

## 1. HiFloat8 W8A8 Cast（已实现，作基准）

见 `scripts/hif8_w8a8_wrap.py`。最简单的 HiFloat8 W8A8：权重 `scale_w = max/16`，激活直转。Qwen3-8B FP16 → W8A8 PPL +0.04，作为下面变体的对照点。

## 2. INT8 W8A8（对称 per-tensor 或 per-channel）

```python
class Int8FakeQuantLinear(nn.Module):
    def __init__(self, ori, w_per_channel=True):
        super().__init__()
        w = ori.weight.detach()
        if w_per_channel:
            # per-output-channel scale
            scale_w = w.abs().amax(dim=1, keepdim=True) / 127.0
        else:
            scale_w = w.abs().max() / 127.0
        q_w = (w / scale_w).round().clamp(-128, 127)
        self.register_buffer("fakequant_weight", (q_w * scale_w).to(w.dtype))
        self.bias = ori.bias

    @torch.no_grad()
    def forward(self, x):
        # 激活 per-tensor 静态量化需要先标定，简化版用 dynamic
        s_a = x.abs().max() / 127.0
        q_x = (x / s_a).round().clamp(-128, 127)
        fq_x = (q_x * s_a).to(x.dtype)
        return F.linear(fq_x, self.fakequant_weight, self.bias)
```

注意 INT8 静态激活量化要先跑标定 batch 收集 scale，本 skill 是 zero-shot 跑 PPL，所以这里给的是 dynamic 版本（每 batch 重算激活 scale），不适合做部署对比，只适合精度上界估算。要做部署对应的静态 INT8，走 `amct_pytorch.quantize` 主流程。

## 3. W4A16（仅权重量化）

```python
class W4A16FakeQuantLinear(nn.Module):
    def __init__(self, ori, group_size=128):
        super().__init__()
        w = ori.weight.detach()
        out, in_ = w.shape
        assert in_ % group_size == 0
        w_g = w.view(out, in_ // group_size, group_size)
        # per-group symmetric, max -> 7
        scale = w_g.abs().amax(dim=-1, keepdim=True) / 7.0
        q = (w_g / scale).round().clamp(-8, 7)
        fq = (q * scale).view(out, in_)
        self.register_buffer("fakequant_weight", fq.to(w.dtype))
        self.bias = ori.bias

    @torch.no_grad()
    def forward(self, x):
        return F.linear(x, self.fakequant_weight, self.bias)
```

`group_size=128` 是 GPTQ/AWQ 论文默认；group=64 / 32 精度更好但开销大。**W4A16 不量化激活，只量化权重**，所以速度收益在内存带宽侧，PPL 通常比 W8A8 略差但仍可用。

## 4. AWQ / GPTQ

这两个需要**校准数据 + 优化过程**，不能直接做"加载即量化"。流程是：

1. 加载 FP16 模型
2. 跑 N 个校准 batch（典型 N=128，wikitext 或 pile）
3. 算每层的 importance（AWQ 用激活幅值，GPTQ 用 Hessian 对角）
4. 优化 scale / clip，得到量化后权重
5. 替换 Linear 为 W4A16/W8A16 的量化版本

接入本 skill 的方式：**先跑 AMCT 主流程做量化保存，得到 `convert` 后的模型，然后 `--model_path` 指向量化模型路径直接跑 PPL**。不要在 PPL 脚本里嵌入校准过程 —— 那是 `amct_pytorch.quantize/convert` 的职责。

如果只想快速看 AWQ/GPTQ 精度，参考 `examples/algorithms/awq/` 和 `examples/algorithms/gptq/` 的端到端脚本。

## 选型快查表

| 量化方式 | 校准数据 | 推理开销 | Qwen3-8B 预期 PPL Δ |
|---|---|---|---|
| HiFloat8 Cast（W8A8） | 否 | 小（NPU cast 算子） | +0.04 ~ +0.10 |
| INT8 W8A8 dynamic | 否 | 中（每次激活 max） | +0.1 ~ +0.5 |
| INT8 W8A8 static | 需 | 小 | +0.05 ~ +0.3 |
| W4A16 group=128 | 否 | 小 | +0.2 ~ +1.0 |
| AWQ W4A16 | 需 | 小 | +0.05 ~ +0.2 |
| GPTQ W4A16 | 需 | 小 | +0.05 ~ +0.2 |

具体数字与模型相关；上表是 Qwen3-8B 量级的经验估计。
