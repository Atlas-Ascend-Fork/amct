# Qwen3-8B SmoothQuant + HiFloat8 测试结果

## 测试对象

| 项 | 值 |
| --- | --- |
| 模型 | `Qwen3-8B` |
| 数据集 | `wikitext-2-raw-v1/test`，从本地 HuggingFace cache 读取 |
| 量化配置 | `amct.HIFP8_SMOOTHQUANT_CFG`，测试 `smooth_strength=0.5` 和 `0.8` |
| 设备 | `npu:0` |
| dtype | `torch.bfloat16` |
| PyTorch | `2.10.0+cpu` |
| torch_npu | `2.10.0` |

## 验证项

- PR 新增的 hifloat8 SmoothQuant 单测通过：
  - `test_hif8_hif8_channel_tensor_sym_smooth_success`
  - `test_hif8_hif8_token_sym_smooth_success`
- Qwen3-8B 上成功执行 `amct.quantize(model, amct.HIFP8_SMOOTHQUANT_CFG)`。
- wikitext2 校准后，模型中插入 `252` 个 `SmoothQuant` 层。
- 完成量化前后 PPL 全量 146 段对比。

## 量化配置

本次使用 PR 176 新增导出的 `amct.HIFP8_SMOOTHQUANT_CFG`。基础配置如下，
分别测试 `smooth_strength=0.5` 和 `smooth_strength=0.8`：

```python
{
    "batch_num": 1,
    "quant_cfg": {
        "weights": {
            "type": "hifloat8",
            "symmetric": True,
            "strategy": "channel",
        },
        "inputs": {
            "type": "hifloat8",
            "symmetric": True,
            "strategy": "tensor",
        },
    },
    "algorithm": {
        "smoothquant": {
            "smooth_strength": 0.5,  # 另测 0.8
        },
    },
    "skip_layers": {"lm_head"},
}
```

## PPL 结果

### 2048-token 全量评估

| 项 | 值 |
| --- | ---: |
| `seqlen` | `2048` |
| `max_samples` | `None` |
| 评估段数 | `146` |
| baseline PPL | `9.7251768112` |
| baseline 评估耗时 | `28.8s` |

| smooth_strength | hifloat8 smoothquant PPL | PPL delta | 相对劣化 | 量化与校准耗时 | 量化后评估耗时 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `0.5` | `9.7718954086` | `+0.0467185974` | `+0.4804%` | `1.8s` | `642.3s` |
| `0.8` | `9.7356834412` | `+0.0105066299` | `+0.1080%` | `2.8s` | `641.9s` |

### 4096-token 全量评估

| 项 | 值 |
| --- | ---: |
| `seqlen` | `4096` |
| `max_samples` | `None` |
| 评估段数 | `73` |
| baseline PPL | `8.9974622726` |
| baseline 评估耗时 | `27.5s` |

| smooth_strength | hifloat8 smoothquant PPL | PPL delta | 相对劣化 | 量化与校准耗时 | 量化后评估耗时 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `0.5` | `9.0287866592` | `+0.0313243866` | `+0.3481%` | `2.2s` | `637.8s` |
| `0.8` | `9.0359315872` | `+0.0384693146` | `+0.4276%` | `2.0s` | `637.3s` |

## 结论

`HIFP8_SMOOTHQUANT_CFG` 在 Qwen3-8B + wikitext2 全量 146 段上可以跑通。
2048-token 全量评估中，`smooth_strength=0.8` 的精度优于默认 `0.5`，
相对劣化约 `0.1080%`；4096-token 全量评估中，`smooth_strength=0.5`
略优于 `0.8`，相对劣化约 `0.3481%`，几乎无损。
