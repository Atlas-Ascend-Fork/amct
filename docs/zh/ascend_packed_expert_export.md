# Ascend Tensorwise Packed Expert 导出说明

## 背景

部分 MoE 模型的 BF16 checkpoint 不使用逐专家权重名，而是把路由专家保存为 packed tensor：

```text
model.layers.0.mlp.experts.gate_up_proj
model.layers.0.mlp.experts.down_proj
```

典型形状为：

```text
gate_up_proj: [num_experts, 2 * moe_intermediate_size, hidden_size]
down_proj:    [num_experts, hidden_size, moe_intermediate_size]
```

而 vLLM-Ascend 的 W8A8_DYNAMIC checkpoint 使用逐专家、二维权重：

```text
model.layers.0.mlp.experts.0.gate_proj.weight
model.layers.0.mlp.experts.0.up_proj.weight
model.layers.0.mlp.experts.0.down_proj.weight
```

因此 tensorwise deploy 必须在导出过程中完成两次一致的名称和 tensor 展开：

1. 在匹配 JSON 量化方案前，展开 index 中的虚拟权重名。
2. 读取每个 shard 后，把真实三维 packed tensor 切分成逐专家二维 tensor。

## 公共实现

公共实现位于：

```text
amct_pytorch/common/models/llm/common/packed_expert_export.py
```

核心接口：

```python
detect_packed_gated_expert_layout(weight_map, config)
PackedGatedExpertLayout.expand_weight_map(weight_map)
PackedGatedExpertLayout.expand_tensors(state_dict, selected_weight_keys)
```

检测函数根据 checkpoint index 和模型 config 自动识别标准 packed gated-expert 格式。当前支持以下配置字段作为专家数量来源：

```text
num_experts
n_routed_experts
num_local_experts
```

中间维优先读取：

```text
moe_intermediate_size
intermediate_size
```

如果 index 中没有同时出现 `gate_up_proj` 和 `down_proj`，则认为不是当前标准格式；如果只出现其中一个，会直接报错，避免静默导出错误 checkpoint。

## 调用链

### 计划阶段

```text
deploy.py:main()
  -> LlmDeployWorkflow.run()
    -> _run_tensorwise()
      -> _prepare_ascend_deploy_plan()
        -> detect_packed_gated_expert_layout()
        -> layout.expand_weight_map()
        -> build_ascend_deploy_plan()
          -> match_deploy_modules()
          -> adapter.get_ascend_deploy_candidates()
          -> adapter.validate_ascend_deploy_selection()
```

展开后的虚拟 index 只用于计划匹配，不会改写源模型的 `model.safetensors.index.json`。

Ascend 量化配置使用逐专家模块名：

```json
{
  "quant_layers": [
    "model.language_model.layers.*.mlp.experts.*.gate_proj",
    "model.language_model.layers.*.mlp.experts.*.up_proj",
    "model.language_model.layers.*.mlp.experts.*.down_proj"
  ]
}
```

### shard 导出阶段

```text
_run_tensorwise()
  -> _convert_tensorwise_shard()
    -> safetensors.load_file()
    -> layout.expand_tensors()
    -> 遍历逐专家二维 tensor
    -> _export_tensor_payload()
      -> quant_payload()
      -> adapt_ascend_payload()
    -> _write_safetensor_file()
```

`gate_up_proj` 沿第二维从中间切分：前半部分是 `gate_proj`，后半部分是 `up_proj`。`down_proj` 只沿第一维按专家切分。

## 共享存储处理

packed tensor 的切片可能共享原始 storage。选中量化的权重后续会被 INT8 tensor 替换，因此可以暂时保留切片；没有被量化、仍以 FLOAT 写出的切片必须 clone，避免 safetensors 保存共享 storage 时失败。

`expand_tensors()` 通过 `selected_weight_keys` 区分两类 tensor：选中的二维权重保留 view，未选中的二维权重 clone 后写出。这也替代了原先只针对 `mtp.` 前缀的特殊 clone 逻辑。

## 模型 adapter 的职责

Qwen3.6 adapter 不再实现 packed checkpoint 的展开函数。它仍然负责模型本身的部署约束，例如：

- 哪些 attention 层允许量化。
- QKV 是否必须成组选择。
- 某层的全部专家和三个 projection 是否必须成组选择。
- MTP 模块是否必须排除。

公共转换器负责 checkpoint 表示转换，adapter 负责模型语义校验，两者边界保持分离。

## 校验与错误

真实 tensor 展开时会校验：

```text
tensor.ndim == 3
tensor.shape[0] == num_experts
gate_up_proj.shape[1] 为偶数
gate_up_proj.shape[1] == 2 * moe_intermediate_size（配置存在时）
down_proj.shape[2] == moe_intermediate_size（配置存在时）
```

不满足时会在当前 shard 量化前抛出 `ValueError`，不会生成不完整的量化 payload。

## 扩展边界

当前公共实现针对标准布局：

```text
experts.gate_up_proj + experts.down_proj
```

如果后续模型使用不同的 packed 命名或不同的切分语义，应新增独立 layout，而不是在 `llm_deploy.py` 中添加模型名分支。workflow 只需要调用检测出的 layout，adapter 也不需要重新实现 tensor 展开。
