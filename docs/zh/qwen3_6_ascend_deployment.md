# Qwen3.6 Ascend 动态 INT8 权重导出

本功能从原始 BF16 checkpoint 导出 W8A8_DYNAMIC 权重，目标部署框架为 vLLM-Ascend v0.23.0。导出采用 tensorwise 模式，逐个 safetensors 分片处理，不执行测评、校准、PTQ 或旋转。

## 输入与环境

需要可运行 AMCT 的 Python 环境、完整原始 BF16 checkpoint、足够的主机内存和独立磁盘空间。导出计算使用 CPU，不需要 NPU。可以通过 `PYTHON_BIN` 指定已有环境的 Python 解释器。

输入 checkpoint 必须满足以下条件：

- 包含完整的 `config.json`、`model.safetensors.index.json` 和索引引用的全部分片。
- 索引中的分片值必须是当前模型目录下的文件名，不能包含子目录或绝对路径。
- 权重必须已经无损转换为 BF16，不支持将 MXFP 权重直接作为本流程输入。
- 不能包含 `rot.weight` 或以 `.rot.weight` 结尾的旋转权重。
- 输出目录必须不存在或为空，不能与输入目录相同，也不能与输入目录互相嵌套。

当前适配基于 40 个语言层、每层 256 个 routed experts、每 4 层一个 full-attention 层以及 1 个 MTP 层的配置。若后续模型配置发生变化，导出逻辑会从 `num_hidden_layers`、`num_experts` 和 `layer_types` 读取实际值，但仍应先核对量化规则是否符合目标模型。

## 导出命令

在仓库根目录执行，路径建议使用绝对路径：

```bash
export QWEN36_BF16_DIR=/data/Qwen3.6-BF16
export QWEN36_ASCEND_DIR=/data/Qwen3.6-AMCT-W8A8
export PYTHON_BIN=/path/to/venv/bin/python
bash examples/deploy_qwen3_6_ascend.sh
```

样例脚本固定使用以下参数：

```text
--model_name qwen3_6_moe
--device cpu
--granularity tensor
--quant_dtype int
--bit_config amct_pytorch/configs/w8a8.yaml
--deploy_format ascend
--quant_layers_config amct_pytorch/configs/deploy/qwen3_6_ascend_w8a8.json
```

`--deploy_format` 的默认值仍为 `legacy`，原有 legacy 导出路径不受影响。只有显式指定 `ascend` 时才生成 vLLM-Ascend 使用的动态量化描述和辅助参数。

## 量化范围

默认规则位于 `amct_pytorch/configs/deploy/qwen3_6_ascend_w8a8.json`。本方案量化：

- 所有语言层的 routed experts：`gate_proj`、`up_proj` 和 `down_proj`。
- `layer_types=full_attention` 的语言层中的 `self_attn.q_proj`、`k_proj` 和 `v_proj`。

以下部分保持 FLOAT：

- linear-attention 层及其投影。
- full-attention 层的 `self_attn.o_proj`。
- shared expert、router gate 和 shared expert gate。
- embedding、最终 norm 和 `lm_head`。
- 视觉塔的全部参数。
- MTP 的全部参数。

Qwen3.6 的源索引将每层 routed experts 保存为 packed tensor：

```text
model.language_model.layers.<layer>.mlp.experts.gate_up_proj
model.language_model.layers.<layer>.mlp.experts.down_proj
```

导出时会按照 `num_experts` 展开成 vLLM checkpoint 使用的独立专家名称：

```text
model.language_model.layers.<layer>.mlp.experts.<expert>.gate_proj.weight
model.language_model.layers.<layer>.mlp.experts.<expert>.up_proj.weight
model.language_model.layers.<layer>.mlp.experts.<expert>.down_proj.weight
```

`gate_up_proj` 沿中间维均分为 gate 和 up，`down_proj` 沿专家维拆分。任意一层只选择部分专家或只选择 gate/up/down 中的一部分都会报错；full-attention 的 Q/K/V 也必须成组选择。

## 自定义量化规则

如需调整范围，可直接调用 deploy CLI，并将 `--quant_layers_config` 指向自定义 JSON。格式如下：

```json
{
  "quant_layers": [
    "model.language_model.layers.*.self_attn.q_proj"
  ],
  "ignore_layers": [
    "model.language_model.layers.*.self_attn.o_proj"
  ]
}
```

规则匹配不带 `.weight` 的完整模块名，大小写敏感，支持 `*`、`?` 等 fnmatch 通配符。`ignore_layers` 优先，未选中的模块保持 FLOAT。任意规则未命中、最终没有模块被选中、选择 linear-attention QKV、部分 Q/K/V 或部分 routed experts 时，导出都会失败。Ascend 模式下 JSON 选择结果覆盖 `quant_target`，两者不求交集。

## 输出产物

成功导出后，输出目录包含：

| 文件 | 含义 |
|---|---|
| `*.safetensors` | 沿用输入分片名；选中 weight 为 INT8，`weight_scale`/`weight_offset` 为 BF16 `[out, 1]` |
| `model.safetensors.index.json` | 所有输出 tensor 到分片的映射，`total_size` 为实际 tensor 数据字节数 |
| `quant_model_description.json` | 量化 weight/scale/offset 标记为 `W8A8_DYNAMIC`，其他 tensor 标记为 `FLOAT` |
| `config.json`、tokenizer 等 | 从原始 checkpoint 按字节复制的支持文件 |
| `deployment_validation.json` | 描述、索引、dtype、shape 和辅助参数的一致性检查报告 |
| `logs/deploy.log` | 默认导出日志 |

量化模块会生成三类 tensor：

```text
<module>.weight         INT8
<module>.weight_scale   BF16 [out, 1]，有限且大于 0
<module>.weight_offset  BF16 [out, 1]，值为 0
```

`quant_model_description.json` 固定包含：

```json
{
  "version": "1.0.0",
  "model_quant_type": "W8A8_DYNAMIC",
  "metadata": {},
  "group_size": 0,
  "is_rot_used": false
}
```

该描述不写入 `optional`，也不生成 input scale、input offset、quant bias、deq scale 或 rot 参数。vLLM-Ascend 会通过 `quant_model_description.json` 自动识别 ModelSlim/Ascend 量化格式。

导出结束时会执行静态验证。`deployment_validation.json` 中 `status=passed` 只表示文件级契约通过，不代表 NPU 服务能够成功加载或推理。失败输出不会自动续写；保留现场用于诊断后，应为下一次运行指定新的空目录。

## vLLM-Ascend 服务验收

以下命令只给出关键参数。TP/EP 拓扑、硬件数量、HCCL、多机网络、最大序列长度和显存占用必须根据实际机器配置。首次验证建议先关闭 MTP：

```bash
export QWEN36_ASCEND_DIR=/data/Qwen3.6-AMCT-W8A8
export QWEN36_TP_SIZE=16

vllm serve "${QWEN36_ASCEND_DIR:?}" \
  --served-model-name qwen3.6-amct \
  --quantization ascend \
  --tensor-parallel-size "${QWEN36_TP_SIZE:?}" \
  --enable-expert-parallel \
  --max-model-len 4096 \
  --trust-remote-code
```

服务启动后执行最小请求：

```bash
curl http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-amct",
    "prompt": "The future of AI is",
    "max_tokens": 32,
    "temperature": 0
  }'
```

基础加载和请求通过后，再停止服务并增加 MTP 参数进行对比验收：

```bash
--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'
```

这里的方法名使用 `qwen3_5_mtp`，因为 checkpoint 的底层架构仍是 Qwen3.5 MoE。需要分别保存 MTP 关闭和开启时的完整启动命令、加载日志及请求结果，并检查：

- 没有缺失专家权重、辅助参数、MTP 权重或维度不匹配。
- 量化 Linear/MoE 选择了 W8A8_DYNAMIC 实现，FLOAT 层没有被错误识别为量化层。
- 开启 MTP 时实际加载并执行 speculative 路径，没有静默回退或禁用。
- 记录 vLLM、vLLM-Ascend、PyTorch、CANN 和驱动版本，以及 TP/EP 拓扑。

## 当前验收状态

| 项目 | 状态 |
|---|---|
| Qwen3.6 模型注册、JSON 选层和 packed expert 展开单元测试 | 已通过 |
| Ascend 参数门禁及 W8A8_DYNAMIC 描述生成测试 | 已通过 |
| 导出样例脚本参数转发及 Bash 语法检查 | 已通过 |
| 完整原始 BF16 checkpoint 导出 | 待真实权重环境执行 |
| vLLM-Ascend v0.23.0，MTP off/on 加载和推理 | 待 NPU 环境执行 |
