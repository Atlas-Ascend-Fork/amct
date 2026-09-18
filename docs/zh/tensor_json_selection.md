# Tensor 导出的 JSON 选层

## 范围

`deploy --granularity tensor` 在 legacy 和 ascend 格式下均使用 JSON 作为唯一选层来源。
本次支持 INT W8A8；显式 W4、16 bit 配置直接失败。block、PTQ、eval 不使用这套新规则。
模型旧 `generate_tensorwise_quant_layers()` / `generate_tensorwise_ignore_layers()`
保留作为独立对照基准，正式 tensor 导出不会调用。

Ascend 参数校验不限制模型名称；导出预检要求模型 adapter 实现候选层和融合约束校验
接口。目前仅 GLM-5.2 提供这些接口，其他模型需要完成对应适配后才能导出 Ascend 格式。

legacy 通过 `_refresh_config()` 更新 `config.json` 的量化元数据。tensor 分支使用
JSON 解析后的实际计划，将用户 ignore 与其他未量化权重模块合并为元数据 ignore，
不调用模型的 `bits_scheme()`。Ascend 保留源 `config.json`，通过独立的
`quant_model_description.json` 描述量化状态。

## 配置

```yaml
# amct_pytorch/configs/deploy/tensor_w8a8.yaml
attn-linear:
  w_bits: 8
  a_bits: 8
mlp:
  w_bits: 8
  a_bits: 8
moe:
  w_bits: 8
  a_bits: 8
```

省略未使用的分组即可。每个选中模块必须找到明确配置的 a/w 位宽；不使用隐式
16 bit 或顶层默认值代替模块配置。兼容读取显式顶层 W8A8 字段，但不能仅靠它
完成选中模块的位宽解析。字段名为 `w_bits`、`a_bits`，不能写成 `w_bit`。

位宽错误在构建模型和写输出前报出，包括未选中分组中出现 W4、被更细项覆盖
的 W4、16 bit、拼写错误、不完整 a/w 对。匹配实际名称后检查选中模块是否缺失策略。
本次 tensor 路径不接受 `attn-cache` 位宽项，也不支持纯反量化免选层分支。

```json
{
  "quant_layers": ["model.layers.*.self_attn.o_proj"],
  "ignore_layers": ["model.layers.78.*"]
}
```

JSON 使用 checkpoint 完整模块名，不包含 `.weight` 后缀。区分大小写，支持
`*`、`?`、`[abc]` 等 fnmatch glob，`*` 可以跨越点号。JSON 不支持 `re:` 正则。
重复规则去重；ignore 优先。quant 规则零命中或最终量化集合为空时严格报错；
ignore 规则零命中时打印警告，并在计划的 `unmatched_ignore_patterns` 中单独记录。
ignore 可以匹配 norm、head 等非量化模块。

实际量化仅处理选中模块的 `.weight`；bias 和其他辅助参数不会因此被量化。
未选中权重不执行目标量化；源 checkpoint 若需解码，其处理仍沿用源格式逻辑。
`--quant_target` 在 tensor 路径不参与选择，传入时会打印提示。

### 位宽角色映射

| checkpoint 路径 | BitPolicy 分组 |
|---|---|
| `*.self_attn.*` | `attn-linear` |
| `*.mlp.experts.<数字>.*` | `moe.routed` |
| `*.mlp.shared_experts.*` / `*.mlp.shared_mlp.*` | `moe.shared` |
| 其他 `*.mlp.*` | `mlp` |

分组下 projection 名称沿用最后一级模块名，例如 `o_proj`。`moe.routed/shared`
可继承 `moe` 配置。无法映射角色时明确报错；本次没有声称任意自定义模型路径都能识别。
JSON 名称命中不代表后端支持；Ascend 仍独立校验候选层、融合配对和专家一致性。

## 三份方案的用途

| JSON | 用途 |
|---|---|
| `glm5_2_tensor_w8a8.json` | 复现 GLM-5.2 旧 INT generate 方案 |
| `hyv3_tensor_w8a8.json` | 复现 Hy3 全部启用 attn-linear/mlp/moe 的旧方案 |
| `glm5_2_ascend_w8a8.json` | 保留用户已实机验证的 Ascend 方案 |

以上文件位于 `amct_pytorch/configs/deploy/`。前两份配置不要求通过 vllm-ascend
服务验证，也不要求与第三份相同。GLM 旧方案保留部分 MTP 量化，并忽略 q_a/kv_a；
现有 Ascend 方案量化 q_a/kv_a、显式排除第 78 层。

配置主样例的结构前提：GLM 包含 dense、MoE、indexer、MTP；Hy3 首层 dense，
后续 MoE，且有一层 MTP。结构不同需调整 quant 规则，避免零命中；ignore 零命中会告警。Hy3 子集方案通过
删除不需要的 quant 规则并将对应模块放入 ignore 来表达，不靠 quant_target 开关。

### 已识别的旧接口清单差异

本地完整 GLM-5.2 索引与旧接口对照：quant 两侧均为 58,785 且集合一致；旧
ignore 为 360，checkpoint 可解析为 359，唯一缺失项为
`model.layers.78.shared_head.head`。JSON 保留这条旧规则；经用户确认，ignore
零命中仅告警。完整索引回归显式核对该唯一缺失项及对应零命中规则，实际存在的
359 个 ignore 模块完全一致，不将不存在的模块计入已命中集合。

## 使用命令

以下为符合对应结构前提的 checkpoint 使用方式。GLM 旧方案中的缺失 ignore
项按上节说明打印告警，不阻断导出。

```bash
python amct_pytorch/cli/llm/deploy.py \
  --model /data/Hy3 \
  --model_name hy_v3 \
  --device cpu \
  --granularity tensor \
  --quant_dtype int \
  --deploy_format legacy \
  --bit_config amct_pytorch/configs/deploy/tensor_w8a8.yaml \
  --quant_layers_config amct_pytorch/configs/deploy/hyv3_tensor_w8a8.json \
  --output_dir /data/Hy3-w8a8
```

GLM 旧方案对应 `--model_name glm5_2` 和 `glm5_2_tensor_w8a8.json`。
需要 Ascend 格式时使用原 `examples/deploy_glm5_2_ascend.sh`，其 JSON 不变，
位宽文件切换为模块级 `tensor_w8a8.yaml`。

## MTP 状态

从模型 config 和真实 checkpoint 模块名识别 MTP；日志不会修改最终选层集合。

| 状态 | 含义 |
|---|---|
| `not_present` | checkpoint 未识别到 MTP 模块 |
| `fully_ignored` | 所有 MTP 模块被 ignore 命中 |
| `not_selected` | MTP 未被选中，但没有显式 ignore |
| `partially_ignored` | 部分 MTP 被 ignore；同时打印剩余选中数量 |
| `selected` | 存在选中 MTP 且没有 ignore |

日志包含 modules/ignored/selected/quantized 数量及最多 20 个选中模块与位宽样例。
`planned` 是计划状态，不代表导出完成；`exported` 在实际权重量化及产物检查后输出。
MTP 未写 ignore 本身只产生提示；Ascend 当前适配不支持 MTP 量化，若最终选中则报错，
由用户通过 JSON 排除。legacy 可以保留旧接口中已有的 MTP linear 选择。

## 验收

- 两模型直接比较 JSON 的 `{module: bit}` 与旧接口字典。
- ignore 在同一有参数模块清单上展开，比较集合；旧 Hy3 `re:` 由测试 oracle
  独立按正则处理，不把无 tensor 的运行时 experts 容器当作 checkpoint 模块。
- 独立结构 fixture 覆盖 GLM indexer/dense/MoE/MTP 变化、Hy3 七个非空 target 组合。
- CPU 小分片导出验证真实 INT8/scale、bias 保留、未选中权重、索引和元数据。
- legacy 元数据的完整 ignore 包括未选中 weight 模块，与用户规则解析的 ignore
  分开维护，不能将两者混为旧接口等价结果。
- 现有 Ascend JSON 与本地完整索引/参考描述对照，仍选中 58,167 个模块。

用户提供过原 Ascend JSON 的实机成功记录，本次代码未重新执行 NPU 加载或精度评测。
本次不以完整模型导出或启动 vllm-ascend 作为 JSON 选层等价验收条件。
