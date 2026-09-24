# GLM-5.2 Ascend 动态 INT8 权重导出

本功能从原始 BF16 checkpoint 导出 W8A8_DYNAMIC 权重，目标部署框架为 vllm-ascend v0.23.0。仅覆盖 deployment，不执行测评、校准、PTQ 或旋转。当前已完成 CPU 小分片、静态产物验证和验收客户端测试。用户已报告使用现有 Ascend JSON 完成 NPU 环境导出并成功拉起服务，吐字和精度正常；该记录不等同于本次代码已重新完成 NPU 验收，也未单独提供 MTP off/on 日志。

## 输入与环境

需要可运行 AMCT 的 Python 环境、完整原始 BF16 checkpoint、足够的主机内存和磁盘空间。导出计算使用 CPU，不需要 NPU。已有环境可通过 `PYTHON_BIN` 指定解释器，无需为运行样例重装依赖。

本地只有 config/index/tokenizer JSON 时，可以进行名称级对账，不能执行真实权重导出。原始 BF16 索引记录的数据量约 1.51 TB；输出还需要独立空间。峰值内存主要由当前输入/输出分片和量化中间 tensor 决定，应在真实机器记录，不以小分片测试推断固定内存上限。

输出目录必须不存在或为空，且不能与输入目录相同或互相嵌套。失败输出不会自动续写；保留诊断后，为下一次运行指定新的独立目录。原始 BF16 `config.json` 原样复制，包括 `moe_router_dtype`，不写入 compressed-tensors 配置。

## 导出命令

在仓库根目录执行，路径建议使用绝对路径：

```bash
export GLM52_BF16_DIR=/data/GLM-5.2
export GLM52_ASCEND_DIR=/data/GLM-5.2-amct-dynamic
export PYTHON_BIN=/path/to/venv/bin/python
bash examples/deploy_glm5_2_ascend.sh
```

脚本固定使用 `--device cpu --granularity tensor --quant_dtype int --deploy_format ascend` 和模块级位宽配置 `amct_pytorch/configs/deploy/tensor_w8a8.yaml`。`--deploy_format` 默认为 `legacy`；两种格式的 tensor 导出均要求 JSON 选层，且本次仅支持 INT W8A8。显式 W4/16 配置在加载模型前报错。

样例规则位于 `amct_pytorch/configs/deploy/glm5_2_ascend_w8a8.json`。如需自定义，直接使用 deploy CLI，将 `--quant_layers_config` 指向自己的文件。规则格式：

```json
{
  "quant_layers": ["model.layers.*.self_attn.o_proj"],
  "ignore_layers": ["model.layers.78.*"]
}
```

规则匹配不带 `.weight` 的完整模块名，大小写敏感，支持 `*`、`?` 等 fnmatch 通配符；ignore 优先，未选中模块保持浮点。quant 规则没有匹配对象或最终无选中模块会报错；ignore 规则零命中仅告警并记录。JSON 选层不与 `quant_target` 求交集。

GLM 限制包括 q_a/kv_a 配对、gate/up 配对、同一 routed MoE 的所有专家及投影统一量化，以及有效 W8A8 位宽。MTP 从模型配置识别，检查 ignore 覆盖和最终量化状态并打印日志，不再自动排除。Ascend 当前适配要求 MTP 保持 FLOAT，选中时明确报错；通过 JSON 显式排除。样例中的 78 仅针对当前 GLM-5.2。MTP 权重仍保留，不生成 rot 参数。

## 输出与静态验收

输出包含以下文件：

| 文件 | 含义 |
|---|---|
| `*.safetensors` | 复用输入分片名；选中 weight 为 INT8，scale/offset 为 BF16 `[out, 1]`，offset 全零 |
| `model.safetensors.index.json` | 所有输出 tensor 的分片映射；Ascend 的 total_size 为 tensor 数据字节数 |
| `quant_model_description.json` | 选中 weight/scale/offset 为 W8A8_DYNAMIC，其余为 FLOAT |
| `config.json`、tokenizer、generation config 等 | 原始支持文件按字节复制 |
| `deployment_validation.json` | header、辅助参数、索引与描述的一致性报告 |
| `logs/deploy.log` | 导出日志（默认位置） |

描述固定使用 `version=1.0.0`、`model_quant_type=W8A8_DYNAMIC`、`metadata={}`、`group_size=0`、`is_rot_used=false`，不写 `optional`。

完整名称级参考为 58,167 个量化模块、174,501 个动态参数、1,418 个 FLOAT 参数，共 175,919 个 tensor。这基于当前 BF16 索引且不增加 MTP 别名，不应代替实际产物校验。

验证器只检查权重 header，并逐个读取 scale/offset，不重新加载全部权重。报告中 `status=passed` 表示静态检查通过；失败时保存 `status=failed` 并使导出命令非零退出。提前中断可能没有报告；目录存在或存在权重文件不代表成功。静态报告不代表 NPU 部署通过。

## NPU 服务验收步骤（可选独立验证）

根据 [v0.23.0 官方 GLM-5.2 指南](https://docs.vllm.ai/projects/ascend/en/v0.23.0/tutorials/models/GLM5.2.html) 选择匹配的硬件、CANN、PyTorch、vLLM 和 vllm-ascend 环境。以下仅展示关键参数，TP/EP、多机、网络和其他环境设置必须按实际机器补齐；不能假设单卡可容纳。

先启动 MTP 关闭的服务并保存完整命令及加载日志：

```bash
vllm serve "${GLM52_ASCEND_DIR:?}" \
  --served-model-name glm52-amct \
  --quantization ascend \
  --tensor-parallel-size "${GLM52_TP_SIZE:?set validated TP size}" \
  --enable-expert-parallel \
  --max-model-len 4096 \
  --trust-remote-code
```

使用标准库客户端请求已启动的服务；客户端不启动、停止或修改服务：

```bash
python3 examples/validate_glm5_2_ascend.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model glm52-amct --mtp-mode off \
  --output-dir /data/glm52-acceptance
```

客户端固定 temperature=0、max_tokens=32，依次执行单 prompt 和双 prompt 批请求，检查 HTTP 成功、choices/index、非空文本和 token usage。通过返回 0，失败返回非零。鉴权可设置 `OPENAI_API_KEY`，请求证据不保存 Authorization header。

停止第一服务后，以相同配置增加以下参数重启，再使用 `--mtp-mode on` 执行客户端：

```bash
--speculative-config '{"method":"deepseek_mtp","num_speculative_tokens":3,"enforce_eager":true}'
```

两次结果分别保存在 `mtp_off/`、`mtp_on/`，包含请求/响应、`acceptance.json` 和 `deployment_acceptance.md`。已有非空证据目录拒绝覆盖。**模式参数仅记录操作者声明，不能证明服务确实使用 MTP。** HTTP 检查通过后仍标记 `pending_log_review`，需补充日志证据：

- MTP 开关与启动命令一致；开启时加载 FLOAT MTP 并实际执行 speculative 路径，未静默禁用。
- 无缺失权重、rot 缺失或维度错误。
- 记录 Linear/MoE 动态量化实现选择以及 MLAPO 回退 warning；动态量化触发的回退本身不作为格式验收失败，但需要记录。
- 记录软硬件版本、导出命令、耗时、主机峰值内存、磁盘占用以及两种模式的加载日志和请求结果。

官方 W8A8 模型包含静态量化及旋转处理，本方案统一动态量化且不旋转。因此参考模型用于量化范围和格式契约对齐，不要求权重数值、精度或吞吐与官方模型一致。真实部署验收不包含数据集精度评测或性能调优。

## 当前验收状态

| 项目 | 状态 |
|---|---|
| CPU 小分片导出、静态验证及故障测试 | 已通过 |
| 官方量化位置与动态描述名称级对账 | 已通过 |
| 导出样例参数与 HTTP 客户端逻辑测试 | 已通过（模拟 HTTP 响应） |
| 完整原始 BF16 权重导出及服务运行 | 用户报告成功；本次代码未重新进行 NPU 验证 |
| MTP 源权重逐值一致性 | 未提供独立记录 |
| vllm-ascend v0.23.0，MTP off/on 分别验收 | 未提供两种模式的独立记录 |
