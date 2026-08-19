# 量化结果 artifact

## 1. 用途

传统 eager 量化流程把校准参数保存在当前 Python 进程的模型对象中。进程结束后，这些参数随对象一起消失；要部署模型，只能在同一进程中继续调用 `convert(model)`。

quant-result artifact 将“已经完成的量化决定”保存到目录中，包括：

- 量化单元身份和构造配置；
- MinMax scale、AWQ scale、GPTQ 修正权重或 LLM PTQ 参数等算法状态；
- tensor 的 dtype、shape、文件大小和 SHA-256；
- 绑定原始模型的 fingerprint；
- 作为完整提交标志的 `quant_manifest.json`。

artifact 不是部署模型，也不会重复保存一份完整大模型。恢复时仍需提供与 fingerprint 匹配的原始浮点模型或 checkpoint。

## 2. 目录结构

```text
quant_result/
├── <unit-0>.safetensors
├── <unit-1>.safetensors
├── ...
└── quant_manifest.json
```

每个 safetensors 文件对应一个可独立校验的量化单元。Classic 路径通常以量化层为 unit；LLM blockwise 路径以某层的 attention、MLP 或 expert wrapper 为 unit，一个 unit 内可以包含多个 QuantLinear 的算法状态。

manifest 记录 producer、artifact UUID、源模型 fingerprint 和所有 unit 条目。每个条目绑定 layer name、原模块类型、量化模块类型、量化配置、非 tensor 参数、tensor 描述、文件名、文件大小、文件 SHA-256 和 entry digest。

## 3. 设计取舍

### 3.1 按 unit 保存，而不是保存整个模型

Classic 量化层和 LLM blockwise 单元的完成时机不同，因此 writer 按 unit 写文件。这样单个单元完成后即可释放中间状态，失败时也能定位到具体层。对 27B 模型，如果等全部层结束后再组装一个大文件，会延长 Tensor 的驻留时间，并使一次写入失败导致整份结果重写。

另一种做法是每个 Tensor 单独写文件。该方案会显著增加文件数量，而且 Tensor 与量化模块、构造配置之间的关系仍需额外索引。当前粒度以“能够独立恢复的最小量化单元”为界：一个 unit 可以包含多个相关 Tensor，但不能跨越两个独立恢复的模块。

### 3.2 Tensor 使用 safetensors，结构信息使用 JSON

量化参数中的 Tensor 写入 safetensors，构造配置、类型和摘要写入 JSON。没有使用 `pickle` 或 `torch.save`，因为 reader 不应在读取外部结果时执行任意 Python 对象反序列化；同时，safetensors 可以直接写文件，不需要先在 Python 堆中构造一份同等大小的 bytes。

也没有把 Tensor 编码进一个 JSON 文件。JSON 适合稳定表达标量和结构，不适合保存大块二进制数据；将二进制转成文本会增加体积和解析开销，也会丢失 safetensors 提供的 dtype、shape 边界。

### 3.3 manifest 最后发布，而不是边写边更新

manifest 表示“这组文件可以被读取”，因此只在全部 unit 写完并复核后发布。如果一开始就创建 manifest，再随着进度修改，进程可能在 manifest 已存在但 unit 不完整时退出，reader 还需要引入额外的进度状态和恢复规则。manifest-last 将判定条件收敛为一条：没有 manifest 的目录一律不是完整 artifact。

manifest 自身通过临时文件和 `os.replace` 发布，但这不能让多个文件成为一个文件系统事务。为此，unit 先各自原子提交，publisher 再验证整个文件集合，最后只用一次原子替换提交 manifest。该顺序避免把“单文件原子写”误当成“目录原子写”。

### 3.4 同时保存 fingerprint 和文件摘要

unit 的 SHA-256 用来判断 artifact 文件有没有损坏或被替换；源模型 fingerprint 用来判断量化参数是否属于当前模型。两者不能互相替代：artifact 文件完整不代表它来自正确模型，源模型匹配也不代表 unit 文件没有变化。

没有只比较模型名称或参数 shape，因为同一结构可以对应不同训练权重；也没有把完整源权重复制进 artifact，因为恢复时本来就需要原始 checkpoint，重复保存会把量化参数目录变成第二份模型。SHA-256 在这里用于完整性和身份绑定，不承担发布者签名功能。

### 3.5 只写入新目录，不覆盖旧结果

writer 要求目标目录尚不存在。原地覆盖旧 artifact 时，新旧 unit 可能在失败后混在一起，额外文件检查也无法区分哪些文件属于本次运行。使用新目录可以让一次运行对应一个明确的文件集合；失败目录保留用于排查，重跑时换用另一个目录。

### 3.6 恢复采用两阶段提交，不逐层原地修改

reader 先验证全部输入，再离线构建和加载所有替换模块，最后才修改模型。逐层读取、逐层替换虽然代码更短，但后续任一层失败都会留下部分量化的模型，调用方无法判断哪些层已经改变。两阶段方案增加了临时模块开销，换来明确的失败语义；提交阶段仍保留回滚，以处理 `setattr` 本身发生异常的情况。

## 4. 写入协议

单个 unit 的写入顺序如下：

1. 拒绝重复 layer name 和文件名碰撞。
2. 严格检查量化配置只能包含可确定 JSON 表达的值。
3. tensor 执行 `detach().cpu().contiguous()`，不保存计算图、设备引用或非连续 view。
4. 将 safetensors 写到目标目录内的临时文件，执行 flush 和 `fsync`。
5. 使用 `os.replace` 原子替换正式 unit 文件。
6. 回读 unit，检查 header、tensor key、dtype、shape 和 entry digest。
7. 记录正式文件的 byte size 和 SHA-256。

全部 unit 写完后，publisher 再检查目录中没有缺文件或额外文件，并重新校验每个 unit。只有全部通过，才以同样的临时文件、`fsync`、`os.replace` 顺序原子发布 `quant_manifest.json`。

该协议称为 manifest-last：manifest 既是索引，也是完整提交标志。进程在写 unit 时被终止，最多留下没有 manifest 的目录；reader 不会把它误判成完整结果。

## 5. 模型绑定和完整性

Classic eager 模型的 fingerprint 包含模型结构配置摘要和原始 `state_dict` 摘要。结构摘要会记录模块类型、参数名、dtype 和 shape；状态摘要绑定每个参数/缓冲区的内容。

LLM checkpoint 不适合把 27B 权重全部加载到内存后再计算摘要。LLM writer 流式计算每个 safetensors shard 的大小和 SHA-256，并把规范化 weight map、shard 描述和 `config.json` 绑定为：

```text
source_model.model_type
source_model.config_sha256
source_model.state_dict_sha256
```

reader 会重新计算当前源模型 fingerprint。模型结构相同但权重不同、少一个 shard、修改 shard 内容或替换 config，都会在加载量化参数前失败。

SHA-256 用于发现损坏和误配，不用于证明文件来自可信发布者。artifact 仍应从可信位置取得，并受普通文件权限保护。

## 6. Classic API

### 6.1 保存

```python
import amct_pytorch as amct

amct.quantize(model, config, quant_result_dir="./quant_result")
for inputs in calibration_loader:
    model(inputs)
```

`quant_result_dir` 必须是尚不存在的新目录。权重量化算法可在模块插入阶段完成并立即发布；需要激活校准的算法会在所有量化层达到定稿条件后的成功顶层 forward 结束时发布 manifest。

当前 Classic artifact 支持 MinMax、AWQ 和 GPTQ 量化模块。其他算法未声明完整 export/load 契约时，会在创建目录前被拒绝。`quant_result_dir=None` 时保持原有行为：只原地修改当前模型、量化参数只保存在内存中、不创建文件，后续仍通过 `convert(model)` 转换该对象。

### 6.2 恢复并转换

```python
fresh_float_model = build_the_same_model()
amct.convert(fresh_float_model, quant_result_dir="./quant_result")
```

传入 artifact 时，`convert` 先验证整个目录和源模型 fingerprint，再离线构建并加载所有替换模块。只有全部成功才修改模型；提交过程中出现异常时，会恢复已替换模块，避免半个模型已转换、半个模型未转换。

未传 `quant_result_dir` 时，`convert(model)` 的行为不变：输入必须是当前进程中已经由 `quantize` 校准完成的模型，并在该对象上原地替换部署模块。接口返回 `None`。

## 7. LLM CLI

LLM PTQ 每次只处理一个 `quant_target`，因此每个 target 使用独立目录：

```bash
python -m amct_pytorch.ptq \
  --model /path/to/checkpoint \
  --model_name qwen3_5 \
  --granularity block \
  --quant_target attn-linear \
  --quant_dtype int \
  --algos gptq_diag \
  --bit_config /path/to/bits.yaml \
  --data_dir /path/to/ptq_data/attn-linear \
  --start_block_idx 0 \
  --end_block_idx 64 \
  --quant_result_dir /path/to/artifacts/attn-linear
```

`eval` 和 `deploy` 可以同时读取多个 target artifact：

```bash
python -m amct_pytorch.eval \
  --model /path/to/checkpoint \
  --model_name qwen3_5 \
  --granularity block \
  --eval_mode quant \
  --quant_result_dir \
    /path/to/artifacts/attn-linear \
    /path/to/artifacts/mlp
```

加载时会从 artifact 恢复模型适配器名称、量化目标、数据类型、bit policy、算法列表及其状态。多个目录必须满足：

- 量化目标互不重复；
- 绑定同一个源 checkpoint；
- 除 `quant_target` 外的构造配置完全一致；
- 每个 artifact 覆盖模型全部层；
- 当前模型适配器发现的 unit 集合、路径、类型和 metadata 与 artifact 一致。

完整 Qwen3.8-27B 示例参见[Qwen3.8-27B 量化结果落盘与部署](../../examples/models/qwen3.8/Qwen3.8-27B.md)。

## 8. 失败边界

以下情况不会静默降级为直接量化，而是停止并报告错误：

- 目录已存在、路径不安全或目录包含额外文件；
- manifest 缺失、schema 不支持、字段类型错误或 artifact UUID 非法；
- unit 缺失、重复、文件 hash/size/digest 不匹配；
- tensor key、dtype、shape 或算法要求的参数不完整；
- artifact 与当前源模型 fingerprint 不一致；
- 多个 LLM artifact 的共享配置不一致；
- 当前安装未注册 manifest 中记录的量化模块；
- 算法支持 reload/fake quant，但没有实现目标 runtime 所需的 deploy export。

发生中断后不要手工补写 manifest 或修改 digest。应保留目录用于排查，选择一个新的空目录重新运行。
