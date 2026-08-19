# Qwen3.8-27B 量化结果落盘与部署

[English](Qwen3.8-27B_en.md)

本文给出 Qwen3.8-27B 从 BF16 基线、校准数据提取、PTQ 量化结果落盘、独立进程恢复，到 deploy checkpoint 导出和 Ascend 910C 推理的完整流程。样例使用 `attn-linear` 和 `mlp` 两个量化目标，并分别生成 artifact；评估和部署阶段同时加载两份 artifact。

## 1. 验证范围

模型使用 [Qwen/Qwen3.8-27B](https://www.modelscope.ai/models/Qwen/Qwen3.8-27B) BF16 权重。其 Transformers 配置类型为 `qwen3_5`，因此 AMCT 参数使用 `--model_name qwen3_5`。

文本骨干包含 64 个 decoder layer：48 层 linear attention 和 16 层 full attention。本样例量化 496 个 Linear：

| 量化目标 | 模块 | 数量 |
| --- | --- | ---: |
| `mlp` | 每层 `gate_proj`、`up_proj`、`down_proj` | 192 |
| `attn-linear` | linear attention 的 5 个投影和 full attention 的 4 个投影 | 304 |
| 合计 | 文本 decoder 内 Linear | 496 |

embedding、norm、lm_head、视觉模块、MTP、卷积、attention 状态更新和其他非 Linear 结构保持源精度。

## 2. 方案选择

### 2.1 从 BF16 权重开始

本样例选择 BF16 checkpoint，而不是 Qwen3.8-27B-FP8。PTQ 需要从未经过本轮量化的权重估计裁剪和 scale，BF16 也能提供统一的精度基线。若从 FP8 checkpoint 再做 INT8/INT4，结果会同时包含上一次 FP8 量化和本次 PTQ 的误差，无法判断掉点来自哪一步；artifact 的 fingerprint 也只能绑定已经量化过的源权重。FP8 转 BF16 只能改变存储类型，不能恢复量化前的信息，因此没有把它当作等价替代。

### 2.2 使用 blockwise，并拆分 attention 与 MLP

27B 模型不适合在 PTQ 时同时保留全部层的浮点权重、校准输入和算法状态。blockwise 每次只构建一层，完成该层的统计和落盘后释放中间对象，峰值内存由单层而不是整模型决定。

`attn-linear` 和 `mlp` 使用独立 artifact，是因为现有提取和 PTQ CLI 每次只处理一个 `quant_target`，两类模块的输入 hook 和参数集合也不同。强行合并会把两个独立失败边界绑在一起，并阻止两张 NPU 并行处理。eval 和 deploy 在读取阶段严格检查两份 artifact 的源 checkpoint 和共享配置，再将它们合并；因此拆分不会放宽一致性要求。

### 2.3 最终采用 `gptq_diag`

MinMax 不需要校准优化，速度快，但本次 W8A8 PPL 为 `7.0270099640`，高于 `gptq_diag` 的 `6.9715690613`。AWQ 的 W8A8 PPL 最低，为 `6.9421286583`，但 AWQ 需要保存并在运行时应用输入通道变换；当前 compressed-tensors dense Linear schema 没有对应字段，因此它可以用于 artifact reload PPL，不能用于本样例的真实 dense deploy。

仓库已有的 `gptq` 使用完整 Hessian，计算和状态规模随输入宽度平方增长。对 64 层、496 个 Linear 的 27B blockwise 实验，完整 Hessian 会显著增加校准内存和时间。本样例使用的 `gptq_diag` 只累计二阶矩对角项并搜索裁剪因子，保留了输入统计对权重误差的影响，同时使每个 QuantLinear 只需持久化 `clip_factor`。因此选择它作为本轮“可落盘、可恢复、可部署”的方案，并使用独立名称，避免把近似实现描述为完整 GPTQ。

### 2.4 使用 mixed W4A8，而不是全 W4 或全部 W8

全 W4 的 MLP 单独测试仍可接受，但 attention 全 W4 的 PPL 为 `8.0783901215`，MLP 与 attention 同时全 W4 为 `8.3023319244`，均超过 `0.5` 的筛选门槛。敏感性实验表明主 QKV 投影的 W4 掉点最大，因此它们保留 W8；MLP、辅助门控和输出投影采用 W4。最终策略得到 `PPL=7.0721402168`，相对 BF16 增加 `0.1518216133`。

全部 W8 的精度更稳，但实测权重 NPU memory 为 27.5743 GB；mixed W4A8 降至 19.0806 GB。选择 mixed 策略是用实测 PPL 约束低比特范围，而不是先规定 W4 覆盖率。当前 runtime 会融合 `in_proj_qkv` 和 `in_proj_z`，因此其中 48 个数值 W4 模块必须使用 W8 ABI；没有强行标成 W4 kernel，因为那会让 metadata 与实际融合模块的参数契约不一致。

### 2.5 artifact 与 deploy checkpoint 分开

artifact 只保存算法结果和重建配置，不复制 27B BF16 权重。这样可以用同一份源 checkpoint 分别进行 PPL 和 deploy，也能先校验算法结果再付出完整 checkpoint 导出的磁盘开销。直接把 PTQ 输出写成最终 deploy checkpoint 会把“算法状态是否可恢复”和“某个 runtime 格式能否加载”合并成一次操作，无法单独验证 Week 3/4 的持久化协议，也不利于以后接入其他 deploy backend。

## 3. 验证环境

| 项目 | PTQ/PPL 环境 | deploy runtime 环境 |
| --- | --- | --- |
| 硬件 | 2 x Ascend910_9362 | 1 x Ascend910_9362 |
| 驱动 / CANN | 25.5.5 / 9.1.0 | 25.5.5 / 9.1.0 |
| PyTorch / torch-npu | 2.7.1 / 2.7.1.post4 | 2.10.0 / 2.10.0.post4 |
| Transformers | 5.12.1 | 5.5.4 |
| vLLM / vLLM-Ascend | - | 0.23.0 / 0.23.0 |

PTQ 和 deploy 可以使用不同 Python 环境，但两边都需要加载对应 CANN 环境，并且 Transformers 必须包含 Qwen3.5 模型实现。AMCT 安装方式参见仓库根目录 [README](../../../README.md#安装验证)。

## 4. 路径准备

从 ModelScope 下载 BF16 模型：

```bash
python - <<'PY'
from modelscope import snapshot_download

snapshot_download("Qwen/Qwen3.8-27B", local_dir="/data/models/Qwen3.8-27B")
PY
```

在 AMCT 仓库根目录设置后续命令使用的路径：

```bash
export MODEL_DIR=/data/models/Qwen3.8-27B
export RUN_DIR=outputs/qwen3_8_27b_gptq_diag_mixed_w4a8
export BIT_CONFIG=examples/models/qwen3.8/qwen3_8_27b_mixed_w4a8.yaml

mkdir -p "${RUN_DIR}"
```

模型目录至少应包含 `config.json`、`model.safetensors.index.json` 和 index 引用的全部 safetensors shard。量化结果会绑定这些文件的 SHA-256；后续换用不同 checkpoint 时必须重新生成 artifact。

## 5. BF16 PPL 基线

```bash
python -m amct_pytorch.eval \
  --model "${MODEL_DIR}" \
  --model_name qwen3_5 \
  --granularity block \
  --device npu:0 \
  --eval_mode bf16 \
  --bit_config amct_pytorch/configs/bf16.yaml \
  --seq_len 4096 \
  --seed 0 \
  --output_dir "${RUN_DIR}/bf16_eval"
```

本次实测使用 WikiText-2 test，得到 `PPL=6.920318603515625`。该值是后续量化结果的同口径基线。

## 6. 提取两个 target 的校准数据

`extract_ptq_data` 每次只处理一个 `quant_target`。以下命令使用 8 个 512-token 样本，分别提取 attention Linear 和 MLP 的逐层输入：

```bash
python -m amct_pytorch.extract_ptq_data \
  --model "${MODEL_DIR}" \
  --model_name qwen3_5 \
  --granularity block \
  --device npu:0 \
  --quant_target attn-linear \
  --nsamples 8 \
  --seq_len 512 \
  --data_dir "${RUN_DIR}/ptq_data/attn-linear" \
  --output_dir "${RUN_DIR}/extract/attn-linear"

python -m amct_pytorch.extract_ptq_data \
  --model "${MODEL_DIR}" \
  --model_name qwen3_5 \
  --granularity block \
  --device npu:1 \
  --quant_target mlp \
  --nsamples 8 \
  --seq_len 512 \
  --data_dir "${RUN_DIR}/ptq_data/mlp" \
  --output_dir "${RUN_DIR}/extract/mlp"
```

两个命令可在两张卡上并行执行，也可以改成同一张卡顺序执行。校准集只用于统计量化参数，WikiText-2 test 只用于 PPL，二者不能混用。

## 7. PTQ 并生成 quant-result artifact

本样例使用 `gptq_diag`。它累计输入二阶矩的对角项，为每个输出通道搜索 `[0.5, 1.0]` 内的权重裁剪比例，并保存 `clip_factor`。`gptq_diag` 是面向 27B blockwise 流程的对角 Hessian 近似；仓库中的 `gptq` 是完整 Hessian GPTQ，两者不是同一个算法。

```bash
python -m amct_pytorch.ptq \
  --model "${MODEL_DIR}" \
  --model_name qwen3_5 \
  --granularity block \
  --device npu:0 \
  --quant_target attn-linear \
  --quant_dtype int \
  --algos gptq_diag \
  --bit_config "${BIT_CONFIG}" \
  --data_dir "${RUN_DIR}/ptq_data/attn-linear" \
  --end_block_idx 64 \
  --quant_result_dir "${RUN_DIR}/artifacts/attn-linear" \
  --output_dir "${RUN_DIR}/ptq/attn-linear"

python -m amct_pytorch.ptq \
  --model "${MODEL_DIR}" \
  --model_name qwen3_5 \
  --granularity block \
  --device npu:1 \
  --quant_target mlp \
  --quant_dtype int \
  --algos gptq_diag \
  --bit_config "${BIT_CONFIG}" \
  --data_dir "${RUN_DIR}/ptq_data/mlp" \
  --end_block_idx 64 \
  --quant_result_dir "${RUN_DIR}/artifacts/mlp" \
  --output_dir "${RUN_DIR}/ptq/mlp"
```

每个 target 生成 64 个 unit safetensors 和一个 `quant_manifest.json`。unit 文件保存该层内部多个 QuantLinear 的 `clip_factor`；manifest 保存模型 fingerprint、量化配置、unit 索引、文件大小、文件 SHA-256 和 entry digest。

writer 先原子写每个 unit，回读校验全部 unit 后，最后原子发布 manifest。因此 `quant_manifest.json` 是 artifact 已完整提交的标志：目录中只有 unit、没有 manifest，表示运行中断，评估和部署都会拒绝读取。

## 8. 从 artifact 恢复并评估 PPL

下面启动一个新的评估进程。量化目标、位宽、算法和必要构造参数从两份 artifact 恢复，不依赖 PTQ 进程中的内存对象：

```bash
python -m amct_pytorch.eval \
  --model "${MODEL_DIR}" \
  --model_name qwen3_5 \
  --granularity block \
  --device npu:0 \
  --eval_mode quant \
  --seq_len 4096 \
  --seed 0 \
  --quant_result_dir \
    "${RUN_DIR}/artifacts/attn-linear" \
    "${RUN_DIR}/artifacts/mlp" \
  --output_dir "${RUN_DIR}/artifact_eval"
```

reader 会依次校验 manifest schema、目录文件集合、unit hash/digest、两份 artifact 的共享配置、64 层完整覆盖，以及当前 BF16 checkpoint 的 fingerprint。任何一项不匹配都会在构建量化 block 前失败。

## 9. 导出 deploy checkpoint

```bash
python -m amct_pytorch.deploy \
  --model "${MODEL_DIR}" \
  --model_name qwen3_5 \
  --granularity block \
  --device npu:0 \
  --quant_result_dir \
    "${RUN_DIR}/artifacts/attn-linear" \
    "${RUN_DIR}/artifacts/mlp" \
  --output_dir "${RUN_DIR}/deploy"
```

deploy 逐层从同一 BF16 checkpoint 构建量化 block、加载对应 unit 的 `clip_factor`、生成 INT4/INT8 权重和 scale，并写出新的 sharded checkpoint。未在量化范围内的 tensor 原样复制，最终重新生成 weight index 和 compressed-tensors `quantization_config`。

Qwen3.8 的 vLLM runtime 会把 `in_proj_qkv` 和 `in_proj_z` 融合为 `in_proj_qkvz`。本策略中前者为 W8、后者为 W4，一个融合模块不能同时使用两种参数 ABI，因此 48 个 `in_proj_z` 保留 W4 数值与 W4 scale，但在 metadata 中使用 W8 ABI。该 fallback 会由导出结果显式记录，不应表述为 400 个模块全部进入 W4 kernel。

## 10. vLLM-Ascend smoke test

在 vLLM-Ascend 环境中加载导出目录：

```bash
export VLLM_PLUGINS=ascend
export ASCEND_RT_VISIBLE_DEVICES=0

python - <<'PY'
import os
from vllm import LLM, SamplingParams

llm = LLM(
    model=os.path.join(os.environ["RUN_DIR"], "deploy"),
    tensor_parallel_size=1,
    max_model_len=256,
    gpu_memory_utilization=0.85,
    enforce_eager=True,
    trust_remote_code=True,
    limit_mm_per_prompt={"image": 0, "video": 0},
)
result = llm.generate(
    ["The capital of France is"],
    SamplingParams(temperature=0.0, max_tokens=8),
)[0].outputs[0]
print(result.text)
print(list(result.token_ids))
PY
```

本次实测输出 token IDs 为 `[11751, 13, 198, 760, 6511, 314, 9564, 369]`。该测试证明 checkpoint 能在真实 910C 上完成加载和生成，不代表生产吞吐、长上下文或并发性能已经验收。

## 11. 实测结果

所有 PPL 使用同一 BF16 checkpoint、WikiText-2 test、seed 0 和 `seq_len=4096`：

| 位宽策略 | 算法 | PPL | 相对 BF16 |
| --- | --- | ---: | ---: |
| BF16 | - | 6.9203186035 | 0 |
| W8A8 | MinMax | 7.0270099640 | +0.1066913605 |
| W8A8 | AWQ | 6.9421286583 | +0.0218100548 |
| W8A8 | `gptq_diag` | 6.9715690613 | +0.0512504578 |
| mixed W4A8（较低 W4 覆盖） | MinMax | 7.3077268600 | +0.3874082565 |
| mixed W4A8（较低 W4 覆盖） | AWQ | 7.2863488197 | +0.3660302162 |
| mixed W4A8（较低 W4 覆盖） | `gptq_diag` | 7.0944299698 | +0.1741113663 |
| 本样例 mixed W4A8 | `gptq_diag` | 7.0721402168 | +0.1518216133 |

本样例配置共覆盖 400 个数值 W4 Linear 和 96 个原生 W8 Linear。真实 deploy 分类为：

| runtime 分类 | 数量 | 说明 |
| --- | ---: | --- |
| 真实 W4 runtime | 352 | W4 数值并使用 W4 kernel 参数契约 |
| W4 数值、W8 ABI fallback | 48 | linear attention 的 `in_proj_z` |
| 原生 W8 | 96 | 主 QKV 等敏感投影 |

deploy 静态验收检查 65 个 shard、2799 个 index key，结果为 0 missing、0 extra、0 duplicate、0 route error；703 个未替换 tensor 与 BF16 源逐元素一致。真实 NPU 权重内存从 W8A8 的 27.5743 GB 降至 19.0806 GB，减少约 30.803%。W4 checkpoint 文件比 W8 大约 0.96%，因为逻辑 INT4 暂存在 INT8 container 中，真正的 4-bit packing 在 vLLM 加载阶段完成；本方案的收益体现在 NPU memory，而不是磁盘包体。

## 12. 更换算法或位宽

- 全 W8A8：将 `BIT_CONFIG` 改为 `amct_pytorch/configs/w8a8.yaml`，使用新的输出目录重新执行 PTQ、artifact eval 和 deploy。
- MinMax：将 `--algos gptq_diag` 改为 `--algos minmax`。
- AWQ：将 `--algos gptq_diag` 改为 `--algos awq`。当前 compressed-tensors dense Linear 没有保存 AWQ 输入通道变换的字段，因此 AWQ 可用于 artifact reload PPL，但不能使用本样例的 dense deploy 导出路径。
- 完整 Hessian GPTQ：使用 `--algos gptq`。其状态和计算量与 `gptq_diag` 不同，不能加载本样例生成的对角近似 artifact。

算法、位宽、模型权重或量化目标变化后，都必须重新生成对应 artifact；不能通过修改 manifest 来复用不匹配的结果。
