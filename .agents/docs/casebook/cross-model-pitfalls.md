# 通用适配陷阱 · L1（跨网络通用经验库）

> 几乎任何网络适配/量化都可能遇到，与具体结构无关。**适配任何新模型前先通读本页**。按结构家族分的经验见 [structure-family-pitfalls.md](structure-family-pitfalls.md)（L2）；模型专属见各个案（L3）。每条：现象 / 根因 / 处理 / 规避；「例」指首遇个案。

## 适配前自检清单（先验，非全集）

> 进新模型时主动过一遍；命中即按对应条目处理，清单外现象照常据证据诊断。

- [ ] 先拉 `model.safetensors.index.json`（modelscope 镜像最稳），列顶层 key + 代表层 key 模式 + 不同签名的层——**用事实定层数/load 路径，不靠 config 深度或 `architectures` 命名推断**。
- [ ] 抓 `config.json` 对照字段表（HF 标准名 vs 代码 attr 名）。
- [ ] BF16 blockwise baseline：先看 chunk0 loss 是否回到合理量级，再看全量 PPL；**PPL 离谱（高/低/quant 反优）第一件事查 blockwise mask/pos/position_embeddings 传递链**。
- [ ] `lm_head` 是否与 `embed_tokens` tied。
- [ ] 判断属哪些 L2 结构家族（见 family-pitfalls 触发信号，可多属），叠加读对应小节。
- [ ] 量化起步：先关闭量化（bits=16）验 wrapper 与 BF16 等价，再直转；先整网再缩 block/unit。

## 评测与适配链路

### BF16 PPL 明显异常（离谱高 / 离谱低 / quant 反优于 BF16）
- 根因：blockwise 路径漏传 `attention_mask` / `position_ids` / `position_embeddings`，单层 block 与真实 decoder 不一致。
- 处理：把 embedding 阶段捕获的上下文显式传入每层 `do_block_forward`；先看 chunk0 loss 是否回到合理量级。
- 规避：BF16 PPL 一旦离谱，**第一件事查 blockwise mask/pos 传递链**。
- 例：`qwen/qwen3-moe`（235B：漏 mask → PPL 466，chunk0 loss 5.47→1.23）。

### checkpoint 是唯一事实源（config 字段 / `architectures` 命名都不可信）
- 现象：HF 只加载出少数层（config `num_hidden_layers` 偏小）；或按 `architectures:["XxxForCausalLM"]` 假设 HF 标准嵌套（`model.layers.*`），实际是扁平结构。
- 处理：动 modeling / load 前**先拉 `model.safetensors.index.json`**（modelscope 镜像最稳），看真实顶层 key + 代表层 key 模式 + 不同签名的层；按事实定层数与 load 路径。
- 规避：**不靠 config 深度、不靠命名约定推断结构**。
- 例：`qwen/qwen3-moe`（235B：config=5 层 / 真实 94 层）、`deepseek/deepseekv4`（扁平 `embed./layers.N./head.`，无 `model.` 前缀）。

## 入口与权重

### `lm_head` 与 `embed_tokens` 权重共享（tied）
- 现象：找不到独立 `lm_head.weight`。处理：adapter 内把 `lm_head` 绑回 `model.embed_tokens`，不按独立 key 去 index 查找。例：`qwen/qwen3-dense`（4B）。

### `Catcher` 必须透传模型特有 forward 字段
- 现象：替换 layer0 后官方 forward 在 embedding 阶段访问某字段（如 `attention_type`）报错。处理：`Catcher` 显式透传该字段，或用框架原版 Catcher + `**kwargs` 通道。例：`qwen/qwen3-dense`（4B）。

## 量化通用

### attn-linear wrapper 不得替换官方 attention kernel
- 现象：关闭量化等价（step8）明显漂移。根因：仅量化 q/k/v/o 时不应改 `attention_interface`。处理：attn-linear 只 wrap 投影 Linear，复用官方 interface。例：`qwen/qwen3-dense`（4B）。

### int activation quant：`ActivationQuantizer` 必须 `is_act=True`
- 现象：`weight_quant()` 收到 activation tensor 触发 shape 断言。根因：实例化 dtype 漏传 `is_act=True`；`dynamic_per_token_quant()` 误用只适用正数的 clamp。处理：传 `is_act=True` + 去掉 signed activation quant 的正数 clamp。例：`longcat/longcat`（Flash-Lite）。

### 通用 PTQ provider tensor-only，不覆盖 attention unit
- 现象：`materialize_gt()` 只转发 tensor batch，但 attention unit 还需 `position_embeddings`/`attention_mask`。处理：最小 PTQ smoke 先用 expert/mlp 单元闭环；做 attention unit PTQ 需把 provider 泛化为支持额外上下文。例：`qwen/qwen3-moe`（235B）。

### PTQ 前先确认算法在 new-path（LLM）ALGO_REGISTRY 中
- 现象：指定 `--algos gptq`（或 awq/smooth_quant）跑 PTQ，算法名「看似登记」却选择失败/不生效。根因：classic 图量化 path 的算法名册 与 LLM new quantization path 的 `ALGO_REGISTRY`（`amct_pytorch/quantization/modules/quant_base.py`）是**两套**——classic 列了名 ≠ LLM PTQ 路径实现了。处理：跑 PTQ 前先确认所选算法在 **new-path `ALGO_REGISTRY`** 内（当前 = `autoround / lac / lwc / omniquant`；gptq/awq/mxfp 视分支移植）；**只认 new-path、忽略 classic 名册**。例：Qwen3-4B PTQ 验证（误选 gptq → 查 new-path 缺 → 自纠改 LWC）。

### 加载 PTQ 参数（eval / deploy）必带与 ptq 一致的 `--algos`
- 现象：评测/部署 PTQ 结果时带了 `--<target>_param_dir` 却漏 `--algos`，`load_module` 报 `KeyError: Submodule '...weight_quantizer.algorithms.<algo>' is not found`。根因：PTQ 保存的参数含 `algorithms.<algo>` 子模块，加载侧 quant 模块只有按相同 `--algos` 构建才会有该子模块，缺则结构不匹配。处理：**eval 与 deploy 加载 PTQ 参数都必须带与 ptq 训练一致的 `--algos`**（不只 deploy）。例：Qwen3-4B PTQ 验证（eval 漏 `--algos lwc` → KeyError）。

## 通用纪律

- 上游 modeling 文件原样保留，只改明显坏掉的 import；接入前先 `python -c "import ..."` 一次。
- 模型专属逻辑收在 `amct_pytorch/common/models/llm/<vendor>/...`，默认不改 workflow / solver / 通用 quant module。
- 量化方案默认：先直转再 PTQ；先整网再缩 block/unit；误差定位只做粗粒度；`delta <= 0.2` 为默认接受阈值；评测口径统一 Wikitext PPL `seq_len=4096`。
