# 网络类型通用陷阱 · L2（按结构家族经验库）

> 仅对某一类结构成立。适配前先按**触发信号**判断模型属哪几类（**可同时属多类**，取并集），读对应小节（**叠加** L1 [cross-model-pitfalls.md](cross-model-pitfalls.md)）。模型专属见各个案（L3）。每条：现象 / 根因 / 处理；「例」指首遇个案。

## MoE 类

> **触发信号**：config `num_experts` / `n_routed_experts` / `num_local_experts` > 0；checkpoint 含 `*.experts.<i>.*` key；有 gate/router 模块。

### expert 磁盘展开 vs 运行时 packed
- 现象：state_dict key / expert 权重形状与运行时模块不一致。根因：checkpoint 逐 expert 展开（`mlp.experts.<i>.{gate,up,down}_proj`），运行时期待 packed 张量。处理：adapter 内重组为 packed `gate_up_proj`/`down_proj`；先确认 `transformers` 版本（布局随版本可能变）。例：`qwen/qwen3-moe`、`qwen/qwen3-next`。

### MoE activation capture 误命中 `gate`
- 现象：capture 报 `AttributeError: 'tuple' object has no attribute 'detach'`。根因：宽泛 `hook_name="mlp"` 同时匹配 `mlp.gate`，gate 返回 tuple。处理：用真实 hidden states 过 gate、选实际命中的 expert 单元做 capture/PTQ，不按名宽抓。例：`qwen/qwen3-moe`（30B）。

### packed-expert PTQ 的 device 迁移（惰性视图 weight 留 CPU）
- 现象：PTQ 训练报 `RuntimeError: Expected all tensors to be on the same device, npu:x and cpu`（quant_linear `F.linear` 处）。根因：自写 experts 量化类只用 `GatedExpertView(materialize=False)` 惰性切 packed 张量，block `.to(npu)` 不迁移该视图引用 → PTQ 时 weight 仍在 CPU。处理：复用 `qwen/moe_common.py` 的 `QuantGatedExperts`（含 `build_ptq_expert_module(materialize=True)` 实体化为真 Parameter 的 PTQ 路径），不要自写只做 `materialize=False` 的 experts 类。例：`hunyuan/hy3-preview`。

### 不存在"冷 expert 无输入"——输入按 (layer, quant_target) 缓存（更正误诊）
- 现象：PTQ报 `AttributeError: 'NoneType' object has no attribute 'size'`（`TensorDataset(inps)`）/ `WARNING: PTQ input file not found`。~~曾误判为"稀疏激活下少样本 calib 覆盖不到部分 expert，未命中 expert 输入为 None"——**该说法错误**~~。**真实机制**：`load_ptq_inps(data_dir, quant_target, layer_idx)` 按 `block_{layer}_{quant_target}_in.pkl` **一层一 target 一个文件**加载，同层所有 expert **共用**同一文件（calibration 在路由前 hook 整个 MoE block 输入）→ **不存在按 expert 区分的"冷 expert"**。`inps is None` 的真正含义：该 `(layer, quant_target)` 输入文件不存在——多为 extract_ptq_data 未产出，或 **yield 的 `unit.kind` 与 extract 产出文件名不匹配**（典型：误 yield `moe.shared` 单元去找从不产出的 `block_*_moe.shared_in.pkl`，见下条 shared_experts）。处理：核对 extract 已跑且 `unit.kind` 与产出文件一致；**不要用 `inps is None → skip` 掩盖**（会静默产出降质模型，应 fail-fast）。例：`hunyuan/hy3-preview`（实为 shared_experts 误 yield，根因见下条；guard 已回退）。

### shared_experts PTQ 策略不一致（design contradiction）
- 现象：PTQ报 `WARNING - PTQ input file not found: block_*_moe.shared_in.pkl`；或 `Skip PTQ unit 'shared_experts': no calibration inputs`；**训练 epoch loss 恒为 0**；参数虽有保存但无实际优化。根因：quant_module 层使用 `build_no_algo_args(args)`（清空 `algos`，意图跳过 PTQ），但 adapter 层 `iter_ptq_units` 显式 `yield make_ptq_unit("moe.shared", ...)`（要求 PTQ）→ 两层设计矛盾。处理：两者必须一致——要么都用 `build_no_algo_args` + 不 yield（**推荐**，shared_experts 密集激活、直接量化效果好），要么都不用 + yield（少数场景需 PTQ 优化）。**优先用通用函数 `apply_quant_to_moe_mlp`**（自动对 shared_experts 调 `build_no_algo_args`）；自写 MoE wrapper 时检查：quant_module 用 `build_no_algo_args` → adapter 不 yield shared_experts。参考：`qwen/qwen3_5_moe.py`、`deepseek/deepseek_v3_2/deepseekv3_2.py`。例：`hunyuan/hy3-preview`（已修复：注释掉 yield，hyv3.py:133-136）。
- **诊断信号**：训练 loss 恒为 0 或 NaN → 立即停止检查适配逻辑（见 model-adapter.md Step 7）。

### MoE per-expert 动态量化的 prefill 性能
- 结论：per-expert 动态量化在 prefill 下平均路由 `M_eff`（≈ `seq*topk/experts`）偏小，单 expert `dynamic_quant + 头开销 > BF16 matmul`，每层 ×experts 放大 → **常为负收益**；attention projection / MLP 多为正。
- 规避：MoE 是否进量化首要看**精度 delta + 下游 infer 路径覆盖**，性能须 infer 侧 packed `MoEGMM` 实测确认；decode(M=1) weight-bound 可能翻正。例：`qwen/qwen3-moe`（30B）、`longcat/longcat`（Flash-Lite）。

## attention wrapper（sdpa 复用）通用

> **触发信号**：adapter 自写 attention wrapper 调用共享 `scaled_dot_product_attention` helper。

### sdpa 同时传 attn_mask 与 is_causal=True
- 现象：量化 attn-linear 路径报 `AssertionError`（`attention_forward.py` 的 `assert attn_mask is None`，仅 `is_causal=True` 分支）。根因：wrapper 把 `attn_mask=attention_mask` 和 `is_causal=self.is_causal` 同时传给 helper，二者互斥。处理：因果 LM 无 padding 时用 `attn_mask=None, is_causal=True`（让 helper 自建因果 mask），对齐 `qwen3` wrapper 写法。例：`hunyuan/hy3-preview`。

## 混合 attention 类（`linear_attention` + `full_attention`）

> **触发信号**：config `layer_types` 同时含 `linear_attention` 与 `full_attention`（或 `full_attention_interval`/周期布局）；存在 `GatedDeltaNet`/`linear_attn` 模块。

### 混合 attention 的 causal mask 路由
- 现象：BF16 PPL 低得离谱（像"看到后文"）。根因：`do_embedding_forward()` 只捕 layer0 mask，而 layer0 是 `linear_attention` 捕到 `None`，后续 `full_attention` 复用即丢 causal mask。处理：adapter 内分别维护 `linear_attention_mask` 与 `full_attention_mask`（后者 `create_causal_mask(...)` 显式构造），wrapper 显式传 mask 时 `is_causal=False`。
- 另：不同 attention 形态不能套同一 attention 量化路径；未就绪时 `linear_attention` kernel 先留 BF16（仍可量化其投影 Linear）。例：`qwen/qwen3-next`、`qwen/qwen3.5-3.6`。

## 接「非 transformers 包内」的自定义 modeling 类

> **触发信号**：`architectures` 类不在当前 transformers 版本（`import` 不到 / 需 `trust_remote_code`）；需自实现 PreTrainedModel 化 + `AutoConfig/AutoModel.register`。例：`deepseek/deepseekv4`、`longcat/longcat`（Next）。

### `trust_remote_code` 顶层模型空载失败 / 多余依赖
- 现象：`from_config(..., trust_remote_code=True)` 被 `flash_attn` 等阻塞。根因：顶层走多模态入口，而 blockwise PTQ 只需 text-only backbone。处理：adapter 内直接实例化动态模块里的文本模型类 + 修 `lm_head` 形状。例：`longcat/longcat`（Next）。

### 空载 OOM：`register_buffer` 实分配未随 meta
- 现象：`init_empty_weights()`（默认 `include_buffers=False`）下全模型构图吃大量内存。根因：上游 `register_buffer(torch.zeros/full/...)` init 即分配实数据。处理：`init_empty_weights(include_buffers=True)`。例：`deepseek/deepseekv4`。

### `@lru_cache` 工厂函数在 meta 上下文污染缓存
- 现象：`block.to(device)` 报 "Cannot copy out of meta tensor"。根因：`precompute_freqs_cis` 等 `@lru_cache` 工厂在 meta 上下文首调缓存了 meta 张量，后续真 CPU 构造命中缓存拿回 meta。处理：离开 meta 上下文 `.cache_clear()`（或 cache key 含 device）。例：`deepseek/deepseekv4`。

### 自定义 Config 双向命名翻译 + 字段撞 HF slot
- 现象：HF 工具读 `config.max_position_embeddings` AttributeError；或自家 `config.dtype` 被 HF `dtype/torch_dtype` 同 slot 覆盖。根因：只把 HF 名翻成内部名、没把 HF 名存回 `self`；自家字段撞 HF 已占 slot。处理：`__init__` 末尾对每个 HF 标准名对称 `self.<hf_name>=...`；撞名字段内部重命名（如 `dtype`→`arch_dtype`）。规避：先列 HF 已占 slot（`dtype/torch_dtype/hidden_size/.../rope_scaling/quantization_config`）。例：`deepseek/deepseekv4`。

### 子模块命名与框架默认不符
- 现象：`block.attn/block.ffn`（非 `self_attn/mlp`），调框架 `apply_quant_to_attn`/`iter_ptq_units` 不报错但什么都不做。处理：adapter 内单点替换 + override `iter_ptq_units`。例：`deepseek/deepseekv4`。

### 看到 `.scale` 配对就实现 FP8 dequant
- 现象：index.json 每个 Linear 有 `name.weight`+`name.scale`，推断 FP8 提议加 dequant。处理：**问用户最终权重格式**，不要从字段推断、替用户做"以后可能要做"的事。例：`deepseek/deepseekv4`。

## MLA 类

> **触发信号**：config 有 `kv_lora_rank` / `q_lora_rank`；attention 低秩压缩 KV（共享 KV、`wkv_a/wkv_b`、Compressor）。

- MLA attention 量化分 `attn-linear`（投影）与 `attn-cache`（q/k/v cache）两路；cache 位宽通常更敏感（常 q/k 16-bit 起步）。MLA 单 head 共享 KV、低秩 `wo`、Compressor/Indexer 等结构细节因模型而异，见各个案（`deepseek/deepseekv4`、`deepseek/deepseek-v3.2`）。
