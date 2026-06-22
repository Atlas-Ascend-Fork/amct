# DeepSeek-V4 (DeepSeek-V4-Pro) · 系列 deepseek · 结构 mla-moe

> **速览**：代码侧适配完成、**等权重**补 baseline/等价/PTQ smoke。结构最特殊：MLA 单 head 共享 KV + `o_groups` 分组低秩 `wo` + Compressor/Indexer 双子模块 + Hyper-Connections 4D hidden + 自定义 forward 签名 `(x, start_pos, input_ids)`。下次遇 MLA+Compressor+Indexer+HC 类模型，首参考本案 + `deepseek_v3_2`(MLA) + `longcat`(Next：PreTrainedModel 化与 AutoModel 注册)。**本案是仓内"接非 transformers 自定义 modeling"L2 经验的首遇例**——下方 L3 只留 v4 独有结构坑，通用接入坑见 L2。modeling 上游源自 DeepSeek-V4-Flash `inference/model.py`，Pro/Flash 结构同源、共用本 adapter（Pro BF16 ppl 1.9946 / Flash 4.1517）。
> **触发信号**：`architectures:["DeepseekV4ForCausalLM"]` 不在 transformers（需自实现注册）；config `kv_lora_rank`（MLA）+ `n_routed_experts`（MoE）。读 [L2 · 接非 transformers 自定义 modeling + MLA 类 + MoE 类](../structure-family-pitfalls.md) + [L1](../cross-model-pitfalls.md)。

## 为什么值得记录

- 这一例和系列默认路径(v3.2 DSA)相比,特别之处是:
  - **MLA 是单 head 共享 KV** + **`o_groups` 维度的分组低秩 wo** —— `wo_a` 通过 `weight.view(g, r, d_in)` 走 `bsgd,grd->bsgr` 的 grouped einsum,不能像普通 Linear 那样直接 wrap
  - **Compressor + Indexer 双子模块** —— Indexer 内部还嵌一个独立的 Compressor(rotate=True),需要做两层 wrap 嵌套
  - **Hyper-Connections (HC)**:hidden state 全程是 4D `[b, s, hc_mult, d]`,Block 之间通过 hc_pre/hc_post 混合;最后的 head 也是 HC-aware
  - **forward 签名是 `(x, start_pos, input_ids)`** —— 跟标准 HF decoder 完全不一样,没有 attention_mask / position_ids / position_embeddings;`input_ids` 在 hash-routed MoE 层(前 `n_hash_layers` 层)需要参与路由
  - **官方 `architectures: ["DeepseekV4ForCausalLM"]` 但当前 transformers 版本里没有这个类**,需要在 `amct_pytorch/common/models/llm/deepseek/deepseek_v4/` 自己实现 PreTrainedModel 化的版本并 `AutoConfig.register / AutoModelForCausalLM.register`

## 参考关系

- 参考系列:DeepSeek (`.agents/docs/casebook/deepseek/README.md`)
- 参考已有模型/案例:`deepseekv32` (MLA + DSA 结构最接近), `qwen3_moe` (MoE wrapper 形态), `longcat_next` (PreTrainedModel 化 + 不在 transformers 包里的模型怎么接)
- 本次主要差异:
  - 没沿用 v3.2 的 `QuantDSA / QuantDeepseekV3MLP` —— v4 的 MLA / Compressor / Indexer 跟 v3.2 DSA 结构差太大,直接复用会语义错位
  - Config 字段需要双向命名翻译(HF 标准命名 ↔ ModelArgs 命名)
  - 子模块命名是 `block.attn / block.ffn`(v3.2 是 `block.self_attn / block.mlp`),`apply_quant_to_attn` 不能直接用 —— 在 adapter 内部直接 `block.attn = QuantV4Attention(args, block.attn)` 单点替换;`apply_quant_to_moe_mlp` 把 `block.ffn` 当 root 传进去,递归内部仍能匹配 `experts/shared_experts`
  - `iter_ptq_units` 必须 override —— 父类按 `self_attn / mlp` 找,v4 找不到

## 适配结论

- **复用的现有抽象**:
  - `BaseModel`(继承)
  - `PtqUnit / make_ptq_unit / iter_indexed_units`
  - `QuantLinear`(全部 Linear 都用 — wq_a/wq_b/wkv/wo_b + Compressor wkv/wgate + Indexer wq_b/weights_proj + Indexer.compressor 内的 wkv/wgate + Expert w1/w2/w3)
  - `QuantGatedMLP`(`QuantV4Expert` 子类化它,通过 facade 适配 w1→gate_proj 等命名差)
  - `ActivationQuantizer`(每个 Linear 输入 + q/kv cache 节点)
  - `apply_quant_to_moe_mlp`(直接复用,把 `block.ffn` 喂进去就行)
  - `Catcher`(框架原版 — 不新建 `_V4Catcher`。v4 的 `Block.forward(x, start_pos=0, input_ids=None)` 改成 keyword 默认值后跟 `Catcher.forward(inp, **kwargs)` 直接对得上;在 adapter 的 `do_embedding_forward` 里手动驱动 `embed → HC 展开 → layer0(h, start_pos=0, input_ids=inputs)` —— kwargs 走 Catcher 的 `**kwargs` 路径就行)
  - `BaseModel.do_block_forward` 的 per-sample state 通道:`self.input_ids` 列表(BaseModel 新增字段,默认 None,其它 adapter 用不到)。v4 在 `do_embedding_forward` 末尾设 `self.input_ids = list(samples)`,继承的 `do_block_forward` forward loop 自动 zip 注入 —— v4 自己的 `do_block_forward` 退化成纯 pass-through(没有 30 行重复实现)
  - `init_empty_weights(include_buffers=True)`(空载构图;v4 的 `register_buffer(torch.zeros / -inf, ...)` 不少,必须 include_buffers)

- **新增的最小适配**(全部限制在 `amct_pytorch/common/models/llm/deepseek/deepseek_v4/`):
  - `modeling/configuration_deepseek_v4.py`:`DeepseekV4Config(PretrainedConfig)`,双向命名翻译 + 嵌套 `rope_scaling` / `quantization_config` 拆包
  - `modeling/modeling_deepseek_v4.py`:`Transformer(nn.Module)` → `DeepseekV4ForCausalLM(DeepseekV4PreTrainedModel)`,**其他类(Block/Attention/Compressor/Indexer/MoE/Expert/Gate/RMSNorm/ParallelHead) 完全保持上游风格**,做浮点对照基准
  - `quant_module.py`:`QuantV4Attention / QuantV4Compressor / QuantV4Indexer / QuantV4Expert`,prefill-only,wrapper 内不持有 cache buffer
  - `deepseekv4.py`:adapter 主体 — `empty_weights_model` / 嵌入与头部加载(含顶层 `hc_head_*` 参数)/ `do_embedding_forward / do_block_forward / do_head_forward` 按 v4 4D HC + start_pos=0 + per-sample input_ids 重写 / `apply_quant_attn / apply_quant_moe_mlp / iter_ptq_units / iter_deploy_bindings` v4 命名 override
  - **sharded 多卡加载（eval 用，真权重大模型）**：`block()` override 双路分流——`sharded_block=True` 走 `_block_sharded`（`_build_block_device_map` + `_dispatch_block` + `_NoMoveAlignDevicesHook`，把单层参数/各 expert 钉在多张 NPU 上、加载即就位，绕开 BaseModel `block()` 的 ~35GB CPU staging；per-tensor 只对浮点 `.to(bf16)`）；`sharded_block=False`（PTQ）仍走 `super().block()` 的 CPU staging。**两路不可统一**：PTQ 要 CPU staging 逐 expert 上卡（per-MLP 学习），而 sharded 的 `AlignDevicesHook` 把 expert 钉死在固定卡，会和 PTQ 的 `.to(device)`/`.cpu()` 循环打架。
  - `__init__.py`:`AutoConfig.register("deepseek_v4", ...)` + `AutoModelForCausalLM.register(DeepseekV4Config, DeepseekV4ForCausalLM, exist_ok=True)`

- **wo_a 单点例外**:wo_a 用 `weight.view(g, r, d_in)` 做 grouped einsum,不能套 `QuantLinear.forward` 的 `F.linear` 调用方式。`QuantV4Attention` 里把 wo_a 保留为原 `nn.Linear`,旁路挂一个 `WeightQuantizer`,在 `_wo_a_apply` 里手动跑 `observe_input + weight_quantizer + 重塑 + einsum`。其他 4 个 MLA Linear 都正常 `QuantLinear`。

- **哪一步最容易出问题**:
  1. `register_buffer(torch.zeros / torch.full(-inf, ...))` 不带 `include_buffers=True` 的话,会真分配 CPU buffer,大模型直接 OOM(v4 的 max_seq_len=1048576,kv_cache 单层就能上百 MB,61 层全建会炸)
  2. 上游 `precompute_freqs_cis` 是 `@lru_cache`,在 meta 上下文里被调用一次后,后面真实跑也会拿到缓存的 meta 张量;按层 build_block 时要按需重算
  3. wo_a 的 grouped einsum 跟 `QuantLinear.forward` 不兼容,直接 wrap 会语义错
  4. `block.attn / block.ffn` 名跟框架默认的 `self_attn / mlp` 不同,如果只调框架的 `apply_quant_to_attn / iter_ptq_units` 不会做任何事,但也不报错 —— 容易漏

- **已知风险(等权重到位后修)**:
  - `BaseModel.block()` 末尾的 `decoder_layer.eval().bfloat16()` 把 v4 里 `set_dtype(fp32)` 声明的子参数(`Block.hc_attn_fn / hc_ffn_fn / hc_*_base / hc_*_scale`、`RMSNorm.weight`、`ParallelHead.weight` 等)也一并 cast 到 bf16。forward 时 `hc_pre` 里 `F.linear(x.float(), hc_fn)` 会撞到 `float != BFloat16` 的 dtype 不匹配。空载 smoke 不触发(没真跑过 forward),真权重跑 BF16 baseline 那一步会立刻报。
  - **现状(已部分缓解)**:`empty_weights_model()` 这条路径已经修好 —— 改用 `AutoModelForCausalLM.from_config(self.config, torch_dtype=torch.bfloat16)`(走 `__init__.py` 里 register 的路由),`from_config` 内部 `set_default_dtype(bf16)` 包住实例化,Block.__init__ 里 `with set_dtype(fp32):` 嵌套生效,fp32 子参数正确停在 fp32(smoke 验证通过:`hc_attn_fn / attn_norm.weight / attn_sink` 等仍 fp32,`wq_a.weight` 仍 bf16,所有 buffer 在 meta)。**仍有 bug 的是 `BaseModel.block(layer_idx)` 的单层构造**,它还显式调 `.bfloat16()`,blockwise PTQ 里 `do_block_forward / build_quant_block` 拿到的就是这条路径出来的 block。
  - **现状更新**：`block()` 已 override 为双路分流（见上「新增的最小适配」）——**eval（sharded）路径 per-tensor 只对浮点 `.to(bf16)`、dtype 正确，V4-Pro/Flash BF16 baseline 已在此路径跑出**；**PTQ 路径仍走 `super().block()` → BaseModel `.eval().bfloat16()` 全量 cast，该 bug 未修**。真权重现已可加载、具备验证条件，待进 PTQ 时在 PTQ 分支按 `set_default_dtype` 方式修（当时「无权重无法验证」已不成立）。

- **最终是否完成**:
  - BF16 baseline:**已出** — DeepSeek-V4-Pro ppl **1.9946**（同一 modeling/Config 也覆盖 V4-Flash，DP-Flash ppl **4.1517**；两变体结构同源、仅 config.json 不同，共用本 adapter）
  - 关闭量化后保持浮点等价:**未做** — 权重未下载
  - 最小 PTQ unit 闭环:**部分** — 在小 config(`n_layers=2, dim=128, n_routed_experts=4`)上跑通空载 smoke:Block 构建(meta + CPU 都过)→ apply_quant_attn / apply_quant_moe_mlp → 25 个 QuantLinear + 19 个 ActivationQuantizer → `iter_ptq_units(attn-linear)` 1 个 unit、`iter_ptq_units(moe)` 4 个 unit → `iter_deploy_bindings` 16 个 binding → `expert.export_ptq_params()` 返回 3 keys。**真权重 + GT 路径未跑过**

## 量化结论

- BF16 baseline:V4-Pro **1.9946** / V4-Flash（DP-Flash）**4.1517**（同一 adapter，加载各自官方 config）
- 第一版直转量化方案:暂未确定;预计起点跟 v3.2 类似(attn-linear w8a8 / attn-cache q/k 16-bit / moe routed w8 + shared w8)
- 直转结果:待跑（权重已可加载）
- 是否进入 PTQ:待直转结果
- 最终采用方案:待
- 最终结果:待

## 官方 checkpoint 实际布局(查 `model.safetensors.index.json` 后核对)

> **关键认知**:`architectures: ["DeepseekV4ForCausalLM"]` 这个名字虽然符合 HF 约定,但**官方 checkpoint 的 state_dict 是扁平结构,不是 HF 标准的 `model.X` 嵌套**。修 modeling / load 路径前先 `curl https://modelscope.cn/api/v1/models/deepseek-ai/DeepSeek-V4-Pro/repo?Revision=master&FilePath=model.safetensors.index.json` 看实际 key,不要靠命名推断。

- **顶层 6 个 key**:`embed.weight / head.weight / norm.weight / hc_head_fn / hc_head_scale / hc_head_base` —— **没有 `model.` 前缀**
- **主 backbone 61 层**(`layers.0..60.*`),四类签名:
  - layer 0/1:hash-routed(`ffn.gate.tid2eid`)+ Compressor 但无 Indexer(compress_ratio=128)
  - layer 2:hash-routed(`ffn.gate.tid2eid`)+ Compressor + Indexer(compress_ratio=4)
  - layer 3,5,7,...,59:score-routed(`ffn.gate.bias`)+ Compressor 但无 Indexer
  - layer 4,6,8,...,60:score-routed(`ffn.gate.bias`)+ Compressor + Indexer
- 跟 config 的 `num_hash_layers=3` 和 `compress_ratios` 数组(index 2/4/6/.../60 = 4)严格对应
- **MTP 模块(`mtp.{m}.*`,m=0..0)**存在于 checkpoint 顶层,结构跟 Block 类似 + `e_proj / h_proj / enorm / hnorm` 四个连接到主 backbone 的额外组件 + 自带 `hc_head_*`。我们当前 modeling **不实例化 MTP**;adapter 的 load 路径全部按特定前缀(`layers.N. / embed. / norm. / head.`)取 key,跟 `mtp.*` 自然不相交,所以**无需显式过滤,strict load 也不会失败** —— MTP 在主 backbone PTQ 走通之前不接(skill 阶段属于"已知未接通,等主 backbone 跑通后再决定")
- **命名验证**(全部跟我们 modeling 对得上):`attn`(不是 self_attn),`ffn`(不是 mlp),`ffn.experts.{e}` 显式逐 expert,`ffn.shared_experts` 单数模块名

## 典型问题

> 问题 0/1/2/3/5/6 已上抽为 L2「接非 transformers 自定义 modeling」+ L1「checkpoint 是唯一事实源」的通用条目（本案即首遇例）；通用规避读 L2/L1。此处**保留 v4 完整 repro** 作为该家族最详尽的参考（问题 4 的 wo_a/HC/扁平布局是 v4 专属 L3，见上文「为什么值得记录」与「官方 checkpoint 实际布局」）。

- **问题 0**(已修):`block(layer_idx).to(device)` 报 "Cannot copy out of meta tensor; no data!"
  - 现象:`empty_weights_model()` 之后,任何 `do_block_forward / build_quant_block` 路径走到 `block = block.to(self.args.device)` 这一行就报错,即使 layer 是单独在真 CPU 上下文里 `Block(config, layer_idx)` 构造出来的
  - 根因:`Attention.__init__` 调 `precompute_freqs_cis(...)`,这个函数是 `@lru_cache(2)`。`empty_weights_model` 的 `with init_empty_weights(include_buffers=True):` 上下文里(等价于 `with torch.device('meta'):`)首次调用时,返回值是 meta 张量,**lru_cache 把这个 meta 张量缓存了**。后续 `block(layer_idx)` 在真 CPU 上下文里再 `Attention.__init__`,`precompute_freqs_cis(...)` 命中缓存,**直接返还那个 meta 张量给 `register_buffer("freqs_cis", ...)`**;block 的 freqs_cis buffer 因此停在 meta,`.to(device)` 试图从 meta 拷贝数据就报这个错
  - 修法:`empty_weights_model()` 收尾时显式 `precompute_freqs_cis.cache_clear()`,把缓存清空;后续 `block(layer_idx)` 在真 CPU 上下文里重新算,把真 CPU 张量填回缓存,所有层共享同一个 cached 真张量(因为参数都一样)
  - 下次如何提前规避:任何模块级 `@lru_cache` / `@cache` 装饰的、返回值为 tensor / device-dependent 对象的工厂函数,如果在 `with torch.device('meta'):` 或 `init_empty_weights` 上下文里被调过,**离开上下文时必须 `.cache_clear()`**。否则 device-dependent 缓存会跨上下文污染。一种安全实践:adapter 的 `empty_weights_model()` 末尾固定加一句把这一族函数的 cache 全清掉;或者把 lru_cache 改成"key 里包含 default_device"

- **问题 1**:空载 OOM
  - 现象:`empty_weights_model()` 走 `init_empty_weights()` (默认 `include_buffers=False`)时,n_layers=61 的全模型构图会消耗大量 CPU 内存(主要是 Attention.kv_cache / Compressor.kv_state / Compressor.score_state / Indexer.kv_cache 这些 `register_buffer(torch.zeros / -inf)`)
  - 根因:v4 大量 `register_buffer` 在 init 阶段就分配实数据,而 `init_empty_weights` 不带 `include_buffers=True` 只把 Parameter 重定向到 meta,Buffer 不动
  - 修法:`init_empty_weights(include_buffers=True)`(等价于 `with torch.device('meta'):`),让 buffer 跟 Parameter 一起到 meta
  - 下次如何提前规避:任何上游模型如果 `register_buffer` 用了实分配 factory(`torch.zeros / torch.full / torch.ones`)且 buffer 大小跟 max_seq_len / max_batch_size 相关,默认开 `include_buffers=True`

- **问题 2**:HF 字段命名翻译漏
  - 现象:`AutoConfig.from_pretrained` 给的 `DeepseekV4Config` 里 `dim` 是默认 4096(应当是 7168),`Block / Attention` 跑起来全是默认值,但又不报错
  - 根因:官方 `config.json` 用 HF 标准名(`hidden_size / num_hidden_layers / qk_rope_head_dim / rms_norm_eps / sliding_window / max_position_embeddings / rope_scaling / quantization_config`),`DeepseekV4Config.__init__` 如果只接收 ModelArgs 名,HF 名走 `**kwargs` 进 `super().__init__()` 只存为旁路属性,不会落到 `self.dim` 等
  - 修法:`__init__` 同时声明两套命名,HF 名做翻译;`rope_scaling` 和 `quantization_config` 拆包;round-trip 用官方 config 验证
  - 下次如何提前规避:接 HF 系模型先抓官方 config.json,把字段一栏一栏对照 ModelArgs / 代码内 attr,缺什么补什么

- **问题 3**:靠 `architectures` 字段推断 HF 嵌套结构
  - 现象:看到官方 config 的 `architectures: ["DeepseekV4ForCausalLM"]` 推断 checkpoint 是 HF 标准嵌套(`model.embed_tokens.weight / model.layers.* / lm_head.weight`),提议给 modeling 套 `self.model = DeepseekV4Model(...)`
  - 根因:HF 命名约定不能保证 checkpoint 用嵌套结构。v4 的 checkpoint 实际是扁平的 —— 顶层直接 `embed.weight / layers.{i}.* / head.weight`
  - 修法:动 modeling 结构前**先拉 `model.safetensors.index.json` 看实际 key**(modelscope 镜像最稳,HF 直链有 LFS 重定向 + 大文件被 WebFetch 截断的问题)
  - 下次如何提前规避:skill Step 1 "结构判断"必须包含一项"列出顶层 key + 一个代表层的 key 模式 + 跟代表层不同的层签名" —— 拿这些**事实**而不是命名推断来定 modeling

- **问题 5**:HF 字段翻译只单向,运行时 AttributeError
  - 现象:Config 用 `if hidden_size is not None: dim = hidden_size` 把 HF 名翻成 ModelArgs 名,但**没有把 HF 名也存回 self**;HF 通用工具(generation / tokenizer max-length / transformers internals)直接读 `config.max_position_embeddings`,挂 AttributeError
  - 根因:翻译只做了"输入归一化",没做"双向暴露"。`super().__init__(**kwargs)` 只会接住 PretrainedConfig 通用的少数字段,不会把所有原始 kwargs 都存为 attr
  - 修法:Config 末尾对每个 HF 标准名做对称的 `self.<hf_name> = <translated_value>`,跟现有 `self.hidden_size = dim` 那一类对齐。当前 v4 已经全部铺开:`hidden_size / num_hidden_layers / num_attention_heads / moe_intermediate_size / num_experts_per_tok / qk_rope_head_dim / rms_norm_eps / sliding_window / scoring_func / routed_scaling_factor / num_hash_layers / num_nextn_predict_layers / max_position_embeddings / rope_scaling / quantization_config`
  - 下次如何提前规避:写双向 Config 时不要"只翻不存"。机械规则 —— 凡是 `__init__` 接受的 HF 名,Config 末尾都要 `self.<hf_name> = ...` 一份,即使 ModelArgs 名也存了

- **问题 6**:`config.dtype` 字段名跟 HF 现代版 `dtype/torch_dtype` 别名撞名
  - 现象:`AutoConfig.from_pretrained(...)` 后 `cfg.dtype` 变成 `torch.bfloat16`(从官方 config 的 `torch_dtype: "bfloat16"` 来),不是我们期望的 `'fp8'`(架构 quant 格式)。我们 `__init__` 末尾的 `self.dtype = dtype` 在 `super().__init__(**kwargs)` 之前被覆盖,因为 modern transformers 把 `dtype` 设成 `torch_dtype` 的同 slot 别名
  - 根因:HF 的 `dtype` 是 runtime 加载 dtype(`torch.bfloat16` 这种 torch.dtype 对象),v4 ModelArgs 的 `dtype` 是架构 quant 格式选项(`"fp8"` / `"bf16"` 字符串)。两个语义同名却不同义,在同一 config 上互斥
  - 修法:把 v4 的语义重命名 `arch_dtype`,modeling 里 `config.dtype == "fp8"` 改成 `config.arch_dtype == "fp8"`。HF 的 `cfg.dtype / cfg.torch_dtype` 维持原义。这条只影响 v4 modeling 的一行 + Config 的字段绑定,不影响 ModelArgs 输入参数名(仍然是 `dtype="fp8"`,内部翻译到 `arch_dtype`)
  - 下次如何提前规避:写双向 Config 时,**先列出 modern HF PretrainedConfig 已经占用的 attribute slot**(`dtype / torch_dtype / hidden_size / num_hidden_layers / num_attention_heads / max_position_embeddings / vocab_size / pad_token_id / bos_token_id / eos_token_id / use_cache / tie_word_embeddings / rope_scaling / quantization_config / ...`),自家 ModelArgs 字段如果撞这些里任何一个,**优先在 Config 内部重命名**,不要靠"在 super 后再赋值"硬抢

## 最终建议

- 下次遇到类似模型(MLA + Compressor + Indexer + HC + 自定义 forward 签名),先参考:本案 + `deepseek_v3_2`(MLA 部分)+ `longcat_next`(PreTrainedModel 化 + AutoModel 注册)
- 先做什么:
  1. 抓官方 `config.json` 做字段表对照(HF 标准名 vs 模型代码 attr 名),写双向 Config
  2. **抓官方 `model.safetensors.index.json`**(走 modelscope 镜像 `https://modelscope.cn/api/v1/models/<owner>/<model>/repo?Revision=master&FilePath=...`,HF 直链 LFS 重定向 + WebFetch 10MB 限制都不稳),解析顶层 key + 代表层 key + 不同签名的层。这一步比读上游源码更直接
  3. 在小 config(`n_layers=2, dim=128, expert<=4`)上跑通"空载 smoke" — Block 构建 + apply_quant_* + iter_ptq_units + export_ptq_params,所有 Linear / ActivationQuantizer 数能对得上,再去找权重
  4. 上游 modeling 文件原样保留,只改明显坏掉的 import(像 statsmodels 那种)
- 不建议一开始做什么:
  - 一上来就裁上游 modeling 的 kv_cache / decode 分支 —— 那是 wrapper 的事,modeling 要做浮点对照基准
  - 直接复用同系列前一代的 quant 模块(v3.2 → v4 直接抄 QuantDSA 这种)—— 结构差异大,语义错位会很隐蔽
  - 在框架级 `quant_apply.py` 改 `name in [...]` 列表去匹配新模型的命名 —— wrapper 在 adapter 里做单点替换 + override `iter_ptq_units` 更干净
  - **靠 HF 命名约定推断 checkpoint 嵌套结构** —— 看 `architectures` 字段不能代替看 `model.safetensors.index.json` 的真实 key
  - **看到 `.scale` 配对就先实现 FP8 dequant 路径** —— 先问用户最终权重是什么格式,不要替用户做"以后可能要做"的事
  - 一上来就把 MTP / 额外 head / 等非主 backbone 模块塞进 modeling —— skill 阶段任务是主链跑通;非主链结构在 load 路径里"自然不被请求"(load 按 `embed./layers.N./norm./head.` 等特定前缀取 key)就行,不需要 modeling 里实例化对应模块
