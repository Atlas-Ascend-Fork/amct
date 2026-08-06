# amct-coder 回归测试投喂语

本文件只保留用例 A～E 的投喂语，不含任何答案、期望行为或历史结果。执行者每次只复制一个代码块到全新会话，不要让被测 Agent 打开本文件，也不要同时提供用例编号、期望行为、历史结果或设计结论。

## 用例 A

```text
规划一个混合改动：新增 torch.ops.amct.fake_quant_v2
NPU 算子，包括 Ascend C kernel、C++/Meta 注册，同时在 amct_pytorch 的
通用 quantizer 中增加 Python 调用。

当前只确定了算子名称和集成范围，输入输出、量化参数、支持 dtype/shape 与目标
SoC 尚未定稿。请做只读分析，先拆分职责、列出前置契约，再给出条件式实施顺序；
不要修改文件，也不要运行测试、构建或 NPU 命令。

为保持本次分析独立，只能按当前作用域 AGENTS.md 和实际触发的 skill 指引读取
repo-map、skill reference、直接相关源码及测试。不要读取
.agents/docs/amct_coder_analysis/ 下的文件，不要查看 git diff、git log、git show
或 git blame，也不要在仓库中搜索本题原句或同义测试描述。若这些分析材料或历史
答案已经出现在当前会话上下文中，请停止作答并明确标记“本轮评测已污染”。
```
7wtokens
只读分析完成。会话上下文干净（未出现任何分析材料或历史答案），未污染。已读：AGENTS.md、amct-ops-dev skill、amct_ops/（hifloat8_cast 全套模板 + ops_build.sh/setup.py/README）、amct_pytorch/quantization/（quant_base.py、dtypes/int*.py、quant_linear.py）、quant_util.py 的 amct_ops 集成范式、tests/amct_ops/。未读 .agents/docs/amct_coder_analysis/，未用任何 git 历史命令。
现状锚点（已确认事实）
- amct_ops/ 仅含 hifloat8_cast 一个算子，可作为唯一模板；ops_build.sh/setup.py 自动发现 <op>/python/<pkg>/ 与 CMake 产物，新增算子无需改它们。
- 命名空间 torch.ops.amct 目前仅注册 encode_to_hifloat8 / decode_from_hifloat8（register.cpp:24-27），fake_quant_v2 名称空闲（README 要求算子名在 amct 内唯一）。
- "通用 quantizer" = amct_pytorch/quantization/：ActivationQuantizer/WeightQuantizer（quant_base.py:104/142）→ quant_obj = DTYPE_REGISTRY 的 dtype 模块 → QuantDequantInt.fake_quant（dtypes/int.py:35）→ int_impl.py 的两个纯 torch 参考实现：
- weight_quant（int_impl.py:72）：scale = abs_max/ qmax，沿 dim=1（行），2D 约束，round_ste（STE 梯度直通），支持 v 偏置与 real_quant（返回 int8/scale/bias）
- dynamic_per_token_quant（int_impl.py:26）：沿 dim=-1 每 token 动态 scale
- amct_pytorch→amct_ops 已有集成范式：hifloat8_supported() 能力探测 + 懒加载 import + 无 amct_ops 时报错引导（quant_util.py:41-97），amct_ops 是 amct_pytorch 的可选依赖。
- 测试范式：tests/amct_ops/test_hifloat8_cast.py（unittest + NPU + 对照 CPU 参考）。
职责拆分
层	交付物
amct_ops/fake_quant_v2/op_kernel/	fake_quant_v2_kernel.cpp + fake_quant_v2_tiling.h
amct_ops/fake_quant_v2/op_extension/	ops.h、register.cpp、fake_quant_v2_torch.cpp
amct_ops/fake_quant_v2/python/	__init__.py + ops.py
amct_ops/ 顶层	CMakeLists.txt（复制改名）、README 算子表加一行
amct_pytorch/	dtypes/int_impl.py、dtypes/int.py（可能新增 helper）
tests/amct_ops/	test_fake_quant_v2.py
前置契约（当前未定稿 → 实施前必须冻结，均阻塞）
1. 数值语义契约（阻塞，决定一切）：fake_quant_v2 必须与哪个 Python 参考逐元素等价？weight_quant（静态权重、行内 abs-max scale、round_ste、clamp−qmax,qmax、乘回 scale）或 dynamic_per_token_quant（每 token），还是定义全新融合语义？"v2" 后缀暗示与某种历史语义存在差异——必须先书面写明 v1→v2 差异点。
2. Schema 签名契约（阻塞）：fake_quant_v2(Tensor input, ...) -> Tensor 的参数与返回必须定稿（input / scale Tensor? 或 kernel 内算 / 偏置 v / bits / 模式枚举 / 返回 dequant 后的 fp 还是 real_quant 的 int8+scale）。schema 是破坏性变更（skill 明确警告），一旦发版不可改签名，只能新增 v3。
3. STE/反向契约（阻塞 Python 集成设计）：round_ste 需要梯度直通（int_impl.py:22 的 .detach() + x）。torch.ops.amct.fake_quant_v2 走 dispatcher 无 autograd 实现；若量化学权重用于反向（non-eval 路径，quant_linear.py:59），需要 torch.autograd.Function 包装保留 STE，否则训练路径直接报错。需确认算子的使用范围是 eval/PTQ（.detach()/eval_mode 缓存）还是含训练。
4. dtype/shape 契约（阻塞）：input dtype（fp16/bf16/fp32？）、scale/v dtype、输出 dtype；rank 约束（weight 路径 2D、act 路径任意 rank 沿最后维）；空张量、非 contiguous 处理。必须同步三处：ops.py docstring、register.cpp TORCH_CHECK、tiling.h 枚举。
5. 量化参数粒度契约：per-channel（行）vs per-token（列）；scale 由 kernel 内部算（动态 reduce，tile 策略复杂）vs 传入（纯 element-wise，kernel 简单）；bits ∈ {4,8}；real_quant 是否一期支持。
6. 目标 SoC 契约：A2/A3（dav-2201，共享编译产物）必做？A5（dav-3510，不同 ISA）是否一期覆盖。kernel 必须运行时取 UB 大小算 tile，否则 A3 浪费一半 UB。
7. 集成开关契约：NPU 上无条件走 fake_quant_v2，还是能力探测 + 回退（仿 hifloat8_supported）？amct_ops 缺失时行为（报错引导 vs 静默回退 torch）。
8. 验收契约：与 CPU 参考逐位一致（round-half-even）还是容差（相对误差阈值）；平台编译矩阵（A2/A3/A5 至少编译通过）；是否含性能门槛（若目的是提性能需 baseline）。
条件式实施顺序
第一步永远是冻结契约（契约 1-8），其后步骤按契约结果分叉。
阶段 0：契约冻结（全局前置，唯一阻塞主线） — 定语义参照、schema 签名、dtype/shape、SoC 范围、STE 策略、集成开关、验收标准。产出：一份 schema 定义 + 语义差异说明，作为 register.cpp、kernel、Python 三方对齐的唯一依据。
阶段 1：amct_ops 算子（路径分叉于契约 2/5/6）
- 1a 复制模板：cp -r hifloat8_cast fake_quant_v2，全量改名（含 project()、目录名、import）
- 1b tiling.h：定义 tiling 结构体（blockNum/numPerCore/tailNumLastCore/tileLength + 模式枚举）
- 1c kernel：按契约 5 分叉——
- scale 传入（静态）：纯 element-wise 三段式，最简，仿 hifloat8 标量循环
- scale 内算（动态）：需沿约简维做 reduce，tile 需按"一行/一 token"组织，复杂度和 UB 预算都变
- 1d extension：ops.h 声明 host 接口 → register.cpp 写 schema + PrivateUse1 + Meta → fake_quant_v2_torch.cpp 解包 + 运行时 tile/block 计算 + launch
- 1e Python 包装 + docstring；1f README 算子表
- 1g 构建验证（ops_build.sh fake_quant_v2，A2 起）→ 阶段 3 补平台矩阵
阶段 2：amct_pytorch 集成（分叉于契约 3/7）
- 2a 按契约 7 写能力探测 + 懒加载 helper（仿 hifloat8_supported/hifloat8_fake_quant，quant_util.py:41/67）
- 2b int_impl.py：weight_quant/dynamic_per_token_quant 的 NPU 分支接入算子，CPU 路径原样保留
- 2c int.py QuantDequantInt.fake_quant 透传生效（无需改签名）
- 2d 按契约 3 决定是否加 torch.autograd.Function STE 包装
- 2e NPU 数值对齐：NPU 算子输出 vs 纯 torch 参考
阶段 3：测试与平台
- 3a tests/amct_ops/test_fake_quant_v2.py（对照契约 8 的参考实现）
- 3b 按契约 6 补 A3/A5 编译覆盖；A2 数值验证
依赖关系：1a→1b→1c→1d→1e→1g（线性）；2a→2b→2c（线性）；1g 与 2b 并行可接；3 依赖 1g 与 2d 的最终语义。
主要风险
- schema 冻结太晚：后期改签名 = 破坏性变更，只能升 v3 → 契约 2 必须阶段 0 定死。
- STE 遗漏：只测 eval 路径、漏反向，训练用户遇到 no autograd 报错 → 契约 3 未冻结前勿写 Python 集成。
- 动态 reduce 低估复杂度：若选"kernel 内算 scale"，tile/UB 预算与静态版差异大，别复用静态版预算注释。
- 集成开关反了：amct_ops 是可选依赖，若默认全量走算子而无回退，未装 amct_ops 的 NPU 用户直接崩 → 契约 7 需仿 hifloat8 范式。
未修改任何文件，未运行构建/测试/NPU 命令。


## 用例 B

```text
请 给出计划：修改 amct_pytorch/workflows/llm_deploy.py，
在生成 model.safetensors.index.json 时补充一个新的量化元数据字段，并同步
config.json 和相关单元测试。

字段名、取值来源和下游消费方目前尚未说明。请做只读分析，在不假设 metadata
schema 的前提下给出职责归属、待确认契约和条件式代码修改计划；不要修改文件，
也不要运行测试、构建、deploy 或模型命令。

为保持本次分析独立，只能按当前作用域 AGENTS.md 和实际触发的 skill 指引读取
repo-map、skill reference、直接相关源码及测试。不要读取
.agents/docs/amct_coder_analysis/ 下的文件，不要查看 git diff、git log、git show
或 git blame，也不要在仓库中搜索本题原句或同义测试描述。若这些分析材料或历史
答案已经出现在当前会话上下文中，请停止作答并明确标记“本轮评测已污染”。
```

3wtokens

只读分析完成。会话上下文干净，未污染。已读：deploy-export skill、amct_pytorch/workflows/llm_deploy.py、common/models/llm/common/deploy_export.py（generate_quant_config）、base.py 接口面、tests/unit_test/workflows/test_llm_deploy.py（全 709 行）及 test_base.py 的相关夹具。未读 .agents/docs/amct_coder_analysis/，未用 git 历史命令，未按题句搜索。
现状锚点
- model.safetensors.index.json 唯一写入口是 _refresh_weight_index（llm_deploy.py:186-200），被 blockwise（:232）与 tensorwise（:279）两条路径共用。产物结构固定 {"metadata": ..., "weight_map": ...}；metadata = 原 index 的 metadata 副本 + 重算 total_size（:187-191）；落盘 json.dump(ensure_ascii=False, indent=2, sort_keys=True)。
- 单分片模型合成 index 时 metadata 仅 {"total_size": ...} 占位（:153）。
- config.json 唯一写入口是 _refresh_config（:157-174）：读写 quantization_config（来自 generate_quant_config，deploy_export.py:52-74，含 format/quant_method/quantization_status/config_groups/ignore/cache 方案）；quant_dtype is None 时反而 pop 掉该键。
- _refresh_weight_index 目前签名只收 (original_index, updated_weight_map)，无 pipeline/量化参数；_refresh_config 才接触 pipeline 的 cache_scheme/bits_scheme。blockwise 中先写 index 后写 config（:232-233）。
- 相关测试集中在 test_llm_deploy.py：test_refresh_weight_index_writes_metadata_and_total_size（:228，断言原 metadata 键保留 + total_size）、test_run_tensorwise_copies_and_rewrites_weight_index（:545，同样断言 metadata 保留）、test_load_weight_index_single_shard_synthesizes（:690）、test_refresh_config_*（:248/:263/:628）。
职责归属
层	承担方
值来源（数据模型）	deploy_export.py（或新增 builder）
装配/落盘	llm_deploy.py _refresh_weight_index + _refresh_config
消费方	下游 infer 仓 / 校验工具
测试	tests/unit_test/workflows/test_llm_deploy.py
待确认契约（不假设 schema，全部阻塞）
1. 命名与位置（契约 A）：字段名（发版即锁死，改名=破坏）；index.json 放 metadata.<name>（与 total_size 并列，HF 只认 metadata.total_size+weight_map，多余键透传安全）；config.json 放 quantization_config.<name> 还是顶层？两文件字段名是否一致？
2. 值来源（契约 B）：单一事实源是哪个——(a) workflow 自身状态（quant_dtype/model_name/is_mx）；(b) pipeline 的 bits_scheme/cache_scheme；(c) generate_quant_config 产物子集；(d) AMCT 工具版本。若取自 (c)，注意 quant_dtype=None 时 config 会 pop 掉 quantization_config，此时该字段写什么。
3. 消费方格式（契约 C）：下游期望 JSON 类型（str/int/dict）、示例值、读取方式；sort_keys=True 下能否稳定比对。
4. 粒度一致性（契约 D）：blockwise/tensorwise 两条路径（共用同一 _refresh_weight_index，天然一致）是否都要写；_load_weight_index 合成的单分片 index（:153）是否也带该字段，还是只在真实导出刷新时写。
5. 无量化行为（契约 E）：quant_dtype=None 路径 config.json 无 quantization_config 时新字段落点与占位。
6. 向后兼容（契约 F）：加键是兼容改动，但需确认旧 infer/严格校验工具对该多余键的容忍度。
条件式代码修改计划
阶段 0（前置，阻塞）：冻结契约 A–F（命名、位置、来源、格式、粒度一致性、无量化行为）。产出样例字段定义与示例 JSON。
阶段 1（值来源层，分叉于契约 B）
- 若来源含方案数据：在 deploy_export.py 新增元数据构建函数（或扩展 generate_quant_config 返回值），schema 定义留在数据模型层。
- 若来源含工具版本：确认版本常量来源。
阶段 2（装配层，分叉于契约 A/B/E）
- 方案 α（字段不依赖 config 派生，可从 self.pipeline/self.* 直接取）：给 _refresh_weight_index 加一个私有元数据构建方法，在 :187 处并入 metadata；签名不变，两条 granularity 路径自动一致。改动最小。
- 方案 β（字段必须与 config.json 同源，如取自 quantization_config）：抽共享 builder，_refresh_weight_index（:186-191）与 _refresh_config（:166-174）都调用；注意 blockwise 当前先写 index 后写 config（:232-233）——若 index 字段依赖 config 计算值，需调整顺序或让共享 builder 先行。两文件分别落盘，靠单一事实源防漂移。
- 方案 γ（tensorwise 的单分片合成路径也要带字段）：改 _load_weight_index（:153），并在阶段 3 补对应断言。
阶段 3（测试同步，test_llm_deploy.py）
- test_refresh_weight_index_writes_metadata_and_total_size（:228）：新增字段存在 + 取值断言，保留 foo/total_size 断言防回归。
- 新增：字段在 tensorwise 路径（:545 测试）也存在。
- test_refresh_config_*：若 config.json 也落字段，补同步一致性断言。
- 契约 E 有分支时补无量化分支断言。
- 若改签名/新增方法：_run_blockwise/_run_tensorwise 测试中的 _refresh_weight_index 是 MagicMock，不受签名影响；但新方法需独立单测。
阶段 4（验证项，不执行）：现有 709 行测试全量通过；index.json 与 config.json 字段跨路径一致；HF 兼容性（多余 metadata 键不破坏 from_pretrained）；sort_keys=True 下产物可稳定 diff。
主要风险
- 命名即契约：字段名一旦被下游读取，改名/改类型即破坏——阶段 0 必须定稿。
- 双文件漂移：index.json 与 config.json 分别落盘，若值来自量化方案、两个方法各算一遍，容易不一致——必须单一事实源 + 共享 builder。
- 调用顺序陷阱：blockwise 先写 index 后写 config，若 index 字段依赖 config 派生值，现有顺序会写空/旧值。
- 无量化路径：quant_dtype=None 时 config 主动 pop 掉 quantization_config，若契约 E 未冻结，字段落点会不一致。
未修改任何文件，未运行任何测试/构建/deploy/模型命令。

## 用例 C

```text
请规划一个新 fp4-like quant dtype：需要实现 fake quant 和
export_deploy，注册 dtype，接入 CLI quant_dtype choices，并补充单元测试。

具体编码格式、weight/activation 适用范围、group/block 粒度和部署 layout 尚未确定。
请做只读分析，先列出必须冻结的 dtype 契约，再给出条件式分层计划和影响范围；
不要修改文件，也不要运行测试、构建、量化、deploy 或模型命令。

为保持本次分析独立，只能按当前作用域 AGENTS.md 和实际触发的 skill 指引读取
repo-map、skill reference、直接相关源码及测试。不要读取
.agents/docs/amct_coder_analysis/ 下的文件，不要查看 git diff、git log、git show
或 git blame，也不要在仓库中搜索本题原句或同义测试描述。若这些分析材料或历史
答案已经出现在当前会话上下文中，请停止作答并明确标记“本轮评测已污染”。
```
2.5wtokens

只读分析完成。会话上下文干净，未污染。已读：deploy-export skill（上轮已加载）、quantization/dtypes/（int.py/mxfp.py/int_impl.py/mxfp_impl.py/init.py）、quant_base.py、bit_policy.py、cli/llm/args.py、workflows/llm_deploy.py、deploy_export.py、tests/unit_test/quantization/test_dtypes.py、test_quant_base.py、test_llm_deploy.py。未读 .agents/docs/amct_coder_analysis/，未用 git 历史命令。
现状锚点
- dtype 框架：DTYPE_REGISTRY（dtypes/init.py），register_dtype() 懒注册 int（QuantDequantInt，int.py:33）与 mxfp（QuantDequantMx，mxfp.py:28）。消费方 WeightQuantizer/ActivationQuantizer（quant_base.py:112/150）以 quant_obj(x) 调 forward；export_deploy() 存在性经 hasattr 校验（quant_base.py:172-177）。
- dtype 类接口：__init__(bits, is_act)、forward(x, v=0.0)（bits==16 直通）、fake_quant(x, v)、export_deploy(x, v) -> {"qweight", "weight_scale"[, "weight_bias"]}。
- CLI：cli/llm/args.py:67 --quant_dtype choices=['int','mxfp']。
- deploy 消费：llm_deploy.py 用 startswith("mx"/"int"/"hif") 判 dtype（:58-60），tensorwise 用白名单 quant_dtype in ["int","mxfp"]（:269），generate_quant_config(is_mx=...) 决定 config 的 format（deploy_export.py:52-74，int→"int-quantized"，mx→"float-quantized"）；skill 已提示该函数对 Linear group 硬编码 num_bits=8/8。quant_payload/export_block_deploy 消费 export_deploy 的 dict，extra 键写为 .weight_scale/.weight_bias（deploy_export.py:123-128, 155-161）。
- 模型/算法层：hyv3.py:175、deepseekv4.py:257 cache_type = "int" if quant_dtype=="int" else "float"；glm5_2 多处 == "mxfp" 硬比较；auto_round.py:91、auto_clip.py:59/71 按 "mxfp"/"int" 分流。
- 测试：test_dtypes.py（forward 形状/dtype 保持、bits16 直通、export_deploy dict、STE 梯度、量化误差界、impl 单测）、test_quant_base.py、test_llm_deploy.py。
必须冻结的 dtype 契约（全部阻塞）
1. 编码格式：fp4 具体位布局（E2M1？对称/非对称？NaN/Inf 处理）、rounding 规则（round_ste vs round-half-even）——决定 fake quant 数值语义与后续任何算子的位级对齐。
2. 适用范围：weight only 还是 weight+activation？决定 is_act 分支（ActivationQuantizer 也走同一 registry）。
3. 粒度与 scale 来源：per-channel/per-token/block（如 32/128）/per-tensor；scale 动态计算 vs 外部传入——决定 fake quant 实现与 export_deploy 的 scale 形状、deploy 侧 weight_dequant 的 block_size 语义。
4. 部署 layout：qweight 容器（int8 容器 + 高/低 4 位打包，如 pack_uint4/f32_to_f4_unpacked，还是 float8？）、scale/bias 键名与形状、是否带 assistance bias（int 的 int4_assistance_bias）——消费方 infer 仓解包方式必须对齐，发版后改 layout=破坏。
5. dtype 命名：注册名（如 fp4）决定 startswith 前缀判断、CLI choices、config format 字符串——发版后改名影响 CLI/配置/日志。
6. 算法兼容分流：auto_round/auto_clip 按 "mxfp"/"int" 分流的算法，新 dtype 走哪条分支或需适配（浮点类型通常复用 mxfp 分支）。
7. deploy 识别与 config 语义：is_* 前缀、tensorwise 白名单（:269）、generate_quant_config 的 format——fp4-like 属浮点量化，若不正确分流会被错标成 "int-quantized"，且 num_bits 硬编码 8/8 会错标位宽（skill 已知 bug 面）。
8. bits 范围：仅 bits=4？还是 4/8/16 通配（保留 bits==16 直通约定）？
影响范围
层	文件
新增	quantization/dtypes/<name>.py + <name>_impl.py
注册	quantization/dtypes/__init__.py
CLI	cli/llm/args.py:67
deploy	workflows/llm_deploy.py（is_*、:269 白名单）、deploy_export.py（format、必要时 weight_dequant 解包）
模型/算法	hyv3.py/deepseekv4.py cache_type、glm5_2 硬比较、auto_round/auto_clip 分流
测试	tests/unit_test/quantization/test_dtypes.py（主）、test_quant_base.py、test_llm_deploy.py（若 deploy 识别改动）、CLI args 测试
条件式分层计划
阶段 0（前置，阻塞）：冻结契约 1–8（编码、范围、粒度、layout、命名、算法分流、deploy 语义、bits 范围）。
阶段 1（数值层，依赖契约 1/2/3/8，与 deploy 解耦）
- <name>_impl.py：fake quant 数学（scale 计算 + 量化/反量化），对齐 round_ste 惯例。
- <name>.py：QuantDequantFp4（仿 mxfp 结构：__init__/forward/fake_quant/export_deploy，必要时 deploy），注册 DTYPE_REGISTRY，register_dtype() 挂入。
阶段 2（集成层，依赖契约 5/6/7）
- CLI：args.py:67 choices 加新名。
- deploy：llm_deploy.py 前缀判断 / tensorwise 白名单（:269）纳入新名；generate_quant_config 的 format 与 num_bits 语义按契约 7 对齐。
- 模型/算法层按契约 6 分流（若复用 mxfp 浮点路径则基本零改动）。
阶段 3（测试层，依赖 1/2/4/8 定稿）
- test_dtypes.py：forward 形状/dtype 保持、bits 直通、量化误差界、STE、export_deploy dict 与 layout 断言、impl 函数单测（对照 mxfp 测试簇）。
- 若有 deploy/CLI 改动，同步 test_llm_deploy.py 与 args 测试。
主要风险
- layout 最贵：契约 4 未冻结前写 export_deploy 属空转；打包/键名/scale 形状一发版即被 infer 仓绑定。
- format 错标：fp4-like 若 is_mx 判断不覆盖，deploy config 会被标成 "int-quantized" 且 num_bits 错标 8（沿用 skill 已知 bug 面）。
- 算法/模型硬比较：== "mxfp"/== "int" 等硬编码处对新 dtype 静默失效（走 else 分支），可能无报错但语义错误——契约 6/7 需逐点核对。
未修改任何文件，未运行测试/构建/量化/deploy/模型命令。

## 用例 D

```text
请评估这个方案：在 amct_pytorch/classic/graph_based 下
新增 optimizer pass，并通过 LLM new-path 的 ALGO_REGISTRY 注册和选择它。

这是对拟议注册方案的 review。请做只读分析，核对 classic graph 的现有入口后
指出问题并给出正确实施计划；不要修改文件，也不要运行测试、构建或图转换命令。

为保持本次分析独立，只能按当前作用域 AGENTS.md 和实际触发的 skill 指引读取
repo-map、skill reference、直接相关源码及测试。不要读取
.agents/docs/amct_coder_analysis/ 下的文件，不要查看 git diff、git log、git show
或 git blame，也不要在仓库中搜索本题原句或同义测试描述。若这些分析材料或历史
答案已经出现在当前会话上下文中，请停止作答并明确标记“本轮评测已污染”。
```
3wtokens
只读 review 完成。会话上下文干净，未污染。已读：model-adapter skill、algorithms/registry_factory.py、algorithms/quant/__init__.py（register_algorithms）、algorithms/quant/（auto_round/auto_clip 注册样例）、quant_base.py（ALGO_REGISTRY 唯一消费方）、classic/graph_based/amct_pytorch/optimizer/__init__.py + graph_optimizer.py + quantize_tool.py 的 pipeline 组装点、prune/calibration/distillation 的 add_pass 位置。未读 .agents/docs/amct_coder_analysis/，未用 git 历史命令。
现状锚点：两套互不相干的机制
classic graph pass（现有入口）
- pass 是实例对象，协议为 run(graph) 或 run(model=..., graph=...)（graph_optimizer.py:58-65 按 inspect.signature 探测），由 GraphOptimizer 顺序执行（add_pass → do_optimizer）。
- 没有 pass registry。新增 pass 的现有入口 = ① 在 optimizer/<name>_pass.py 写类 → ② optimizer/__init__.py 导出（__all__+module+class 三处，见 :18-156）→ ③ 在 pipeline 构建点按硬编码顺序 optimizer.add_pass(...)（quantize_tool.py:138/291/316/557/624-755/858/875、prune_helper.py:98-105、prune_interface.py:540-555、quant_calibration_interface.py:80、distillation_interface.py:126/318-329）。顺序与配置开关显式写死（如 get_fusion_switch() gating）。
新路径 ALGO_REGISTRY（注册/选择）
- ALGO_REGISTRY = Registry("algo")（algorithms/registry_factory.py:20），条目为 nn.Module 子类，@register(name=..., targets=(...))。现有 targets：weight（lwc/autoround）、activation（lac）、structure（omniquant/flatquant）。
- 唯一消费方是 quant_base.py:29-68 get_algo_names_by_target/build_algorithms_by_target：按 args.algos（CLI --algos）+ targets 元数据过滤 → 以 (args, *ctor_args) 实例化进 quantizer 的 nn.ModuleDict → 在 forward 里 algo(x)（quant_base.py:124-126 / :153-163）。钩子契约：forward/quantize(x, quant_obj)/export_deploy(x, quant_obj)/observe_input/trainable_params。
- 新路径 workflow（llm_eval/llm_ptq/llm_deploy）只调 register_algorithms()（algorithms/quant/init.py:28）拉入注册，从不构造 GraphOptimizer，也没有 onnx-like Graph 对象（走 blockwise 张量路径）。
问题点（拟议方案为何不成立）
1. 对象模型不符：graph pass 不是 nn.Module，没有 forward(x)。即使 @ALGO_REGISTRY.register 语法上能注册，build_algorithms_by_target 选中后会在 quantizer 组装/forward 阶段调 algo(x) → 直接失败；若注册时不带 targets 或 target 不在 {weight, activation, structure}，则被 quant_base.py:33-40 过滤掉，静默死代码。
2. 选择路径不通：ALGO_REGISTRY 的选择只存在于 quantizer 组装这一条路径（quant_base.py 是唯一 import 方）。没有任何新路径代码"选中一个 pass 去跑 GraphOptimizer"。注册了也无法被选择执行。
3. 顺序/生命周期语义缺失：classic pass 依赖严格全局顺序与 pipeline 级实例（构造参数是 config/records/device）；ALGO_REGISTRY 按 target 无序、按 quantizer 构造。混入会丢掉 pass 顺序与图生命周期语义。
4. 分层边界破坏（model-adapter skill 边界规则）：classic graph_based 是独立旧路径，新增 pass 有自有入口；把它塞进新路径算法注册/选择 = 两个机制耦合。要么把 pass 改造成张量级 nn.Module 算法（语义从"改图"变"改张量"，不再是 pass），要么把 GraphOptimizer 跑进新路径 workflow（违反"不在 workflows 塞模型/机制特有逻辑"）。
正确实施计划（按 pass 真实语义二选一，不可混）
分支 A — 它确实是作用于 Graph 的 classic pass（拟议方案的本意）：
1. 在 classic/graph_based/amct_pytorch/optimizer/<name>_pass.py 实现 pass 类，协议对齐 run(graph)/run(model, graph)（graph_optimizer.py:58-65）。
2. optimizer/__init__.py 三处导出。
3. 在正确的 pipeline 构建点插入 optimizer.add_pass(...)——按目标场景选 quantize_tool.py（fusion/quant/retrain/fakequant 各 pipeline）、quant_calibration_interface.py、prune_interface.py 或 distillation_interface.py；顺序要卡在依赖它/被它依赖的相邻 pass 之间，配置开关沿用现有 gating 模式（如 Configuration().get_*_switch()）。
4. "选择"用 classic 侧既有机制（调用哪个 pipeline 接口 / 配置开关），不引入 registry。
5. 测试落 classic 对应 tests/ 目录。
分支 B — 需求本质是新路径张量级算法（在 quantizer 内对权重/激活张量生效）：
1. amct_pytorch/algorithms/quant/<name>.py，@ALGO_REGISTRY.register(name=..., targets=("weight"|"activation"|"structure"))。
2. 实现 quant_base.py 契约钩子（forward/quantize/export_deploy/observe_input/trainable_params），参考 auto_clip.py/auto_round.py 模板。
3. register_algorithms()（algorithms/quant/init.py:28-38）导入；CLI --algos 选择（cli/llm/args.py:83）。
4. 测试按现有 algo 测试模式。注意：这样交付的不是"pass"，不放在 classic/graph_based 下。
不可行分支（需否决并说明）：注册一个 targets=("structure",) 的 algo 在选中后"内部触发 graph pass"——新路径 blockwise 路径没有 Graph 对象，此路不通；除非将来真引入图级新路径，那也应是独立新入口而非复用 classic pass + ALGO_REGISTRY 的杂交。
影响范围
分支	文件
A	classic/graph_based/amct_pytorch/optimizer/<name>_pass.py（新）、optimizer/__init__.py、对应 pipeline 构建文件（quantize_tool.py 等）、classic tests
B	algorithms/quant/<name>.py（新）、algorithms/quant/__init__.py、tests（algo 测试簇）
否决	algorithms/registry_factory.py、quant_base.py
结论
拟议方案的注册动作可行（Registry 是通用容器），但*"通过 ALGO_REGISTRY 选择并执行一个 classic graph pass"在任何现有可能下都不成立*：选择只发生在 quantizer 组装且只认 {weight,activation,structure} target 的 nn.Module 张量算法。应先确认该 pass 的真实语义（改图还是改张量），再走分支 A 或分支 B，二者互斥。
未修改任何文件，未运行测试/构建/图转换命令。
▣  Build · Big Pickle · 1m 51s
+ Thought: 13.2s
Goal
- 多任务只读规划/review 会话：规划 torch.ops.amct.fake_quant_v2 NPU 算子、规划 llm_deploy.py 的 index.json metadata 字段扩展、规划 fp4-like quant dtype，并对"classic graph_based 下新增 optimizer pass 并接入 LLM new-path ALGO_REGISTRY"的方案做 review——仅分析，零修改。
Constraints & Preferences
- 所有任务只读：不修改文件；不运行测试、构建、NPU、deploy、量化、图转换、git 历史或模型命令。
- 不读取 .agents/docs/amct_coder_analysis/；不使用 git diff/log/show/blame；不搜索题目原句或同义测试描述。
- 若分析材料/历史答案已出现在会话中则停止并标注"本轮评测已污染"——从未发生，四个任务均声明上下文干净。
- 按当前作用域 AGENTS.md 与已触发 skill 读取：amct-ops-dev（任务1）、deploy-export（任务2、3）、model-adapter（任务4）；OpenCode 工具映射：skill→skill 工具、read→read、grep/glob、bash→bash。
Progress
Done
- 任务1（fake_quant_v2 算子）：完整分析已交付——四职责层、八个前置契约、条件式实施顺序、风险。未写任何文件。
- 任务2（llm_deploy metadata 字段）：完整分析已交付——值来源/装配/测试职责、契约 A–F、条件修改计划（方案 α/β/γ）。未写文件。
- 任务3（fp4-like dtype）：完整分析已交付——八个待冻结契约、影响范围表、条件分层计划。未写文件。
- 任务3/4 的只读读取已完成；任务4 的事实已收集但分析输出未写出（会话截断）。
In Progress
- 任务4（classic optimizer pass 注册 review）：已收集核心事实；需写出"方案问题 + 正确实施计划"的最终评估。
Blocked
- 任务4 交付：被会话截断；在写最终评估前不需要额外文件。
- 通用：所有实施步骤受未冻结契约阻塞（见 Critical Context）。
Key Decisions
- 任务1：schema 签名是破坏性变更（只能升 v3），须最先冻结；amct_pytorch 集成仿 hifloat8 范式（能力探测 + 懒加载 + 回退），amct_ops 是可选依赖；分层必须保持——kernel 无 Torch/ACL，extension 无算法，amct_pytorch 不 import kernel。
- 任务2：字段名/位置= 发布契约；index.json 与 config.json 必须单一事实源防双文件漂移；注意 blockwise 先写 index 后写 config（llm_deploy.py:232-233）的顺序陷阱。
- 任务3：编码格式与部署 layout 最贵（下游 infer 仓绑定，发版后改=破坏）；正确识别 is_mx/format 语义（避免错标 int-quantized、num_bits 错标 8）；算法层硬比较 == "mxfp" 是静默失效风险点。
- 任务4 初步发现：classic graph passes 与 ALGO_REGISTRY 算法是两个不同机制（见 Critical Context）——这是 review 的关键结论，待完整书写。
Next Steps
- 任务4：完成 review 输出——指出 classic optimizer 机制与 ALGO_REGISTRY 消费方目标不匹配的问题，给出正确实施计划（存疑时待用户确认目标）。
- 任务1–3：各自在冻结前置契约（任务1：语义/schema/STE/dtype-shape/目标 SoC/集成开关/验收；任务2：字段名/来源/消费方格式/粒度/无量化路径；任务3：编码格式/适用范围/粒度/layout/命名/算法分流/一致性/bit bits）后进入相应实施阶段。
- 需要计划/实施时，可按序重读各任务的条件计划。
Critical Context
- 任务1：amct_ops/ 仅含 hifloat8_cast 模板算子；分层为 op_kernel/op_extension/python；TORCH_LIBRARY_FRAGMENT(amct) 位于 op_extension/register.cpp；NPU_ARCH dav-2201(A2/A3)/dav-3510(A5) 由 CMakeLists + ops_build.sh 控制；ops_build.sh/setup.py 自动发现新算子无需修改；torch.ops.amct 目前只有 encode/decode 两个注册（register.cpp:24-27）；QuantDequantInt.fake_quant（dtypes/int.py:35）→ int_impl.py：weight_quant（2D 沿 dim=1、round_ste、real_quant 可选）与 dynamic_per_token_quant（沿 dim=-1）；hifloat8_fake_quant 在 quant_util.py:67 是 amct_ops 回退范式。当前未定义：fake_quant_v2 的数值语义、签名、dtype/shape、目标 SoC、STE 策略。
- 任务2：model.safetensors.index.json 唯一写入口 _refresh_weight_index（llm_deploy.py:186-200，结构 metadata+weight_map，保留原 metadata 键+重算 total_size，sort_keys indent=2）；单分片合成 index 在 :153 处只有 total_size 占位；config.json 唯一写入口 _refresh_config（:157-174）读写 quantization_config（来自 deploy_export.py 的 generate_quant_config），quant_dtype=None 时 pop 掉该键；测试：test_llm_deploy.py:228（metadata 保留+tota_size）、:545、:690、:231/:248/:263/:628 config 测试。
- 任务3：DTYPE_REGISTRY（dtypes/init.py）懒注册 int/mxfp；消费方 Quantizer（quant_base.py:112/150，quant_obj(x)），export_deploy() 存在性经 :172-177 hasattr 校验，返回 {"qweight","weight_scale", "weight_bias"}，extra 键落盘为 .weight_scale/.weight_bias（deploy_export.py:123-128,155-161）；CLI choices 在 cli/llm/args.py:67；llm_deploy.py 用 startswith("mx"/"int"/"hif") 分发（:58-60），tensorwise 白名单 quant_dtype in ["int","mxfp"]（:269）；hyv3.py:175/deepseekv4.py:257 cache_type 按 == "int"；auto_round.py:91 / auto_clip.py:59,71 按 "mxfp"/"int" 分流；部署解包基于 mxfp_impl 语义（pack_uint4/up_size/block_size 等）。
- 任务4：ALGO_REGISTRY = Registry("algo")（algorithms/registry_factory.py:20）；注册算法位于 algorithms/quant/：auto_round.py:60、omniquant.py:24、flatquant.py:92、auto_clip.py:23,77，集中 register_algorithms()（algorithms/quant/init.py:28）；消费方为 quant_base.py（get_algo_names_by_target :29-40 / build_algorithms_by_target :43 起），通过 ALGO_REGISTRY.get_item(algo_name).metadata["targets"] 按 target 选择算法，metadata 缺失 targets 会抛 ValueError（:33-38）——即 ALGO_REGISTRY 期望的是量化目标算法（算法名/目标/步骤），而非图转换实例。classic graph_based 另有一套机制：amct_pytorch/classic/graph_based/amct_pytorch/optimizer/ 的 GraphOptimizer（graph_optimizer.py：add_pass/clear_pass/do_optimizer 顺序执行），pass 通过 optimizer.add_pass(opt.*Pass(...)) 手动装配（quantize_tool.py：如 :138 ConvBnFusionPass、:316 InsertCaliQuantPass、:711-718 量化 pass 链）；两机制不互通——图 pass 依赖 Configuration/recorder/device 上下文按序执行，而 ALGO_REGISTRY 无此接口。这是方案 review 的核心论点，最终评估待书写。
Relevant Files
- 任务1：amct_ops/hifloat8_cast/（CMakeLists.txt、op_kernel/*、op_extension/{ops.h,register.cpp,hifloat8_cast_torch.cpp}、python/hifloat8_cast/{init.py,ops.py}）、amct_ops/{ops_build.sh,setup.py,ops_init.py,README.md}、amct_pytorch/common/utils/quant_util.py（:41-97 回退范式）、amct_pytorch/quantization/dtypes/{int.py,int_impl.py}、amct_pytorch/quantization/modules/quant_base.py、tests/amct_ops/test_hifloat8_cast.py、tests/amct_ops/README.md
- 任务2：amct_pytorch/workflows/llm_deploy.py（_:153、:157-174、:186-200、:232-233、:269、:279）、amct_pytorch/common/models/llm/common/deploy_export.py、amct_pytorch/common/models/llm/common/base.py、tests/unit_test/workflows/test_llm_deploy.py、tests/unit_test/common/models/llm/common/test_base.py
- 任务3：amct_pytorch/quantization/dtypes/{__init__.py,int.py,int_impl.py,mxfp.py,mxfp_impl.py}、quantization/modules/quant_base.py、bit_policy.py（_ALLOWED_BITS=(4,8,16)）、cli/llm/args.py:67、workflows/llm_deploy.py、common/models/llm/common/deploy_export.py、hyv3.py、tests/unit_test/quantization/test_dtypes.py、tests/unit_test/quantization/modules/test_quant_base.py
- 任务4：amct_pytorch/common/utils/registry_factory.py、amct_pytorch/algorithms/registry_factory.py、amct_pytorch/algorithms/quant/*.py（auto_round/omniquant/flatquant/auto_clip）、amct_pytorch/classic/graph_based/amct_pytorch/optimizer/{__init__.py,graph_optimizer.py}、amct_pytorch/classic/graph_based/amct_pytorch/quantize_tool.py、amct_pytorch/quantization/modules/quant_base.py
▣  Compaction · Big Pickle · 37.8s
+ Thought: 5.3s
任务4 的 review 已交付，结论落在分支选择上——而分支 A/B 取决于该 pass 的真实语义（改图还是改张量），方案描述里没有给出，我无法替你推断。任务1-3 的实施也全部被"冻结前置契约"卡住（需要你拍板数值语义、schema、字段名、dtype 等）。所以先确认任务4 的方向：
# Questions
任务4 的新 pass 本质是哪种？决定走哪个实施分支（二者互斥，我的 review 结论：不能靠 ALGO_REGISTRY 选择/执行 classic graph pass）。
只是 review，不实施
+ Thought: 1.3s
明白，任务4 到此为止（review 交付即完成，不实施）。
当前状态汇总：
- 任务1-3（fake_quant_v2 算子 / llm_deploy metadata / fp4-like dtype）：规划已交付，实施停留在"冻结前置契约"节点，等你确认各契约项后再进入。
- 任务4（classic pass 注册 review）：已交付 review 并确认不实施。
- 全部只读，无任何文件修改。
需要继续时，告诉我先推进哪个任务的契约冻结即可

## 用例 E

```text
请帮我运行 Qwen3 的 bf16 和 W4A8 quant eval，拿到两组 PPL
并比较结果。当前只给出了模型族和目标精度，尚未提供模型路径、device、
quant_target、quant_dtype、bit config、数据源和输出目录。

请先做只读规划，给出职责归属、缺失输入和执行/判读顺序；不要修改文件，也不要
实际运行评测或量化命令。

为保持本次分析独立，只能按当前作用域 AGENTS.md 和实际触发的 skill 指引读取
repo-map、skill reference、直接相关源码及测试。不要读取
.agents/docs/amct_coder_analysis/ 下的文件，不要查看 git diff、git log、git show
或 git blame，也不要在仓库中搜索本题原句或同义测试描述。若这些分析材料或历史
答案已经出现在当前会话上下文中，请停止作答并明确标记“本轮评测已污染”。
```

2wtokens

只读规划完成。会话上下文干净，未污染。已读：quant-workflow / quant-run / direct-quant-eval 三 skill、casebook L1 + qwen 系列总览 + qwen3-dense 个案、metrics-and-thresholds.md、examples/eval.sh、configs/w4a8.yaml、cli/llm/args.py、Qwen3 适配注册（qwen3 / qwen3_moe）。未读 .agents/docs/amct_coder_analysis/，未用 git 历史命令。
现状核查（阶段判定）
- Qwen3 已适配：qwen3（dense，4B/8B）、qwen3_moe（30B-A3B/235B-A22B）均已注册（common/models/llm/__init__.py:33-35）。→ 阶段 3（已适配，可直接进量化实验环），无需 model-adapter。
- 无本地产物：outputs/ 下无 Qwen3 结果、无 progress.md → 无运行产物可复用，只有 casebook 结论（见下）。
- 位宽配置齐备：configs/w4a8.yaml（w4/a8，dense 直接可用）、configs/bf16.yaml。评测入口 python -m amct_pytorch.eval（examples/eval.sh 权威模板）。
- casebook 复用信号（qwen3-dense.md，seq_len=4096 口径）：
- Qwen3-4B：BF16=12.377；W4A8 int=19.9143（delta≈+7.5）、W4A8 mxfp=13.808/13.066（delta≈+1.4/+0.7）——均远超 0.2，直转不达标
- Qwen3-8B：BF16=8.9464；仅 W8A8-int 记录（delta≈-0.08），无 W4A8 先例
- 系列总览：Qwen3 首推第一轮 attn-linear + mlp W8A8-int；qwen3_moe 走 attn + moe W8A8-int
职责归属（skill / 代理分工）
职责	Skill / 代理
编排	quant-workflow（supervisor）
方案设计	scheme-recommendation（quant-analyzer）
执行	quant-run（quant-implementer）
判读	direct-quant-eval（quant-reviewer）
不涉及	model-adapter / deploy-export
缺失输入（用户需补齐 7 项）
1. --model：Qwen3 checkpoint 本地路径（含 config.json + safetensors.index.json；不可达时 modelscope/HF 镜像）
2. --model_name：qwen3（dense）还是 qwen3_moe——"Qwen3"未指定变体；且 4B/8B（dense）与 30B-A3B/235B（MoE）适配结论不同（注意：缺省会误用 deepseek，必填）
3. --device npu:N：卡号；前置自检需 NPU device 可用
4. --quant_dtype：int|mxfp——W4A8 两种都可做，quant 模式必填（缺报 KeyError）
5. --quant_target：dense Qwen3 仅 mlp/attn-linear/attn-cache（moe 会 raise）；casebook 首推 attn-linear mlp
6. 数据源：Wikitext2 评测数据可达性（--data_dir 或 HF_ENDPOINT/modelscope/本地路径）
7. --output_dir：默认 ./outputs（progress/日志落点）
非缺失：--bit_config（configs/w4a8.yaml、configs/bf16.yaml 已存在）、--seq_len 4096（统一口径，默认值）。
执行顺序（quant-run / implementer）
1. 前置自检：amct_pytorch 可导入、NPU device 可用、模型与评测数据可达；任一不满足 → 写 BLOCKED 停下，不臆造继续。
2. 复用确认（硬门#2）：先汇报 casebook 已有 4B/8B 结论；用户已明确要"跑两组 PPL"，故两条命令都本会话重跑以保证同代码/同 checkpoint 口径（casebook 数值仅作对照）。
3. 方案确认（硬门#1）：quant_dtype + quant_target 未指定 → 先让 scheme-recommendation 出推荐卡、用户确认后才起跑（CLI default 不得自动执行）。
4. 命令 1 — bf16 baseline：python -m amct_pytorch.eval --model <path> --model_name qwen3 --device npu:N --granularity block --eval_mode bf16 --bit_config amct_pytorch/configs/bf16.yaml --seq_len 4096
5. 命令 2 — W4A8 quant：同上，--eval_mode quant --quant_target mlp attn-linear --quant_dtype <int|mxfp> --bit_config amct_pytorch/configs/w4a8.yaml（--granularity block 必填，否则 model 不真量化）
6. 防超时：大模型 eval 用 nohup ... > run.log 2>&1 & + 轮询 PPL:，不单命令阻塞等结束。
7. 拿到两组 PPL: → 写 progress（只记录不判达标）。
判读顺序（direct-quant-eval / reviewer）
1. 口径核对：同 seq_len=4096、同模型/代码版本；口径不同不横比。
2. 算 delta：delta = ppl_quant - ppl_bf16（指标见 metrics-and-thresholds.md）。
3. 判读：delta ≤ 0.2 达标可接受；delta > 0.2 不达标 → 建议缩范围或转 PTQ 升级判读；delta ≈ 0 或逐位相同 → 查是否 granularity=model/quant_target 未真落算子；quant 明显优于 bf16 → 先查链路。
4. 交互门：直转结果出来后不默认升级，确认"接受 / 改方案重测 / 升级 PTQ"。
5. 输出"直转量化判读卡"（当前方案 / 口径 / ppl 两组 / delta / 是否达标 / 异常检查 / 下一步）。
先验预期（给判读的参考）
casebook 明确显示 Qwen3-4B 的 W4A8 直转 int 与 mxfp 均远超 delta=0.2，8B 无 W4A8 先例。故本轮"两组 PPL 比较"大概率结论为不达标，直转跑完后大概率需进入 algorithm-recommendation（lwc/lac/autoround/omniquant，仅认 new-path ALGO_REGISTRY）→ PTQ 升级。若用户目标就是验证 W4A8 掉点规模，可据此决定是否重跑同一尺寸之外的变体。
未修改任何文件，未运行评测/量化命令。 需要我继续时，请补齐上述缺失输入，或先指定 quant_dtype/quant_target 让我出方案推荐卡。