# amct-coder 回归测试

第二次测试，测试性能

本文件只保存测试 10～14 的无答案投喂语。执行者每次只复制一个代码块到全新会话，不要让被测 Agent 打开本文件，也不要同时提供用例编号、期望行为、历史结果或设计结论，目前还没有尝试重新测试。

## 用例 A

```text
请规划一个混合改动：新增 torch.ops.amct.fake_quant_v2
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

. 职责拆分
依据 repo-map 分层边界与两个 playbook，本任务横跨两个明确职责域，需各自承接：
域	承接方	范围
amct_ops/<op>/ 算子实现	amct-ops-dev	Ascend C kernel、tiling、schema 注册、PrivateUse1 + Meta、torch binding、Python wrapper、CMake/构建、NPU 数值测试
amct_pytorch 通用 quantizer 接入	ops-boundary（本 skill 主子能力）	能力探测、延迟 import、fallback/报错策略、消费点接入、上层单测
统一核对	本 skill	五个接口面一致性 + 兼容矩阵 + repo-map/文档更新判断
当前可复用的事实基础（已核对源码）：
- amct_ops 仅有 hifloat8_cast 一个算子，torch.ops.amct 当前只有 encode_to_hifloat8 / decode_from_hifloat8（amct_ops/hifloat8_cast/op_extension/register.cpp:24-27），fake_quant_v2 无重名冲突。
- 通用 quantizer 的 fake_quant 最终落到 quant_obj(x)：ActivationQuantizer.fake_quant / WeightQuantizer.fake_quant（amct_pytorch/quantization/modules/quant_base.py:118,179），quant_obj 来自 DTYPE_REGISTRY（当前 int/mxfp）。
- INT fake-quant 的现有 Python 数值语义在 amct_pytorch/quantization/dtypes/int_impl.py：weight_quant（per-channel 对称、dim==2 校验、round_ste + clamp [-2^(b-1), 2^(b-1)-1]）与 dynamic_per_token_quant（per-token 对称）。
- 上层调 amct_ops 的既有模式在 amct_pytorch/common/utils/quant_util.py:41-97：hifloat8_supported() liveness 探测（lru_cache）→ 优先原生 torch_npu → 调用点延迟 import amct_ops → 缺失抛友好 ImportError，不让 import amct_pytorch 提前失败。
- 算子模板：amct_ops/hifloat8_cast/ 全目录，构建走 ops_build.sh（自动发现算子），SoC 映射 A2/A3→dav-2201、A5→dav-3510，Meta 只推 shape 不做实算（register.cpp:53-64）。
- 测试锚点：tests/amct_ops/test_hifloat8_cast.py（NPU 数值）、上层 fallback 测试 tests/amct_pytorch/test_hifloat8_fallback.py（存在但本次未读内容）。
2. 前置契约（冻结前均为待确认项，不能据此锁死 schema/消费点）
按 ops-boundary「先冻结共享契约」清单，当前仅确认算子名 fake_quant_v2 与集成范围（NPU op + 通用 quantizer），以下 7 项全部待定稿：
1. schema：参数名/顺序/默认值、Tensor/Scalar/list、输出数量。候选形态取决于粒度：per-channel 需要外传 scale（+zero_point/v，参考 weight_quant(tensor, bits, v)）；per-token 动态量化则无外部 scale 参数。
2. 数值语义：对称/非对称、粒度（per-tensor/per-channel/per-group/per-token）、round 模式、clamp 边界、特殊值。Python oracle 待定：int_impl.weight_quant / dynamic_per_token_quant 的语义是否就是目标语义（注意它们含 torch.abs(tensor).max() 的 scale 计算——scale 计算是否进算子内还是上层算好传入，必须明确）。
3. tensor 契约：device（PrivateUse1/NPU）、dtype、rank（weight 限定 2 维？）、shape、stride/contiguous、空 tensor、非连续输入处理。
4. PyTorch 语义：是否 mutation/alias；是否 STE 反传——PTQ 训练路径需要，而现有 hifloat8_cast 是 torch.no_grad 纯 forward，若 fake_quant_v2 需训练用，round 处的梯度语义（C++ autograd 注册 or 上层包 (round(x)-x).detach()+x）必须在契约里定；torch.compile/fake tensor 行为（Meta 已含）。
5. 兼容策略：v2 命名暗示存在或不存在的 v1 变体；与 int dtype 现有 weight_quant Python 路径的取舍（替换 or 并存），以及旧路径是否保留。
6. fallback：优先顺序待定——现有模式为「原生 torch_npu → amct_ops → 友好报错」；INT 路径目前无原生 NPU fake-quant 参考，需决定是 Python 参考实现兜底还是明确 NPU-only 报错。上层不得因可选 wheel 缺失破坏 CPU/源码环境 import。
7. 目标 SoC/CANN：A2/A3/A5 覆盖范围、最小 CANN 版本、真实硬件验证要求；kernel 需按 GetCoreMemSize() 运行时自适应 UB（不能写死 tileLength）。
3. 条件式实施顺序
契约冻结前，具体文件仅列为候选锚点；顺序按依赖展开：
- 阶段 0｜冻结共享契约（ops-boundary 牵头，跨域确认）：锁死 schema、数值语义、dtype/shape、STE、fallback、SoC 范围，并确定可验证的 Python oracle。此阶段未完成前，不进入阶段 1/2 的具体落点。
- 阶段 1｜算子实现（amct-ops-dev，契约冻结后）：cp -r amct_ops/hifloat8_cast amct_ops/fake_quant_v2 → 重命名替换 → 写 op_kernel（文件头注释 UB 预算、host 算 tile、双缓冲）→ register.cpp 写 TORCH_LIBRARY_FRAGMENT(amct) schema + PrivateUse1 TORCH_CHECK + Meta shape-only → *_torch.cpp 用 c10_npu::getCurrentNPUStream() launch → Python 薄 wrapper + docstring → 更新 amct_ops/README.md「支持的算子」表 → bash ops_build.sh --soc {ascend910b,ascend910_93,ascend950} fake_quant_v2 → tests/amct_ops/test_fake_quant_v2.py 对照 oracle 的 NPU 数值测试。
- 阶段 2｜上层接入（ops-boundary，依赖阶段 1 wheel 可导入）：核对 import amct_ops.fake_quant_v2 成功且 torch.ops.amct.fake_quant_v2 已注册 → 在 quantizer 侧选定消费点（候选：int_impl.py 的 weight/activation fake-quant 路径内做能力探测分支，或 QuantDequantInt.fake_quant 分流；最终由阶段 0 契约决定，非确定落点）→ 采用与 hifloat8_supported() 一致的能力探测 + 调用点延迟 import + 缺 wheel 友好报错 → 上层单测（import 不因缺 amct_ops 失败 / 各路径可控）。
- 阶段 3｜统一核对：五接口面（C++ schema、PrivateUse1/Meta、Python wrapper、上层调用、测试 oracle）签名一致；输出兼容矩阵（新旧 wheel/调用方组合）；判断 repo-map 是否更新。
4. 待确认项与边界提示
- repo-map：本次新增算子不改变框架边界/核心流程，repo-map 无 amct_ops section，默认不更新；amct_ops/README.md 必须更新。
- BLOCKER 遵守：上层消费点不绕过 schema/Meta/import 边界；不在通用 quantizer 加模型专属分支；fake_quant_v2 若不引入新 dtype/algorithm 注册闭环则仅属算子接入。
- 证据等级：阶段 1 编译成功≠NPU 数值正确，A3/A5 至少保证编译通过；缺 NPU/CANN 环境下本规划不宣称任何验证。
本轮仅读取：repo-map、ops-boundary + amct-ops-dev 两个 skill、hifloat8_cast 模板全目录、quant_base/int/int_impl/quant_util 相关切片、两个测试/构建入口文件。未读取 .agents/docs/amct_coder_analysis/，未使用 git diff/log/show/blame，未搜索题面原句。

消耗6.5wtoken

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
分析完成。以下为只读规划结论。
1. 职责归属
本次改动完全落在 amct-coder 的 deploy-export-code 子能力 OWN 范围内（llm_deploy.py + deploy_export.py），无需路由外部 skill。改动面与已确认的活跃调用链：
面	候选落点（契约冻结前仅列候选）
index 生成	LlmDeployWorkflow._refresh_weight_index（llm_deploy.py:186-200）
config 生成	_refresh_config（:157-174）→ generate_quant_config（deploy_export.py:52-74）
测试	tests/unit_test/workflows/test_llm_deploy.py（index：test_refresh_weight_index_writes_metadata_and_total_size :228；config：:248/:263/:628；tensorwise 集成：:545/:642）；helper：tests/unit_test/common/models/llm/common/test_deploy_export.py（:80-118）
关键边界事实：_refresh_config_tensor（:176-184）与 _convert_tensor（:110-115）无运行时调用方（blockwise/tensorwise 主路径均调 _refresh_config），不得把二者描述为当前行为路径。
2. 待确认契约（字段名/来源/消费者未说明，全部待定稿）
按 playbook 门禁 #2 与 BLOCKER 首条，冻结前不得写入 index/config：
1. 字段名：index 与 config 中的字段名及层级（index 的 metadata 子对象？config 的 quantization_config 子对象？还是顶层？）。
2. 值来源：候选包括 args/CLI 参数、pipeline adapter 方法（现有模式为 bits_scheme()/cache_scheme()）、quant_dtype 推导、或 export_deploy() payload 汇总。来源未定，不能在 workflow 内发明取值逻辑。
3. 类型：str/dict/int/float/list。
4. 适用路径：blockwise / tensorwise / bf16 / int / mxfp 哪些路径写、哪些不写。
5. 覆盖策略：与 original_index["metadata"] 中同名键冲突时覆盖、保留还是报错。
6. 下游消费者：谁读取该字段（infer 仓 / vLLM / transformers loader），决定 schema 兼容性。
7. 权威来源：index 与 config 两处若表达同一语义，须明确谁是权威、如何防复制漂移（门禁 #3）。
8. 旧产物兼容性：旧产物缺字段的行为、新产物对旧消费者的影响。
9. 测试断言：新字段默认值、缺失兼容、两路径一致性。
3. 条件式代码修改计划
- 阶段 0｜冻结契约：确认上表 9 项；同时确认值来源在 workflow 可及范围（来自 pipeline/adapter 或 args）。若来源需要 adapter 新增方法，转 model-adapter review，workflow 不写模型分支（门禁 #7）。
- 阶段 1｜index 侧：在 _refresh_weight_index 构造 metadata 的调用点（llm_deploy.py:187-191 附近）写入新字段；必须同步处理两处调用方（:232 blockwise、:279 tensorwise），确认取值在调用前已从来源取得并传入/挂到实例上。遵守：保留原 metadata、重算 total_size、weight_map 不丢项。
- 阶段 2｜config 侧：若字段进 quantization_config，修改 generate_quant_config（deploy_export.py:52-74）——注意其唯一调用方是 :166，改签名需同步 _refresh_config 与 helper 测试；若进顶层，改 _refresh_config。确认 bf16 路径（quant_dtype is not None 分支，见测试 :628）的字段行为。
- 阶段 3｜测试同步：test_refresh_weight_index_writes_metadata_and_total_size 增新字段断言；config 三处测试（:248/:263/:628）与 test_deploy_export.py 的 generate_quant_config 测试（:80-118）增断言；tensorwise 集成测试（:545/:642）验证两条路径字段一致；若含默认/缺失兼容语义，补对应用例。
- 阶段 4｜验证：pytest tests/unit_test/workflows/test_llm_deploy.py + pytest tests/unit_test/common/models/llm/common/test_deploy_export.py。真实模型导出需权重与运行环境，未执行时仅声称单测覆盖，不升级为端到端结论。
- repo-map：本次为局部字段变更，不改变框架边界/核心流程语义，默认不更新；若下游消费契约确立且公共产物格式变化，再评估。
4. 待确认项与证据等级
- 本次分析未改动任何文件，未运行测试/构建/deploy 命令。
- 本轮仅读取：repo-map（复用上轮）、deploy-export-code playbook、llm_deploy.py（全文）、deploy_export.py（1-120 行）、test_llm_deploy.py（全文）、test_deploy_export.py（通过 glob 确认存在 + grep 定位断言）。未读取 .agents/docs/amct_coder_analysis/，未使用 git diff/log/show/blame，未搜索题面原句。

2wtokens
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
1. 职责归属
主职责落在 amct-coder 的 dtype-quant-module 子能力（OWN：quantization/dtypes/、quantization/modules/、bit_policy.py）；CLI、deploy、NPU 属下游闭环，仅在真实契约变化时进入。关键事实链（已核对源码）：
- 注册：DTYPE_REGISTRY（dtypes/__init__.py:22），register_dtype() 通过 import 链注册 int+mxfp（:32-33），_REGISTERED 幂等。新 dtype 不入该 import 链则 registry 不可达。
- 模板：QuantDequantInt（dtypes/int.py，export_deploy 返回 qweight/weight_scale/weight_bias）与 QuantDequantMx（dtypes/mxfp.py，block_size=32，bits=4 时 E2M1 打包 + e8m0 共享指数，export_deploy 返回 qweight/weight_scale）；两者 forward 在 bits==16 直通。
- quantizer 消费：ActivationQuantizer/WeightQuantizer 均 DTYPE_REGISTRY.get(args.quant_dtype) 实例化（modules/quant_base.py:112,150），fake_quant 落 quant_obj(x)，export_deploy 走 quant_obj.export_deploy(x, v) 与 algorithm quantize() hook。
- CLI：--quant_dtype choices 在 cli/llm/args.py:67，当前 ['int', 'mxfp']。
- deploy 消费：llm_deploy.py 用 startswith("mx"/"int"/"hif") 分类（:58-60），tensorwise 用 quant_dtype in ["int", "mxfp"] 精确分支（:269），generate_quant_config 以 is_mx 二分 float/int 格式（deploy_export.py:58）——新名字必须逐处判断。
2. 必须冻结的 dtype 契约
按 playbook「先冻结 dtype 契约」九项 + 防重复门禁，冻结前不臆造 FP4 语义：
1. registry 名 + CLI 值：quant_dtype 字符串（候选如 float4/fp4/float4_e2m1），须与现有 int/mxfp 不冲突。
2. 支持 bit 集合：{4} 还是 {4,8}；bits==16 在 fake quant / quant module / deploy 的直通语义。
3. weight / activation / cache 适用范围：不能从"fp4-like"推断 weight-only 还是 weight+activation。
4. 编码格式：纯 FP4 E2M1（per-element 编码 + per-channel/group scale）还是 MXFP4（每 32 元素共享 e8m0 指数）。已有冲突面须先澄清：新路径 mxfp 的 bits=4 已是 MXFP4（mxfp.py:76-79 + mxfp_impl.py f4 打包）；classic 层已有 FLOAT4_E2M1 用法（classic/quantize_op/minmax_module.py、linear_awq_module.py、utils.py，不同注册表体系）；experimental 有 MXFP4（MiniMax-M2.5/mxfp4_quantizer、deepseek-v4/mx_quantize.py）。不同代码体系不能直接复用注册表，且须说明新 dtype 与这些既有实现的区别。
5. 粒度：per-token / per-channel / group / block、量化维度、block size、非整除行为（mxfp.quant 现按 block_size unflatten 末维，不整除报错）。
6. real quant layout：packing 顺序、scale/bias dtype 与 shape、payload 字段名、反量化契约（参考 mxfp_impl.weight_dequant / unpack_mxfloat4_to_fp32）。
7. algorithm hook：是否支持 rounding offset v、自定义 quantize()/export_deploy()（WeightQuantizer.export_deploy 已能消费二者）。
8. deploy config：qtype/format 归属、blockwise/tensorwise 支持范围、generate_quant_config 的 is_mx 二分是否要扩展、下游消费者。
9. NPU kernel：是否需要；需要则保留 Python 语义基线，kernel/schema/Meta 转 amct-ops-dev（上层软依赖归 ops-boundary）。
3. 条件式分层计划
契约冻结前所有落点均为候选；按依赖展开：
- 阶段 0｜冻结契约：上述 9 项 + 与 classic/experimental/现有 mxfp bits=4 的重复与差异声明。
- 阶段 1｜dtype/impl 层：新增 dtypes/<new>.py + <new>_impl.py（fake quant、real quant、packing/dequant、输入约束、bits==16 直通），复用 int.py/mxfp.py 及 impl 模式。
- 阶段 2｜注册：dtypes/__init__.py 的 register_dtype() 加入 import 链；验证重复注册/未知名/导入顺序。
- 阶段 3｜quant module：registry 构造已满足则不改 quant_base.py（BLOCKER：不新增 dtype 分支）；仅在接口不兼容时修改。
- 阶段 4｜bit policy / CLI：cli/llm/args.py:67 choices 扩展；仅允许真实支持的 bit/weight/activation 组合进入入口。
- 阶段 5｜deploy（若参与）：按冻结 payload 更新 llm_deploy.py 前缀分类与 tensorwise 精确分支、generate_quant_config 格式判定；此阶段完整读取 deploy-export-code。
- 阶段 6｜algorithm / 模型侧审计：仅当现有算法显式区分 dtype 时逐个确认支持/拒绝/fallback；模型侧依赖（如 deepseek_v4 的 expert fp4）需兼容性 review，结构实现转 model-adapter。
- 阶段 7｜NPU 加速（若需）：Python 路径作语义基线/fallback，转 ops-boundary + amct-ops-dev。
4. 影响范围与测试
面	文件
dtype/impl	quantization/dtypes/（新文件 + __init__.py）
quant module	quantization/modules/quant_base.py（仅接口不兼容时）
CLI	cli/llm/args.py:67
deploy	workflows/llm_deploy.py、common/models/llm/common/deploy_export.py
常量复用	common/utils/vars.py（已有 FLOAT4_E2M1/MXFP4_E2M1/group size 映射，classic 体系）
repo-map：新增 dtype 属新能力，不改框架边界/核心流程/算法分流，默认不更新；若 deploy 公共产物格式变化再评估。
5. 证据等级与待确认
- 本次未修改文件、未运行测试/构建/量化/deploy/模型命令。
- 已确认：注册链、两个 dtype 模板、quantizer 消费、CLI choices、deploy 分类、classic/experimental 既有 FP4 基础设施（防重复）。
- 待确认：见契约 1-9。真实 eval/deploy/NPU 数值未执行前，不把单测覆盖表述为端到端已验证。
- 本轮仅读取：repo-map（复用）、dtype-quant-module playbook、int.py/mxfp.py/int_impl.py/mxfp_impl.py/__init__.py/vars.py 相关切片、cli/llm/args.py 相关行、test_dtypes.py（全文）、test_llm_module_entrypoints.py（全文）、FP4 相关既有引用。未读取 .agents/docs/amct_coder_analysis/，未使用 git diff/log/show/blame，未搜索题面原句。

消耗2.5wtoken
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

拟议方案不成立（命中 BLOCKER）
「在 classic/graph_based 下新增 optimizer pass，再用 LLM new-path ALGO_REGISTRY 注册并通过 --algos 选择」——这是 classic-graph BLOCKER 首条与 amct-coder 全局 BLOCKER「不混用 LLM new-path ALGO_REGISTRY 与 classic AlgorithmRegistry」的直接违反。三个注册/选择体系互不相通，方案在契约层就无法成立。
问题清单（附源码证据）
1. classic graph_based pass 根本没有 pass 注册表，不是"注册/选择"模型
graph_based pass 的调度方式是显式编排：GraphOptimizer.add_pass()（graph_based/amct_pytorch/optimizer/graph_optimizer.py:22,35）与 ModelOptimizer.add_pass()（model_optimizer.py:22,35），实际调用点散落在 quantize_tool.py（如 :138-140、:316、:711-718、:749-755）、quant_calibration_interface.py、prune_interface.py、distillation_interface.py 等。pass 通过 opt.<Pass> 使用，需同步 optimizer/__init__.py 的 module import、class import 与 __all__（optimizer/__init__.py:72,116）。"注册到 ALGO_REGISTRY"这个概念对 graph_based 不存在。
2. ALGO_REGISTRY 契约与 graph pass 完全不相容
ALGO_REGISTRY = Registry("algo")（algorithms/registry_factory.py:20），其条目是 torch.nn.Module 子类 + targets=("weight","activation","structure") 元数据（例：algorithms/quant/auto_clip.py:23-28 的 LWC），由 quant_base.py 的 get_algo_names_by_target / build_algorithms_by_target 以 nn.ModuleDict 实例化（:29-68）并挂在 quantizer 的 forward 内执行。graph pass 实现的是 match_pattern(node/module, ...) + do_pass(graph, node, model)，不是可被 (args) 构造的 nn.Module——硬塞进 ALGO_REGISTRY 后无法被 _build_algorithm 实例化，--algos 分流（activation/weight/structure）也无从选中它。
3. --algos 无图级语义
--algos 只作用于 LLM new-path 的 quantizer 算法组，与 pass 顺序、optimizer 调度、图拓扑完全不相关。即使用 --algos 声明该 pass，也不会在任何图流程里被执行，形成"注册了但永无 add_pass() 调用点"的死代码（classic-graph BLOCKER 第二条）。
4. 归因澄清：classic eager 的 AlgorithmRegistry = Algorithm()（algorithms/__init__.py:45，register(name, src_op, quant_op, deploy_op)）与 LLM ALGO_REGISTRY 是两个不同对象，两者都不是 graph pass 的入口；experimental flatquant 走的是 classic AlgorithmRegistry（experimental/flatquant/__init__.py:7-11），也不构成先例。
正确实施计划（条件式）
契约未冻结前，pass 具体位置/执行对象均按条件展开：
先定执行对象：ONNX graph node → BaseFusionPass + GraphOptimizer；PyTorch module → BaseModuleFusionPass + ModelOptimizer。两套 pass 基类、run/do_pass 签名和执行对象不能混用。
实现 pass：match_pattern + do_pass（及必要 setup/teardown），维护 graph 拓扑、anchor、initializer 与 module 状态一致。
接入命名空间：同步 optimizer/__init__.py 的 module import、class import、__all__，使调用方可写 opt.<NewPass>。
确定唯一真实编排点：反查目标 pass 前后相邻 pass 的数据不变量，在 quantize_tool.py 等真实 add_pass() 序列中插入，不能把目录存在当已接通。
按需扩展依赖：仅当 pass 确实读写 Configuration、RecordFileParser、record、custom_op/proto 时才扩展对应语义。
测试：tests/amct_pytorch/testcase_python/optimizer/test_<pass>.py 覆盖 match 成功/失败、图修改、pass 顺序、异常输入；classic 测试可能依赖 ONNX/protobuf/NPU，缺依赖时说明原因。
若真实意图是 LLM new-path 功能：则应作为带 targets 元数据的 nn.Module 算法落在 amct_pytorch/algorithms/ 下，而不是 graph_based pass——两个能力域不可互相替代。
待确认项与证据等级
待确认：pass 的执行对象（ONNX/module）、目标领域（quant/prune/distill 等）、相邻 pass 与数据依赖、调用方命名空间。
本次仅读取：repo-map（复用）、classic-graph playbook、registry_factory.py、algorithms/__init__.py（grep 命中）、auto_clip.py（1-40 行）、quant_base.py（复用上轮读取）、graph_based 目录结构与 add_pass 调用点、optimizer/__init__.py 导出模式。未读取 .agents/docs/amct_coder_analysis/，未使用 git diff/log/show/blame，未搜索题面原句。未修改文件、未运行任何测试/构建/图转换命令。

消耗约1wtokens

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
规划完成。以下为只读规划结论。
1. 职责归属
环节	承接方
实际执行 eval 命令	quant-run（implementer）
结果判读（delta 是否达标、下一步）	direct-quant-eval（reviewer）
方案/算法设计（仅当直转掉点需升级 PTQ）	scheme-recommendation / algorithm-recommendation
编排与契约核对	amct-coder
本次用户仅给定模型族（Qwen3）与目标精度（bf16 / W4A8），其余输入全部缺失，命令无法落定。
2. 缺失输入（执行前置契约）
1. --model：Qwen3 checkpoint 目录路径（必填）。
2. --model_name：已注册适配器名，应为 qwen3（repo-map 当前注册集合含 qwen3；examples/eval.sh 示例用的是 qwen3_5）。缺省会默认 deepseek 误用，必填。
3. --device：npu:N（当前设备号）。
4. --quant_target：mlp / moe / attn-linear / attn-cache 之一（cli/llm/args.py:52）。一次聚焦一个角色，多角色分别跑。
5. --quant_dtype：W4A8 通常 int（weight int4 + activation int8 走 QuantDequantInt）；若目标为 MX 体系则 mxfp。quant 模式必填，缺报 KeyError: '' is not registered in 'dtype'。
6. --bit_config：quant 用 amct_pytorch/configs/w4a8.yaml，bf16 用 amct_pytorch/configs/bf16.yaml（两者均已在 configs/ 确认存在）。
7. 数据源：wikitext2/pileval 等评测数据可达性。离线容器不可达属环境/调用方责任——需 HF_ENDPOINT 镜像 / modelscope / 本地数据路径；仍不可达按 progress 协议写 BLOCKED 停下。
8. 输出目录/工作区：run log 与 progress 落点（用于后台轮询与结果记录）。
3. 执行顺序（条件式，命令未落定前不运行）
- 步骤 1｜bf16 baseline（quant-run，按 examples/eval.sh 模板）：
python -m amct_pytorch.eval --model <path> --model_name qwen3 --device npu:N --granularity block --eval_mode bf16 --bit_config amct_pytorch/configs/bf16.yaml --seq_len 4096 → ppl_bf16。
- 步骤 2｜W4A8 直转评测（quant-run）：
python -m amct_pytorch.eval --model <path> --model_name qwen3 --device npu:N --granularity block --eval_mode quant --quant_target <target> --quant_dtype int --bit_config amct_pytorch/configs/w4a8.yaml --seq_len 4096 → ppl_quant。
- 长命令防超时：大模型 eval 单段约 20min+，须 nohup ... > run.log 2>&1 & 后台启动，再多次短轮询 grep -E "PPL:|Error" run.log + ps -p <pid> 至结束，不能单条阻塞等待。
- 步骤 3｜判读（direct-quant-eval）：算 delta = ppl_quant − ppl_bf16，默认阈值 delta ≤ 0.2。达标→记录；未达标→按范围缩窄或升级 PTQ（extract_ptq_data → ptq --algos lwc/lac/... → 结果评测，注意 --data_dir 一致与 --algos 一致）。
- 步骤 4｜progress 归档：写 ppl_bf16 / ppl_quant / ppl_algo、命令摘要、产物路径与状态块；不判达标（判读归 reviewer）。
4. 待确认项与证据等级
- 待确认：上表 8 项缺失输入；W4A8 的 quant_target 覆盖范围（是否含 attn-cache）。
- 本次仅读取：repo-map（复用）、quant-run skill、examples/eval.sh（权威模板）、amct_pytorch/configs/ 文件清单、cli/llm/args.py 相关行。未读取 .agents/docs/amct_coder_analysis/，未使用 git diff/log/show/blame，未搜索题面原句。
- 未修改文件、未运行任何评测/量化/构建命令；两组 PPL 尚未产生，不宣称任何精度结论。
消耗约6ktokens