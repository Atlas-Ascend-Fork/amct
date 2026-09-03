# AMCT 量化 Agent 测试集

## 测试 1：BATQuant 与 MXFP4 Group-wise Quantization

**测试语：**

在 AMCT 量化框架中开发 BATQuant 算法，接入现有 PTQ pipeline，并支持 MXFP4 group-wise quantization。请跟踪并说明 Agent 从需求解析、代码定位、方案设计到验证设计的处理数据流，检查其是否符合 AMCT 现有分层和调用链预期。本次只输出任务规划，不修改代码，不执行量化，不写入任何文件。

### Agent 处理结果

#### 一、需求澄清

这部分是我瞎选的
| 问题 | 回答 |
|------|------|
| BATQuant 应接入哪条量化管道？ | 经典管道 (Classic) |
| MXFP4 group-wise 量化范围？ | 仅权重量化 (weight-only) |
| BATQuant 的校准/训练机制？ | 基于 Hessian 的 block 优化 |
| BATQuant 中 "block" 的粒度？ | Transformer 层级别 |

#### 二、方案对比

##### 方案 A：类 GPTQ 逐层量化 + 块级上下文修正

核心思路：BATQuant 作为 `BaseQuantizeModule` 子类，每个 Linear 层独立收集 Hessian，但在同一 Transformer 层的所有 Linear 完成校准后，执行一次块级联合修正。

```
quantize() → 每个 Linear 替换为 BATQuant(Linear)
calibration → 每个 BATQuant.forward() 收集 Hessian
             → 同层所有 BATQuant 校准完成后，触发块级修正
convert()   → 替换为 NpuWeightQuantizedLinear
```

| 维度 | 评价 |
|------|------|
| 与现有架构兼容性 | 高 — 复用 BaseQuantizeModule + AlgorithmRegistry |
| 块级联合优化 | 支持 — 校准完成后触发块级修正 |
| MXFP4 group-wise | 支持 — 复用现有工具链 |
| 复杂度 | 中 |

##### 方案 B：块级整体替换（修改 Pass 基础设施）

核心思路：新增 `InsertBlockQuantizeModulePass`，将整个 Transformer 层（attention + FFN）替换为一个 `BATBlockQuant` 模块。

| 维度 | 评价 |
|------|------|
| 与现有架构兼容性 | 低 — 需修改 pass 基础设施 |
| 块级联合优化 | 天然支持 — 整层作为单模块 |
| MXFP4 group-wise | 支持 |
| 复杂度 | 高 |

##### 方案 C：两阶段管线（预提取 + 逐层量化）

核心思路：
1. 阶段 1：新增 pre-quantization pass，遍历整个模型收集块级 Hessian 信息，存入共享 buffer
2. 阶段 2：标准逐层 BATQuant，从共享 buffer 读取块级上下文进行权重优化

| 维度 | 评价 |
|------|------|
| 与现有架构兼容性 | 中 — 新增 pass 但不修改现有 |
| 块级联合优化 | 支持 — 通过共享 buffer 传递上下文 |
| MXFP4 group-wise | 支持 |
| 复杂度 | 高 |

##### 推荐方案 A

理由：
1. 最小侵入性 — 不修改 pass 基础设施
2. 块级修正可在 `BaseQuantizeModule.forward()` 的 `cur_batch == batch_num` 触发点中实现（与 GPTQ 的 Cholesky 优化在同一位点）
3. MXFP4 group-wise 已有完整工具链可直接复用
4. 符合 AMCT 现有分层：注册 → 校准 → 转换

## 测试 2：基于 INT4 扩展 HIF4 路径

**测试语：**

基于 AMCT 现有 INT4 权重量化实现扩展 HIF4 路径，要求原有 INT4 行为、导出格式和测试结果不受影响。请跟踪并说明 Agent 从现有实现分析、兼容性边界识别、改动范围设计到回归验证设计的处理数据流，检查其是否满足向后兼容预期。本次只输出任务规划，不修改代码，不执行测试，不写入任何文件。
# Todos
探索 AMCT INT4 权重量化完整路径：配置→量化→导出→部署
识别 INT4 兼容性边界：配置验证、dtype 矩阵、导出格式、测试用例
分析 HIF4 与 INT4 的差异点和可复用点
提出 2-3 种 HIF4 扩展方案并给出推荐
设计回归验证方案确保向后兼容
输出完整任务规划和数据流分析
+ Thought: 663ms
Explore Task — Trace INT4 weight quantization full path
↳ 40 toolcalls · 4m 41s
Explore Task — Find INT4 test cases and HiFloat8 patterns
↳ 43 toolcalls · 5m 18s
Explore Task — Explore INT4 dtype matrix and vars
↳ 45 toolcalls · 4m 13s
ctrl+x down view subagents
+ Thought: 21.6s
# Todos
探索 AMCT INT4 权重量化完整路径：配置→量化→导出→部署
识别 INT4 兼容性边界：配置验证、dtype 矩阵、导出格式、测试用例
分析 HIF4 与 INT4 的差异点和可复用点
提出 2-3 种 HIF4 扩展方案并给出推荐
设计回归验证方案确保向后兼容
输出完整任务规划和数据流分析
+ Thought: 488ms
已完成全面探索。在设计 HIF4 扩展方案前，需要澄清一个关键定义：
# Questions
HIF4 的具体数据格式是什么？它与现有 FLOAT4_E2M1（软件查表，max=6.0）和 MXFP4_E2M1（MX 格式共享指数）的区别在哪里？
自定义格式
+ Thought: 98ms
# Questions
请描述 HIF4 的具体格式参数：
1. 指数位数和尾数位数（如 E2M1）？
2. max_norm（最大可表示值）？
3. 是否有原生 NPU 类型支持（torch_npu.hifloat4）还是纯软件实现？
4. 是否需要 shared exponent（MX 格式）还是 per-tensor/per-channel scale？
5. 与 INT4 的核心区别是量化方式（浮点 vs 整数）还是存储格式？
你自己设定吧
+ Thought: 23.0s
# Todos
探索 AMCT INT4 权重量化完整路径：配置→量化→导出→部署
识别 INT4 兼容性边界：配置验证、dtype 矩阵、导出格式、测试用例
分析 HIF4 与 INT4 的差异点和可复用点
提出 2-3 种 HIF4 扩展方案并给出推荐
设计回归验证方案确保向后兼容
输出完整任务规划和数据流分析
+ Thought: 628ms
好的，以下是我基于代码库分析设计的完整任务规划。
一、HIF4 格式定义
基于 AMCT 现有模式（HiFloat8 参考 + FLOAT4_E2M1 参考），HIF4 定义如下：
属性
常量名
格式
max_norm
量化类型
存储 dtype
scale 策略
支持算法
与 INT4 核心区别
二、INT4 兼容性边界识别
通过代码分析，以下是 HIF4 扩展必须保持不动的 INT4 边界：
2.1 dtype 常量与矩阵层（vars.py）
┌─────────────────────────────────────────────────────────┐
│ SUPPORT_WEIGHT_QUANT_DTYPE    → 新增 'hifloat4'         │
│ SUPPORT_INPUT_QUANT_DTYPE     → 不加（仅权重量化）       │
│ ALGORITHM_SUPPORTED_QUANT_TYPE_COMB                     │
│   → 新增 'NOT_QUANTIZE hifloat4': ['awq','gptq','minmax']│
│   → 新增 'int8 hifloat4': ['minmax','smoothquant']      │
│ ALLOWED_WEIGHT_DTYPES         → 新增 hifloat4 条目       │
│ WTS_PER_*_SUPPORT_COMBINATION → 新增 hifloat4 组合       │
│ GROUP_SIZE_SUPPORTED_DTYPE/MAP → 新增 hifloat4           │
└─────────────────────────────────────────────────────────┘
兼容性约束：所有 INT4 现有条目 零修改。新增条目以 'hifloat4' 为 key，不改动 'int4' 的任何值。
2.2 配置解析层（fields.py + parser.py）
检查点
WeightsCfgField.check()
WTS_ASYMMETRIC_DTYPE
check_group_size()
check_quant_op_constraint()
_check_quant_dtype_comb_rules()
兼容性约束：所有 if quant_data_comb in ['NOT_QUANTIZE int4', 'int8 int4'] 的分支需追加 hifloat4 组合，但 不修改原有条件。
2.3 量化核心层（quant_util.py + utils.py）
函数	INT4 路径
get_int_quant_scope('int4')	返回 7 或 8
get_float_quant_scope()	不包含 INT4
calculate_scale_offset()	INT4 走整数分支
convert_dtype()	round().clamp(-8,7).to(int32)
quant_tensor()	round().clamp(-8,7).to(int32)
quant_dequant_tensor()	通用 dequant 路径
2.4 校准算子层（quantize_op/）
文件
linear_awq_module.py:86
gptq_module.py
smooth_quant_module.py
minmax_module.py
2.5 部署算子层（deploy_op/）
文件
weight_npu_quant_module.py:37
weight_npu_quant_module.py:107
npu_quantization_linear.py:201
npu_quantization_linear.py:262
三、扩展方案对比
方案 1：独立 dtype 新增（推荐）
核心思路：HIF4 作为独立 dtype 加入 AMCT，与 INT4 共享部署路径，但有独立的量化/反量化逻辑。
vars.py:     HIFLOAT4 = 'hifloat4'  (新常量)
              → 加入 SUPPORT_WEIGHT_QUANT_DTYPE
              → 新增 ALGORITHM_SUPPORTED_QUANT_TYPE_COMB 条目
quant_util.py: → convert_dtype() 新增 HIFLOAT4 分支 (float_to_fp4e2m1)
              → quant_tensor() 新增 HIFLOAT4 分支
              → 不修改任何 INT4 分支
deploy_op/weight_npu_quant_module.py: → 白名单追加 HIFLOAT4
              → int4pack 路径追加 HIFLOAT4
维度
INT4 兼容性
代码复用
改动文件数
风险
方案 2：INT4 参数化扩展
核心思路：不新增 dtype，而是在 INT4 配置中添加一个 float_mode: True 参数，复用 INT4 全部路径，仅在 scale 计算时切换 quant_scope。
# 配置示例
{'weights': {'type': 'int4', 'symmetric': True, 'strategy': 'channel', 'float_mode': True}}
维度
INT4 兼容性
代码复用
改动文件数
风险
方案 3：继承 FLOAT4_E2M1 扩展
核心思路：HIF4 本质是 FLOAT4_E2M1 的"HiFloat 原生路径版"。将 HIF4 作为 FLOAT4_E2M1 的一个子模式。
维度
INT4 兼容性
代码复用
改动文件数
风险
推荐方案 1：独立 dtype 新增最干净，与 HiFloat8 的扩展模式完全一致，且 INT4 兼容性最强（零侵入）。
# Todos
探索 AMCT INT4 权重量化完整路径：配置→量化→导出→部署
识别 INT4 兼容性边界：配置验证、dtype 矩阵、导出格式、测试用例
分析 HIF4 与 INT4 的差异点和可复用点
提出 2-3 种 HIF4 扩展方案并给出推荐
设计回归验证方案确保向后兼容
输出完整任务规划和数据流分析
+ Thought: 2ms
四、回归验证方案（确保向后兼容）
4.1 回归测试矩阵
测试层级	测试文件
单元测试 — 配置验证	test_fields.py
 	test_parser.py
 	test_parser.py
 	test_parser.py
单元测试 — 量化核心	test_quant_util.py
 	test_quant_util.py
 	test_quant_util.py
 	test_quant_util.py
单元测试 — dtype	test_dtypes.py
 	test_dtypes.py
集成测试 — AWQ	test_awq.py
集成测试 — GPTQ	test_gptq.py
集成测试 — MinMax	test_minmax.py
集成测试 — SmoothQuant	test_smoothquant.py
集成测试 — Config	test_config.py
集成测试 — 端到端	test_quantize.py
部署测试	test_deploy_kv_quant.py
4.2 新增 HIF4 测试矩阵
测试层级
单元 — config
 
 
 
单元 — quant_util
 
 
 
集成 — AWQ
 
集成 — GPTQ
集成 — Config
端到端
4.3 向后兼容性验证流程
Phase 1: 基线建立
  └→ 运行全部 INT4 现有测试，记录基线结果（pass/fail + 数值）
  
Phase 2: HIF4 代码合入
  └→ 运行全部 INT4 测试 → 必须全部通过（零回归）
  └→ 运行全部 HIF4 新增测试 → 必须全部通过
  
Phase 3: 交叉验证
  └→ 同时存在 INT4 + HIF4 配置的模型量化 → 互不干扰
  └→ 验证 ALGORITHM_SUPPORTED_QUANT_TYPE_COMB 无冲突
  
Phase 4: 导出格式验证
  └→ INT4 导出 → npu_convert_weight_to_int4pack → 与基线 bit-identical
  └→ HIF4 导出 → 同样走 int4pack（但 scale 值不同）
五、完整数据流分析：Agent 处理流程
5.1 需求解析阶段
输入: "基于 INT4 扩展 HIF4 路径"
  │
  ├→ Agent 解析: HIF4 = 浮点4-bit量化 (E2M1, max=6.0)
  │              与 INT4 (整数4-bit, max=7) 的核心差异是 quant_scope
  │
  ├→ 约束提取: 原有 INT4 行为零影响
  │              导出格式复用 int4pack
  │              测试结果不回归
  │
  └→ 范围界定: 仅权重量化, AWQ/GPTQ/MinMax 算法
5.2 代码定位阶段
Agent 搜索链:
  grep('int4') → 353 matches across amct_pytorch/
    │
    ├→ vars.py: dtype 常量定义 → 入口点
    ├→ config.py: 预定义配置 → 扩展点
    ├→ fields.py: 配置验证 → 条件扩展
    ├→ parser.py: 形状约束 → 条件扩展
    ├→ quant_util.py: 量化/反量化 → 新增分支
    ├→ utils.py (quantize_op): scale 计算 → 新增浮点路径
    ├→ linear_awq_module.py: AWQ 特殊处理 → 追加类型
    ├→ weight_npu_quant_module.py: 部署打包 → 追加类型
    └→ npu_quantization_linear.py: W4A8 部署 → 追加类型
5.3 方案设计阶段
决策树:
  HIF4 是新 dtype 还是 INT4 模式?
    → 新 dtype (方案1) [推荐, 与 HiFloat8 模式一致]
  
  HIF4 走整数还是浮点量化路径?
    → 浮点路径 (QUANT_SCOPE['hifloat4'] = 6.0)
    → 走 get_float_quant_scope() 而非 get_int_quant_scope()
  
  HIF4 部署打包格式?
    → 复用 int4pack (npu_convert_weight_to_int4pack)
    → 与 INT4 相同的 NPU 算子 (npu_weight_quant_batchmatmul)
  
  HIF4 配置入口?
    → type: 'hifloat4' (独立 dtype 字符串)
    → algorithm: {'awq': {...}} (复用现有算法)
5.4 改动范围精确清单（按优先级）
#	文件
1	common/utils/vars.py
2	common/utils/quant_util.py
3	classic/quantize_op/utils.py
4	classic/quantize_op/linear_awq_module.py
5	classic/deploy_op/weight_npu_quant_module.py
6	classic/deploy_op/npu_quantization_linear.py
7	common/config/config.py
8	algorithms/__init__.py
5.5 向后兼容性验证检查点
Check 1: vars.py 零修改验证
  → diff INT4 现有 6 个 map 条目 → 必须完全相同
  → 新增条目使用新 key 'hifloat4'，不改 'int4' 的值

Check 2: 量化输出隔离验证
  → INT4 quant_tensor() → torch.int32, [-8,7]
  → HIF4 quant_tensor() → torch.float32, [-6,6]
  → 两个路径无交叉

Check 3: 导出格式验证
  → INT4 deploy → int4pack (unchanged)
  → HIF4 deploy → int4pack (same NPU op, different scale values)
  → 导出文件 bit-identical 对比

Check 4: 测试回归验证
  → 运行全部 INT4 测试 (353 个引用点的测试用例)
  → 每个测试必须保持原结果

Check 5: 配置解析验证
  → INT4 配置通过 → HIF4 配置通过 → 互不干扰
  → skip_layers 对两种 dtype 都生效
六、总结
维度
推荐方案
核心改动
INT4 兼容性
复用度
回归风险
验证策略
# Todos
探索 AMCT INT4 权重量化完整路径：配置→量化→导出→部署
识别 INT4 兼容性边界：配置验证、dtype 矩阵、导出格式、测试用例
分析 HIF4 与 INT4 的差异点和可复用点
提出 2-3 种 HIF4 扩展方案并给出推荐
设计回归验证方案确保向后兼容
输出完整任务规划和数据流分析
+ Thought: 1.5s
任务规划完成。以上输出覆盖了：
需求解析 → HIF4 定义为 E2M1 浮点 4-bit（quant_scope=6.0），与 INT4（整数 4-bit，quant_scope=7）的核心差异
代码定位 → 353 个 INT4 引用点的完整扫描，8 个核心改动文件精确定位
方案设计 → 推荐独立 dtype 方案（与 HiFloat8 扩展模式一致），8 个文件全部为追加式修改
验证设计 → 4 阶段回归验证 + INT4/HIF4 测试矩阵，确保向后兼容
所有改动均不修改 INT4 现有分支，通过 dtype 字符串隔离实现完全解耦


## 测试 3：BATQuant 敏感层自动识别

**测试语：**

为 BATQuant 设计敏感层自动识别逻辑，根据 activation error 或 Hessian 近似自动决定跳过量化的层。请跟踪并说明 Agent 从指标选择、数据采集、阈值或排序策略、跳层决策到效果验证设计的处理数据流，检查其是否符合现有 PTQ pipeline 的职责边界。本次只输出任务规划，不修改代码，不运行实验，不写入任何文件。

### Agent 处理结果

#### 一、PTQ Pipeline 职责边界分析

通过对 6 层架构的逐一审查，建立如下职责矩阵：

| 层级 | 职责 | DECIDES | EXECUTES | CANNOT |
|------|------|---------|----------|--------|
| Config (`fields.py`) | 语法校验 | 值域、类型、组合合法性 | — | 看不到模型 |
| Parser (`parser.py`) | 层级资格判定 | op 类型匹配、dtype 兼容、shape 约束、用户 skip_layers | — | **不做敏感性分析** |
| Pass (`optimizer/`) | 模块替换 | 选择 deploy op（weight-only vs W+A） | 机械安装 quant_op / deploy_op | 分析模型内容 |
| QuantOp (`quantize_op/`) | 校准计算 | 无（始终执行 forward） | 计算 scale/offset/Hessian/AWQ search | 自主跳过自己 |
| Algorithm (`algorithms/`) | 算法函数 | 无（无状态函数） | 提供 search_scale/compute_mse 等 | 与 pipeline 通信 |
| Workflow (`workflows/`) | 编排调度 | 块级迭代、结果缓存 | 调用 solver/model | 改变量化决策 |

**关键发现**：当前整个 pipeline 中**没有任何层**具备"基于敏感性自动跳过层"的能力。skip_layers 完全由用户静态指定。

#### 二、各算法已有的可复用统计量

| 算法 | 已有统计量 | 位置 | 粒度 | 敏感性潜力 |
|------|-----------|------|------|-----------|
| **GPTQ** | Hessian 矩阵 `H = X^T X` | `gptq_module.py:223` | `(cin, cin)` | **极强** — `diag(H)` 是经典敏感度指标 |
| **GPTQ** | Hessian 对角线排序 | `gptq_module.py:119` | `(cin,)` | 直接可用 — 已用于列排序 |
| **AWQ** | 激活均值 `inputs_mean` | `awq.py:84` | `(cin,)` | 中等 — 通道级激活幅度 |
| **AWQ** | 最优重建损失 `min_loss` | `awq.py:105` | 标量/层 | 强 — 层级量化敏感度 |
| **SmoothQuant** | smooth_factor | `smooth_quant_module.py:101` | `(1, cin)` | 强 — `abs(log(factor))` 越大越敏感 |
| **MinMax** | 激活范围 `data_max - data_min` | `minmax_module.py:120-123` | `(1, cin)` | 弱代理 — 范围大不一定敏感 |
| **OFMR** | 权重量化损失矩阵 | `ofmr_quant_module.py:219` | `(11, cout)` | **极强** — 逐通道 MSE 曲线 |
| **OFMR** | `compute_mse()` 函数 | `ofmr_quant_module.py:72-85` | 可复用 | 直接可用的 MSE 工具 |
| **BlockwiseSolver** | 重建损失 `total_loss` | `blockwise_solver.py:77` | 标量/层 | 强 — 层级 MSE |

**核心缺口**：没有统一的敏感性分析工具，各算法各自计算和丢弃统计量。

#### 三、设计：敏感层自动识别逻辑

##### 3.1 插入点分析

当前 `quantize()` 的调用链：

```
quantize(model, config)
  │
  ├→ [1] parse_config(model, config, AlgorithmRegistry)
  │      → detail_config: {层名: 配置}  ← skip_layers 在此生效
  │
  ├→ [2] InsertQuantizeModulePass(detail_config)
  │      → 机械安装 quant_op  ← 无决策能力
  │
  ├→ [3] 用户运行校准数据 forward()
  │      → quant_op.forward() 计算 Hessian/scale/loss
  │
  └→ [4] convert(model)
         → 替换为 deploy_op
```

可选插入点：

| 位置 | 方案 | 优劣 |
|------|------|------|
| A. `parse_config()` 内部 | 在 config 层做敏感性分析 | ❌ 违反职责 — config 层看不到模型数据 |
| B. `InsertQuantizeModulePass` 之后、校准之前 | 新增 pre-calibration sensitivity pass | ✅ 不侵入现有层，独立新组件 |
| C. 校准完成后、`convert()` 之前 | 新增 post-calibration skip pass | ✅ 可利用校准统计数据，但已完成量化计算 |
| D. `quantize()` 内部编排 | 在 `quantize()` 函数中插入新步骤 | ✅ 最自然的编排点 |

**推荐位置 D**：在 `quantize()` 函数中，在 `parse_config()` 和 `InsertQuantizeModulePass` 之间插入一个可选的 sensitivity analysis 步骤。

##### 3.2 整体数据流设计

```
quantize(model, config)
  │
  ├→ [1] parse_config(model, config, AlgorithmRegistry)
  │      → detail_config (所有合格层)
  │
  ├→ [2] ★ SensitivityAnalyzer.analyze(model, detail_config, calib_data)
  │      │  ← 新增步骤，可选
  │      │
  │      ├→ 2a. 轻量校准：对每个层收集统计量
  │      │     - 运行 1-batch 校准数据
  │      │     - 每个层的 forward hook 收集 Hessian 近似 或 激活统计
  │      │
  │      ├→ 2b. 敏感性评分：对每个层计算 sensitivity_score
  │      │     - GPTQ 算法：score = ||diag(H)|| / cin
  │      │     - AWQ 算法：score = activation_range / weight_range
  │      │     - 通用 fallback：score = activation_var
  │      │
  │      ├→ 2c. 阈值判定：score > threshold → 标记为 sensitive
  │      │     - 固定阈值模式：用户指定 threshold
  │      │     - 自适应模式：score > mean + k * std → sensitive
  │      │
  │      └→ 2d. 合并到 skip_layers：detail_config 中移除 sensitive 层
  │
  ├→ [3] InsertQuantizeModulePass(detail_config)
  │      → 只安装非敏感层的 quant_op
  │
  ├→ [4] 用户运行校准 forward()
  │
  └→ [5] convert(model)
```

##### 3.3 核心组件设计

**组件 1：`SensitivityAnalyzer`（新增类）**

位置：`amct_pytorch/classic/quantize_op/sensitivity_analyzer.py`（新文件）

```python
class SensitivityAnalyzer:
    """敏感层自动识别器。

    职责: 在 quantize() 流程中，基于轻量校准数据，
          评估每个层的量化敏感性，输出需要跳过的层列表。

    不修改任何现有 quant_op、algorithm、deploy_op 的代码。
    """

    def __init__(self, method='hessian_diag', threshold=None, k=2.0):
        """
        method: 敏感性度量方法
            - 'hessian_diag': Hessian 对角线范数（推荐，最准确）
            - 'act_range': 激活范围比（轻量）
            - 'act_var': 激活方差（最轻量）
            - 'auto': 根据算法自动选择
        threshold: 固定阈值（None 则使用自适应模式）
        k: 自适应模式下的标准差倍数
        """

    def analyze(self, model, detail_config, calibration_loader=None):
        """
        输入:
          model: 待量化模型
          detail_config: parse_config() 输出的层配置字典
          calibration_loader: 校准数据 DataLoader（可选）
        输出:
          auto_skip_layers: set[str] — 自动识别的敏感层名称集合
        """
```

**组件 2：统计量收集器（Hook-based）**

```python
class _StatsCollector:
    """每个层的统计量收集器。"""

    def __init__(self):
        self.input_cache = None

    def hook_fn(self, module, input, output):
        """注册为 module.register_forward_hook()"""
        self.input_cache = input[0].detach()
```

**组件 3：敏感性评分函数**

```python
def _compute_sensitivity(method, input_tensor, weight_tensor=None):
    """
    'hessian_diag' (GPTQ 风格):
        H_approx = input^T @ input  # (cin, cin)
        score = diag(H_approx).norm() / cin

    'act_range' (MinMax 风格):
        act_range = input.abs().amax(dim=batch_dims) - input.abs().amin(dim=batch_dims)
        score = act_range.mean()

    'act_var' (最轻量):
        score = input.var()
    """
```

**组件 4：阈值判定**

```python
def _apply_threshold(scores, threshold=None, k=2.0):
    """
    固定阈值: auto_skip = {name for name, s in scores.items() if s > threshold}
    自适应: auto_skip = {name for name, s in scores.items() if s > mean + k * std}
    """
```

##### 3.4 `quantize()` 函数修改

```python
# amct_pytorch/classic/quantize.py

def quantize(model, config=None):
    if config is None:
        config = set_default_config()
    layer_config = parse_config(model, config, AlgorithmRegistry)

    # ★ 新增: 敏感层自动识别（可选）
    auto_skip = config.get('auto_skip', None)
    if auto_skip is not None:
        analyzer = SensitivityAnalyzer(
            method=auto_skip.get('method', 'auto'),
            threshold=auto_skip.get('threshold', None),
            k=auto_skip.get('k', 2.0),
        )
        skip_set = analyzer.analyze(model, layer_config)
        for layer_name in skip_set:
            layer_config.pop(layer_name, None)
            LOGGER.logi(f'Auto-skip sensitive layer: {layer_name}')

    optimizer = ModelOptimizer()
    optimizer.add_pass(InsertQuantizeModulePass(layer_config))
    optimizer.do_optimizer(model)
```

##### 3.5 配置格式扩展

```python
config = {
    'batch_num': 1,
    'quant_cfg': {
        'weights': {'type': 'int4', 'symmetric': True, 'strategy': 'channel'}
    },
    'algorithm': {'gptq': {}},
    'skip_layers': {'lm_head'},          # 静态跳过（现有机制）
    'auto_skip': {                         # ★ 新增：动态敏感层跳过
        'method': 'hessian_diag',         # 敏感性度量方法
        'threshold': None,                # 固定阈值（None=自适应）
        'k': 2.0,                         # 自适应: score > mean + k*std
    }
}
```

#### 四、架构合规性验证

##### 4.1 职责边界检查

| 检查项 | 结果 | 说明 |
|--------|------|------|
| Config 层是否做算法决策？ | ✅ 不违反 | `auto_skip` 仅声明参数，不含敏感性分析逻辑 |
| Parser 层是否做敏感性分析？ | ✅ 不违反 | `parse_config()` 不变，分析在 `quantize()` 中独立执行 |
| Pass 层是否分析模型内容？ | ✅ 不违反 | `InsertQuantizeModulePass` 仍接收预过滤 config |
| QuantOp 是否自主跳过？ | ✅ 不违反 | QuantOp 不知道 auto-skip 的存在 |
| Algorithm 是否与 pipeline 通信？ | ✅ 不违反 | Algorithm 保持无状态函数角色 |
| 新增组件是否侵入现有层？ | ✅ 不侵入 | `SensitivityAnalyzer` 是独立新类 |

##### 4.2 向后兼容性检查

| 检查项 | 结果 | 说明 |
|--------|------|------|
| `auto_skip` 未配置时行为不变？ | ✅ | `config.get('auto_skip', None)` 为 None 时跳过 |
| `skip_layers` 静态跳过仍生效？ | ✅ | `parse_config()` 先执行，auto_skip 在其后追加 |
| INT4 行为不受影响？ | ✅ | auto_skip 是算法无关的通用机制 |
| 现有测试不回归？ | ✅ | 所有现有测试不包含 `auto_skip` 配置 |
| deploy_op 格式不变？ | ✅ | 仅影响 `detail_config` 的层集合 |

##### 4.3 与 graph-based 管道敏感性机制的对比

| 维度 | graph-based (已有) | classic auto_skip (新增) |
|------|-------------------|------------------------|
| 管道 | ONNX graph | PyTorch nn.Module |
| 敏感性度量 | CosineSimilarity / MSE | Hessian diag / act_range / act_var |
| 触发方式 | `AccuracyBasedAutoCalibration` 全流程 | `auto_skip` 配置项 |
| 回退机制 | BinarySearchStrategy 逐层回退 | 一次性跳过（不回退） |
| 适用范围 | graph-based 专用 | classic 管道通用 |

#### 五、改动范围精确清单

| # | 文件 | 改动类型 | 具体内容 | 侵入现有逻辑？ |
|---|------|---------|---------|--------------|
| 1 | `classic/quantize_op/sensitivity_analyzer.py` | **新文件** | `SensitivityAnalyzer` 类 + `_StatsCollector` + 评分函数 | ❌ |
| 2 | `classic/quantize.py` | **追加 5 行** | `auto_skip` 配置读取 + `SensitivityAnalyzer` 调用 | ❌ |
| 3 | `common/config/fields.py` | **追加字段** | `AutoSkipField` 验证 method/threshold/k 参数 | ❌ |
| 4 | `common/config/fields.py` | **追加字段** | `QuantConfig.__init__` 读取 `auto_skip` | ❌ |
| 5 | `tests/unit_test/classic/test_sensitivity_analyzer.py` | **新文件** | 单元测试 | ❌ |

#### 六、数据流追踪总结

```
Agent 处理数据流:

[需求解析]
  "BATQuant 敏感层自动识别"
  → 解析: 需要基于 activation error / Hessian 近似自动决定跳过
  → 约束: 符合 PTQ pipeline 职责边界
  → 范围: classic 管道 (quantize/convert API)

[代码定位]
  grep('skip_layers') → parser.py:42-59 (check_skip_layer)
  grep('auto_calibration') → graph_based/ (已有敏感性参考)
  grep('hessian') → gptq_module.py:223 (已有 Hessian 计算)
  grep('sensitivity') → auto_calibration/ (已有 CosineSimilarity/MSE)

[方案设计]
  方案 A: parse_config() 内做 → 违反 config 层职责
  方案 B: quant_op 自主跳过 → 违反 quant_op 职责
  方案 C: 校准后跳过 → 已完成计算, 浪费
  方案 D: quantize() 编排新步骤 → ✅ 最佳, 独立新组件

[架构验证]
  所有层职责边界未被违反:
    Config 层 → 仅声明 auto_skip 参数
    Parser 层 → 不变
    Pass 层 → 接收预过滤 config
    QuantOp 层 → 不知道 auto_skip
    Algorithm 层 → 无状态函数

[兼容性验证]
  auto_skip=None → 走原有路径 → 零影响
  skip_layers 静态跳过仍生效 → 向后兼容
  不修改任何现有文件逻辑 → 零回归风险
```

## 测试 4：AMCT CLI 增加 GPTQ

**测试语：**

基于 AMCT CLI 框架增加 GPTQ 量化算法，支持 INT4、INT8、MXFP4 等数据类型，并将优化后的权重落盘到文件中。请跟踪并说明 Agent 处理数据流，检查其是否符合现有 CLI、workflow、algorithm 和 deploy 的分层预期。本次只输出任务规划，不修改代码，不执行量化，不生成权重文件，也不写入任何文件。

### Agent 处理结果

#### 一、架构现状：GPTQ 在 AMCT 中的位置

##### 1.1 两套独立的算法系统

| 维度 | 经典 Pipeline | 新型 LLM Workflow/CLI |
|------|-------------|---------------------|
| 注册表 | `AlgorithmRegistry` (`algorithms/__init__.py`) | `ALGO_REGISTRY` (`algorithms/registry_factory.py`) |
| 注册方式 | `AlgorithmRegistry.register(name, src_op, quant_op, deploy_op)` | `@ALGO_REGISTRY.register(name=..., targets=...)` 装饰器 |
| 已有算法 | minmax, awq, **gptq**, smoothquant, mxquant, ofmr, cast, quantile | autoround, omniquant, LAC, LWC, flatquant |
| GPTQ 存在？ | **YES** — 完整实现 (`GPTQuant` 类) | **NO** — 未注册 |
| 入口 | `amct_pytorch.quantize(model, config)` | CLI `--algos` → `ALGO_REGISTRY` 查找 |
| 配置 | Dict-based (`INT4_GPTQ_WEIGHT_QUANT_CFG`) | YAML bit_config + `--algos` 标志 |

##### 1.2 CLI 当前算法支持

CLI `--algos` 参数为自由字符串列表（无 choices 限制），通过 `ALGO_REGISTRY.get_item(algo_name)` 查找。当前注册的算法：

| 算法 | 类 | Target |
|------|-----|--------|
| `autoround` | `AutoRound` | `("weight",)` |
| `lwc` | `LWC` | `("weight",)` |
| `lac` | `LAC` | `("activation",)` |
| `omniquant` | `OmniQuant` | `("structure",)` |
| `flatquant` | `FlatQuant` | `("structure",)` |

`--algos gptq` 当前会触发 `KeyError`。

##### 1.3 经典 GPTQ 实现核心

`classic/quantize_op/gptq_module.py` 中的 `GPTQuant` 类：

- **Hessian 累积**：`update_hessian(input_data)` 累积 `H = X^T @ X`（输入激活的协方差矩阵）
- **权重优化**：`get_opt_weight_and_quant_factor()` 核心 GPTQ 算法：
  1. 克隆权重为 float32
  2. 按 Hessian 对角值排序列
  3. Cholesky 分解求 Hessian 逆
  4. 逐列量化 + 误差传播（block_size=128）
  5. 恢复列顺序
- **量化**：`cal_scale_offset_static()` 基于权重 min/max 计算 scale/offset（非 activation-aware）
- **MXFP4 特殊处理**：返回 `(None, None)`，跳过传统 scale/offset

#### 二、新型 GPTQ 算法接口设计

##### 2.1 新型算法接口契约

基于 AutoRound、LWC 的分析，新型权重量化算法必须/可选实现：

| 方法 | 签名 | 必需 | 说明 |
|------|------|------|------|
| `__init__` | `(self, args, w_bits)` | YES | 构造器，args 为配置 namespace，w_bits 为权重量化位宽 |
| `forward` | `(self, weight: Tensor) -> Tensor` | YES | 变换权重张量 |
| `observe_input` | `(self, x, weight=None)` | 可选 | 校准时累积统计量 |
| `quantize` | `(self, weight, quant_obj) -> Tensor` | 可选 | 覆盖默认 fake-quantization（**仅一个**算法可定义） |
| `export_deploy` | `(self, weight, quant_obj) -> dict` | 可选 | 部署导出（需配合 `quantize()`） |
| `trainable_params` | `() -> list[nn.Parameter]` | 可选 | 可训练参数（GPTQ 为一次性算法，返回空列表） |
| `export_ptq_params` / `load_ptq_params` | 序列化/反序列化 | 可选 | 参数持久化 |

##### 2.2 GPTQ 在新接口中的角色

GPTQ 与 AutoRound/LWC 的本质区别：

| 维度 | AutoRound/LWC | GPTQ |
|------|-------------|------|
| 优化方式 | 迭代训练（多 epoch） | 一次性优化（单次 Hessian 逆 + 逐列量化） |
| 数据依赖 | 需要 GT loss 反向传播 | 需要输入激活计算 Hessian |
| 时机 | 训练循环中 | 校准后、训练前 |
| 可训练参数 | 有（rounding offsets, clip factors） | 无 |

**关键适配点**：GPTQ 需要一个 **post-calibration hook** 来在 Hessian 累积完成后执行权重优化。

##### 2.3 注册声明

```python
@ALGO_REGISTRY.register(
    name="gptq",
    description="GPTQ weight-only quantization with Hessian-based optimization",
    targets=("weight",),
)
class GPTQWeightAlgo(nn.Module):
    def __init__(self, args, w_bits): ...
    def observe_input(self, x, weight=None): ...     # 累积 Hessian
    def post_calibrate(self): ...                      # 权重优化（新增 hook）
    def forward(self, weight): ...                     # 应用 fake_quant
    def quantize(self, weight, quant_obj): ...         # 覆盖默认量化
    def export_deploy(self, weight, quant_obj): ...    # 部署导出
    def export_ptq_params(self): ...                   # 序列化
    def load_ptq_params(self, params): ...             # 反序列化
```

#### 三、核心组件设计

##### 3.1 `GPTQWeightAlgo` 类

```python
@ALGO_REGISTRY.register(
    name="gptq",
    description="GPTQ weight-only quantization with Hessian-based optimization",
    targets=("weight",),
)
class GPTQWeightAlgo(nn.Module):
    def __init__(self, args, w_bits):
        super().__init__()
        self.args = args
        self.bits = w_bits
        self.perc_damp = 0.01
        self.block_size = 128
        self.group_size = getattr(args, 'group_size', None)
        self._hessian = None          # 累积 Hessian: [in_features, in_features]
        self._nsamples = 0
        self._optimized = False       # 权重是否已优化
        self._optimized_weight = None # 优化后的权重缓存

    def observe_input(self, x, weight=None):
        """校准阶段：从输入激活累积 Hessian (H = X^T @ X)"""
        if self._optimized:
            return
        if self._hessian is None:
            self._hessian = torch.zeros(x.shape[-1], x.shape[-1], device=x.device)
        batch = x.reshape(-1, x.shape[-1])
        self._hessian += batch.T @ batch
        self._nsamples += batch.shape[0]

    def post_calibrate(self):
        """校准完成后：执行 GPTQ 权重优化（Hessian 逆 + 逐列量化 + 误差传播）"""
        if self._optimized:
            return
        self._optimized = True
        # 1. 克隆权重为 float32
        # 2. 阻尼 Hessian 对角: H += damp * I
        # 3. Cholesky 分解求 Hessian 逆
        # 4. 按 Hessian 对角值排序列
        # 5. 逐列量化 + 误差传播
        # 6. 恢复列顺序
        # 7. 缓存优化后权重

    def forward(self, weight):
        """应用 fake_quant 到优化后的权重"""
        if self._optimized and self._optimized_weight is not None:
            weight = self._optimized_weight
        return weight  # 默认 fake_quant 由 WeightQuantizer 处理

    def quantize(self, weight, quant_obj):
        """覆盖默认量化：GPTQ 使用基于权重 min/max 的量化"""
        if self._optimized and self._optimized_weight is not None:
            weight = self._optimized_weight
        w_min, w_max = weight.min(), weight.max()
        if self.bits == 4 and "mx" in str(getattr(self.args, 'quant_dtype', '')):
            return quant_obj(weight)  # MXFP4 由 quant_obj 处理
        # INT4/INT8: 计算 scale/offset
        scale = (w_max - w_min) / (2 ** self.bits - 1)
        offset = torch.round(-w_min / scale)
        return quant_obj(weight, scale, offset)

    def export_deploy(self, weight, quant_obj):
        """部署导出：使用优化后的权重"""
        if self._optimized and self._optimized_weight is not None:
            weight = self._optimized_weight
        return quant_obj.export_deploy(weight)

    def export_ptq_params(self):
        return {
            'hessian': self._hessian,
            'nsamples': self._nsamples,
            'optimized_weight': self._optimized_weight,
        }

    def load_ptq_params(self, params):
        self._hessian = params['hessian']
        self._nsamples = params['nsamples']
        self._optimized_weight = params.get('optimized_weight')
        self._optimized = self._optimized_weight is not None
```

##### 3.2 `BlockwiseSolver` 扩展

在 `_optimize_block()` 训练循环前插入 `post_calibrate()` 调用：

```python
def solve(self, data_loader, forward_kwargs=None):
    # Phase 1: 校准（observe_input 累积 Hessian）
    for data_batch in data_loader:
        unit_inp, unit_gt = data_batch
        outputs = self.model(unit_inp, **kwargs)
        loss = self._reconstruction_loss(outputs, unit_gt)

    # Phase 2: post_calibrate（新增）
    for module in self.model.modules():
        algo_fn = getattr(module, "post_calibrate", None)
        if callable(algo_fn):
            algo_fn()

    # Phase 3: 训练（GPTQ 已优化权重，其他算法继续训练）
    for epoch in range(self.args.epochs):
        avg_loss = self._optimize_block(data_loader, forward_kwargs=forward_kwargs)
```

**注意**：当前 `BlockwiseSolver` 将校准和训练合并在同一个循环中。需要将循环拆分为两阶段：先校准（所有 batch），再训练（多 epoch）。

##### 3.3 `WeightQuantizer.observe_input()` 调用链

当前 `WeightQuantizer` 已有 `observe_input()` 方法（`quant_base.py:190-194`）：

```python
def observe_input(self, x, weight=None):
    for algo in self.algorithms.values():
        observe_input = getattr(algo, "observe_input", None)
        if callable(observe_input):
            observe_input(x, weight)
```

**此方法已存在，但未在 `BlockwiseSolver` 中被调用**。需要在 solver 的校准阶段显式调用。

#### 四、完整数据流追踪

##### 4.1 CLI 命令示例

```bash
# Step 1: 提取校准数据
llm extract_ptq_data \
  --model_path /path/to/Qwen3.5-9B \
  --calib_data pileval \
  --quant_target attn-linear \
  --data_dir /tmp/gptq_test

# Step 2: GPTQ 量化
llm ptq \
  --model_path /path/to/Qwen3.5-9B \
  --calib_data pileval \
  --quant_target attn-linear \
  --algos gptq \
  --quant_dtype int \
  --w_bits 4 \
  --data_dir /tmp/gptq_test

# Step 3: 评测
llm eval \
  --model_path /path/to/Qwen3.5-9B \
  --quant_target attn-linear \
  --eval_mode quant \
  --data_dir /tmp/gptq_test
```

##### 4.2 数据流追踪

```
[CLI 入口]
  args.algos = ["gptq"]
  args.quant_dtype = "int"
  args.w_bits = 4
  args.quant_target = ["attn-linear"]

[LlmPtqWorkflow.setup()]
  → _register_components() → register_algorithms() → 导入 GPTQWeightAlgo → ALGO_REGISTRY 注册
  → _build_pipeline() → MODEL_REGISTRY.get("qwen3_5")(args) → 构建模型
  → build_quant_block() → 包装 QuantQwen3Attn → 内含 WeightQuantizer
  → WeightQuantizer._init_algo() → build_algorithms_by_target(args, "weight", 4)
    → ALGO_REGISTRY.get_item("gptq") → 实例化 GPTQWeightAlgo(args, 4)

[LlmPtqWorkflow._run_blockwise()]
  for layer_idx in range(start, end):
    block = pipeline.build_quant_block(layer_idx)
    unit = next(pipeline.iter_ptq_units(layer_idx, block))

    # 加载校准数据
    unit_batch = _prepare_unit_batch(unit)
    → load_ptq_inps("block_{i}_attn_in.pkl") → [N, seq_len, hidden_dim] 隐藏状态
    → materialize_gt(inps, original_block) → 原始 float 输出（GPTQ 不使用）

    # Solver 求解
    solver = SOLVER_REGISTRY.get("block")(args, block)
    solver.solve(data_loader, forward_kwargs)

      [BlockwiseSolver.solve() 内部]

      Phase 1: 校准
        for data_batch in data_loader:
          unit_inp, unit_gt = data_batch
          # WeightQuantizer.forward() → observe_input() → GPTQ.observe_input()
          outputs = model(unit_inp, **kwargs)
          # GPTQ.observe_input(): _hessian += batch.T @ batch

      Phase 2: post_calibrate (新增)
        for module in model.modules():
          if hasattr(module, 'post_calibrate'):
            module.post_calibrate()
        # GPTQ.post_calibrate():
        #   1. Cholesky 分解求 Hessian 逆
        #   2. 逐列量化 + 误差传播
        #   3. 缓存 _optimized_weight

      Phase 3: 训练
        for epoch in range(epochs):
          for data_batch in data_loader:
            outputs = model(unit_inp, **kwargs)
            # GPTQ.forward(): 使用 _optimized_weight → 返回 fake_quant 结果
            loss = MSE(outputs, unit_gt)
            loss.backward()
            optimizer.step()

    # 导出
    finalize()
      → model.export_ptq_params()
      → GPTQWeightAlgo.export_ptq_params() → {hessian, nsamples, optimized_weight}
      → 保存 layer_{i}_attn.pt
```

#### 五、数据类型支持矩阵

| 数据类型 | 量化方式 | GPTQ 适配点 | quant_obj |
|---------|---------|------------|-----------|
| **INT4** | `cal_scale_offset_static()` | 权重 min/max → scale/offset | `QuantDequantInt(bits=4)` |
| **INT8** | `cal_scale_offset_static()` | 权重 min/max → scale/offset | `QuantDequantInt(bits=8)` |
| **MXFP4** | `cal_scale_offset_static()` → `(None, None)` | 跳过 scale/offset，由 quant_obj 处理 | `QuantDequantMx(bits=4)` |
| **HiFloat8** | `cal_scale_offset_static()` | 权重 min/max → scale/offset | 自定义 quant_obj |
| **Float8_E4M3FN** | `cal_scale_offset_static()` | 权重 min/max → scale/offset | 自定义 quant_obj |

**MXFP4 特殊处理**：经典 GPTQ 对 MXFP4 返回 `(None, None)`，新实现中由 `quant_obj`（`QuantDequantMx`）处理 group-wise 量化。

#### 六、改动范围精确清单

| # | 文件 | 改动类型 | 具体内容 |
|---|------|---------|---------|
| 1 | `algorithms/quant/gptq.py` | **新文件** | `GPTQWeightAlgo` 类，实现 Hessian 累积 + 权重优化 + fake_quant |
| 2 | `algorithms/quant/__init__.py` | **追加 import** | `from .gptq import GPTQWeightAlgo` |
| 3 | `algorithms/__init__.py` | **追加算法名** | `BUILT_IN_ALGORITHM` 追加 `'gptq'`（若需要） |
| 4 | `common/optimization/blockwise_solver.py` | **追加阶段** | `solve()` 拆分为校准 → `post_calibrate()` → 训练 |
| 5 | `common/optimization/base_solver.py` | **追加方法** | `post_calibrate()` 基类方法（可选） |
| 6 | `common/utils/vars.py` | **无需修改** | `ALGORITHM_SUPPORTED_QUANT_TYPE_COMB` 已包含 `'gptq'` |
| 7 | `cli/llm/args.py` | **无需修改** | `--algos` 为自由字符串，自动支持 |
| 8 | `workflows/llm_ptq.py` | **无需修改** | 算法通过 `WeightQuantizer._init_algo()` 自动加载 |

**向后兼容**：所有改动为追加式。`gptq` 为新增到 `ALGO_REGISTRY` 的算法名，不影响现有 `autoround`/`lwc`/`lac`/`omniquant`/`flatquant` 的任何逻辑。经典 pipeline 的 `GPTQuant` 保持不变。

#### 七、数据流追踪总结

```
Agent 处理数据流:

[需求解析]
  "基于 AMCT CLI 框架增加 GPTQ 量化算法"
  → 核心: 将经典 GPTQuant 移植到新型 ALGO_REGISTRY 接口
  → 支持: INT4、INT8、MXFP4 等数据类型

[架构分析]
  经典 Pipeline: AlgorithmRegistry → GPTQuant (完整实现)
  新型 Workflow: ALGO_REGISTRY → 无 GPTQ
  → 需要在 ALGO_REGISTRY 中注册新型 GPTQ

[接口设计]
  GPTQWeightAlgo (nn.Module):
    observe_input() → 累积 Hessian
    post_calibrate() → 权重优化（新增 solver hook）
    forward() → 应用 fake_quant
    quantize() → 覆盖默认量化（基于权重 min/max）

[数据流追踪]
  extract_ptq_data: 保存 block_{i}_attn_in.pkl (输入激活)
  ptq: WeightQuantizer → GPTQWeightAlgo.observe_input() → 累积 Hessian
       → post_calibrate() → 优化权重
       → forward() → fake_quant → MSE loss → 训练
  finalize: export_ptq_params() → 保存优化后权重

[改动范围]
  新文件: 1 (algorithms/quant/gptq.py)
  修改文件: 2 (algorithms/quant/__init__.py, blockwise_solver.py)
  无需修改: 5+ (vars.py, args.py, llm_ptq.py 等)

[数据类型支持]
  INT4: QuantDequantInt(bits=4) + 权重 scale/offset
  INT8: QuantDequantInt(bits=8) + 权重 scale/offset
  MXFP4: QuantDequantMx(bits=4) + quant_obj 处理
  向后兼容: 所有现有算法不受影响
```

## 测试 5：Attention 与 MLP 联合校准

**测试语：**

为 AMCT CLI 框架增加 Attention 模块和 MLP 模块一起进行量化校准的数据通路，使用 Qwen3.5-9B 进行测试，并给出联合量化与分开量化的精度比较。请跟踪并说明 Agent 处理数据流。本次只输出任务规划，不修改代码，不运行模型或量化实验，不写入任何文件。

### Agent 处理结果

#### 一、当前架构分析：什么阻止了联合校准

##### 1.1 三个硬约束

| # | 约束 | 位置 | 具体代码 |
|---|------|------|---------|
| C1 | `quant_target` 只能为单值 | `llm_ptq.py:45-46` | `if len(args.quant_target) != 1: raise ValueError(...)` |
| C2 | `iter_ptq_units()` 互斥分支 | `base.py:264-290` | attn 分支 `return` 后不会执行 mlp 分支 |
| C3 | 内层循环逐 unit 独立求解 | `llm_ptq.py:195-213` | 每个 unit 独立创建 solver、独立 forward、独立保存 |

##### 1.2 分开校准的完整数据流

```
[分开校准 — attn]

  extract_ptq_data --quant_target attn-linear
    → Qwen3_5.build_quant_block(): 只包装 QuantQwen3Attn
    → hook 捕获 attn 输入 → 保存 block_{i}_attn_in.pkl

  ptq --quant_target attn-linear
    → iter_ptq_units(): yield PtqUnit(kind="attn", module=self_attn)
    → _prepare_unit_batch(): load attn_in.pkl → materialize_gt(self_attn)
    → BlockwiseSolver.solve(): 仅优化 attn 内 LWC/LAC/AutoRound 参数
    → finalize(): 保存 layer_{i}_attn.pt

[分开校准 — mlp]

  extract_ptq_data --quant_target mlp
    → Qwen3_5.build_quant_block(): 只包装 QuantQwen3MLP
    → hook 捕获 mlp 输入 → 保存 block_{i}_mlp_in.pkl

  ptq --quant_target mlp
    → iter_ptq_units(): yield PtqUnit(kind="mlp", module=mlp)
    → _prepare_unit_batch(): load mlp_in.pkl → materialize_gt(mlp)
    → BlockwiseSolver.solve(): 仅优化 mlp 内 LWC/LAC/AutoRound 参数
    → finalize(): 保存 layer_{i}_mlp.pt
```

**问题**：attn 的校准不感知 mlp 的输入分布，mlp 的校准不感知 attn 的输出扰动。两者的量化误差无法在联合 forward 中相互补偿。

#### 二、设计：联合校准数据通路

##### 2.1 方案选择

| 方案 | 描述 | 优劣 |
|------|------|------|
| A. 放宽 quant_target 为多值 | 允许 `--quant_target attn-linear mlp`，内层循环依次处理 | 最小改动，但仍逐 unit 独立求解，**无联合优化** |
| B. 新增 `joint` quant_target | 新增 `--quant_target joint`，yield 一个包含 attn+mlp 的复合 PtqUnit | 中等改动，实现联合 forward + 联合 loss |
| C. 新增 `JointBlockSolver` | 保持 quant_target 单值，但新增 solver 类型支持全 block 联合优化 | 最干净，solver 层面解决 |

**推荐方案 B + C 组合**：新增 `joint` quant_target 作为编排层入口，配合 `JointBlockSolver` 实现全 block 联合优化。理由：
1. 方案 A 只是放宽约束，不改变"逐 unit 独立求解"的本质
2. 方案 B 在 `iter_ptq_units()` 层面 yield 复合 unit，自然承载 attn+mlp 的联合上下文
3. 方案 C 在 solver 层面实现联合 forward + 联合 loss，是核心差异点

##### 2.2 整体数据流设计

```
[联合校准 — 完整流程]

  Step 1: extract_ptq_data --quant_target joint
    │
    ├→ Qwen3_5.build_quant_block(layer_idx):
    │    包装 QuantQwen3Attn + QuantQwen3MLP（两者同时包装）
    │
    ├→ hook 注册: 对 attn 输入和 mlp 输入分别捕获
    │    → 保存 block_{i}_attn_in.pkl  (attn 的输入)
    │    → 保存 block_{i}_mlp_in.pkl   (mlp 的输入，来自 attn 输出)
    │
    └→ 关键: mlp_in 来自 attn 的输出，因此两者数据天然关联

  Step 2: ptq --quant_target joint
    │
    ├→ iter_ptq_units(layer_idx, block):
    │    yield PtqUnit(kind="joint", name="block",
    │                  module=block,           ← 整个 decoder layer
    │                  metadata={"attn_name": "self_attn", "mlp_name": "mlp"})
    │
    ├→ _prepare_unit_batch(unit):
    │    load_unit_inputs(): 同时加载 attn_in.pkl + mlp_in.pkl
    │    materialize_gt(): 对整个 block 做 forward → 得到 block 输出
    │
    ├→ JointBlockSolver.solve():
    │    model = block (整个 decoder layer)
    │    forward: block(attn_in) → block_output
    │    loss = MSE(block_output, block_gt)
    │    backward: 梯度流过 attn + mlp 两者
    │    optimizer: 收集 attn 和 mlp 中所有可训练参数
    │
    └→ finalize(): 导出 attn + mlp 的量化参数
         → 保存 layer_{i}_joint_attn.pt + layer_{i}_joint_mlp.pt
```

##### 2.3 核心组件设计

**组件 1：`iter_ptq_units()` 扩展**

在 `Qwen3_5` 适配器中（或 `BaseModel` 中）新增 `joint` 分支：

```python
def iter_ptq_units(self, layer_idx, block):
    if self.quant_target == "joint":
        yield PtqUnit(
            kind="joint",
            name="block",
            layer_idx=layer_idx,
            module=block,
            metadata={"attn_name": "self_attn", "mlp_name": "mlp"}
        )
        return
    # ... 现有 attn/mlp/moe 分支保持不变
```

**组件 2：`load_unit_inputs()` 扩展**

当 `kind == "joint"` 时，同时加载 attn 和 mlp 的输入：

```python
@staticmethod
def load_unit_inputs(data_dir, unit):
    if unit.kind == "joint":
        attn_inps = load_ptq_inps(data_dir, "attn", unit.layer_idx)
        mlp_inps = load_ptq_inps(data_dir, "mlp", unit.layer_idx)
        return (attn_inps, mlp_inps)
    return load_ptq_inps(data_dir, unit.kind, unit.layer_idx)
```

**组件 3：`JointBlockSolver`（新增）**

```python
@SOLVER_REGISTRY.register(name="joint_block", description="Joint block calibration solver")
class JointBlockSolver(BaseSolver):
    """联合 block 级校准求解器。

    与 BlockwiseSolver 的区别:
    - model 是整个 decoder layer (attn + mlp)
    - forward 通过完整 block
    - loss 基于 block 输出 vs block ground truth
    - 梯度同时更新 attn 和 mlp 的量化参数
    """

    def solve(self, data_loader, forward_kwargs=None):
        # 收集 attn + mlp 中所有可训练参数
        ...

    def _optimize_block(self, data_loader, forward_kwargs=None):
        for data_batch in data_loader:
            block_inp = ...
            outputs = self.model(block_inp, **kwargs)
            loss = F.mse_loss(outputs, block_gt)
            loss.backward()
            self.step()
        ...
```

**组件 4：`build_quant_block()` 修改**

在 `Qwen3_5.build_quant_block()` 中，当 `quant_target == "joint"` 时同时包装 attn 和 mlp：

```python
def build_quant_block(self, layer_idx):
    block = self.block(layer_idx)
    if self.quant_target == "joint":
        apply_quant_to_attn(self.args, block, QuantQwen3Attn)
        apply_quant_to_moe_mlp(self.args, block.mlp, QuantQwen3MLP)
    elif "attn-linear" in self.quant_target:
        apply_quant_to_attn(self.args, block, QuantQwen3Attn)
    elif "mlp" in self.quant_target:
        apply_quant_to_moe_mlp(self.args, block.mlp, QuantQwen3MLP)
    return block
```

**组件 5：CLI args 扩展**

```python
parser.add_argument("--quant_target", nargs="+", default=[],
                    choices=["mlp", "moe", "attn-linear", "attn-cache", "joint"],
                    help="Quantization target(s). 'joint' enables joint attn+mlp calibration.")
```

**组件 6：`LlmPtqWorkflow` 放宽约束**

```python
# 原: if len(args.quant_target) != 1: raise ValueError(...)
# 改: 允许 'joint' 单值，不允许 'joint' 与其他 target 混用
if len(args.quant_target) != 1:
    raise ValueError("ptq only supports a single quant_target.")
if args.quant_target[0] == "joint" and len(args.quant_target) > 1:
    raise ValueError("'joint' target cannot be combined with other targets.")
self.quant_target = args.quant_target[0]
```

##### 2.4 `materialize_gt()` 对 joint 的适配

```python
def materialize_gt(self, inps, ori_module, kwargs=None):
    if isinstance(inps, tuple):
        attn_inps, mlp_inps = inps
        gts = []
        with torch.no_grad():
            for attn_x, mlp_x in zip(attn_inps_loader, mlp_inps_loader):
                gt = ori_module(attn_x, **kwargs)
                gts.append(gt.detach())
        return torch.cat(gts, dim=0)
    # 非 joint 模式: 走原有路径
    ...
```

#### 三、Qwen3.5-9B 特殊考虑

##### 3.1 模型结构

| 属性 | 值 |
|------|-----|
| 注册名 | `qwen3_5` |
| 适配器类 | `Qwen3_5` (`qwen/qwen3_5/qwen3_5.py`) |
| 模型类型 | Dense (非 MoE) |
| quant_target | `attn-linear` / `mlp` / `joint` |
| 层数 | 约 48 层（9B 参数量） |

##### 3.2 Qwen3_5 适配器需修改的点

| 文件 | 修改 | 说明 |
|------|------|------|
| `qwen3_5.py` `build_quant_block()` | 新增 `joint` 分支 | 同时包装 attn + mlp |
| `qwen3_5.py` `iter_ptq_units()` | 新增 `joint` 分支 | yield 复合 unit |
| `base.py` `iter_ptq_units()` | 新增 `joint` 通用分支 | 在 base 层支持 |
| `base.py` `load_unit_inputs()` | 新增 `joint` 分支 | 加载双份输入 |
| `llm_ptq.py` `_prepare_unit_batch()` | 适配 joint 输入格式 | 处理复合元组 |
| `llm_ptq.py` `_run_blockwise()` | 选择 JointBlockSolver | 当 quant_target=="joint" |

#### 四、联合 vs 分开校准精度对比方案

##### 4.1 对比实验矩阵

| 实验 | quant_target | extract_ptq_data | ptq | eval |
|------|-------------|-----------------|-----|------|
| **A: FP16 baseline** | — | — | — | `--eval_mode bf16` |
| **B: 分开校准 attn** | `attn-linear` | `--quant_target attn-linear` | `--quant_target attn-linear` | `--eval_mode quant --quant_target attn-linear` |
| **C: 分开校准 mlp** | `mlp` | `--quant_target mlp` | `--quant_target mlp` | `--eval_mode quant --quant_target mlp` |
| **D: 分开校准 attn+mlp** | B+C 依次执行 | 两次 extract | 两次 ptq | eval 同时检测 |
| **E: 联合校准** | `joint` | `--quant_target joint` | `--quant_target joint` | `--eval_mode quant --quant_target joint` |

##### 4.2 评测指标

| 指标 | 说明 | 获取方式 |
|------|------|---------|
| **WikiText-2 PPL** | 语言建模困惑度 | `llm_eval --eval_mode bf16/quant` |
| **逐层 MSE** | 每层 block 输出的量化误差 | `BlockwiseSolver._reconstruction_loss` 日志 |
| **校准耗时** | 联合 vs 分开的 wall-clock time 对比 | 日志时间戳 |

##### 4.3 预期结果假设

| 场景 | 预期 | 理由 |
|------|------|------|
| attn PPL | 联合 ≈ 分开 | attn 校准时 mlp 未量化，输入分布差异小 |
| mlp PPL | 联合 < 分开 | mlp 的输入来自 attn 输出，联合校准时 attn 量化误差被 mlp 感知并补偿 |
| 总体 PPL | 联合 < 分开 | 联合 loss 捕获跨模块误差传播，量化参数全局更优 |
| 校准耗时 | 联合 ≈ 分开 | 联合只多一次 forward（GT 生成），优化 epoch 数相同 |

#### 五、改动范围精确清单

| # | 文件 | 改动类型 | 具体内容 |
|---|------|---------|---------|
| 1 | `cli/llm/args.py` | **追加 choice** | `quant_target` choices 追加 `"joint"` |
| 2 | `workflows/llm_ptq.py` | **放宽约束 + 新分支** | `_run_blockwise()` 当 joint 时选 JointBlockSolver；`_prepare_unit_batch()` 适配 joint 输入 |
| 3 | `common/models/llm/common/base.py` | **追加分支** | `iter_ptq_units()` 新增 `joint` yield；`load_unit_inputs()` 新增 joint 加载 |
| 4 | `common/models/llm/qwen/qwen3_5/qwen3_5.py` | **追加分支** | `build_quant_block()` 新增 joint 同时包装 attn+mlp |
| 5 | `common/optimization/joint_block_solver.py` | **新文件** | `JointBlockSolver` 类 |
| 6 | `common/optimization/__init__.py` | **追加 import** | 导入 JointBlockSolver 触发注册 |
| 7 | `common/datasets/ptq_provider.py` | **追加分支** | `materialize_gt()` 适配 joint 复合输入 |
| 8 | `workflows/llm_extract_ptq_data.py` | **追加分支** | extract 流程支持 joint 双 hook |

**向后兼容**：所有改动为追加式。`joint` 为新增 quant_target 值，不影响现有 `attn-linear`/`mlp`/`moe`/`attn-cache` 的任何逻辑。

#### 六、数据流追踪总结

```
Agent 处理数据流:

[需求解析]
  "Attention + MLP 联合校准数据通路"
  → 核心: 让 attn 和 mlp 在同一个 forward pass 中联合优化
  → 测试: Qwen3.5-9B, 对比 joint vs separate PPL

[代码定位]
  llm_ptq.py:45-46 → len(quant_target)==1 硬约束
  base.py:264-290  → iter_ptq_units() 互斥 return
  llm_ptq.py:195-213 → 内层循环逐 unit 独立求解
  blockwise_solver.py → model = unit.module (单子模块)
  qwen3_5.py → build_quant_block() 按 quant_target 条件包装

[方案设计]
  选项 A: 放宽 quant_target 多值 → 仍逐 unit 独立, 无联合优化
  选项 B: 新增 joint quant_target → yield 复合 unit, 联合 forward
  选项 C: 新增 JointBlockSolver → solver 层面联合优化
  → 推荐 B+C 组合: 编排层 + 求解层协同

[数据流对比]
  分开: attn_in → attn_solver → attn_params | mlp_in → mlp_solver → mlp_params
  联合: (attn_in, mlp_in) → joint_solver(block) → (attn_params, mlp_params)
         ↑ 一个 forward, 一个 loss, 梯度流过两者

[验证设计]
  5 组实验: baseline / separate attn / separate mlp / separate both / joint
  指标: WikiText-2 PPL + 逐层 MSE + 耗时
  预期: 联合 PPL ≤ 分开 PPL (mlp 受益最大)
```

## 测试 6：Kimi-K3 PTQ 量化流程

**测试语：**

实现 Kimi-K3 的 PTQ 量化流程，支持 FP16 baseline、FlatQuant 加 INT4 权重量化，并输出精度对比结果。请跟踪并说明 Agent 处理数据流，检查其是否符合 AMCT 端到端量化流程及人工确认门预期。本次只输出任务规划，不修改代码，不运行模型或量化实验，不写入任何文件。

### Agent 处理结果

#### 一、架构现状分析

##### 1.1 Kimi-K3 在 AMCT 中的现状

| 维度 | 状态 |
|------|------|
| 模型适配器 | **不存在** — 无 `MODEL_REGISTRY` 注册，无适配器代码 |
| FlatQuant 注册 | 已注册 `Qwen3Attention` / `Qwen3MLP`（若 Kimi-K3 使用同结构可复用） |
| CLI 支持 | 无 — `--model_name kimi_k3` 会触发 `KeyError` |
| 预定义配置 | 无 — 无 `INT4_FLAT_QUANT_CFG_KIMI_K3` |
| 测试用例 | 无 |

**结论**：Kimi-K3 需要从零创建模型适配器。

##### 1.2 FlatQuant 两套实现对比

| 维度 | 实验性实现 (`experimental/flatquant/`) | 新 pipeline (`algorithms/quant/flatquant.py`) |
|------|--------------------------------------|---------------------------------------------|
| 完整度 | 完整：校准、训练、重参数化、NPU 部署 | 最小核心：仅变换矩阵 |
| 注册表 | `AlgorithmRegistry`（经典） | `ALGO_REGISTRY`（新型） |
| 校准 | 专用 `cali_flat_quant()` 逐层 MSE 优化 | 依赖 `BlockwiseSolver` 通用训练循环 |
| NPU 算子 | `npu_kronecker_quant`（融合变换+量化） | 无 |
| 模型支持 | LlamaAttention/LlamaMLP, Qwen3Attention/Qwen3MLP | 通用 |
| KV 缓存量化 | 支持 | 不支持 |
| 配置 | `FlatConfig` + `INT4_FLAT_QUANT_CFG` | `bit_policy` |

**关键发现**：FlatQuant 的完整流程（校准→重参数化→NPU 部署）仅在实验性实现中可用。新 pipeline 的 FlatQuant 只提供变换矩阵核心，缺少校准循环和部署路径。

##### 1.3 AMCT 端到端量化流程

```
[标准 CLI 端到端流程]

  Step 1: 模型适配（新增/注册适配器）
    → MODEL_REGISTRY 注册
    → build_quant_block() 实现
    → iter_ptq_units() 实现

  Step 2: extract_ptq_data（提取校准数据）
    → cli/llm/extract_ptq_data.py
    → workflows/llm_extract_ptq_data.py
    → 保存 block_{i}_{target}_in.pkl

  Step 3: ptq（量化训练）
    → cli/llm/ptq.py
    → workflows/llm_ptq.py
    → BlockwiseSolver.solve() 逐层优化

  Step 4: eval（精度评测）
    → cli/llm/eval.py
    → workflows/llm_eval.py
    → 输出 PPL 等指标

  Step 5: deploy（部署导出）
    → cli/llm/deploy.py
    → workflows/llm_deploy.py
    → 导出量化权重
```

##### 1.4 人工确认门（Agent 编排层）

AMCT 的人工确认门不在 Python 代码中，而在 Agent 编排层（`.agents/skills/quant-workflow/SKILL.md`）：

| 门 | 触发时机 | 内容 |
|----|---------|------|
| **方案确认** | 执行量化前 | 确认 quant_target / bits / dtype / algos |
| **复用确认** | 复用已有结果前 | 确认跳过已存在的中间产物 |
| **升级确认** | 直转→PTQ 升级时 | 解释为什么不能停在直转 |
| **直转→PTQ 确认** | 直转掉点时 | 说明 PTQ 的必要性 |
| **deploy 精度门** | 导出前 | 必须有 PPL 精度结果 |

#### 二、Kimi-K3 模型适配器设计

##### 2.1 架构假设

Kimi-K3 由 Moonshot AI 开发，基于 Qwen 架构。假设：
- 使用 `Qwen3Attention` 和 `Qwen3MLP`（或相似结构）
- Dense 模型（非 MoE）
- 权重文件格式为 HuggingFace safetensors

##### 2.2 适配器类设计

```python
@MODEL_REGISTRY.register(
    name="kimi_k3",
    task="llm",
    family="kimi",
    description="Kimi-K3 model adapter (Qwen-based dense LLM)"
)
class KimiK3(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.textconfig = Qwen3Config  # 复用 Qwen3 配置（假设结构相同）
        self.num_layers = self.config.num_hidden_layers
        self.cls = Qwen3DecoderLayer  # 复用 Qwen3 decoder layer
        self.model = self.empty_weights_model()
        self.parse_quant_mode()

    def parse_quant_mode(self):
        if "moe" in self.quant_target:
            raise ValueError("Kimi-K3 is a dense model, does not support quant_target='moe'.")

    def get_layer_weight_prefix(self, layer_idx: int) -> str:
        return f"model.layers.{layer_idx}."

    def build_quant_block(self, layer_idx):
        decoder_layer = self.block(layer_idx)
        if "attn-linear" in self.quant_target or "attn-cache" in self.quant_target:
            apply_quant_to_attn(self.args, decoder_layer, QuantQwen3Attn)
        if "mlp" in self.quant_target:
            apply_quant_to_moe_mlp(self.args, decoder_layer, cls=QuantQwen3MLP)
        return decoder_layer
```

##### 2.3 FlatQuant 复用分析

由于 FlatQuant 已注册 `Qwen3Attention` → `FlatQuantAttention` 和 `Qwen3MLP` → `FlatQuantMLP`：

| 条件 | 结果 |
|------|------|
| Kimi-K3 使用 `Qwen3Attention`/`Qwen3MLP` | ✅ FlatQuant 直接复用，无需额外注册 |
| Kimi-K3 使用自定义 Attention/MLP | ❌ 需要在 `experimental/flatquant/__init__.py` 中新增注册 |

**关键验证点**：需确认 Kimi-K3 的 `config.json` 中 `architectures` 字段是否包含 `Qwen3ForCausalLM` 或等效标识。

#### 三、FlatQuant + INT4 集成路径

##### 3.1 两条路径选择

| 路径 | 描述 | 优劣 |
|------|------|------|
| **A. 实验性路径** | 使用 `experimental/flatquant/` 完整实现 | ✅ 完整校准+部署 ❌ 需 `--experimental` 构建 |
| **B. 新 pipeline 路径** | 使用 `algorithms/quant/flatquant.py` + CLI | ✅ 标准 CLI ❌ 缺少校准循环和 NPU 部署 |

**推荐路径 A**：实验性路径提供完整的 FlatQuant 校准、重参数化和 NPU 部署流程，是唯一能端到端完成 FlatQuant + INT4 量化的路径。

##### 3.2 实验性 FlatQuant 数据流

```
[FlatQuant 校准流程 — 实验性实现]

  Step 1: 构建 FlatQuant 模型
    → FlatQuantAttention(module) 替换原始 attention
    → FlatQuantMLP(module) 替换原始 mlp
    → 每个 FlatQuantizedLinear 包装 q/k/v/o/up/gate/down_proj

  Step 2: 校准（cali_flat_quant）
    Phase 1: 收集第一层输入
      → Catcher 捕获 embedding 输出 → fp_inps[0]

    Phase 2: 逐层校准
      for layer_idx in range(num_layers):
        1. FP 参考前向: ori_mode=True → fp_outs[i]
        2. 量化校准: ori_mode=False → cali_layer()
           → 初始化 diag_scale (基于 ln_smax/up_smax/down_smax)
           → 收集可训练参数: trans.linear + diag_scale + clip_factor_w + clip_factor_a
           → AdamW 优化 15 epoch, MSE loss, cosine LR
        3. 交换 fp_inps/fp_outs, offload layer

  Step 3: 重参数化（reparameterize）
    → 将变换矩阵折叠到权重中
    → 将 diag_scale 折叠到 LayerNorm 权重中
    → 转换为 NpuFlatQuantAttention / NpuFlatQuantMLP

  Step 4: 部署
    → NPU 算子: npu_kronecker_quant (融合变换+量化)
    → 导出 INT4 权重 + 变换矩阵
```

##### 3.3 INT4 权重量化配置

```python
INT4_FLAT_QUANT_CFG = {
    'quant_cfg': {
        'inputs':  {'type': 'int4', 'symmetric': True, 'strategy': 'token'},
        'weights': {'type': 'int4', 'symmetric': True, 'strategy': 'channel'},
    },
    'algorithm': {
        'flatquant': {
            'use_kcache_quant': False, 'k_bits': 16,
            'use_vcache_quant': False, 'v_bits': 16,
            'use_o_quant': False,
            'use_down_quant': True,
            'add_diag': True, 'lac': False, 'lwc': True, 'diag_alpha': 0.3,
            'epochs': 15, 'cali_bsz': 4, 'flat_lr': 5e-3, 'cali_trans': True,
        },
    },
    'skip_layers': {'lm_head'}
}
```

#### 四、端到端量化流程设计

##### 4.1 实验矩阵

| 实验 | 量化方案 | 产出 |
|------|---------|------|
| **A: FP16 baseline** | 无量化 | PPL_bf16 |
| **B: FlatQuant + INT4** | FlatQuant 校准 + INT4 权重 | PPL_flatquant_int4 |

##### 4.2 完整执行流程

```
[Step 0: 环境准备]
  bash build.sh --torch --experimental  # 构建含实验特性的包
  pip install build_out/amct_pytorch-*.tar.gz

[Step 1: FP16 baseline 评测]
  python -m amct_pytorch.cli.llm.eval \
    --model_path /path/to/Kimi-K3 \
    --calib_data pileval \
    --quant_target attn-linear \
    --eval_mode bf16

  → 输出: PPL_bf16 (baseline 参考值)

[Step 2: FlatQuant + INT4 校准]
  # 使用实验性 FlatQuant 路径（专用校准循环）
  python -c "
  from amct_pytorch.experimental.flatquant import FlatQuantAttention, FlatQuantMLP
  from amct_pytorch.experimental.flatquant.config import INT4_FLAT_QUANT_CFG
  from amct_pytorch.experimental.flatquant.flat_quant_module.train_utils import cali_flat_quant

  # 1. 加载 Kimi-K3 模型
  model = load_kimi_k3_model()

  # 2. 替换 attention 和 MLP 为 FlatQuant 版本
  for layer in model.model.layers:
      layer.self_attn = FlatQuantAttention(layer.self_attn, layer_config=INT4_FLAT_QUANT_CFG)
      layer.mlp = FlatQuantMLP(layer.mlp, layer_config=INT4_FLAT_QUANT_CFG)

  # 3. 校准（逐层 MSE 优化）
  cali_flat_quant(model, cali_dataloader, device)

  # 4. 重参数化
  for layer in model.model.layers:
      layer.self_attn.reparameterize()
      layer.mlp.reparameterize()

  # 5. 导出
  save_model(model)
  "

[Step 3: FlatQuant + INT4 精度评测]
  python -m amct_pytorch.cli.llm.eval \
    --model_path /path/to/Kimi-K3-FlatQuant-INT4 \
    --calib_data pileval \
    --quant_target attn-linear \
    --eval_mode quant

  → 输出: PPL_flatquant_int4

[Step 4: 精度对比]
  计算 delta = PPL_flatquant_int4 - PPL_bf16
  判断是否达标（默认 delta ≤ 0.2）
```

#### 五、人工确认门设计

##### 5.1 四个强制确认门

| # | 门 | 触发时机 | 检查内容 | 不通过处理 |
|---|---|---------|---------|-----------|
| **G1** | 模型适配确认 | Step 1 完成后 | Kimi-K3 适配器已注册、`build_quant_block()` 可正常构建 | 修复适配器后重试 |
| **G2** | 方案确认 | Step 2 执行前 | 确认 FlatQuant + INT4 配置（epochs=15, cali_bsz=4, flat_lr=5e-3） | 调整参数后重新确认 |
| **G3** | 精度确认 | Step 3 完成后 | PPL_flatquant_int4 与 PPL_bf16 的 delta ≤ 0.2 | 决定是否调整方案或进入 PTQ 升级 |
| **G4** | deploy 确认 | Step 4 执行前 | 精度达标 + 重参数化成功 + NPU 算子可用 | 不达标则不导出 |

##### 5.2 确认门与数据流的交互

```
[G1: 模型适配确认]
  检查: MODEL_REGISTRY.get("kimi_k3") 是否存在
  检查: build_quant_block() 是否能构建 QuantQwen3Attn/QuantQwen3MLP
  门输出: 通过 → 继续; 不通过 → 修复适配器

[G2: 方案确认]
  展示: INT4_FLAT_QUANT_CFG 完整配置
  展示: 预计校准时间（~15 epoch × 128 samples × 48 layers）
  门输出: 用户确认 → 执行; 用户拒绝 → 调整配置

[G3: 精度确认]
  展示: PPL_bf16, PPL_flatquant_int4, delta
  判断: delta ≤ 0.2 → 通过; delta > 0.2 → 考虑 PTQ 升级
  门输出: 通过 → 继续; 不通过 → 升级到 PTQ（需再次确认）

[G4: deploy 确认]
  检查: 重参数化是否成功（权重已折叠）
  检查: NPU 算子是否可用（npu_kronecker_quant）
  门输出: 通过 → 导出; 不通过 → 修复后重试
```

#### 六、改动范围精确清单

| # | 文件 | 改动类型 | 具体内容 |
|---|------|---------|---------|
| 1 | `common/models/llm/kimi/kimi_k3/kimi_k3.py` | **新文件** | `KimiK3` 适配器类，`@MODEL_REGISTRY.register(name="kimi_k3")` |
| 2 | `common/models/llm/__init__.py` | **追加 import** | `from .kimi.kimi_k3.kimi_k3 import KimiK3` |
| 3 | `experimental/flatquant/__init__.py` | **可能追加注册** | 若 Kimi-K3 使用非 Qwen3 模块，需新增 `AlgorithmRegistry.register('flatquant', ...)` |
| 4 | `experimental/flatquant/config.py` | **可能新增配置** | 若需 Kimi-K3 专用 FlatQuant 配置 |
| 5 | `cli/llm/args.py` | **无需修改** | `--model_name` 通过 `MODEL_REGISTRY` 自动支持 |
| 6 | `workflows/llm_ptq.py` | **无需修改** | 算法通过 `WeightQuantizer._init_algo()` 自动加载 |
| 7 | `common/utils/vars.py` | **无需修改** | FlatQuant 不在 `ALGORITHM_SUPPORTED_QUANT_TYPE_COMB` 中 |

**向后兼容**：所有改动为追加式。Kimi-K3 为新增模型适配器，不影响现有模型的任何逻辑。

#### 七、数据流追踪总结

```
Agent 处理数据流:

[需求解析]
  "实现 Kimi-K3 的 PTQ 量化流程"
  → 核心: 创建模型适配器 + 集成 FlatQuant + INT4
  → 评测: FP16 baseline vs FlatQuant INT4 精度对比

[架构分析]
  Kimi-K3: 无现有实现 → 需创建适配器
  FlatQuant: 实验性实现完整 (校准+重参数化+NPU), 新 pipeline 仅核心
  → 推荐实验性路径 (完整端到端)

[适配器设计]
  KimiK3(BaseModel):
    注册: MODEL_REGISTRY.register(name="kimi_k3")
    复用: Qwen3Config, Qwen3DecoderLayer (假设结构相同)
    关键: build_quant_block() → QuantQwen3Attn + QuantQwen3MLP

[FlatQuant 集成]
  前提: Kimi-K3 使用 Qwen3Attention/Qwen3MLP → FlatQuant 已注册
  流程: cali_flat_quant() → 逐层 MSE 优化 → reparameterize() → NPU 部署

[端到端流程]
  Step 1: 构建 KimiK3 适配器 → G1 确认门
  Step 2: FP16 baseline eval → PPL_bf16
  Step 3: FlatQuant + INT4 校准 → G2 确认门 (方案)
  Step 4: 精度评测 → PPL_flatquant_int4 → G3 确认门 (精度)
  Step 5: 重参数化 + NPU 部署 → G4 确认门 (deploy)

[人工确认门]
  G1: 模型适配 → 注册成功 + 构建正常
  G2: 方案确认 → INT4_FLAT_QUANT_CFG 配置
  G3: 精度确认 → delta ≤ 0.2
  G4: deploy 确认 → 重参数化 + NPU 算子

[改动范围]
  新文件: 1 (kimi_k3.py)
  可能修改: 2-3 (__init__.py, flatquant/__init__.py, config.py)
  无需修改: 4+ (args.py, llm_ptq.py, vars.py 等)

[数据类型支持]
  INT4: FlatQuant WeightQuantizer (per-channel symmetric)
  INT4 activation: FlatQuant ActivationQuantizer (per-token symmetric)
  KV cache: 默认不量化 (k_bits=16, v_bits=16)
```
