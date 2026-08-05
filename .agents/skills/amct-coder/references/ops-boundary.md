# 子能力：ops-boundary

## 简介

解决 AMCT 的 Python 量化代码应该怎样调用 NPU 自定义算子的问题，明确上层调用、可选依赖、兼容处理和异常行为与底层算子实现之间的职责边界。本子能力负责 `amct_pytorch` 一侧的接入；算子内核、C++ 扩展和构建实现交给 `amct-ops-dev`。

## 触发与职责拆分

`amct_pytorch` 调用 `amct_ops`、可选 wheel、`torch.ops.amct.*` 签名、Python wrapper/import、fallback 或跨包兼容 review 时使用。

本子能力只 OWN `amct_pytorch` 侧调用边界，PRODUCE 上层调用契约、软依赖和 fallback 策略。Ascend C kernel、C++ extension、schema/PrivateUse1/Meta 注册、算子 CMake、`ops_build.sh` 和 wheel 实现归 `amct-ops-dev`。

| 任务实质 | 处理方式 |
| --- | --- |
| 只新增/修改 kernel、C++、schema、Meta、CMake 或 ops 构建 | 第一阶段直接转 `amct-ops-dev`，不要先用本 playbook 模拟实现 |
| 只修改 `amct_pytorch` 调用、可选 import、fallback 或报错时机 | 本子能力承接 |
| 新增算子并接入 `amct_pytorch` | 混合任务：先冻结共享契约，再由 `amct-ops-dev` 实现算子，本子能力接上层调用，最后统一核对签名 |
| review 跨包方案 | 本子能力检查边界；算子内部实现结论以 `amct-ops-dev` 为准 |

## 先冻结共享契约

只有算子名和集成范围时，以下内容均为待确认项，不能据此锁定 dtype、algorithm 或具体 Python 文件：

1. schema：参数名/顺序/默认值、Tensor/Scalar/list 类型、输出数量。
2. 数值语义：公式、量化粒度、round/clamp、特殊值、确定性和误差口径。
3. tensor 契约：device、dtype、rank/shape、stride/contiguous、广播和空 tensor。
4. PyTorch 语义：是否 mutation/alias、是否需要 autograd/STE、fake tensor/`torch.compile` 行为。
5. 兼容策略：旧 schema 是否保留、调用方与 wheel 的版本组合、缺 wheel 的报错时机。
6. fallback：原生 `torch_npu`、Python 参考实现、可选 NPU op 或明确报错的优先级。
7. 目标 SoC 和 CANN：A2/A3/A5 范围、编译与真实硬件验证要求。

契约未冻结时，`int_impl.py`、某个 algorithm 或 wrapper 只能列为候选锚点。`fake_quant` 等名字不能替代 weight/activation、INT/MXFP 或 shape 语义。

## 按动作选择源码锚点

| 当前动作 | 本子能力首轮锚点 | 条件扩展/外部 skill |
| --- | --- | --- |
| 可选 import/fallback | `amct_pytorch` 中直接调用函数、其 import 位置和 fallback 单测 | dtype/algorithm 语义确实变化时读对应 playbook |
| schema 兼容 review | 上层调用签名、Python wrapper、`register.cpp` schema/Meta 声明 | kernel/extension 实现判断转 `amct-ops-dev` |
| wheel/import 问题 | 上层报错点、`amct_ops/<op>/python/.../__init__.py`、导入测试 | `ops_build.sh`/setup/CMake 修改转 `amct-ops-dev`；公开安装说明读 `public-api-tests` |
| 新算子混合接入 | 已冻结的共享契约、Python 参考语义、候选上层消费点 | `amct-ops-dev` 完成算子后再确定最终调用文件 |

不要因为需要核对 schema 就接管 `amct_ops` 实现，也不要因为任务包含 kernel 就把 `amct_pytorch` fallback 一并交给外部 skill。

## 当前接通事实

- 当前 `amct_ops` 只有 `hifloat8_cast`，通过 `amct_ops.hifloat8_cast` 和 `torch.ops.amct.*` 暴露 encode/decode。
- Python 子包 `__init__.py` 加载 `.so` 后 re-export 薄 wrapper；wrapper 本身只透传 dispatcher。
- C++ 使用 `TORCH_LIBRARY_FRAGMENT(amct, ...)` 定义 schema，注册 `PrivateUse1` 实算和 `Meta` 形状推导。Meta 不是 CPU fallback，也不做数值计算。
- `ops_build.sh` 自动发现 `<op>/CMakeLists.txt` 和 `<op>/python/<pkg>/`，汇集到 `staging/` 后构建统一 wheel；单个新算子通常不需要修改通用脚本。
- A2 `ascend910b` 与 A3 `ascend910_93` 映射 `dav-2201`，A5 `ascend950` 映射 `dav-3510`；硬件实现细节由 `amct-ops-dev` 维护。
- 当前 HiFloat8 上层先探测原生 `torch_npu`；不支持时在调用点延迟 import `amct_ops`。wheel 缺失会抛友好 `ImportError`，不是静默 CPU 回退，也不会让 `import amct_pytorch` 提前失败。

## 混合任务实施顺序

1. 冻结共享契约，并确认 Python 参考实现或可验证的数值 oracle。
2. 路由 `amct-ops-dev` 实现 schema、PrivateUse1、Meta、kernel、wrapper、构建和 NPU 数值测试。
3. 核对算子 wheel 可导入、schema 已注册，再在 `amct_pytorch` 选择最终消费点。
4. 上层采用延迟 import 或显式能力探测，按契约选择原生实现、Python fallback、amct_ops 或调用时错误。
5. 统一检查五个接口面：C++ schema、PrivateUse1/Meta 函数签名、Python wrapper、上层调用、测试 oracle。
6. schema 变化时说明新旧 wheel/调用方兼容矩阵；破坏性变更优先新函数名并保留旧接口。
7. 最后分层验证，不用编译成功替代 NPU 数值正确性，也不用 Meta 成功替代实算。

## 测试矩阵

| 层 | 最小断言 |
| --- | --- |
| 上层 import | 未安装 `amct_ops` 时 `import amct_pytorch` 成功；只在命中功能时 fallback 或报错 |
| fallback | 原生路径、amct_ops 路径、Python 路径（若定义）、wheel 缺失错误分别可控 |
| schema/wrapper | 参数、默认值、输出 shape/dtype、错误类型在五个接口面一致 |
| Meta | fake tensor/`torch.compile` 可推导 shape/dtype，不触发真实 NPU 计算 |
| wheel | `.so` 和 Python 包进入 wheel，import 后 `torch.ops.amct.<fn>` 存在 |
| NPU 数值 | dtype/shape/边界/特殊值/误差对照 Python 或 CPU oracle |
| 平台 | 按目标范围验证 A2/A3/A5 编译；真实运行平台单独记录 |

从仓库根目录使用：

- 构建：`bash amct_ops/ops_build.sh [--soc <soc>] [<op>]`
- wheel 内容：`python3 -m zipfile -l amct_ops/dist/amct_ops-*.whl`
- staging 导入：`PYTHONPATH=amct_ops/staging python3 -c "import amct_ops.<op>; import torch; print(torch.ops.amct.<fn>)"`
- NPU 单测：`PYTHONPATH=amct_ops/staging python3 -m unittest tests.amct_ops.test_<op>`

## BLOCKER

- 共享 schema、数值、dtype/shape、fallback 或 SoC 契约未冻结，却提前确定上层实现文件或 kernel 方案。
- `amct_pytorch` 写入 kernel/C++ 逻辑，或 `amct_ops` import `amct_pytorch` 算法/workflow。
- 上层模块 import 时硬依赖可选 ops wheel，破坏 CPU/源码环境导入。
- 把 Meta 当 CPU 实算 fallback，或默认要求 NPU-only 算子注册 CPU/CUDA backend。
- schema 改签名但 wrapper、上层调用、Meta、测试或兼容说明未同步。
- Python wrapper 承载 reshape/量化算法以绕过 schema，或未知语义下静默走错误 fallback。
- 只验证构建/import 就声称数值或多 SoC 已通过。

## 自验

- 纯上层改动：运行命中的 fallback/import 单测；当前 HiFloat8 参考为 `tests/amct_pytorch/test_hifloat8_fallback.py`。
- 算子实现、构建和 NPU 测试按 `amct-ops-dev` 执行。
- 涉及 dtype/algorithm/public 安装面时，追加对应子能力测试并说明每层证据等级。
