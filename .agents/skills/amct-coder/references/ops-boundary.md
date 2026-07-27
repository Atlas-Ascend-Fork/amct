# 子能力：ops-boundary

## 触发

用户提到 `amct_ops`、NPU 自定义算子、`torch.ops.amct.*`、ops 构建、wheel、CMake、Meta kernel、PrivateUse1、Ascend C kernel，或 `amct_pytorch` 需要调用 NPU 算子时使用。

## 边界

`amct-coder` 负责：

- 判断 `amct_pytorch` 与 `amct_ops` 的调用边界。
- 检查 Python 侧是否正确 import/load `amct_ops`。
- 检查构建入口、packaging、测试命令和上层调用契约。
- 在 review 中指出 kernel/schema/Meta 缺失风险。

真正新增或修改 Ascend C kernel、C++ extension、schema、Meta backend、CMake 细节时，转 `amct-ops-dev`。

## 必读文件

- `amct_ops/README.md`
- `amct_ops/ops_build.sh`
- `amct_ops/setup.py`
- `amct_ops/ops_init.py`
- `amct_ops/hifloat8_cast/CMakeLists.txt`
- `amct_ops/hifloat8_cast/op_extension/register.cpp`
- `amct_ops/hifloat8_cast/op_extension/hifloat8_cast_torch.cpp`
- `amct_ops/hifloat8_cast/python/hifloat8_cast/ops.py`
- `.agents/skills/amct-ops-dev/SKILL.md`
- `tests/amct_ops/`

## 上下游依赖

- 上游：`amct_pytorch` 只依赖 Python wrapper 或 `torch.ops.amct.*` schema，不依赖 kernel 源码细节。
- ops 内部依赖链是 `ops_build.sh` -> CMake -> Ascend C/C++ extension -> `TORCH_LIBRARY_FRAGMENT` schema -> backend impl -> Python wrapper -> tests。
- 下游：dtype、algorithm 或 workflow 调用 ops 时必须保留 CPU/source import 可用性；packaging 需要 wheel、staging 和 import 路径闭环。
- 真正新增/修改 Ascend C kernel、C++ extension、Meta backend 或 CMake 细节时，转 `amct-ops-dev`。

## 执行步骤

1. 先判断需求是上层调用边界、packaging/build 检查，还是 kernel/C++ 开发。
2. 上层调用只改 `amct_pytorch` 的软依赖、fallback、wrapper 调用和测试，不写 kernel 逻辑。
3. schema 或 wrapper 变更时，检查 `TORCH_LIBRARY_FRAGMENT`、Python API、旧接口兼容和 Meta backend 风险。
4. build 变更时，检查 `ops_build.sh`、算子本地 `CMakeLists.txt`、wheel 内容和生成目录是否被提交。
5. 需要硬件实现时停下并进入 `amct-ops-dev`，本 playbook 只保留边界检查结论。
6. 补导入/schema、wheel 内容、CPU fallback 或 NPU 单测的最小验证。

## 上层调用规则

- `amct_pytorch` 可以通过 Python 包装或 `torch.ops.amct.<fn>` 调用算子，但不应包含 kernel/C++ 实现。
- 算子算法语义属于上层量化算法/数据类型；kernel 只实现硬件计算。
- 若 dtype 需要 NPU 加速，先保留 Python fallback 或明确依赖 `amct_ops`，不要让 import 阶段硬失败。

## 构建规则

- 构建统一走 `cd amct_ops && bash ops_build.sh [--soc <soc>] [<op>]`。
- `--soc` 映射：`ascend910b/ascend910_93 -> dav-2201`，`ascend950 -> dav-3510`。
- wheel 产物在 `amct_ops/dist/`。
- `build/`、`dist/`、`staging/`、`<op>/build/` 是构建产物，不提交。

## BLOCKER

- 新算子没有注册 `TORCH_LIBRARY_FRAGMENT(amct, ...)`。
- 没有 `TORCH_LIBRARY_IMPL(amct, Meta, ...)`，导致 fake tensor / torch.compile 形状推导失败。
- schema 改签名但未保留旧接口或迁移说明。
- Python wrapper 做了大量 reshape/算法逻辑，绕过 C++ schema 约束。
- `ops_build.sh` 为单个算子写特殊逻辑，而不是放入算子自己的 `CMakeLists.txt`。
- `amct_pytorch` import 时强依赖未安装的 ops wheel，导致 CPU/源码测试无法 import。

## 自验

- 仅构建检查：`cd amct_ops && bash ops_build.sh <op>`。
- wheel 内容：`python3 -m zipfile -l dist/amct_ops-*.whl`。
- 导入/schema：`PYTHONPATH=amct_ops/staging python -c "import amct_ops.<op>; import torch; print(torch.ops.amct.<fn>)"`。
- 单测：`PYTHONPATH=amct_ops/staging python3 -m unittest tests.amct_ops.test_<op>`。
- 需要 NPU/CANN 的验证没跑时必须明确说明。
