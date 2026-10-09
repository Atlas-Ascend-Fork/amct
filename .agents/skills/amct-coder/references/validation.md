# 验证规则：validation

## 通用

本文件只约束交付说明里「已跑/未跑验证」两项的判定标准：

- 证据写到哪一级，按下方「证据等级」表述；不得用低级证据替代高级结论。
- 缺 NPU、CANN、模型权重、外网数据集、protobuf 生成物时，不要声称已完成对应验证；按「验证失败的处理」的「环境缺失」行写成「X 未执行：缺 Y」，不得写成已验证或跳过。

## 证据等级

结论按下面等级分别表述，不能互相替代：

1. **源码存在**：实现或注册能在源码中定位。
2. **单测覆盖**：测试直接断言了目标行为；只测 wrapper 分派不等于覆盖 `PtqUnit` 或 PTQ I/O。
3. **集成闭环**：输入、GT、保存、回载等跨模块路径实际跑通。
4. **真实环境验证**：真实模型、数据集、NPU/CANN 或目标硬件结果已取得。

要断言“集成闭环”，必须同时给出注册、活跃调用方和直接测试证据；找不到调用方的 helper、hook 或导出函数记为“源码存在但当前未接通”，只能停在第 1 级。

计划中的测试写为“建议运行”或“未运行”，不能表述为已经覆盖。「唯一根因」的判定条件是完整相关链路已被证据排除。

## 最小静态验证

- Python 语法：`python -m py_compile <files>`。
- Python 风格检查：对本次改动文件运行 `ruff check <changed files>` 和 `ruff format --check <changed files>`，只验证、不改文件。
- Python 风格修复：确认需要自动修复时运行 `ruff check --fix <changed files>` 和 `ruff format <changed files>`；对应 `.pre-commit-config.yaml` 的 `ruff-check --fix` / `ruff-format` hook，格式选项来自 `pyproject.toml` 的 `[tool.ruff.format]`。
- pre-commit：环境已就绪时运行 `pre-commit run --files <changed files>`（**不要用 `--all-files`**，避免全仓存量问题掩盖本次改动信号）。未安装 pre-commit 时**不主动安装**——安装有网络、环境和权限副作用；在交付说明中报告「pre-commit 未执行：环境未就绪」即可。仅当用户明确要求完整提交前检查时，经确认后再安装（`python -m pip install pre-commit` + `pre-commit install`）。
- 版权：检查新增代码包含仓库既有的 Apache 2.0 版权声明，现有版权头未被删除或破坏。
- public API：`pytest tests/unit_test/test_public_api.py`（其中 ast 解析的静态检查无需构建环境，import 行为用例在未构建源码树时自动跳过）。

## 验证失败的处理

| 失败类型 | 判定 | 处理 | 交付措辞 |
| --- | --- | --- | --- |
| 静态检查失败（py_compile/ruff） | 失败点在本次改动文件内 | ruff 的存量告警（stash 对照确认非本次引入）不修，如实引用 | 「静态检查通过；X 为存量问题（对照 HEAD 确认）」 |
| 本次改动引入的单测失败 | 失败用例断言的是本次改动目标或其直接下游 | 修复实现或测试（实现错误改实现，断言过时改断言并说明），重跑至绿 | 「单测通过；修正了 X（原因）」 |
| 与本次改动无关的单测失败 | 失败用例与改动面无调用关系 | 不顺手修（范围控制，另开 issue）；验证证据标注该失败与本改动的关系判断依据 | 「本次改动相关用例全过；Y 失败与本改动无关（依据：调用链不相交），已单独记录」 |
| 环境缺失（NPU/CANN/权重/数据集/pre-commit） | 验证目标需要外部资源 | 不假装已验证，不阻塞可执行部分；能 mock 的走单测级 | 「X 未执行：缺 Y」——不得写成已验证或跳过 |
| 间歇性失败（flake） | 同一用例重跑结果不一致 | 重跑 2~3 次确认；仍复现按无关失败或本次引入分类 | 标注重跑次数与结果分布 |

## 按变更类型选验证

先按本次实际改动选择最小验证，再叠加场景测试索引。

| 变更类型 | 最小验证 | 说明 |
| --- | --- | --- |
| docs/examples 纯措辞 | `git diff --check`，必要时检查链接和示例命令是否仍存在 | 不声称运行代码或覆盖实现行为 |
| docs/examples 新增 API/CLI 承诺 | 对应实现 owner 的单测 + 公共接口与打包检查 | 文档承诺必须能被源码和测试支撑 |
| Python 局部实现 | `python -m py_compile <changed .py>` + 命中场景单测 | 若改 public API、workflow 或 solver，再叠加对应集成测试 |
| CLI/parser/default 行为 | parser 测试、命令 help 或最小 workflow dry-run | 默认值变化按兼容性风险说明 |
| 保存/加载/文件格式 | producer 和 consumer 双侧单测或 smoke | 只测保存不等于回载闭环 |
| 构建、packaging、ops 边界 | 对应 build/import 测试 | 缺 CANN/NPU/protobuf 时明确阻塞 |

## 公共接口与打包检查

用户可见改动需要同时核对源码实现、入口转发、安装包和文档示例：

- 公共 API：分别检查普通 import、顶层属性访问、通配符 import、源模块 `__all__` 和顶层 `__all__`；懒加载 API 不能只证明属性可达。
- CLI：核对 parser 参数、默认值、choices、帮助文本和 `python -m` 入口是否与实际 registry/workflow 一致。
- 打包：按改动面检查——运行依赖变更查 `requirements.txt`，打包变更查 `setup.py`、`pyproject.toml`、`build.sh`、`package_data` 和 `AMCT_EXPERIMENTAL` 开关；不提交构建产物或 generated staging。
- 配置口径：文档中的 BF16、weight-only、W4A4、dtype 和 bit policy 必须与 parser 和实际每个 role 的配置一致。
- 文档示例：新增 API、CLI 参数、dtype 或 workflow 行为时同步更新，并确认示例命令在当前 parser 中存在。

## 场景测试索引

先按实现符号、导入关系或目标行为在 `tests/` 中搜索直接断言，确认文件仍存在并覆盖本次改动，再运行对应测试。

新增行为没有直接断言时，在所属测试模块补覆盖，不把邻近测试通过当成该行为已验证。

| 场景操作手册 | 当前测试定位提示 |
| --- | --- |
| `quant-dtype-module` | `tests/unit_test/quantization/test_dtypes.py`、`tests/unit_test/quantization/modules/`；按 dtype/quantizer/BitPolicy 符号定位 |
| `llm-algorithm` | `tests/unit_test/algorithms/`；注册参考 `test_algorithms_registry_factory.py`，分流参考 `tests/unit_test/quantization/modules/test_quant_base.py` |
| `model-adapter-review` | `tests/unit_test/common/models/llm/`；按目标 adapter、wrapper 或 `PtqUnit` 符号定位，跨端行为再查 mocked workflow 测试 |
| `llm-cli-dataflow` | `tests/unit_test/workflows/`、`tests/unit_test/common/datasets/`、`tests/unit_test/common/evaluate/`；按 workflow、provider/I/O、PPL 行为定位 |
| `llm-solving-lifecycle` | `tests/unit_test/common/optimization/`、`tests/unit_test/common/models/llm/common/test_ptq_params.py`、`tests/unit_test/workflows/test_llm_ptq.py` |
| `llm-deploy` | workflow：`tests/unit_test/workflows/test_llm_deploy.py`；helper：`tests/unit_test/common/models/llm/common/test_deploy_export.py` |
| NPU 算子上层接入 | 上层 fallback/import 参考 `tests/amct_pytorch/test_hifloat8_fallback.py`；算子硬件测试在 `tests/amct_ops/` |
| 公共接口与打包 | `tests/unit_test/test_public_api.py`、`tests/unit_test/test_packaging.py`、`tests/unit_test/cli/test_llm_module_entrypoints.py` |

同一文件被多个场景引用时（如 `test_llm_ptq.py` 同时覆盖 `llm-cli-dataflow` 的 ptq 命令和 `llm-solving-lifecycle` 的求解链路）按本次改动选择相关用例，命令详见下节。

常用定向命令，按实际改动选取：

- dtype 与 quantizer 分流：`pytest tests/unit_test/quantization/test_dtypes.py tests/unit_test/quantization/modules/test_quant_base.py`。
- workflow：`pytest tests/unit_test/workflows/test_llm_eval.py tests/unit_test/workflows/test_llm_extract_ptq_data.py tests/unit_test/workflows/test_llm_ptq.py`，只涉及其中一个命令时只选该文件。
- 数据保存与加载：`pytest tests/unit_test/common/datasets/test_ptq_io.py tests/unit_test/common/datasets/test_ptq_provider.py`。
- PPL：`pytest tests/unit_test/common/evaluate/test_eval_ppl.py`。
- solver 与 factory：`pytest tests/unit_test/common/optimization/test_base_solver.py tests/unit_test/common/optimization/test_blockwise_solver.py tests/unit_test/common/optimization/test_factory.py`。
- 导出：`pytest tests/unit_test/workflows/test_llm_deploy.py tests/unit_test/common/models/llm/common/test_deploy_export.py`。
- public API / packaging：`pytest tests/unit_test/test_public_api.py tests/unit_test/test_packaging.py`。
- CLI 入口：`pytest tests/unit_test/cli/test_llm_module_entrypoints.py`；parser/default 行为还需确认有相应断言。

算法自身、classic custom algorithm/config、graph pass、quantize/save、prune/distill/retrain/cali 等按符号与领域定位。跨端任务按生产者和消费者补充上述验证，测试直接 import 未接通模块不证明正常注册入口已接通。

## 构建

- 查看选项：`bash build.sh --help`。
- 构建 amct_pytorch 包：`bash build.sh --torch`。
- 完整包：`bash build.sh --pkg`。
- 单元测试构建：`bash build.sh -u`。
- `bash build.sh` 不带构建选项不会产分发包。

`amct_ops` 构建走 `cd amct_ops && bash ops_build.sh [--soc <soc>] [<op>]`。
