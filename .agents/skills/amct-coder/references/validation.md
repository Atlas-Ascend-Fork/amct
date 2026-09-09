# AMCT Coder 验证规则

实际编码在修改后读取本文件并执行匹配的最小验证；实现规划、行为诊断和各类 review 只在判断验证方案或既有验证证据时读取相关部分。

## 通用

实际编码任务结束前必须说明：

- 命中的场景操作手册；未分类时说明源码探查结果。
- 实际改动文件。
- 是否触及敏感层：workflow、solver、quant module、public API、ops build。
- 已跑验证。
- 未跑验证及阻塞原因。

缺 NPU、CANN、模型权重、外网数据集、protobuf 生成物时，不要声称已完成对应验证。

## 证据等级

结论按下面等级分别表述，不能互相替代：

1. **源码存在**：实现或注册能在源码中定位。
2. **单测覆盖**：测试直接断言了目标行为；只测 wrapper 分派不等于覆盖 `PtqUnit` 或 PTQ I/O。
3. **集成闭环**：输入、GT、保存、回载等跨模块路径实际跑通。
4. **真实环境验证**：真实模型、数据集、NPU/CANN 或目标硬件结果已取得。

计划中的测试写为“建议运行”或“未运行”，不能表述为已经覆盖。问题诊断只有在完整相关链路被证据排除后才使用“唯一根因”。

## 最小静态验证

- Python 语法：`python -m py_compile <files>`。
- Python 风格检查：对本次改动文件运行 `ruff check <changed files>` 和 `ruff format --check <changed files>`，只验证、不改文件。
- Python 风格修复：确认需要自动修复时运行 `ruff check --fix <changed files>` 和 `ruff format <changed files>`；对应 `.pre-commit-config.yaml` 的 `ruff-check --fix` / `ruff-format` hook，格式选项来自 `pyproject.toml` 的 `[tool.ruff.format]`。
- pre-commit：准备提交且环境未就绪时，先运行 `python -m pip install pre-commit` 和 `pre-commit install`；再运行 `pre-commit run --files <changed files>`。**不要用 `--all-files`**，避免全仓存量问题掩盖本次改动信号。
- C/C++：用仓库 `.clang-format` 检查或格式化本次改动文件。
- 版权：检查新增代码包含仓库既有的 Apache 2.0 版权声明，现有版权头未被删除或破坏。
- public API：跑静态 API 测试。

本节只定义修改后的检查动作。代码风格、版权和贡献要求必须已在修改前按 [project-rules.md](project-rules.md) 建立约束，不能等运行本节时才首次得知。

## 按变更类型选验证

先按本次实际改动选择最小验证，再叠加场景测试索引。不要为了看起来完整而运行无关大集合，也不要用低一级证据替代高一级结论。

| 变更类型 | 最小验证 | 说明 |
| --- | --- | --- |
| 只改 `SKILL.md` 或 reference | `quick_validate.py .agents/skills/amct-coder`、Markdown 链接检查、`git diff --check` | forward-test 只有在路由语义或场景规则明显变化时才需要 |
| docs/examples 纯措辞 | `git diff --check`，必要时检查链接和示例命令是否仍存在 | 不声称运行代码或覆盖实现行为 |
| docs/examples 新增 API/CLI 承诺 | 对应实现 owner 的单测 + 公共接口与打包检查 | 文档承诺必须能被源码和测试支撑 |
| Python 局部实现 | `python -m py_compile <changed .py>` + 命中场景单测 | 若改 public API、workflow 或 solver，再叠加对应集成测试 |
| CLI/parser/default 行为 | parser 测试、命令 help 或最小 workflow dry-run | 默认值变化按兼容性风险说明 |
| 保存/加载/文件格式 | producer 和 consumer 双侧单测或 smoke | 只测保存不等于回载闭环 |
| 构建、packaging、ops 边界 | 对应 build/import 测试；ops 内部实现按 `amct-ops-dev` | 缺 CANN/NPU/protobuf 时明确阻塞 |

## Skill / Reference 自验

修改 `.agents/skills/amct-coder/` 时，除 Markdown 静态检查外执行：

1. 使用当前客户端 `skill-creator` 的 `scripts/quick_validate.py .agents/skills/amct-coder` 校验 frontmatter、名称和基本结构。
2. 解析 `SKILL.md` 与 reference 内部 Markdown 链接，确认目标存在；核对导航表中的直接 reference 与实际设计一致。
3. 为命中的单 reference 保留正常题、边界路由题、未知契约题和伪活跃调用链题；跨 reference 至少覆盖 dtype+deploy、algorithm+solver、data+adapter、model+external skill 中受本次修改影响的组合。
4. cold 与 warm forward-test 分开。cold 测试使用物理隔离的新会话，只投喂原始用户题目和可发现 skill，不泄露预期答案、诊断或 reviewer 材料。
5. 原始 prompt、Agent 输出和 reviewer 评分分文件保存，记录源码 commit、实际读取资料、测试模式和 rubric；不要把审查结论追加进被测 Agent 可读取的答案文件。

`quick_validate.py`、链接检查和 `git diff --check` 只证明结构静态有效，不能替代 forward-test。当前工作区含测试答案或预期结论时，明确标记 cold 测试未执行，不用同会话重跑冒充独立行为证据。

## 越界自查

改完按实际调用关系逐个确认改动文件所在层、是否越层以及上下游契约是否闭环。四个高频问题：

1. 改动是否停在声明的落点层，没有顺手改上下游？
2. 是否先复用了现有抽象（`QuantLinear` / `QuantGatedMLP` / family helper）再新增？
3. 关闭量化时原浮点行为是否保住？
4. 是否有没必要的框架级重构或 solver 改动？

## 公共接口与打包检查

用户可见改动需要同时核对源码实现、入口转发、安装包和文档示例：

- 公共 API：分别检查普通 import、顶层属性访问、通配符 import、源模块 `__all__` 和顶层 `__all__`；懒加载 API 不能只证明属性可达。
- CLI：核对 parser 参数、默认值、choices、帮助文本和 `python -m` 入口是否与实际 registry/workflow 一致。
- 打包：检查 `setup.py`、`pyproject.toml`、`build.sh`、`package_data` 和 `AMCT_EXPERIMENTAL` 开关；不提交构建产物或 generated staging。
- 配置口径：文档中的 BF16、weight-only、W4A4、dtype 和 bit policy 必须与 parser 和实际每个 role 的配置一致。
- 文档示例：新增 API、CLI 参数、dtype 或 workflow 行为时同步更新，并确认示例命令在当前 parser 中存在。

## 场景测试索引

这里集中维护测试定位提示与命令；场景手册只描述需要验证的行为。先按实现符号、导入关系或目标行为在 `tests/` 中搜索直接断言，确认文件仍存在并覆盖本次改动，再运行对应测试。下表的目录是搜索范围，不要求整目录执行；文件改名后以搜索结果为准。

通用执行方式：`pytest <已确认的测试文件>`；需要缩小范围时使用 `pytest <已确认的测试文件> -k <相关测试名>`。新增行为没有直接断言时，在所属测试模块补覆盖，不把邻近测试通过当成该行为已验证。

| 场景操作手册 | 当前测试定位提示 |
| --- | --- |
| `quant-dtype-modules` | `tests/unit_test/quantization/test_dtypes.py`、`tests/unit_test/quantization/modules/`；按 dtype/quantizer/BitPolicy 符号定位 |
| `llm-algorithm` | `tests/unit_test/algorithms/`；注册参考 `test_algorithms_registry_factory.py`，分流参考 `tests/unit_test/quantization/modules/test_quant_base.py` |
| `model-adapter` | `tests/unit_test/common/models/llm/`；按目标 adapter、wrapper 或 `PtqUnit` 符号定位，跨层行为再查 mocked workflow 测试 |
| `llm-data-workflow` | `tests/unit_test/workflows/`、`tests/unit_test/common/datasets/`、`tests/unit_test/common/evaluate/`；按 workflow、provider/I/O、PPL 行为定位 |
| `ptq-solver-params` | `tests/unit_test/common/optimization/`、`tests/unit_test/common/models/llm/common/test_ptq_params.py`、`tests/unit_test/workflows/test_llm_ptq.py` |
| `deploy-export-impl` | workflow：`tests/unit_test/workflows/test_llm_deploy.py`；helper：`tests/unit_test/common/models/llm/common/test_deploy_export.py` |
| `amct-ops-dev` 上层接入 | 上层 fallback/import 参考 `tests/amct_pytorch/test_hifloat8_fallback.py`；算子硬件测试在 `tests/amct_ops/`，按 `amct-ops-dev` 执行 |
| 公共接口与打包 | `tests/unit_test/test_public_api.py`、`tests/unit_test/test_packaging.py`、`tests/unit_test/cli/test_llm_module_entrypoints.py` |

常用定向命令，按实际改动选取：

- dtype 与 quantizer 分流：`pytest tests/unit_test/quantization/test_dtypes.py tests/unit_test/quantization/modules/test_quant_base.py`。
- workflow：`pytest tests/unit_test/workflows/test_llm_eval.py tests/unit_test/workflows/test_llm_extract_ptq_data.py tests/unit_test/workflows/test_llm_ptq.py`，只涉及其中一个命令时只选该文件。
- 数据保存与加载：`pytest tests/unit_test/common/datasets/test_ptq_io.py tests/unit_test/common/datasets/test_ptq_provider.py`。
- PPL：`pytest tests/unit_test/common/evaluate/test_eval_ppl.py`。
- solver 与 factory：`pytest tests/unit_test/common/optimization/test_base_solver.py tests/unit_test/common/optimization/test_blockwise_solver.py tests/unit_test/common/optimization/test_factory.py`。
- 导出：`pytest tests/unit_test/workflows/test_llm_deploy.py tests/unit_test/common/models/llm/common/test_deploy_export.py`。
- public API / packaging：`pytest tests/unit_test/test_public_api.py tests/unit_test/test_packaging.py`。
- CLI 入口：`pytest tests/unit_test/cli/test_llm_module_entrypoints.py`；parser/default 行为还需确认有相应断言。

算法自身、classic custom algorithm/config、graph pass、quantize/save、prune/distill/retrain/cali 等按符号与领域定位；不维护逐算法或逐 pass 的文件清单。跨契约任务按生产者和消费者补充上述验证，测试直接 import 未接通模块不证明正常注册入口已接通。

## LLM adapter / wrapper 完成标准

满足前不要认为完成：

1. `MODEL_REGISTRY` 可取到模型。
2. 至少一层 float block 可构建。
3. 至少一层 quant block 可构建。
4. 至少一个 `PtqUnit` 可枚举。
5. 关闭量化后 quant block 与 float block 前向等价或足够接近。
6. 至少一个 unit 输入/GT/参数导出路径闭环。

## PTQ smoke 健康信号

- unit 输入加载成功。
- GT materialize 成功。
- solver 至少跑一个 batch。
- loss 不是恒 0 或 NaN。
- 参数真实更新或统计 buffer 真实写入。
- 参数保存后可回载。

## PPL 口径

Wikitext PPL 默认 `seq_len=4096`。历史结果不是同口径时，只能参考，不能直接比较。

BF16 PPL 异常时第一优先级检查 mask/position/position_embeddings 传递链。

## 构建

- 查看选项：`bash build.sh --help`。
- 构建 amct_pytorch 包：`bash build.sh --torch`。
- 完整包：`bash build.sh --pkg`。
- 单元测试构建：`bash build.sh -u`。
- `bash build.sh` 不带构建选项不会产分发包。

`amct_ops` 构建走 `cd amct_ops && bash ops_build.sh [--soc <soc>] [<op>]`。
