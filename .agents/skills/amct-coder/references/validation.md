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
- Python 风格：对本次改动文件运行仓库 Ruff 配置对应的检查和格式化。
- pre-commit：准备提交且环境未就绪时，先运行 `python -m pip install pre-commit` 和 `pre-commit install`；再运行 `pre-commit run --files <changed files>`。**不要用 `--all-files`**，避免全仓存量问题掩盖本次改动信号。
- C/C++：用仓库 `.clang-format` 检查或格式化本次改动文件。
- 版权：检查新增代码包含仓库既有的 Apache 2.0 版权声明，现有版权头未被删除或破坏。
- public API：跑静态 API 测试。

本节只定义修改后的检查动作。代码风格、版权和贡献要求必须已在修改前按 [project-rules.md](project-rules.md) 建立约束，不能等运行本节时才首次得知。

## 越界自查

改完对照 [layer-rules.md](layer-rules.md) §按层检查规则，逐个改动文件确认所在层的检查项都过。四个高频问题：

1. 改动是否停在声明的落点层，没有顺手改上下游？
2. 是否先复用了现有抽象（`QuantLinear` / `QuantGatedMLP` / family helper）再新增？
3. 关闭量化时原浮点行为是否保住？
4. 是否有没必要的框架级重构或 solver 改动？

## 场景测试索引

| 场景操作手册 | 推荐最小测试 |
| --- | --- |
| `dtype-quant-module` | `pytest tests/unit_test/quantization/test_dtypes.py tests/unit_test/quantization/modules/test_quant_base.py` |
| `llm-algorithm` | `pytest tests/unit_test/algorithms tests/unit_test/quantization/modules/test_quant_base.py` |
| `llm-model-unit`（契约对齐 / review / 局部修复） | `pytest tests/unit_test/common/models/llm`，必要时 mocked workflow 测试 |
| `llm-data-workflow` | `pytest tests/unit_test/workflows tests/unit_test/common/datasets tests/unit_test/common/evaluate` |
| `ptq-solver-params` | `pytest tests/unit_test/common/optimization tests/unit_test/workflows/test_llm_ptq.py` |
| `deploy-export-code` | `pytest tests/unit_test/workflows/test_llm_deploy.py` |
| `classic-graph` | `pytest tests/unit_test/classic tests/amct_pytorch/testcase_python/<area>` |
| `ops-boundary` | `cd amct_ops && bash ops_build.sh <op>`；NPU 上跑 `tests/amct_ops` |
| `public-api-tests` | `pytest tests/unit_test/test_public_api.py tests/unit_test/test_packaging.py` |

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
