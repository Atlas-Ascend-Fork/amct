# 子能力：public-api-tests

## 触发

用户提到公开 API、`__all__`、顶层 `amct_pytorch`、package、build、setup、requirements、examples/docs、测试选择、CI/pre-commit 时使用。

## 源码锚点（按需选择）

以下路径不是全文必读清单。先按 public API、CLI、packaging、docs/examples 或测试入口选择当前用户可见契约的直接锚点。

- `amct_pytorch/__init__.py`
- `amct_pytorch/classic/graph_based/amct_pytorch/__init__.py`
- `setup.py`
- `pyproject.toml`
- `build.sh`
- `tests/unit_test/test_public_api.py`
- `tests/unit_test/test_packaging.py`
- `tests/unit_test/cli/test_llm_module_entrypoints.py`
- 改动命中的 docs/examples

## 上下游依赖

- 上游：public API、CLI、build 和 docs/examples 的变化来自 dtype、algorithm、model adapter、workflow、classic/graph_based 或 ops 的用户可见改动。
- 平级：本 playbook 不定义业务语义，只检查导出、懒加载、packaging、文档样例和测试入口是否闭环。
- 下游：用户安装包、`import amct_pytorch`、CLI module、examples 和 CI/pre-commit 依赖这里的契约稳定。
- 若只是内部 wrapper bug 或单测补充，不需要强行更新 docs/examples。

## 执行步骤

1. 先判断改动是否用户可见：新 API、新 CLI 参数、新 dtype/算法可选项、新 build 行为、新 ops 用户入口或 docs/examples 行为变化。
2. 用户可见 API 先区分下节五种导入/导出契约，再决定同步源模块 `__all__`、顶层 re-export 或 `__getattr__` 懒加载。
3. CLI 改动先确认复用 helper 的真实符号和稳定 I/O，再同步 choices、help、module entrypoint 和解析测试。
4. packaging 改动要检查 `setup.py`、`pyproject.toml`、`build.sh`、package_data 和实验特性开关。
5. docs/examples 只在用户使用方式变化时更新，并保持命令与当前 build/install 规则一致。
6. 补 public API、packaging、CLI entrypoint 或 docs/example smoke 的最小测试。

## 用户目标到 CLI/config 的口径门

用户用自然语言描述精度、位宽或量化范围时，不能直接映射到名称相近的现有命令。计划、review 和 docs/example 必须展开并核对：

| 用户口径 | 必须证明的配置事实 |
| --- | --- |
| FP16 / BF16 baseline | CLI 是否精确支持该计算 dtype；只支持 `eval_mode=bf16` 时不能把结果称为 FP16，必须显式报告差异 |
| weight-only | bit policy 中所有 activation/cache role 保持 16 bit 或未量化，只有目标 weight role 低于 16 bit |
| W4A4 | weight 与 activation role 都是 4 bit，不能描述成 INT4 weight-only |
| INT4 / MXFP4 等 dtype | `quant_dtype`、bit policy、algorithm 和 deploy route 的组合全部受支持，而不是只匹配位宽 |
| baseline vs quant | 模型、数据集、`seq_len`、block 范围和除量化配置外的 eval 参数一致 |

当前 LLM CLI 的 baseline 入口是否只有 BF16、位宽是否只由 `--bit_config` 表达、target/dtype choices 是否接受拟议值，都以本轮 `args.py` 和 parser 测试为准。CLI 无法精确表达用户要求时，只读规划明确列出支持口径与请求口径的差异；实际编码若要新增用户可见选择，先冻结兼容性和默认行为。不要用不存在的 `--w_bits` / `--a_bits` 或名称相近的参数拼出示例命令。

## 五种导入与导出契约

以下行为相互关联但不等价，review 和测试必须逐项说明实际承诺了哪一种：

| 契约 | 核对重点 |
| --- | --- |
| 普通 import | `import amct_pytorch` 是否保持成功、启动开销和可选依赖边界；不能用随后访问属性的测试代替 |
| 顶层属性访问 | `amct_pytorch.<name>` 是显式绑定还是由 `__getattr__` 延迟解析；首次访问失败和缓存行为是什么 |
| wildcard import | `from amct_pytorch import *` 会遍历顶层 `__all__` 并解析其中的名字；加入懒加载名可能在此时触发 graph/可选依赖 |
| 源模块 `__all__` | 名字是否由定义它的模块正式导出；不能只从顶层可达性反推 |
| 顶层 `__all__` | 是否承诺 wildcard export；名字必须有显式绑定或可验证的懒解析路径，并评估兼容性变化 |

graph_based 名字通过 `__getattr__` 可达，不表示它已经进入顶层 `__all__`，也不表示普通 import、属性访问和 wildcard import 具有相同依赖行为。若要把懒加载名加入顶层 `__all__`，必须明确接受、规避或测试 wildcard import 触发 graph 依赖的结果。

阅读现有测试时先确认断言集合的来源。参数化测试遍历固定的 `NON_GRAPH_INTERFACES`，不等于验证所有顶层 `__all__` 成员；只有测试直接从 AST/模块读取完整 `__all__` 时，才能声称是全量守卫。

## 未知 helper 门禁

“复用现有内部 helper”但未给出符号、模块或稳定输入输出时：

1. 先搜索候选定义和活跃调用方，区分“名称相似的 helper”与“题目所指 helper”。
2. 找到多个候选时列出候选及其差异；helper 身份会改变 CLI、输出或依赖边界时，停在契约确认或条件式计划，不替用户选择。
3. helper 未确认前，不把某个候选的参数、默认值、输出格式、异常和依赖写成新命令的确定契约。
4. helper 确认后，才决定复用通用 parser、建立专用 parser，或是否承诺对应 Python public API。

## public API 规则

- 顶层 `amct_pytorch` 显式导出 classic `quantize/convert/algorithm_register` 和 config 常量。
- graph_based API 通过 `__getattr__` 懒加载，避免 LLM-only import 被 ONNX/protobuf 依赖拖垮。
- 新 public API 要同步源模块 `__all__`、顶层 re-export 或懒加载路径、测试。
- 新 CLI module 要确保 `python -m amct_pytorch.<cmd>` 和 `amct_pytorch/cli/llm/<cmd>.py` 对齐。

## packaging 规则

- `bash build.sh` 不带 `--torch`、`--pkg` 或 `-u` 不产分发包。
- `AMCT_EXPERIMENTAL=TRUE` 时才打包 `amct_pytorch.experimental`。
- package_data 中 graph_based 依赖 proto/csv/lib。
- pip 版本大于 25.2 安装源码包时可能需要 `--no-build-isolation`。

## 测试选择

- LLM workflow：`tests/unit_test/workflows/`。
- quantization dtype/module：`tests/unit_test/quantization/`。
- LLM algorithms：`tests/unit_test/algorithms/`。
- common model adapter：`tests/unit_test/common/models/llm/`。
- classic eager：`tests/unit_test/classic/`、`tests/amct_pytorch/test_*.py`。
- graph_based：`tests/amct_pytorch/testcase_python/<area>/`。
- ops：`tests/amct_ops/`，需要 NPU。
- public API / packaging：`tests/unit_test/test_public_api.py`、`tests/unit_test/test_packaging.py`。

## docs/examples 同步

需要同步 docs/examples 的情况：

- 新增用户可见 API 或 CLI 参数。
- 新增预定义 config 或量化 dtype。
- 新增 public workflow 行为。
- 新增 amct_ops 用户可调用算子。

不需要同步的情况：

- 仅修内部 wrapper bug。
- 仅补 unit test。
- 不改变用户接口的局部重构。

## BLOCKER

- `__all__` 中声明了名字但没有绑定。
- 把顶层属性可达误写成已进入顶层 `__all__`，或忽略 wildcard import 对懒加载/可选依赖的影响。
- 源模块新增 public API 但顶层不可达。
- CLI 参数帮助、choices 与实际 registry 不一致。
- FP16/BF16、weight-only/W4A4 或 dtype/bit policy 口径不一致，却在命令、文档或精度比较中当作同一配置。
- 示例命令使用当前 parser 不存在的参数，或未证明 bit policy 中每个 role 与用户目标一致。
- helper 身份或稳定 I/O 未确认，却据候选 helper 固定 CLI 参数、输出格式或退出码。
- 构建产物或 generated staging 被提交。
- 测试只覆盖 import，不覆盖注册/行为。

## 自验

- `pytest tests/unit_test/test_public_api.py tests/unit_test/test_packaging.py`
- CLI 改动：`pytest tests/unit_test/cli/test_llm_module_entrypoints.py`
- 精度/位宽口径：增加 parser + `BitPolicy` 断言，逐 role 证明 baseline、weight-only 或 W4A4 的实际配置。
- 涉及懒加载或顶层 `__all__`：分别覆盖普通 import 不加载可选依赖、属性访问解析、wildcard import 行为、源模块与顶层 `__all__` 声明。
- 构建脚本改动：至少 `bash build.sh --help`，必要时 `bash build.sh --torch`。
