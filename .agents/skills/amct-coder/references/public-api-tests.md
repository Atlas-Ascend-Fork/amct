# 子能力：public-api-tests

## 触发

用户提到公开 API、`__all__`、顶层 `amct_pytorch`、package、build、setup、requirements、examples/docs、测试选择、CI/pre-commit 时使用。

## 必读文件

- `amct_pytorch/__init__.py`
- `amct_pytorch/classic/graph_based/amct_pytorch/__init__.py`
- `setup.py`
- `pyproject.toml`
- `build.sh`
- `AGENTS.md`
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
2. 用户可见 API 要同步源模块 `__all__`、顶层 re-export 或 `__getattr__` 懒加载。
3. CLI 改动要同步 choices、help、module entrypoint 和解析测试。
4. packaging 改动要检查 `setup.py`、`pyproject.toml`、`build.sh`、package_data 和实验特性开关。
5. docs/examples 只在用户使用方式变化时更新，并保持命令与当前 build/install 规则一致。
6. 补 public API、packaging、CLI entrypoint 或 docs/example smoke 的最小测试。

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
- 源模块新增 public API 但顶层不可达。
- CLI 参数帮助、choices 与实际 registry 不一致。
- 构建产物或 generated staging 被提交。
- 测试只覆盖 import，不覆盖注册/行为。

## 自验

- `pytest tests/unit_test/test_public_api.py tests/unit_test/test_packaging.py`
- CLI 改动：`pytest tests/unit_test/cli/test_llm_module_entrypoints.py`
- 构建脚本改动：至少 `bash build.sh --help`，必要时 `bash build.sh --torch`。
