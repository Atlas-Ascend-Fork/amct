# 子能力：public-api-packaging

解决 public API、安装打包与用户文档承诺的修改位置、跨端约定和验证问题：新 API 如何导出与懒加载、packaging 清单如何同步、文档示例如何与实现一致。本子能力负责「用户可见面」的一致性闭环。

## 职责边界

- 承接：`amct_pytorch/__init__.py` 的导出与懒加载、`setup.py`/`pyproject.toml`/`build.sh`/`package_data`、`requirements.txt`（运行依赖安装入口）、`AMCT_EXPERIMENTAL` 开关、examples 与 `docs/` 中的 API/CLI 承诺、版本兼容声明。

## 实现定位

- 主要入口：`amct_pytorch/__init__.py`（`__all__`、`__getattr__` PEP 562 懒加载）、`setup.py`、`pyproject.toml`、`build.sh`。
- 关键符号：`__all__`、`_PRUNING_EXPORTS`、`__getattr__`、`get_package_data()`、`AMCT_EXPERIMENTAL`。
- 迁移后沿符号与打包脚本重定位。

## 当前事实

每条事实附一行「核对」命令；作为判断依据前先执行对应命令，结果与断言冲突时以命令结果为准。发现不符时不改本节，把差异写进交付说明（判定见 [conditional-rules「当前事实与核对命令纪律」](conditional-rules.md#当前事实与核对命令纪律)）。

- classic eager 接口（`quantize`/`convert`/`algorithm_register`）与 config 常量在模块顶层直接 import；pruning 与 graph_based 接口走 `__getattr__` 懒加载（PEP 562），使 LLM-only 使用可 `import amct_pytorch` 而不拉入 onnx/protobuf。（核对：`rg -n "^from|^import|def __getattr__" amct_pytorch/__init__.py`）
- `AMCT_EXPERIMENTAL` 环境变量（`TRUE`）控制 experimental 目录是否进入安装包。（核对：`rg -n "AMCT_EXPERIMENTAL" setup.py`）
- 构建入口：`bash build.sh --torch`（amct_pytorch 包）/ `--pkg`（完整包）/ `-u`（单测构建）；`amct_ops` 走独立 `ops_build.sh`。（核对：`bash build.sh --help`）

## 前置检查

### 新 public API 闭环

新增或修改 public API（`__all__` 成员、顶层函数、config 常量）前逐项确定：

| 跨端约定 | 必须回答的问题 |
| --- | --- |
| 导出形态 | 顶层直接 import 还是 `__getattr__` 懒加载；判据是依赖重量（重依赖必须懒加载，避免 LLM-only 用户被迫装 onnx/protobuf） |
| `__all__` 一致性 | 源模块 `__all__`、顶层 `__all__`、通配符 import、普通 import、顶层属性访问五面一致；懒加载 API 不能只验证属性可达 |
| 向后兼容 | 既有签名与语义无声变化即破坏；改名/删除需弃用期或显式破坏声明 |
| 测试 | `tests/unit_test/test_public_api.py` 有直接断言；静态 ast 检查无需构建环境，import 行为用例在未构建源码树时自动跳过 |
| 文档 | `docs/` 与 examples 的 API 签名同步；示例命令在当前实现中真实可执行 |

### packaging 同步

- 新增第三方依赖按用途落对入口（运行依赖 → `requirements.txt`；构建依赖 → `pyproject.toml` `[build-system]`；可选依赖 → `setup.py` `extras_require`），细则与「不改变现有打包机制」约束见 [coding-rules](coding-rules.md#依赖与放置位置)；核对仓内声明支持的环境区间内可得，部分环境可得的走可选依赖，不静默变硬依赖。
- 数据文件、配置模板进包必须经 `package_data`；不提交构建产物或 generated staging。
- experimental 目录的进包条件是 `AMCT_EXPERIMENTAL=TRUE`，非默认路径；新特性默认落 experimental 时按 coding-rules 的放置位置规则执行。
- 打包变化跑 `tests/unit_test/test_packaging.py`。

### 文档承诺

文档是承诺：`docs/` 与 examples 中写出的 API、CLI 参数、dtype 取值、bit policy 必须与源码和 parser 一致。

- 新增/变更公开行为时同步对应文档；确认示例命令在当前 parser 中存在。
- 配置口径（BF16、weight-only、W4A4、dtype）与 parser 和各 role 实际配置一致。
- 验证按 [validation.md](validation.md#按变更类型选验证) 的「docs/examples 新增 API/CLI 承诺」行：对应实现 owner 的单测 + 公共接口与打包检查；纯措辞改动只查链接与命令存在性，不声称覆盖实现行为。

## 禁止项

- 新 API 只在源模块定义，未进顶层 `__all__` 或未接懒加载分派。
- 懒加载 API 只验证了属性可达，未验证普通/通配符 import 与 `__all__` 一致。
- 既有 public API 签名、语义或默认值无声变化。
- 新依赖未同步安装清单，或部分环境依赖静默变硬依赖。
- 打包清单（package_data、experimental 开关）与实际产物不一致。
- 文档承诺了源码中不存在或行为不符的 API/CLI。

## 自验

- 测试定位与命令见 [validation.md](validation.md#场景测试索引)「公共接口与打包」行；按改动选 `test_public_api.py` / `test_packaging.py` / `test_llm_module_entrypoints.py`。
- 新增导出：五面一致性（源 `__all__`、顶层 `__all__`、普通 import、通配符 import、属性访问）逐项断言。
- 真实安装/多环境验证未执行时明确说明，不把单测通过表述为安装包已验证。
