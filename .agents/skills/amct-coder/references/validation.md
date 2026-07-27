# AMCT Coder 验证规则

## 通用

所有编码任务结束前必须说明：

- 命中的子能力。
- 实际改动文件。
- 是否触及敏感层：workflow、solver、quant module、public API、ops build。
- 已跑验证。
- 未跑验证及阻塞原因。

缺 NPU、CANN、模型权重、外网数据集、protobuf 生成物时，不要声称已完成对应验证。

## 最小静态验证

- Python 语法：`python -m py_compile <files>`。
- ruff/pre-commit 可用时：`pre-commit run --files <changed files>`。
- C/C++：遵循 `.clang-format`，行宽 120，4 空格缩进。
- public API：跑静态 API 测试。

## 场景测试索引

| 子能力 | 推荐最小测试 |
| --- | --- |
| `dtype-quant-module` | `pytest tests/unit_test/quantization/test_dtypes.py tests/unit_test/quantization/modules/test_quant_base.py` |
| `llm-algorithm` | `pytest tests/unit_test/algorithms tests/unit_test/quantization/modules/test_quant_base.py` |
| `llm-model-unit` | `pytest tests/unit_test/common/models/llm`，必要时 mocked workflow 测试 |
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
