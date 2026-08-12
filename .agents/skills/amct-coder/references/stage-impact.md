# LLM 阶段影响检查

只在修改 LLM CLI/workflow、阶段产物或跨阶段契约时读取。阶段描述数据流影响，不决定代码职责归属；classic、`amct_ops` 和纯 public API/packaging 任务标记为「不适用」，不必逐阶段核对。

| 阶段 | 主要检查 |
| --- | --- |
| eval | 输入、PPL 口径、bit policy、adapter forward |
| extract_ptq_data | hook、离线输入命名、kwargs、`PtqUnit.kind` |
| ptq | unit 输入/GT、算法参数、solver、参数保存/回载 |
| deploy | PTQ 参数加载、deploy binding、dtype/algo payload、safetensors/config |

判断受影响阶段时按实际写入和消费点判断，不按关键词。同一改动可能横跨多个阶段：dtype 与 quant module 影响 eval/ptq/deploy，算法只有产生可持久化参数时才影响 deploy，离线数据格式同时约束 extract 保存端和 ptq 加载端。

跨阶段改动按数据流顺序处理：先冻结上游产物契约，再改下游消费方。上游契约不清时回到提供契约的层确认，不要在下游写兜底分支。
