# LLM 阶段影响检查

只在修改 LLM CLI/workflow、阶段产物或跨阶段契约时读取。阶段描述数据流影响，不决定代码职责归属；classic、`amct_ops` 和纯 public API/packaging 任务标记为「不适用」，不必逐阶段核对。

| 阶段 | 入口与输入 | 写入或状态变化 | 下游消费者 |
| --- | --- | --- | --- |
| eval | eval CLI、模型输入、bit policy、adapter forward | PPL/日志；quant 模式重建并回载量化状态 | 精度比较与算法判读 |
| extract_ptq_data | extract CLI、校准样本、adapter hook | unit 输入、per-sample kwargs、`PtqUnit.kind` 对应离线文件 | PTQ provider |
| ptq | PTQ CLI、quant block/unit、离线输入/kwargs、GT | algorithm 状态、优化权重、`layer_<idx>_<unit.save_name>.pt` | quant eval、deploy 参数回载 |
| deploy | deploy CLI、PTQ 参数、deploy binding、dtype/algo payload | safetensors、index、`quantization_config` | 推理仓及真实部署运行时 |

判断受影响阶段时按实际写入和消费点判断，不按关键词。同一改动可能横跨多个阶段：dtype 与 quant module 影响 eval/ptq/deploy，算法只有产生可持久化参数时才影响 deploy，离线数据格式同时约束 extract 保存端和 ptq 加载端。

跨阶段改动按数据流顺序处理：先冻结上游产物契约，再改下游消费方。上游契约不清时回到提供契约的层确认，不要在下游写兜底分支。

## 强制阶段证据表

命中本手册时，计划、review 和最终说明都要为四个阶段逐项标记 `affected` 或 `not applicable`；每个 `affected` 阶段填写：

1. 活跃入口与 caller，不以文件、helper 或注册项存在代替。
2. 输入契约：CLI/config、对象结构、文件名、shape/dtype、quant/observe 状态。
3. 写入或副作用：buffer、权重、参数文件、safetensors/config；无写入也要明确。
4. 下游消费者及重建条件：target、dtype、bit policy、`--algos`、unit/save name。
5. 兼容与失败语义：旧产物、缺文件、strict load、部分结果和断点续跑。
6. 当前证据等级：源码存在、单测覆盖、集成闭环或真实环境验证。

只要上游写入和下游读取的字段、命名、状态或结构任一项未对齐，就不能声称跨阶段闭环。算法“没有新参数”不自动等于 deploy 不受影响：若它原地修改权重、改变 dtype payload 或要求自定义 real quant hook，仍需标记 PTQ/deploy 为 `affected`。反之，只有可持久化状态或部署数值语义真实变化时才扩展 deploy，不按算法关键词盲改。
