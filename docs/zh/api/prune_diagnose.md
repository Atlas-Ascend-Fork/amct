# prune\_diagnose<a name="ZH-CN_TOPIC_0000002600000501"></a>

## 产品支持情况<a name="zh-cn_topic_0000002600000501_section105000010001"></a>

<a name="zh-cn_topic_0000002600000501_table105000010002"></a>

| 产品                                        | 是否支持 |
| ------------------------------------------- | -------- |
| Ascend 950PR/Ascend 950DT                   | √        |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √        |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √        |



## 功能说明<a name="zh-cn_topic_0000002600000501_section105000010003"></a>

剪枝可行性诊断：识别各域（cnn/dense/moe）可剪目标、做固定率剪枝 dry-run、做 acc 二分搜索 dry-run，返回诊断报告。诊断全程在模型副本上进行（dry-run），不修改传入的原模型。

## 函数原型<a name="zh-cn_topic_0000002600000501_section105000010004"></a>

```python
prune_diagnose(model, data=None, config=None, prune_ratio=0.5, tolerance=0.05)
```

## 参数说明<a name="zh-cn_topic_0000002600000501_section105000010005"></a>

<a name="zh-cn_topic_0000002600000501_table105000010006"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002600000501_row105000010001"><th class="cellrowborder" valign="top" width="13.78%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000002600000501_p105000010001"><a name="zh-cn_topic_0000002600000501_p105000010001"></a><a name="zh-cn_topic_0000002600000501_p105000010001"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="8.01%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000002600000501_p105000010002"><a name="zh-cn_topic_0000002600000501_p105000010002"></a><a name="zh-cn_topic_0000002600000501_p105000010002"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="78.21%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000002600000501_p105000010003"><a name="zh-cn_topic_0000002600000501_p105000010003"></a><a name="zh-cn_topic_0000002600000501_p105000010003"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002600000501_row105000010002"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000501_p105000010004"><a name="zh-cn_topic_0000002600000501_p105000010004"></a><a name="zh-cn_topic_0000002600000501_p105000010004"></a>model</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000501_p105000010005"><a name="zh-cn_topic_0000002600000501_p105000010005"></a><a name="zh-cn_topic_0000002600000501_p105000010005"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000501_p105000010006"><a name="zh-cn_topic_0000002600000501_p105000010006"></a><a name="zh-cn_topic_0000002600000501_p105000010006"></a>含义：待诊断的模型（在副本上 dry-run，原模型不变）。</p>
<p id="zh-cn_topic_0000002600000501_p105000010007"><a name="zh-cn_topic_0000002600000501_p105000010007"></a><a name="zh-cn_topic_0000002600000501_p105000010007"></a>数据类型：torch.nn.Module。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002600000501_row105000010003"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000501_p105000010008"><a name="zh-cn_topic_0000002600000501_p105000010008"></a><a name="zh-cn_topic_0000002600000501_p105000010008"></a>data</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000501_p105000010009"><a name="zh-cn_topic_0000002600000501_p105000010009"></a><a name="zh-cn_topic_0000002600000501_p105000010009"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000501_p105000010010"><a name="zh-cn_topic_0000002600000501_p105000010010"></a><a name="zh-cn_topic_0000002600000501_p105000010010"></a>含义：校准/评估数据；方差类方法与 acc 二分搜索需要，为 None 时跳过对应检查。</p>
<p id="zh-cn_topic_0000002600000501_p105000010011"><a name="zh-cn_topic_0000002600000501_p105000010011"></a><a name="zh-cn_topic_0000002600000501_p105000010011"></a>默认值：None。</p>
<p id="zh-cn_topic_0000002600000501_p105000010012"><a name="zh-cn_topic_0000002600000501_p105000010012"></a><a name="zh-cn_topic_0000002600000501_p105000010012"></a>数据类型：Tensor、list、DataLoader 等可迭代对象，或 None。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002600000501_row105000010004"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000501_p105000010013"><a name="zh-cn_topic_0000002600000501_p105000010013"></a><a name="zh-cn_topic_0000002600000501_p105000010013"></a>config</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000501_p105000010014"><a name="zh-cn_topic_0000002600000501_p105000010014"></a><a name="zh-cn_topic_0000002600000501_p105000010014"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000501_p105000010015"><a name="zh-cn_topic_0000002600000501_p105000010015"></a><a name="zh-cn_topic_0000002600000501_p105000010015"></a>含义：剪枝配置；None 时使用 cnn/dense/moe 三域默认方法。</p>
<p id="zh-cn_topic_0000002600000501_p105000010016"><a name="zh-cn_topic_0000002600000501_p105000010016"></a><a name="zh-cn_topic_0000002600000501_p105000010016"></a>默认值：None。</p>
<p id="zh-cn_topic_0000002600000501_p105000010017"><a name="zh-cn_topic_0000002600000501_p105000010017"></a><a name="zh-cn_topic_0000002600000501_p105000010017"></a>数据类型：dict、PruneConfig 或 None。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002600000501_row105000010005"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000501_p105000010018"><a name="zh-cn_topic_0000002600000501_p105000010018"></a><a name="zh-cn_topic_0000002600000501_p105000010018"></a>prune_ratio</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000501_p105000010019"><a name="zh-cn_topic_0000002600000501_p105000010019"></a><a name="zh-cn_topic_0000002600000501_p105000010019"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000501_p105000010020"><a name="zh-cn_topic_0000002600000501_p105000010020"></a><a name="zh-cn_topic_0000002600000501_p105000010020"></a>含义：固定率剪枝 dry-run 所用的剪枝率。</p>
<p id="zh-cn_topic_0000002600000501_p105000010021"><a name="zh-cn_topic_0000002600000501_p105000010021"></a><a name="zh-cn_topic_0000002600000501_p105000010021"></a>默认值：0.5。</p>
<p id="zh-cn_topic_0000002600000501_p105000010022"><a name="zh-cn_topic_0000002600000501_p105000010022"></a><a name="zh-cn_topic_0000002600000501_p105000010022"></a>数据类型：float。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002600000501_row105000010006"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000501_p105000010023"><a name="zh-cn_topic_0000002600000501_p105000010023"></a><a name="zh-cn_topic_0000002600000501_p105000010023"></a>tolerance</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000501_p105000010024"><a name="zh-cn_topic_0000002600000501_p105000010024"></a><a name="zh-cn_topic_0000002600000501_p105000010024"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000501_p105000010025"><a name="zh-cn_topic_0000002600000501_p105000010025"></a><a name="zh-cn_topic_0000002600000501_p105000010025"></a>含义：acc 二分搜索 dry-run 所用的可接受精度损失上界。</p>
<p id="zh-cn_topic_0000002600000501_p105000010026"><a name="zh-cn_topic_0000002600000501_p105000010026"></a><a name="zh-cn_topic_0000002600000501_p105000010026"></a>默认值：0.05。</p>
<p id="zh-cn_topic_0000002600000501_p105000010027"><a name="zh-cn_topic_0000002600000501_p105000010027"></a><a name="zh-cn_topic_0000002600000501_p105000010027"></a>数据类型：float。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000002600000501_section105000010007"></a>

返回 DiagnosisReport（dataclass），主要字段：
- targets：各域可剪目标数量（dict，键为 cnn/dense/moe）。
- prune_works：固定率剪枝是否实际生效（参数量是否下降，bool）。
- prune_reduction：实测权重削减比例（float）。
- prune_forward_ok：剪枝后前向是否正常（bool；无样本时为 None）。
- search_works：acc 二分搜索是否跑通并选出剪枝率（bool）。
- search_chosen_ratio：搜索选出的剪枝率（float；未选出为 None）。
- notes：诊断过程中的提示/告警信息（list）。

可调用 `report.summary()` 打印人类可读摘要，或用 `report.any_domain_detected` 判断是否检测到任一可剪域。

## 调用示例<a name="zh-cn_topic_0000002600000501_section105000010008"></a>

```python
import amct_pytorch as amct
from amct_pytorch.pruning import prune_diagnose

# 在用户已实例化的模型 + 校准数据上做剪枝可行性诊断（dry-run，不改原模型）
report = prune_diagnose(model, data=calib, prune_ratio=0.5, tolerance=0.05)
print(report.summary())
```
