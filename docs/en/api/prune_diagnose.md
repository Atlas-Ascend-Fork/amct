# prune\_diagnose<a name="ZH-CN_TOPIC_0000002600000501"></a>

<!-- md-trans-meta sourceCommit=4a47814eed18048e444f9a2acc42d9864ddc477f translatedAt=2026-10-16T08:28:48.548Z pushedAt=2026-11-05T01:12:03.418Z nl=2 -->

## Supported Products<a name="zh-cn_topic_0000002600000501_section105000010001"></a>

<a name="zh-cn_topic_0000002600000501_table105000010002"></a>

| Product | Supported |
| ------------------------------------------- | -------- |
| Ascend 950PR&950DT products | √ |
| Atlas A3 products | √ |
| Atlas A2 products | √ |

## Description<a name="zh-cn_topic_0000002600000501_section105000010003"></a>

Pruning feasibility diagnosis: identifies prunable targets in each domain (**cnn**/**dense**/**moe**), performs fixed-rate pruning **dry-run** and **acc** binary search dry-run, and returns a diagnosis report. The entire diagnosis is performed on a model copy (dry-run) without modifying the passed-in original model.

## Prototype<a name="zh-cn_topic_0000002600000501_section105000010004"></a>

```python
prune_diagnose(model, data=None, config=None, prune_ratio=0.5, tolerance=0.05)
```

## Parameters<a name="zh-cn_topic_0000002600000501_section105000010005"></a>

<a name="zh-cn_topic_0000002600000501_table105000010006"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002600000501_row105000010001"><th class="cellrowborder" valign="top" width="13.78%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000002600000501_p105000010001"><a name="zh-cn_topic_0000002600000501_p105000010001"></a><a name="zh-cn_topic_0000002600000501_p105000010001"></a>Parameter</p></th>
<th class="cellrowborder" valign="top" width="8.01%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000002600000501_p105000010002"><a name="zh-cn_topic_0000002600000501_p105000010002"></a><a name="zh-cn_topic_0000002600000501_p105000010002"></a>Input/Output</p></th>
<th class="cellrowborder" valign="top" width="78.21%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000002600000501_p105000010003"><a name="zh-cn_topic_0000002600000501_p105000010003"></a><a name="zh-cn_topic_0000002600000501_p105000010003"></a>Description</p></th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002600000501_row105000010002"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000501_p105000010004"><a name="zh-cn_topic_0000002600000501_p105000010004"></a><a name="zh-cn_topic_0000002600000501_p105000010004"></a>model</p></td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000501_p105000010005"><a name="zh-cn_topic_0000002600000501_p105000010005"></a><a name="zh-cn_topic_0000002600000501_p105000010005"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000501_p105000010006"><a name="zh-cn_topic_0000002600000501_p105000010006"></a><a name="zh-cn_topic_0000002600000501_p105000010006"></a>Meaning: Model to be diagnosed (dry-run is performed on a copy, and the original model remains unchanged).</p>
<p id="zh-cn_topic_0000002600000501_p105000010007"><a name="zh-cn_topic_0000002600000501_p105000010007"></a><a name="zh-cn_topic_0000002600000501_p105000010007"></a>Data type: torch.nn.Module.</p></td>
</tr>
<tr id="zh-cn_topic_0000002600000501_row105000010003"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000501_p105000010008"><a name="zh-cn_topic_0000002600000501_p105000010008"></a><a name="zh-cn_topic_0000002600000501_p105000010008"></a>data</p></td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000501_p105000010009"><a name="zh-cn_topic_0000002600000501_p105000010009"></a><a name="zh-cn_topic_0000002600000501_p105000010009"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000501_p105000010010"><a name="zh-cn_topic_0000002600000501_p105000010010"></a><a name="zh-cn_topic_0000002600000501_p105000010010"></a>Meaning: Calibration/evaluation data. Required by variance-based methods and acc binary search; when it is None, the corresponding checks are skipped.</p>
<p id="zh-cn_topic_0000002600000501_p105000010011"><a name="zh-cn_topic_0000002600000501_p105000010011"></a><a name="zh-cn_topic_0000002600000501_p105000010011"></a>Default value: None.</p>
<p id="zh-cn_topic_0000002600000501_p105000010012"><a name="zh-cn_topic_0000002600000501_p105000010012"></a><a name="zh-cn_topic_0000002600000501_p105000010012"></a>Data type: Tensor, list, DataLoader, or other iterable objects, or None.</p></td>
</tr>
<tr id="zh-cn_topic_0000002600000501_row105000010004"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000501_p105000010013"><a name="zh-cn_topic_0000002600000501_p105000010013"></a><a name="zh-cn_topic_0000002600000501_p105000010013"></a>config</p></td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000501_p105000010014"><a name="zh-cn_topic_0000002600000501_p105000010014"></a><a name="zh-cn_topic_0000002600000501_p105000010014"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000501_p105000010015"><a name="zh-cn_topic_0000002600000501_p105000010015"></a><a name="zh-cn_topic_0000002600000501_p105000010015"></a>Meaning: Pruning configuration. When it is None, the default methods for the three domains (cnn/dense/moe) are used.</p>
<p id="zh-cn_topic_0000002600000501_p105000010016"><a name="zh-cn_topic_0000002600000501_p105000010016"></a><a name="zh-cn_topic_0000002600000501_p105000010016"></a>Default value: None.</p>
<p id="zh-cn_topic_0000002600000501_p105000010017"><a name="zh-cn_topic_0000002600000501_p105000010017"></a><a name="zh-cn_topic_0000002600000501_p105000010017"></a>Data type: dict, PruneConfig, or None.</p></td>
</tr>
<tr id="zh-cn_topic_0000002600000501_row105000010005"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000501_p105000010018"><a name="zh-cn_topic_0000002600000501_p105000010018"></a><a name="zh-cn_topic_0000002600000501_p105000010018"></a>prune_ratio</p></td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000501_p105000010019"><a name="zh-cn_topic_0000002600000501_p105000010019"></a><a name="zh-cn_topic_0000002600000501_p105000010019"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000501_p105000010020"><a name="zh-cn_topic_0000002600000501_p105000010020"></a><a name="zh-cn_topic_0000002600000501_p105000010020"></a>Meaning: Pruning ratio used by the fixed-rate pruning dry-run.</p>
<p id="zh-cn_topic_0000002600000501_p105000010021"><a name="zh-cn_topic_0000002600000501_p105000010021"></a><a name="zh-cn_topic_0000002600000501_p105000010021"></a>Default value: 0.5.</p>
<p id="zh-cn_topic_0000002600000501_p105000010022"><a name="zh-cn_topic_0000002600000501_p105000010022"></a><a name="zh-cn_topic_0000002600000501_p105000010022"></a>Data type: float.</p></td>
</tr>
<tr id="zh-cn_topic_0000002600000501_row105000010006"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000501_p105000010023"><a name="zh-cn_topic_0000002600000501_p105000010023"></a><a name="zh-cn_topic_0000002600000501_p105000010023"></a>tolerance</p></td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000501_p105000010024"><a name="zh-cn_topic_0000002600000501_p105000010024"></a><a name="zh-cn_topic_0000002600000501_p105000010024"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000501_p105000010025"><a name="zh-cn_topic_0000002600000501_p105000010025"></a><a name="zh-cn_topic_0000002600000501_p105000010025"></a>Meaning: Upper bound of acceptable accuracy loss used by the acc binary search dry-run.</p>
<p id="zh-cn_topic_0000002600000501_p105000010026"><a name="zh-cn_topic_0000002600000501_p105000010026"></a><a name="zh-cn_topic_0000002600000501_p105000010026"></a>Default value: 0.05.</p>
<p id="zh-cn_topic_0000002600000501_p105000010027"><a name="zh-cn_topic_0000002600000501_p105000010027"></a><a name="zh-cn_topic_0000002600000501_p105000010027"></a>Data type: float.</p></td>
</tr>
</tbody>
</table>

## Return Value<a name="zh-cn_topic_0000002600000501_section105000010007"></a>

Returns a **DiagnosisReport** (dataclass) with the following main fields:

- **targets**: Number of prunable targets in each domain (dict, keyed by cnn/dense/moe).
- **prune_works**: Whether fixed-rate pruning actually takes effect (whether the parameter count decreases, bool).
- **prune_reduction**: Measured weight reduction ratio (float).
- **prune_forward_ok**: Whether the forward pass is normal after pruning (bool; None when no samples are available).
- **search_works**: Whether the acc binary search runs through and selects a pruning ratio (bool).
- **search_chosen_ratio**: Pruning ratio selected by the search (float; None if not selected).
- **notes**: Hints/warnings collected during diagnosis (list).

You can call `report.summary()` to print a human-readable summary, or use `report.any_domain_detected` to determine whether any prunable domain is detected.

## Example<a name="zh-cn_topic_0000002600000501_section105000010008"></a>

```python
import amct_pytorch as amct
from amct_pytorch.pruning import prune_diagnose

# Perform pruning feasibility diagnosis (dry-run, without modifying the original model) on the user-instantiated model and calibration data.
report = prune_diagnose(model, data=calib, prune_ratio=0.5, tolerance=0.05)
print(report.summary())
```
