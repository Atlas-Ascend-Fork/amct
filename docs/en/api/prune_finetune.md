# prune\_finetune<a name="ZH-CN_TOPIC_0000002600000401"></a>

<!-- md-trans-meta sourceCommit=4a47814eed18048e444f9a2acc42d9864ddc477f translatedAt=2026-10-16T08:29:41.602Z pushedAt=2026-11-05T01:12:03.419Z nl=2 -->

## Supported Products<a name="zh-cn_topic_0000002600000401_section104000010001"></a>

<a name="zh-cn_topic_0000002600000401_table104000010002"></a>

| Product | Supported |
| ------------------------------------------- | -------- |
| Ascend 950PR&950DT products | √ |
| Atlas A3 products | √ |
| Atlas A2 products | √ |

## Description<a name="zh-cn_topic_0000002600000401_section104000010003"></a>

Performs a small number of in-place gradient steps to fine-tune the model and recover the accuracy lost by structured pruning. It can be called directly, or wrapped as `finetune_fn=lambda m: prune_finetune(m, train_data, ...)` and passed to `amct.prune` / the automatic pruning API.

## Prototype<a name="zh-cn_topic_0000002600000401_section104000010004"></a>

```python
prune_finetune(model, data, loss_fn=None, steps=200, lr=1e-4, weight_decay=0.0, warmup=0, optimizer=None, batch_adapter=None, device=None, grad_clip=1.0, log_every=0)
```

## Parameters<a name="zh-cn_topic_0000002600000401_section104000010005"></a>

<a name="zh-cn_topic_0000002600000401_table104000010006"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002600000401_row104000010001"><th class="cellrowborder" valign="top" width="13.78%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000002600000401_p104000010001"><a name="zh-cn_topic_0000002600000401_p104000010001"></a><a name="zh-cn_topic_0000002600000401_p104000010001"></a>Parameter</p></th>
<th class="cellrowborder" valign="top" width="8.01%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000002600000401_p104000010002"><a name="zh-cn_topic_0000002600000401_p104000010002"></a><a name="zh-cn_topic_0000002600000401_p104000010002"></a>Input/Output</p></th>
<th class="cellrowborder" valign="top" width="78.21%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000002600000401_p104000010003"><a name="zh-cn_topic_0000002600000401_p104000010003"></a><a name="zh-cn_topic_0000002600000401_p104000010003"></a>Description</p></th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002600000401_row104000010002"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000401_p104000010004"><a name="zh-cn_topic_0000002600000401_p104000010004"></a><a name="zh-cn_topic_0000002600000401_p104000010004"></a>model</p></td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000401_p104000010005"><a name="zh-cn_topic_0000002600000401_p104000010005"></a><a name="zh-cn_topic_0000002600000401_p104000010005"></a>Input/Output</p></td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000401_p104000010006"><a name="zh-cn_topic_0000002600000401_p104000010006"></a><a name="zh-cn_topic_0000002600000401_p104000010006"></a>Meaning: Model to be fine-tuned, with weights updated in-place.</p>
<p id="zh-cn_topic_0000002600000401_p104000010007"><a name="zh-cn_topic_0000002600000401_p104000010007"></a><a name="zh-cn_topic_0000002600000401_p104000010007"></a>Data type: torch.nn.Module.</p></td>
</tr>
<tr id="zh-cn_topic_0000002600000401_row104000010003"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000401_p104000010008"><a name="zh-cn_topic_0000002600000401_p104000010008"></a><a name="zh-cn_topic_0000002600000401_p104000010008"></a>data</p></td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000401_p104000010009"><a name="zh-cn_topic_0000002600000401_p104000010009"></a><a name="zh-cn_topic_0000002600000401_p104000010009"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000401_p104000010010"><a name="zh-cn_topic_0000002600000401_p104000010010"></a><a name="zh-cn_topic_0000002600000401_p104000010010"></a>Meaning: Training batch sequence, reused cyclically until the number of steps specified by <code>steps</code> is reached.</p>
<p id="zh-cn_topic_0000002600000401_p104000010011"><a name="zh-cn_topic_0000002600000401_p104000010011"></a><a name="zh-cn_topic_0000002600000401_p104000010011"></a>Data type: Sequence[Any].</p></td>
</tr>
<tr id="zh-cn_topic_0000002600000401_row104000010004"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000401_p104000010012"><a name="zh-cn_topic_0000002600000401_p104000010012"></a><a name="zh-cn_topic_0000002600000401_p104000010012"></a>loss_fn</p></td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000401_p104000010013"><a name="zh-cn_topic_0000002600000401_p104000010013"></a><a name="zh-cn_topic_0000002600000401_p104000010013"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000401_p104000010014"><a name="zh-cn_topic_0000002600000401_p104000010014"></a><a name="zh-cn_topic_0000002600000401_p104000010014"></a>Meaning: Custom loss function <code>loss_fn(model, batch) -> Tensor</code>; when None, the built-in default loss is used (a dict containing <code>input_ids</code> uses the causal LM self-supervised loss, and an <code>(x, y)</code> tuple uses the cross-entropy classification loss).</p>
<p id="zh-cn_topic_0000002600000401_p104000010015"><a name="zh-cn_topic_0000002600000401_p104000010015"></a><a name="zh-cn_topic_0000002600000401_p104000010015"></a>Default value: None.</p>
<p id="zh-cn_topic_0000002600000401_p104000010016"><a name="zh-cn_topic_0000002600000401_p104000010016"></a><a name="zh-cn_topic_0000002600000401_p104000010016"></a>Data type: Callable[[nn.Module, Any], torch.Tensor] or None.</p></td>
</tr>
<tr id="zh-cn_topic_0000002600000401_row104000010005"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000401_p104000010017"><a name="zh-cn_topic_0000002600000401_p104000010017"></a><a name="zh-cn_topic_0000002600000401_p104000010017"></a>steps</p></td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000401_p104000010018"><a name="zh-cn_topic_0000002600000401_p104000010018"></a><a name="zh-cn_topic_0000002600000401_p104000010018"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000401_p104000010019"><a name="zh-cn_topic_0000002600000401_p104000010019"></a><a name="zh-cn_topic_0000002600000401_p104000010019"></a>Meaning: Number of gradient update steps; when ≤0, returns an empty result directly without training.</p>
<p id="zh-cn_topic_0000002600000401_p104000010020"><a name="zh-cn_topic_0000002600000401_p104000010020"></a><a name="zh-cn_topic_0000002600000401_p104000010020"></a>Default value: 200.</p>
<p id="zh-cn_topic_0000002600000401_p104000010021"><a name="zh-cn_topic_0000002600000401_p104000010021"></a><a name="zh-cn_topic_0000002600000401_p104000010021"></a>Data type: int.</p></td>
</tr>
<tr id="zh-cn_topic_0000002600000401_row104000010006"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000401_p104000010022"><a name="zh-cn_topic_0000002600000401_p104000010022"></a><a name="zh-cn_topic_0000002600000401_p104000010022"></a>lr</p></td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000401_p104000010023"><a name="zh-cn_topic_0000002600000401_p104000010023"></a><a name="zh-cn_topic_0000002600000401_p104000010023"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000401_p104000010024"><a name="zh-cn_topic_0000002600000401_p104000010024"></a><a name="zh-cn_topic_0000002600000401_p104000010024"></a>Meaning: Learning rate of the default AdamW optimizer.</p>
<p id="zh-cn_topic_0000002600000401_p104000010025"><a name="zh-cn_topic_0000002600000401_p104000010025"></a><a name="zh-cn_topic_0000002600000401_p104000010025"></a>Default value: 1e-4.</p>
<p id="zh-cn_topic_0000002600000401_p104000010026"><a name="zh-cn_topic_0000002600000401_p104000010026"></a><a name="zh-cn_topic_0000002600000401_p104000010026"></a>Data type: float.</p></td>
</tr>
<tr id="zh-cn_topic_0000002600000401_row104000010007"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000401_p104000010027"><a name="zh-cn_topic_0000002600000401_p104000010027"></a><a name="zh-cn_topic_0000002600000401_p104000010027"></a>weight_decay</p></td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000401_p104000010028"><a name="zh-cn_topic_0000002600000401_p104000010028"></a><a name="zh-cn_topic_0000002600000401_p104000010028"></a>input</p></td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000401_p104000010029"><a name="zh-cn_topic_0000002600000401_p104000010029"></a><a name="zh-cn_topic_0000002600000401_p104000010029"></a>Meaning: weight decay of the default AdamW optimizer.</p>
<p id="zh-cn_topic_0000002600000401_p104000010030"><a name="zh-cn_topic_0000002600000401_p104000010030"></a><a name="zh-cn_topic_0000002600000401_p104000010030"></a>Default value: 0.0.</p>
<p id="zh-cn_topic_0000002600000401_p104000010031"><a name="zh-cn_topic_0000002600000401_p104000010031"></a><a name="zh-cn_topic_0000002600000401_p104000010031"></a>Data type: float.</p></td>
</tr>
<tr id="zh-cn_topic_0000002600000401_row104000010008"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000401_p104000010032"><a name="zh-cn_topic_0000002600000401_p104000010032"></a><a name="zh-cn_topic_0000002600000401_p104000010032"></a>warmup</p></td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000401_p104000010033"><a name="zh-cn_topic_0000002600000401_p104000010033"></a><a name="zh-cn_topic_0000002600000401_p104000010033"></a>input</p></td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000401_p104000010034"><a name="zh-cn_topic_0000002600000401_p104000010034"></a><a name="zh-cn_topic_0000002600000401_p104000010034"></a>Meaning: number of linear warmup steps. Within the first <code>warmup</code> steps, the learning rate increases linearly from 0 to <code>lr</code>.</p>
<p id="zh-cn_topic_0000002600000401_p104000010035"><a name="zh-cn_topic_0000002600000401_p104000010035"></a><a name="zh-cn_topic_0000002600000401_p104000010035"></a>Default value: 0.</p>
<p id="zh-cn_topic_0000002600000401_p104000010036"><a name="zh-cn_topic_0000002600000401_p104000010036"></a><a name="zh-cn_topic_0000002600000401_p104000010036"></a>Data type: int.</p></td>
</tr>
<tr id="zh-cn_topic_0000002600000401_row104000010009"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000401_p104000010037"><a name="zh-cn_topic_0000002600000401_p104000010037"></a><a name="zh-cn_topic_0000002600000401_p104000010037"></a>optimizer</p></td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000401_p104000010038"><a name="zh-cn_topic_0000002600000401_p104000010038"></a><a name="zh-cn_topic_0000002600000401_p104000010038"></a>input</p></td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000401_p104000010039"><a name="zh-cn_topic_0000002600000401_p104000010039"></a><a name="zh-cn_topic_0000002600000401_p104000010039"></a>Meaning: custom torch optimizer. When it is None, AdamW is automatically built based on the trainable parameters.</p>
<p id="zh-cn_topic_0000002600000401_p104000010040"><a name="zh-cn_topic_0000002600000401_p104000010040"></a><a name="zh-cn_topic_0000002600000401_p104000010040"></a>Default value: None.</p>
<p id="zh-cn_topic_0000002600000401_p104000010041"><a name="zh-cn_topic_0000002600000401_p104000010041"></a><a name="zh-cn_topic_0000002600000401_p104000010041"></a>Data type: torch.optim.Optimizer or None.</p></td>
</tr>
<tr id="zh-cn_topic_0000002600000401_row104000010010"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000401_p104000010042"><a name="zh-cn_topic_0000002600000401_p104000010042"></a><a name="zh-cn_topic_0000002600000401_p104000010042"></a>batch_adapter</p></td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000401_p104000010043"><a name="zh-cn_topic_0000002600000401_p104000010043"></a><a name="zh-cn_topic_0000002600000401_p104000010043"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000401_p104000010044"><a name="zh-cn_topic_0000002600000401_p104000010044"></a><a name="zh-cn_topic_0000002600000401_p104000010044"></a>Meaning: Batch adapter that maps a batch to <code>(args, kwargs)</code> for model forward pass; the default loss uses its output <code>.loss</code>.</p>
<p id="zh-cn_topic_0000002600000401_p104000010045"><a name="zh-cn_topic_0000002600000401_p104000010045"></a><a name="zh-cn_topic_0000002600000401_p104000010045"></a>Default value: None.</p>
<p id="zh-cn_topic_0000002600000401_p104000010046"><a name="zh-cn_topic_0000002600000401_p104000010046"></a><a name="zh-cn_topic_0000002600000401_p104000010046"></a>Data type: BatchAdapter or None.</p></td>
</tr>
<tr id="zh-cn_topic_0000002600000401_row104000010011"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000401_p104000010047"><a name="zh-cn_topic_0000002600000401_p104000010047"></a><a name="zh-cn_topic_0000002600000401_p104000010047"></a>device</p></td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000401_p104000010048"><a name="zh-cn_topic_0000002600000401_p104000010048"></a><a name="zh-cn_topic_0000002600000401_p104000010048"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000401_p104000010049"><a name="zh-cn_topic_0000002600000401_p104000010049"></a><a name="zh-cn_topic_0000002600000401_p104000010049"></a>Meaning: Target device; when None, it is inferred automatically from the model.</p>
<p id="zh-cn_topic_0000002600000401_p104000010050"><a name="zh-cn_topic_0000002600000401_p104000010050"></a><a name="zh-cn_topic_0000002600000401_p104000010050"></a>Default value: None.</p>
<p id="zh-cn_topic_0000002600000401_p104000010051"><a name="zh-cn_topic_0000002600000401_p104000010051"></a><a name="zh-cn_topic_0000002600000401_p104000010051"></a>Data type: torch.device / str or None.</p></td>
</tr>
<tr id="zh-cn_topic_0000002600000401_row104000010012"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000401_p104000010052"><a name="zh-cn_topic_0000002600000401_p104000010052"></a><a name="zh-cn_topic_0000002600000401_p104000010052"></a>grad_clip</p></td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000401_p104000010053"><a name="zh-cn_topic_0000002600000401_p104000010053"></a><a name="zh-cn_topic_0000002600000401_p104000010053"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000401_p104000010054"><a name="zh-cn_topic_0000002600000401_p104000010054"></a><a name="zh-cn_topic_0000002600000401_p104000010054"></a>Meaning: Maximum norm for gradient clipping; when falsy (None/0), clipping is disabled.</p>
<p id="zh-cn_topic_0000002600000401_p104000010055"><a name="zh-cn_topic_0000002600000401_p104000010055"></a><a name="zh-cn_topic_0000002600000401_p104000010055"></a>Default value: 1.0.</p>
<p id="zh-cn_topic_0000002600000401_p104000010056"><a name="zh-cn_topic_0000002600000401_p104000010056"></a><a name="zh-cn_topic_0000002600000401_p104000010056"></a>Data type: float or None.</p></td>
</tr>
<tr id="zh-cn_topic_0000002600000401_row104000010013"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000401_p104000010057"><a name="zh-cn_topic_0000002600000401_p104000010057"></a><a name="zh-cn_topic_0000002600000401_p104000010057"></a>log_every</p></td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000401_p104000010058"><a name="zh-cn_topic_0000002600000401_p104000010058"></a><a name="zh-cn_topic_0000002600000401_p104000010058"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000401_p104000010059"><a name="zh-cn_topic_0000002600000401_p104000010059"></a><a name="zh-cn_topic_0000002600000401_p104000010059"></a>Meaning: When > 0, downsamples the returned <code>loss_history</code> (the loss of each step is always recorded).</p>
<p id="zh-cn_topic_0000002600000401_p104000010060"><a name="zh-cn_topic_0000002600000401_p104000010060"></a><a name="zh-cn_topic_0000002600000401_p104000010060"></a>Default value: 0.</p>
<p id="zh-cn_topic_0000002600000401_p104000010061"><a name="zh-cn_topic_0000002600000401_p104000010061"></a><a name="zh-cn_topic_0000002600000401_p104000010061"></a>Data type: int.</p></td>
</tr>
</tbody>
</table>

## Return Value<a name="zh-cn_topic_0000002600000401_section104000010007"></a>

Returns a **dict** containing the following keys:

- `steps`: Number of gradient steps actually executed (int).
- `initial_loss`: Loss of the first step, or **None** when there are no steps.
- `final_loss`: Loss of the last step, or **None** when there are no steps.
- `loss_history`: List of per-step losses (a downsampled list when `log_every > 0`).

Model weights are updated in-place, so there is no need to receive the returned model object.

## Example<a name="zh-cn_topic_0000002600000401_section104000010008"></a>

```python
import amct_pytorch as amct
from amct_pytorch.pruning import prune_finetune

# model is the pruned torch.nn.Module of the user; calib is the calibration/training batch sequence.
result = prune_finetune(model, calib, steps=300, lr=1e-5, warmup=20)
print(result["initial_loss"], "->", result["final_loss"])
```
