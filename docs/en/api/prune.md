# prune<a name="ZH-CN_TOPIC_0000002600000201"></a>

<!-- md-trans-meta sourceCommit=4a47814eed18048e444f9a2acc42d9864ddc477f translatedAt=2026-10-16T08:30:48.542Z pushedAt=2026-11-05T01:12:03.420Z nl=2 -->

## Supported Products<a name="zh-cn_topic_0000002600000201_section102000010001"></a>

<a name="zh-cn_topic_0000002600000201_table102000010002"></a>

| Product | Supported |
| ------------------------------------------- | -------- |
| Ascend 950PR&950DT products | √ |
| Atlas A3 products | √ |
| Atlas A2 products | √ |

## Description<a name="zh-cn_topic_0000002600000201_section102000010003"></a>

Main entry for structured pruning. It performs in-place pruning on **torch.nn.Module** (dense FFN intermediate dimension / CNN channel / MoE expert) and synchronously rewrites the model `config` dimensions (**intermediate_size** / **num_experts**). When **tolerance** is passed, it automatically searches for the maximum pruning rate based on the upper bound of accuracy loss. When **size_budget** is passed, it prunes according to the size budget. The two are mutually exclusive.

## Prototype<a name="zh-cn_topic_0000002600000201_section102000010004"></a>

```python
prune(model, config=None, data=None, batch_adapter=None, tolerance=None,
      evaluator=None, eval_iterations=1, eval_data=None, ratio_grid=None,
      size_budget=None, finetune_fn=None, quant_fn=None, report=None)
```

## Parameters<a name="zh-cn_topic_0000002600000201_section102000010005"></a>

<a name="zh-cn_topic_0000002600000201_table102000010006"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002600000201_row102000010001"><th class="cellrowborder" valign="top" width="13.78%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000002600000201_p102000010001"><a name="zh-cn_topic_0000002600000201_p102000010001"></a><a name="zh-cn_topic_0000002600000201_p102000010001"></a>Parameter</p></th>
<th class="cellrowborder" valign="top" width="8.01%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000002600000201_p102000010002"><a name="zh-cn_topic_0000002600000201_p102000010002"></a><a name="zh-cn_topic_0000002600000201_p102000010002"></a>Input/Output</p></th>
<th class="cellrowborder" valign="top" width="78.21%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000002600000201_p102000010003"><a name="zh-cn_topic_0000002600000201_p102000010003"></a><a name="zh-cn_topic_0000002600000201_p102000010003"></a>Description</p></th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002600000201_row102000010002"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000201_p102000010004"><a name="zh-cn_topic_0000002600000201_p102000010004"></a><a name="zh-cn_topic_0000002600000201_p102000010004"></a>model</p></td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000201_p102000010005"><a name="zh-cn_topic_0000002600000201_p102000010005"></a><a name="zh-cn_topic_0000002600000201_p102000010005"></a>Input/Output</p></td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000201_p102000010006"><a name="zh-cn_topic_0000002600000201_p102000010006"></a><a name="zh-cn_topic_0000002600000201_p102000010006"></a>Meaning: Model to be pruned. The caller is responsible for instantiating it (for example, via `from_pretrained`); pruning is performed in place.</p>
<p id="zh-cn_topic_0000002600000201_p102000010007"><a name="zh-cn_topic_0000002600000201_p102000010007"></a><a name="zh-cn_topic_0000002600000201_p102000010007"></a>Data type: torch.nn.Module.</p></td>
</tr>
<tr id="zh-cn_topic_0000002600000201_row102000010003"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000201_p102000010008"><a name="zh-cn_topic_0000002600000201_p102000010008"></a><a name="zh-cn_topic_0000002600000201_p102000010008"></a>config</p></td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000201_p102000010009"><a name="zh-cn_topic_0000002600000201_p102000010009"></a><a name="zh-cn_topic_0000002600000201_p102000010009"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000201_p102000010010"><a name="zh-cn_topic_0000002600000201_p102000010010"></a><a name="zh-cn_topic_0000002600000201_p102000010010"></a>Meaning: Pruning configuration that overrides the default pruning rate and method for each domain (cnn/dense/moe); when None, the default PruneConfig is used.</p>
<p id="zh-cn_topic_0000002600000201_p102000010011"><a name="zh-cn_topic_0000002600000201_p102000010011"></a><a name="zh-cn_topic_0000002600000201_p102000010011"></a>Default value: None.</p>
<p id="zh-cn_topic_0000002600000201_p102000010012"><a name="zh-cn_topic_0000002600000201_p102000010012"></a><a name="zh-cn_topic_0000002600000201_p102000010012"></a>Data type: dict / PruneConfig / None.</p></td>
</tr>
<tr id="zh-cn_topic_0000002600000201_row102000010004"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000201_p102000010013"><a name="zh-cn_topic_0000002600000201_p102000010013"></a><a name="zh-cn_topic_0000002600000201_p102000010013"></a>data</p></td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000201_p102000010014"><a name="zh-cn_topic_0000002600000201_p102000010014"></a><a name="zh-cn_topic_0000002600000201_p102000010014"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000201_p102000010015"><a name="zh-cn_topic_0000002600000201_p102000010015"></a><a name="zh-cn_topic_0000002600000201_p102000010015"></a>Meaning: Calibration data (DataLoader / tensor / dict), required by variance-based methods and also used as the default evaluation set.</p>
<p id="zh-cn_topic_0000002600000201_p102000010016"><a name="zh-cn_topic_0000002600000201_p102000010016"></a><a name="zh-cn_topic_0000002600000201_p102000010016"></a>Default value: None.</p>
<p id="zh-cn_topic_0000002600000201_p102000010017"><a name="zh-cn_topic_0000002600000201_p102000010017"></a><a name="zh-cn_topic_0000002600000201_p102000010017"></a>Data type: Any.</p></td>
</tr>
<tr id="zh-cn_topic_0000002600000201_row102000010005"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000201_p102000010018"><a name="zh-cn_topic_0000002600000201_p102000010018"></a><a name="zh-cn_topic_0000002600000201_p102000010018"></a>batch_adapter</p></td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000201_p102000010019"><a name="zh-cn_topic_0000002600000201_p102000010019"></a><a name="zh-cn_topic_0000002600000201_p102000010019"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000201_p102000010020"><a name="zh-cn_topic_0000002600000201_p102000010020"></a><a name="zh-cn_topic_0000002600000201_p102000010020"></a>Meaning: Callback that unpacks a custom batch into forward args and kwargs.</p>
<p id="zh-cn_topic_0000002600000201_p102000010021"><a name="zh-cn_topic_0000002600000201_p102000010021"></a><a name="zh-cn_topic_0000002600000201_p102000010021"></a>Default value: None.</p>
<p id="zh-cn_topic_0000002600000201_p102000010022"><a name="zh-cn_topic_0000002600000201_p102000010022"></a><a name="zh-cn_topic_0000002600000201_p102000010022"></a>Data type: Callable / None.</p></td>
</tr>
<tr id="zh-cn_topic_0000002600000201_row102000010006"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000201_p102000010023"><a name="zh-cn_topic_0000002600000201_p102000010023"></a><a name="zh-cn_topic_0000002600000201_p102000010023"></a>tolerance</p></td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000201_p102000010024"><a name="zh-cn_topic_0000002600000201_p102000010024"></a><a name="zh-cn_topic_0000002600000201_p102000010024"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000201_p102000010025"><a name="zh-cn_topic_0000002600000201_p102000010025"></a><a name="zh-cn_topic_0000002600000201_p102000010025"></a>Meaning: Acceptable upper bound of accuracy loss; when specified, enables accuracy-driven automatic pruning and searches ratio_grid for the maximum pruning rate within the tolerance.</p>
<p id="zh-cn_topic_0000002600000201_p102000010026"><a name="zh-cn_topic_0000002600000201_p102000010026"></a><a name="zh-cn_topic_0000002600000201_p102000010026"></a>Default value: None (fixed pruning rate mode).</p>
<p id="zh-cn_topic_0000002600000201_p102000010027"><a name="zh-cn_topic_0000002600000201_p102000010027"></a><a name="zh-cn_topic_0000002600000201_p102000010027"></a>Data type: float / None.</p></td>
</tr>
<tr id="zh-cn_topic_0000002600000201_row102000010007"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000201_p102000010028"><a name="zh-cn_topic_0000002600000201_p102000010028"></a><a name="zh-cn_topic_0000002600000201_p102000010028"></a>evaluator</p></td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000201_p102000010029"><a name="zh-cn_topic_0000002600000201_p102000010029"></a><a name="zh-cn_topic_0000002600000201_p102000010029"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000201_p102000010030"><a name="zh-cn_topic_0000002600000201_p102000010030"></a><a name="zh-cn_topic_0000002600000201_p102000010030"></a>Meaning: Source of accuracy, shared with quantization. It can be Callable[[model], float] or any object that exposes evaluate(model) and returns a scalar. When it is None, the default JS output distribution similarity evaluation is used (previously the default was "fidelity"; the acceptance boundaries of the same tolerance differ under the two metrics, so existing tolerance/size_budget search results that do not explicitly pass an evaluator will change accordingly. To preserve the old behavior, explicitly pass evaluator="fidelity"). It can also be "fidelity" / "js" / "ppl". "auto" selects classification accuracy or PPL based on the evaluation data and model metadata, falls back to JS when it cannot be determined, and records the reason. The task presets "preservation" / "causal_lm" / "classification" represent fidelity, PPL, and labeled top-1 accuracy, respectively. For explicit classification, eval_data is (inputs, labels) or a dictionary containing labels. auto/JS preserve the positional tuple as model arguments; labels must be explicitly provided via a dictionary labels or batch_adapter, with one int64 class per sample. tolerance is the absolute drop relative to the measured accuracy of the original model. JS accumulates only at positions marked valid by attention_mask (falls back to all positions when no mask is present) and caches the reference distribution in fp16 (with a numerical noise floor on the order of 1e-3). PPL uses unshifted labels (defaulting to input_ids) and ignores positions masked by -100 and attention_mask. When 0 &lt;= tolerance &lt; 1, the upper bound of the allowed PPL ratio is 1 / (1 - tolerance).</p>
<p id="zh-cn_topic_0000002600000201_p102000010031"><a name="zh-cn_topic_0000002600000201_p102000010031"></a><a name="zh-cn_topic_0000002600000201_p102000010031"></a>Default value: None.</p>
<p id="zh-cn_topic_0000002600000201_p102000010032"><a name="zh-cn_topic_0000002600000201_p102000010032"></a><a name="zh-cn_topic_0000002600000201_p102000010032"></a>Data type: Callable / object exposing evaluate(model) / str / None.</p></td>
</tr>
<tr id="zh-cn_topic_0000002600000201_row102000010008"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000201_p102000010033"><a name="zh-cn_topic_0000002600000201_p102000010033"></a><a name="zh-cn_topic_0000002600000201_p102000010033"></a>eval_iterations</p></td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000201_p102000010034"><a name="zh-cn_topic_0000002600000201_p102000010034"></a><a name="zh-cn_topic_0000002600000201_p102000010034"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000201_p102000010035"><a name="zh-cn_topic_0000002600000201_p102000010035"></a><a name="zh-cn_topic_0000002600000201_p102000010035"></a>Meaning: Number of evaluation iterations in tolerance mode.</p>
<p id="zh-cn_topic_0000002600000201_p102000010036"><a name="zh-cn_topic_0000002600000201_p102000010036"></a><a name="zh-cn_topic_0000002600000201_p102000010036"></a>Default value: 1.</p>
<p id="zh-cn_topic_0000002600000201_p102000010037"><a name="zh-cn_topic_0000002600000201_p102000010037"></a><a name="zh-cn_topic_0000002600000201_p102000010037"></a>Data type: int.</p></td>
</tr>
<tr id="zh-cn_topic_0000002600000201_row102000010009"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000201_p102000010038"><a name="zh-cn_topic_0000002600000201_p102000010038"></a><a name="zh-cn_topic_0000002600000201_p102000010038"></a>eval_data</p></td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000201_p102000010039"><a name="zh-cn_topic_0000002600000201_p102000010039"></a><a name="zh-cn_topic_0000002600000201_p102000010039"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000201_p102000010040"><a name="zh-cn_topic_0000002600000201_p102000010040"></a><a name="zh-cn_topic_0000002600000201_p102000010040"></a>Meaning: Evaluation data passed to automatic pruning in tolerance mode; falls back to data by default.</p>
<p id="zh-cn_topic_0000002600000201_p102000010041"><a name="zh-cn_topic_0000002600000201_p102000010041"></a><a name="zh-cn_topic_0000002600000201_p102000010041"></a>Default value: None.</p>
<p id="zh-cn_topic_0000002600000201_p102000010042"><a name="zh-cn_topic_0000002600000201_p102000010042"></a><a name="zh-cn_topic_0000002600000201_p102000010042"></a>Data type: Any.</p></td>
</tr>
<tr id="zh-cn_topic_0000002600000201_row102000010010"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000201_p102000010043"><a name="zh-cn_topic_0000002600000201_p102000010043"></a><a name="zh-cn_topic_0000002600000201_p102000010043"></a>ratio_grid</p></td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000201_p102000010044"><a name="zh-cn_topic_0000002600000201_p102000010044"></a><a name="zh-cn_topic_0000002600000201_p102000010044"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000201_p102000010045"><a name="zh-cn_topic_0000002600000201_p102000010045"></a><a name="zh-cn_topic_0000002600000201_p102000010045"></a>Meaning: Candidate pruning rates (ascending) in tolerance mode; when None, the default grid is used.</p>
<p id="zh-cn_topic_0000002600000201_p102000010046"><a name="zh-cn_topic_0000002600000201_p102000010046"></a><a name="zh-cn_topic_0000002600000201_p102000010046"></a>Default value: None.</p>
<p id="zh-cn_topic_0000002600000201_p102000010047"><a name="zh-cn_topic_0000002600000201_p102000010047"></a><a name="zh-cn_topic_0000002600000201_p102000010047"></a>Data type: Any / None.</p></td>
</tr>
<tr id="zh-cn_topic_0000002600000201_row102000010011"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000201_p102000010048"><a name="zh-cn_topic_0000002600000201_p102000010048"></a><a name="zh-cn_topic_0000002600000201_p102000010048"></a>size_budget</p></td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000201_p102000010049"><a name="zh-cn_topic_0000002600000201_p102000010049"></a><a name="zh-cn_topic_0000002600000201_p102000010049"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000201_p102000010050"><a name="zh-cn_topic_0000002600000201_p102000010050"></a><a name="zh-cn_topic_0000002600000201_p102000010050"></a>Meaning: When given (retention ratio, (0,1]), prunes by size budget; mutually exclusive with tolerance.</p>
<p id="zh-cn_topic_0000002600000201_p102000010051"><a name="zh-cn_topic_0000002600000201_p102000010051"></a><a name="zh-cn_topic_0000002600000201_p102000010051"></a>Default value: None.</p>
<p id="zh-cn_topic_0000002600000201_p102000010052"><a name="zh-cn_topic_0000002600000201_p102000010052"></a><a name="zh-cn_topic_0000002600000201_p102000010052"></a>Data type: float / None.</p></td>
</tr>
<tr id="zh-cn_topic_0000002600000201_row102000010012"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000201_p102000010053"><a name="zh-cn_topic_0000002600000201_p102000010053"></a><a name="zh-cn_topic_0000002600000201_p102000010053"></a>finetune_fn</p></td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000201_p102000010054"><a name="zh-cn_topic_0000002600000201_p102000010054"></a><a name="zh-cn_topic_0000002600000201_p102000010054"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000201_p102000010055"><a name="zh-cn_topic_0000002600000201_p102000010055"></a><a name="zh-cn_topic_0000002600000201_p102000010055"></a>Meaning: Callback that performs (lightweight) fine-tuning recovery and re-evaluation after pruning at each candidate rate in tolerance mode, enabling the search to select a more aggressive usable rate.</p>
<p id="zh-cn_topic_0000002600000201_p102000010056"><a name="zh-cn_topic_0000002600000201_p102000010056"></a><a name="zh-cn_topic_0000002600000201_p102000010056"></a>Default value: None.</p>
<p id="zh-cn_topic_0000002600000201_p102000010057"><a name="zh-cn_topic_0000002600000201_p102000010057"></a><a name="zh-cn_topic_0000002600000201_p102000010057"></a>Data type: `Callable[[nn.Module], Any]` / None.</p></td>
</tr>
<tr id="zh-cn_topic_0000002600000201_row102000010013"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000201_p102000010058"><a name="zh-cn_topic_0000002600000201_p102000010058"></a><a name="zh-cn_topic_0000002600000201_p102000010058"></a>quant_fn</p></td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000201_p102000010059"><a name="zh-cn_topic_0000002600000201_p102000010059"></a><a name="zh-cn_topic_0000002600000201_p102000010059"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000201_p102000010060"><a name="zh-cn_topic_0000002600000201_p102000010060"></a><a name="zh-cn_topic_0000002600000201_p102000010060"></a>Meaning: Callback for quantizing the model during tolerance search (quantization-aware search).</p>
<p id="zh-cn_topic_0000002600000201_p102000010061"><a name="zh-cn_topic_0000002600000201_p102000010061"></a><a name="zh-cn_topic_0000002600000201_p102000010061"></a>Default value: None.</p>
<p id="zh-cn_topic_0000002600000201_p102000010062"><a name="zh-cn_topic_0000002600000201_p102000010062"></a><a name="zh-cn_topic_0000002600000201_p102000010062"></a>Data type: Callable[[nn.Module], Any] / None.</p></td>
</tr>
<tr id="zh-cn_topic_0000002600000201_row102000010014"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000201_p102000010063"><a name="zh-cn_topic_0000002600000201_p102000010063"></a><a name="zh-cn_topic_0000002600000201_p102000010063"></a>report</p></td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000201_p102000010064"><a name="zh-cn_topic_0000002600000201_p102000010064"></a><a name="zh-cn_topic_0000002600000201_p102000010064"></a>Input/Output</p></td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000201_p102000010065"><a name="zh-cn_topic_0000002600000201_p102000010065"></a><a name="zh-cn_topic_0000002600000201_p102000010065"></a>Meaning: Statistics sink provided by the caller. When passed in, pruning statistics are written into it and can be read later via <code>report.as_dict()</code> (no process-level global state).</p>
<p id="zh-cn_topic_0000002600000201_p102000010066"><a name="zh-cn_topic_0000002600000201_p102000010066"></a><a name="zh-cn_topic_0000002600000201_p102000010066"></a>Default value: None.</p>
<p id="zh-cn_topic_0000002600000201_p102000010067"><a name="zh-cn_topic_0000002600000201_p102000010067"></a><a name="zh-cn_topic_0000002600000201_p102000010067"></a>Data type: PruneReport (from amct_pytorch.pruning import) / None.</p></td>
</tr>
</tbody>
</table>

## Return Value<a name="zh-cn_topic_0000002600000201_section102000010007"></a>

None. Pruning is an in-place operation. Pruning statistics are obtained through the `report=PruneReport()` sink provided by the caller (read `report.as_dict()` afterwards, which contains fields such as **backend**, **params_before**, **params_after**, **per_layer_sparsity**, **warnings**, and **budget_unreachable**) (no process-level global state).

## Supported Pruning Structures and Constraints<a name="zh-cn_topic_0000002600000201_section102000010009"></a>

Pruning is performed only when the producer<->consumer API can be verified; other structures are skipped.

<a name="zh-cn_topic_0000002600000201_table102000010010"></a>

<table><thead align="left"><tr id="row102001010001"><th class="cellrowborder" valign="top" width="13%" id="mcps1.2.4.1.1"><p id="p102001010001"><a name="p102001010001"></a><a name="p102001010001"></a>Domain</p></th>
<th class="cellrowborder" valign="top" width="42%" id="mcps1.2.4.1.2"><p id="p102001010002"><a name="p102001010002"></a><a name="p102001010002"></a>Supported Structures</p></th>
<th class="cellrowborder" valign="top" width="45%" id="mcps1.2.4.1.3"><p id="p102001010003"><a name="p102001010003"></a><a name="p102001010003"></a>Constraints/Automatic Exclusion</p></th>
</tr>
</thead>
<tbody><tr id="row102001010004"><td class="cellrowborder" valign="top" width="13%" headers="mcps1.2.4.1.1 "><p id="p102001010004"><a name="p102001010004"></a><a name="p102001010004"></a>dense</p></td>
<td class="cellrowborder" valign="top" width="42%" headers="mcps1.2.4.1.2 "><p id="p102001010005"><a name="p102001010005"></a><a name="p102001010005"></a>Three Linear structures gate/up/down_proj; fused gate_up_proj (Phi-3/GLM-4); two adjacent Linear/Conv1D (including Bloom style). Llama/Qwen2/Mistral/Qwen3 can be pruned without manually configuring skip_layers.</p></td>
<td class="cellrowborder" valign="top" width="45%" headers="mcps1.2.4.1.3 "><p id="p102001010006"><a name="p102001010006"></a><a name="p102001010006"></a>Only the intermediate dimension (gate/up.out_features + down.in_features) is pruned, while the hidden/residual width remains unchanged; attention projections q/k/v/o are automatically excluded.</p></td>
</tr>
<tr id="row102001010007"><td class="cellrowborder" valign="top" width="13%" headers="mcps1.2.4.1.1 "><p id="p102001010007"><a name="p102001010007"></a><a name="p102001010007"></a>cnn</p></td>
<td class="cellrowborder" valign="top" width="42%" headers="mcps1.2.4.1.2 "><p id="p102001010008"><a name="p102001010008"></a><a name="p102001010008"></a>producer Conv2d (`groups=1`) -> optional BatchNorm2d -> channel-matching consumer Conv2d/Linear, with synchronized scaling.</p></td>
<td class="cellrowborder" valign="top" width="45%" headers="mcps1.2.4.1.3 "><p id="p102001010008"><a name="p102001010009"></a><a name="p102001010009"></a>Convs directly fed by a residual add are subject to automatic exclusion; Concat consumers and grouped/depthwise convolutions do not support pruning (0 targets).</p></td>
</tr>
<tr id="row102001010010"><td class="cellrowborder" valign="top" width="13%" headers="mcps1.2.4.1.1 "><p id="p102001010010"><a name="p102001010010"></a><a name="p102001010010"></a>moe</p></td>
<td class="cellrowborder" valign="top" width="42%" headers="mcps1.2.4.1.2 "><p id="p102001010011"><a name="p102001010011"></a><a name="p102001010011"></a>nn.ModuleList experts + nn.Linear gate; fused batched experts (MixtralExperts/Qwen3MoeExperts + *TopKRouter); grouped router (n_group/topk_group); shared experts + sigmoid routing (noaux_tc); same-level two-tensor fused experts (GraniteMoE); nested router-bias (Ernie4.5).</p></td>
<td class="cellrowborder" valign="top" width="45%" headers="mcps1.2.4.1.3 "><p id="p102001010012"><a name="p102001010012"></a><a name="p102001010012"></a>Routed experts are removed as a whole, with hidden input/output unchanged; shared (resident) experts are subject to automatic exclusion (their FFN still belongs to the dense domain, so when pruning only MoE, dense must be explicitly fixed to `prune_ratio: 0.0`).</p></td>
</tr>
</tbody>
</table>

## Example<a name="zh-cn_topic_0000002600000201_section102000010008"></a>

```python
import amct_pytorch as amct
from amct_pytorch.pruning import PruneReport

# Fixed pruning rate: prune 50% of the intermediate dimension in the dense domain (automatically avoiding attention projections).
cfg = {"methods": {"dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.5}}}}
rep = PruneReport()
amct.prune(model, cfg, data=calib, report=rep)
print(rep.as_dict())

# Tolerance-driven: allow a maximum loss of 2%, automatically search for and apply the maximum pruning rate that satisfies the tolerance.
rep2 = PruneReport()
amct.prune(model, data=calib, tolerance=0.02, report=rep2)
print(rep2.as_dict())
```

## Predefined Configurations<a name="zh-cn_topic_0000002600000201_section102000010010"></a>

Ready-to-use preset configurations (dict) that can be passed directly as the `config` parameter. Each preset encapsulates the corresponding pruning method and hyperparameters for different network structures (CNN, Dense, MoE). All constants in the following table are exported at the top level of `amct_pytorch` and can be referenced directly as `amct.<constant name>`.

<a name="zh-cn_topic_0000002600000201_table102000010011"></a>

| Configuration Constant | Applicable Domain | Key kwargs (Default Value) | Description |
| -------- | ------ | -------------------- | ---- |
| CNN_RECONSTRUCT_PRUNE_CFG | cnn | prune_ratio=0.30 | CNN convolution channel reconstruction pruning. After pruning, least-squares reconstruction is performed on downstream weights. |
| CNN_VARIANCE_PRUNE_CFG | cnn | prune_ratio=0.30 | CNN convolution channel variance pruning. Removes channels with the lowest information based on activation variance. |
| DENSE_LOWVAR_PRUNE_CFG | dense | prune_ratio=0.50 | Dense (fully connected) low-variance pruning. Removes neurons with the lowest variance. |
| MOE_MASSVAR_PRUNE_CFG | moe | prune_ratio=0.50, boundary=10 | MoE mass-variance (mass_variance) pruning. Combines weight mass and variance, with boundary controlling the boundary. |
| MOE_OUTPUT_MERGE_PRUNE_CFG | moe | keep_ratio=0.50, selector="calib_nll" | MoE output-space expert merging. Uses the calib-NLL selector to choose between merging and dropping. |
| SENSITIVITY_ALLOC_PRUNE_CFG | dense | prune_ratio=0.50; allocation: strategy="sensitivity", ref_ratio=0.50, min_ratio=0.05, max_ratio=0.90, guard="calib_nll" | Allocates pruning rates by per-layer sensitivity (prune more in robust layers and less in sensitive layers), with the calib_nll guard falling back to uniform allocation. Requires data= to be passed. |
| FULL_STRUCTURED_PRUNE_CFG | cnn + dense + moe | cnn: prune_ratio=0.30; dense: prune_ratio=0.50; moe: prune_ratio=0.50, boundary=10 | A combined preset that performs structured pruning on CNN, Dense, and MoE structures simultaneously in one pass. |

> Note: The "Applicable Domain" in the table refers to the domains listed in `methods` for that preset. Other domains are automatically fixed to `prune_ratio: 0.0` and are not pruned incidentally. All the above constants enable `"missing_data_policy": "warn_skip"` (warn and skip the corresponding method when calibration data is missing). `MOE_VARIANCE_MENU_CFG`, `DENSE_RECOVERY_MENU_CFG`, and `CNN_RECOVERY_MENU_CFG` are dedicated configurations for the guard-style menu mode (menu): when `prune` detects that the configuration contains `menu`, it switches to menu-based selection, measures each candidate on `eval_data`, and applies the best one, without requiring an additional switch. All three are imported from `amct_pytorch.pruning` and are not exported at the top level of `amct_pytorch`.

```python
# Pass the predefined pruning configuration directly as the second argument.
amct.prune(model, amct.DENSE_LOWVAR_PRUNE_CFG, data=calib)
```
