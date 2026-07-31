# accuracy\_based\_auto\_prune<a name="ZH-CN_TOPIC_0000002600000101"></a>

## 产品支持情况<a name="zh-cn_topic_0000002600000101_section101000010001"></a>

<a name="zh-cn_topic_0000002600000101_table101000010002"></a>

| 产品                                        | 是否支持 |
| ------------------------------------------- | -------- |
| Ascend 950PR/Ascend 950DT                   | √        |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √        |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √        |



## 功能说明<a name="zh-cn_topic_0000002600000101_section101000010003"></a>

在剪枝率网格上搜索精度损失 ≤ tolerance 的最大剪枝率并应用，返回 AutoTuneResult；无满足者则不剪枝。精度由 `evaluator(model) -> float` 给出，未提供时使用默认免标签 top-1 保真度指标。

## 函数原型<a name="zh-cn_topic_0000002600000101_section101000010004"></a>

```python
accuracy_based_auto_prune(
    model,
    config=None,
    *,
    data=None,
    tolerance=0.02,
    evaluator=None,
    eval_iterations=1,
    eval_data=None,
    ratio_grid=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
    batch_adapter=None,
    safe_skip_attention=True,
    finetune_fn=None,
    quant_fn=None,
    apply=True,
)
```

## 参数说明<a name="zh-cn_topic_0000002600000101_section101000010005"></a>

<a name="zh-cn_topic_0000002600000101_table101000010006"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002600000101_row101000010001"><th class="cellrowborder" valign="top" width="13.78%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000002600000101_p101000010001"><a name="zh-cn_topic_0000002600000101_p101000010001"></a><a name="zh-cn_topic_0000002600000101_p101000010001"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="8.01%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000002600000101_p101000010002"><a name="zh-cn_topic_0000002600000101_p101000010002"></a><a name="zh-cn_topic_0000002600000101_p101000010002"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="78.21%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000002600000101_p101000010003"><a name="zh-cn_topic_0000002600000101_p101000010003"></a><a name="zh-cn_topic_0000002600000101_p101000010003"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002600000101_row101000010002"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000101_p101000010004"><a name="zh-cn_topic_0000002600000101_p101000010004"></a><a name="zh-cn_topic_0000002600000101_p101000010004"></a>model</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000101_p101000010005"><a name="zh-cn_topic_0000002600000101_p101000010005"></a><a name="zh-cn_topic_0000002600000101_p101000010005"></a>输入/输出</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000101_p101000010006"><a name="zh-cn_topic_0000002600000101_p101000010006"></a><a name="zh-cn_topic_0000002600000101_p101000010006"></a>含义：待剪枝模型；当 <code>apply=True</code> 时被原地（in-place）修改。</p>
<p id="zh-cn_topic_0000002600000101_p101000010007"><a name="zh-cn_topic_0000002600000101_p101000010007"></a><a name="zh-cn_topic_0000002600000101_p101000010007"></a>数据类型：torch.nn.Module。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002600000101_row101000010003"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000101_p101000010008"><a name="zh-cn_topic_0000002600000101_p101000010008"></a><a name="zh-cn_topic_0000002600000101_p101000010008"></a>config</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000101_p101000010009"><a name="zh-cn_topic_0000002600000101_p101000010009"></a><a name="zh-cn_topic_0000002600000101_p101000010009"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000101_p101000010010"><a name="zh-cn_topic_0000002600000101_p101000010010"></a><a name="zh-cn_topic_0000002600000101_p101000010010"></a>含义：剪枝配置；每个 method 的 <code>prune_ratio</code> 会被搜索到的值覆盖。为 <code>None</code> 时使用 CNN/Dense/MoE 三域默认配置。</p>
<p id="zh-cn_topic_0000002600000101_p101000010011"><a name="zh-cn_topic_0000002600000101_p101000010011"></a><a name="zh-cn_topic_0000002600000101_p101000010011"></a>默认值：None。</p>
<p id="zh-cn_topic_0000002600000101_p101000010012"><a name="zh-cn_topic_0000002600000101_p101000010012"></a><a name="zh-cn_topic_0000002600000101_p101000010012"></a>数据类型：dict / PruneConfig / None。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002600000101_row101000010004"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000101_p101000010013"><a name="zh-cn_topic_0000002600000101_p101000010013"></a><a name="zh-cn_topic_0000002600000101_p101000010013"></a>data</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000101_p101000010014"><a name="zh-cn_topic_0000002600000101_p101000010014"></a><a name="zh-cn_topic_0000002600000101_p101000010014"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000101_p101000010015"><a name="zh-cn_topic_0000002600000101_p101000010015"></a><a name="zh-cn_topic_0000002600000101_p101000010015"></a>含义：校准数据（方差类方法需要）；当未提供 <code>eval_data</code> 时也作为默认保真度指标的评估集。</p>
<p id="zh-cn_topic_0000002600000101_p101000010016"><a name="zh-cn_topic_0000002600000101_p101000010016"></a><a name="zh-cn_topic_0000002600000101_p101000010016"></a>默认值：None。</p>
<p id="zh-cn_topic_0000002600000101_p101000010017"><a name="zh-cn_topic_0000002600000101_p101000010017"></a><a name="zh-cn_topic_0000002600000101_p101000010017"></a>数据类型：可迭代对象（batch 序列）。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002600000101_row101000010005"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000101_p101000010018"><a name="zh-cn_topic_0000002600000101_p101000010018"></a><a name="zh-cn_topic_0000002600000101_p101000010018"></a>tolerance</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000101_p101000010019"><a name="zh-cn_topic_0000002600000101_p101000010019"></a><a name="zh-cn_topic_0000002600000101_p101000010019"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000101_p101000010020"><a name="zh-cn_topic_0000002600000101_p101000010020"></a><a name="zh-cn_topic_0000002600000101_p101000010020"></a>含义：可接受精度损失的上界，量纲与 <code>evaluator</code> 一致（默认保真度指标下为 top-1 翻转比例）。</p>
<p id="zh-cn_topic_0000002600000101_p101000010021"><a name="zh-cn_topic_0000002600000101_p101000010021"></a><a name="zh-cn_topic_0000002600000101_p101000010021"></a>默认值：0.02。</p>
<p id="zh-cn_topic_0000002600000101_p101000010022"><a name="zh-cn_topic_0000002600000101_p101000010022"></a><a name="zh-cn_topic_0000002600000101_p101000010022"></a>数据类型：float。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002600000101_row101000010006"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000101_p101000010023"><a name="zh-cn_topic_0000002600000101_p101000010023"></a><a name="zh-cn_topic_0000002600000101_p101000010023"></a>evaluator</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000101_p101000010024"><a name="zh-cn_topic_0000002600000101_p101000010024"></a><a name="zh-cn_topic_0000002600000101_p101000010024"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000101_p101000010025"><a name="zh-cn_topic_0000002600000101_p101000010025"></a><a name="zh-cn_topic_0000002600000101_p101000010025"></a>含义：精度来源：Callable[[model], float] 或 AutoCalibrationEvaluatorBase（与量化共用）；为 None 时用默认 top-1 保真度。</p>
<p id="zh-cn_topic_0000002600000101_p101000010026"><a name="zh-cn_topic_0000002600000101_p101000010026"></a><a name="zh-cn_topic_0000002600000101_p101000010026"></a>默认值：None。</p>
<p id="zh-cn_topic_0000002600000101_p101000010027"><a name="zh-cn_topic_0000002600000101_p101000010027"></a><a name="zh-cn_topic_0000002600000101_p101000010027"></a>数据类型：Callable / AutoCalibrationEvaluatorBase。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002600000101_row101000010007"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000101_p101000010028"><a name="zh-cn_topic_0000002600000101_p101000010028"></a><a name="zh-cn_topic_0000002600000101_p101000010028"></a>eval_iterations</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000101_p101000010029"><a name="zh-cn_topic_0000002600000101_p101000010029"></a><a name="zh-cn_topic_0000002600000101_p101000010029"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000101_p101000010030"><a name="zh-cn_topic_0000002600000101_p101000010030"></a><a name="zh-cn_topic_0000002600000101_p101000010030"></a>含义：传给 <code>evaluator.evaluate</code> 的迭代次数。</p>
<p id="zh-cn_topic_0000002600000101_p101000010031"><a name="zh-cn_topic_0000002600000101_p101000010031"></a><a name="zh-cn_topic_0000002600000101_p101000010031"></a>默认值：1。</p>
<p id="zh-cn_topic_0000002600000101_p101000010032"><a name="zh-cn_topic_0000002600000101_p101000010032"></a><a name="zh-cn_topic_0000002600000101_p101000010032"></a>数据类型：int。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002600000101_row101000010008"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000101_p101000010033"><a name="zh-cn_topic_0000002600000101_p101000010033"></a><a name="zh-cn_topic_0000002600000101_p101000010033"></a>eval_data</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000101_p101000010034"><a name="zh-cn_topic_0000002600000101_p101000010034"></a><a name="zh-cn_topic_0000002600000101_p101000010034"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000101_p101000010035"><a name="zh-cn_topic_0000002600000101_p101000010035"></a><a name="zh-cn_topic_0000002600000101_p101000010035"></a>含义：默认保真度指标使用的评估批；未提供时回退到 <code>data</code>。</p>
<p id="zh-cn_topic_0000002600000101_p101000010036"><a name="zh-cn_topic_0000002600000101_p101000010036"></a><a name="zh-cn_topic_0000002600000101_p101000010036"></a>默认值：None。</p>
<p id="zh-cn_topic_0000002600000101_p101000010037"><a name="zh-cn_topic_0000002600000101_p101000010037"></a><a name="zh-cn_topic_0000002600000101_p101000010037"></a>数据类型：可迭代对象（batch 序列）。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002600000101_row101000010009"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000101_p101000010038"><a name="zh-cn_topic_0000002600000101_p101000010038"></a><a name="zh-cn_topic_0000002600000101_p101000010038"></a>ratio_grid</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000101_p101000010039"><a name="zh-cn_topic_0000002600000101_p101000010039"></a><a name="zh-cn_topic_0000002600000101_p101000010039"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000101_p101000010040"><a name="zh-cn_topic_0000002600000101_p101000010040"></a><a name="zh-cn_topic_0000002600000101_p101000010040"></a>含义：候选剪枝率（升序）。</p>
<p id="zh-cn_topic_0000002600000101_p101000010041"><a name="zh-cn_topic_0000002600000101_p101000010041"></a><a name="zh-cn_topic_0000002600000101_p101000010041"></a>默认值：(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)。</p>
<p id="zh-cn_topic_0000002600000101_p101000010042"><a name="zh-cn_topic_0000002600000101_p101000010042"></a><a name="zh-cn_topic_0000002600000101_p101000010042"></a>数据类型：float 序列。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002600000101_row101000010010"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000101_p101000010043"><a name="zh-cn_topic_0000002600000101_p101000010043"></a><a name="zh-cn_topic_0000002600000101_p101000010043"></a>batch_adapter</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000101_p101000010044"><a name="zh-cn_topic_0000002600000101_p101000010044"></a><a name="zh-cn_topic_0000002600000101_p101000010044"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000101_p101000010045"><a name="zh-cn_topic_0000002600000101_p101000010045"></a><a name="zh-cn_topic_0000002600000101_p101000010045"></a>含义：将一个 data batch 映射为模型前向所需的 <code>(args, kwargs)</code>；为 <code>None</code> 时使用合理的默认解析。</p>
<p id="zh-cn_topic_0000002600000101_p101000010046"><a name="zh-cn_topic_0000002600000101_p101000010046"></a><a name="zh-cn_topic_0000002600000101_p101000010046"></a>默认值：None。</p>
<p id="zh-cn_topic_0000002600000101_p101000010047"><a name="zh-cn_topic_0000002600000101_p101000010047"></a><a name="zh-cn_topic_0000002600000101_p101000010047"></a>数据类型：Callable / None。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002600000101_row101000010011"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000101_p101000010048"><a name="zh-cn_topic_0000002600000101_p101000010048"></a><a name="zh-cn_topic_0000002600000101_p101000010048"></a>safe_skip_attention</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000101_p101000010049"><a name="zh-cn_topic_0000002600000101_p101000010049"></a><a name="zh-cn_topic_0000002600000101_p101000010049"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000101_p101000010050"><a name="zh-cn_topic_0000002600000101_p101000010050"></a><a name="zh-cn_topic_0000002600000101_p101000010050"></a>含义：是否将注意力子模块（self_attn/attention/attn）并入 <code>skip_layers</code>，不参与剪枝。</p>
<p id="zh-cn_topic_0000002600000101_p101000010051"><a name="zh-cn_topic_0000002600000101_p101000010051"></a><a name="zh-cn_topic_0000002600000101_p101000010051"></a>默认值：True。</p>
<p id="zh-cn_topic_0000002600000101_p101000010052"><a name="zh-cn_topic_0000002600000101_p101000010052"></a><a name="zh-cn_topic_0000002600000101_p101000010052"></a>数据类型：bool。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002600000101_row101000010012"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000101_p101000010053"><a name="zh-cn_topic_0000002600000101_p101000010053"></a><a name="zh-cn_topic_0000002600000101_p101000010053"></a>finetune_fn</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000101_p101000010054"><a name="zh-cn_topic_0000002600000101_p101000010054"></a><a name="zh-cn_topic_0000002600000101_p101000010054"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000101_p101000010055"><a name="zh-cn_topic_0000002600000101_p101000010055"></a><a name="zh-cn_topic_0000002600000101_p101000010055"></a>含义：可选的剪枝后恢复回调；提供时每个候选会在评估前对剪枝副本微调一次，使搜索衡量"剪枝+恢复"后的可达精度（每个候选微调一次，更慢）。</p>
<p id="zh-cn_topic_0000002600000101_p101000010056"><a name="zh-cn_topic_0000002600000101_p101000010056"></a><a name="zh-cn_topic_0000002600000101_p101000010056"></a>默认值：None。</p>
<p id="zh-cn_topic_0000002600000101_p101000010057"><a name="zh-cn_topic_0000002600000101_p101000010057"></a><a name="zh-cn_topic_0000002600000101_p101000010057"></a>数据类型：Callable[[torch.nn.Module], Any]。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002600000101_row101000010013"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000101_p101000010058"><a name="zh-cn_topic_0000002600000101_p101000010058"></a><a name="zh-cn_topic_0000002600000101_p101000010058"></a>quant_fn</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000101_p101000010059"><a name="zh-cn_topic_0000002600000101_p101000010059"></a><a name="zh-cn_topic_0000002600000101_p101000010059"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000101_p101000010060"><a name="zh-cn_topic_0000002600000101_p101000010060"></a><a name="zh-cn_topic_0000002600000101_p101000010060"></a>含义：可选的回调，在评估前（在 <code>finetune_fn</code> 之后）作用于剪枝副本，使搜索在该状态下决策；返回的模型仍为仅剪枝模型。</p>
<p id="zh-cn_topic_0000002600000101_p101000010061"><a name="zh-cn_topic_0000002600000101_p101000010061"></a><a name="zh-cn_topic_0000002600000101_p101000010061"></a>默认值：None。</p>
<p id="zh-cn_topic_0000002600000101_p101000010062"><a name="zh-cn_topic_0000002600000101_p101000010062"></a><a name="zh-cn_topic_0000002600000101_p101000010062"></a>数据类型：Callable[[torch.nn.Module], Any]。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002600000101_row101000010014"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000101_p101000010063"><a name="zh-cn_topic_0000002600000101_p101000010063"></a><a name="zh-cn_topic_0000002600000101_p101000010063"></a>apply</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000101_p101000010064"><a name="zh-cn_topic_0000002600000101_p101000010064"></a><a name="zh-cn_topic_0000002600000101_p101000010064"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000101_p101000010065"><a name="zh-cn_topic_0000002600000101_p101000010065"></a><a name="zh-cn_topic_0000002600000101_p101000010065"></a>含义：为 <code>True</code> 时将选定剪枝率原地应用到 model；为 <code>False</code> 时仅搜索、不改动原模型。</p>
<p id="zh-cn_topic_0000002600000101_p101000010066"><a name="zh-cn_topic_0000002600000101_p101000010066"></a><a name="zh-cn_topic_0000002600000101_p101000010066"></a>默认值：True。</p>
<p id="zh-cn_topic_0000002600000101_p101000010067"><a name="zh-cn_topic_0000002600000101_p101000010067"></a><a name="zh-cn_topic_0000002600000101_p101000010067"></a>数据类型：bool。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000002600000101_section101000010007"></a>

返回 `AutoTuneResult`（dataclass），其关键字段：

- `chosen_ratio`：选定的剪枝率；容差内无可行率时为 `None`。
- `tolerance`：本次搜索使用的容差。
- `baseline_quality` / `pruned_quality` / `quality_drop`：基准精度、选定率下的精度、以及精度损失。
- `params_before` / `params_after` / `weight_reduction`：剪枝前后的参数量及实际削减比例。
- `applied`：是否已将剪枝原地应用到模型（受 `apply` 控制）。
- `report`：剪枝报告 `PruneReport`（未应用时为 `None`）。
- `trials`：各候选剪枝率的试探结果列表。
- `summary()`：返回一行可读摘要。

## 调用示例<a name="zh-cn_topic_0000002600000101_section101000010008"></a>

```python
import amct_pytorch as amct

# model 为用户提供的 torch.nn.Module，calib 为校准/评估数据（batch 序列）
result = amct.accuracy_based_auto_prune(
    model,
    data=calib,
    tolerance=0.02,   # 最多可接受 2% 精度损失
)
# 原地剪枝完成，查看结果摘要
print(result.summary())
```
