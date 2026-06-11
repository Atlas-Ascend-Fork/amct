# size\_budget\_prune<a name="ZH-CN_TOPIC_0000002600000301"></a>

## 产品支持情况<a name="zh-cn_topic_0000002600000301_section103000010001"></a>

<a name="zh-cn_topic_0000002600000301_table103000010002"></a>

| 产品                                        | 是否支持 |
| ------------------------------------------- | -------- |
| Ascend 950PR/Ascend 950DT                   | √        |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √        |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √        |



## 功能说明<a name="zh-cn_topic_0000002600000301_section103000010003"></a>

按目标尺寸/保留比例（size budget）反推并应用各域剪枝率。在候选剪枝率网格上二分搜索出参数量满足目标上界的最小剪枝率，并报告该点的实际尺寸与精度。

## 函数原型<a name="zh-cn_topic_0000002600000301_section103000010004"></a>

```python
size_budget_prune(model, config=None, *, data=None, target_keep_ratio=None,
                  target_params=None, evaluator=None,
                  eval_iterations=1, eval_data=None, ratio_grid=DEFAULT_RATIO_GRID,
                  batch_adapter=None, safe_skip_attention=True, finetune_fn=None,
                  quant_fn=None, apply=True, budget_scope="total")
```

## 参数说明<a name="zh-cn_topic_0000002600000301_section103000010005"></a>

<a name="zh-cn_topic_0000002600000301_table103000010006"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002600000301_row103000010001"><th class="cellrowborder" valign="top" width="13.78%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000002600000301_p103000010001"><a name="zh-cn_topic_0000002600000301_p103000010001"></a><a name="zh-cn_topic_0000002600000301_p103000010001"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="8.01%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000002600000301_p103000010002"><a name="zh-cn_topic_0000002600000301_p103000010002"></a><a name="zh-cn_topic_0000002600000301_p103000010002"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="78.21%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000002600000301_p103000010003"><a name="zh-cn_topic_0000002600000301_p103000010003"></a><a name="zh-cn_topic_0000002600000301_p103000010003"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002600000301_row103000010002"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000301_p103000010004"><a name="zh-cn_topic_0000002600000301_p103000010004"></a><a name="zh-cn_topic_0000002600000301_p103000010004"></a>model</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000301_p103000010005"><a name="zh-cn_topic_0000002600000301_p103000010005"></a><a name="zh-cn_topic_0000002600000301_p103000010005"></a>输入/输出</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000301_p103000010006"><a name="zh-cn_topic_0000002600000301_p103000010006"></a><a name="zh-cn_topic_0000002600000301_p103000010006"></a>含义：待剪枝模型；当 apply=True 时原地修改。</p>
<p id="zh-cn_topic_0000002600000301_p103000010007"><a name="zh-cn_topic_0000002600000301_p103000010007"></a><a name="zh-cn_topic_0000002600000301_p103000010007"></a>数据类型：torch.nn.Module。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002600000301_row103000010003"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000301_p103000010008"><a name="zh-cn_topic_0000002600000301_p103000010008"></a><a name="zh-cn_topic_0000002600000301_p103000010008"></a>config</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000301_p103000010009"><a name="zh-cn_topic_0000002600000301_p103000010009"></a><a name="zh-cn_topic_0000002600000301_p103000010009"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000301_p103000010010"><a name="zh-cn_topic_0000002600000301_p103000010010"></a><a name="zh-cn_topic_0000002600000301_p103000010010"></a>含义：剪枝配置，可为 dict / PruneConfig / None；None 时使用三域默认配置。</p>
<p id="zh-cn_topic_0000002600000301_p103000010011"><a name="zh-cn_topic_0000002600000301_p103000010011"></a><a name="zh-cn_topic_0000002600000301_p103000010011"></a>默认值：None。</p>
<p id="zh-cn_topic_0000002600000301_p103000010012"><a name="zh-cn_topic_0000002600000301_p103000010012"></a><a name="zh-cn_topic_0000002600000301_p103000010012"></a>数据类型：dict、PruneConfig 或 None。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002600000301_row103000010004"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000301_p103000010013"><a name="zh-cn_topic_0000002600000301_p103000010013"></a><a name="zh-cn_topic_0000002600000301_p103000010013"></a>data</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000301_p103000010014"><a name="zh-cn_topic_0000002600000301_p103000010014"></a><a name="zh-cn_topic_0000002600000301_p103000010014"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000301_p103000010015"><a name="zh-cn_topic_0000002600000301_p103000010015"></a><a name="zh-cn_topic_0000002600000301_p103000010015"></a>含义：校准数据；当未提供 eval_data 时同时用作默认精度评估集。</p>
<p id="zh-cn_topic_0000002600000301_p103000010016"><a name="zh-cn_topic_0000002600000301_p103000010016"></a><a name="zh-cn_topic_0000002600000301_p103000010016"></a>默认值：None。</p>
<p id="zh-cn_topic_0000002600000301_p103000010017"><a name="zh-cn_topic_0000002600000301_p103000010017"></a><a name="zh-cn_topic_0000002600000301_p103000010017"></a>数据类型：Sequence 或 None。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002600000301_row103000010005"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000301_p103000010018"><a name="zh-cn_topic_0000002600000301_p103000010018"></a><a name="zh-cn_topic_0000002600000301_p103000010018"></a>target_keep_ratio</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000301_p103000010019"><a name="zh-cn_topic_0000002600000301_p103000010019"></a><a name="zh-cn_topic_0000002600000301_p103000010019"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000301_p103000010020"><a name="zh-cn_topic_0000002600000301_p103000010020"></a><a name="zh-cn_topic_0000002600000301_p103000010020"></a>含义：参数保留比例（如 0.7 表示保留 70%、剪掉 30%）；与 target_params 二者择一使用。</p>
<p id="zh-cn_topic_0000002600000301_p103000010021"><a name="zh-cn_topic_0000002600000301_p103000010021"></a><a name="zh-cn_topic_0000002600000301_p103000010021"></a>默认值：None。</p>
<p id="zh-cn_topic_0000002600000301_p103000010022"><a name="zh-cn_topic_0000002600000301_p103000010022"></a><a name="zh-cn_topic_0000002600000301_p103000010022"></a>数据类型：float 或 None。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002600000301_row103000010006"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000301_p103000010023"><a name="zh-cn_topic_0000002600000301_p103000010023"></a><a name="zh-cn_topic_0000002600000301_p103000010023"></a>target_params</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000301_p103000010024"><a name="zh-cn_topic_0000002600000301_p103000010024"></a><a name="zh-cn_topic_0000002600000301_p103000010024"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000301_p103000010025"><a name="zh-cn_topic_0000002600000301_p103000010025"></a><a name="zh-cn_topic_0000002600000301_p103000010025"></a>含义：目标参数量的直接上界；与 target_keep_ratio 二者择一使用。</p>
<p id="zh-cn_topic_0000002600000301_p103000010026"><a name="zh-cn_topic_0000002600000301_p103000010026"></a><a name="zh-cn_topic_0000002600000301_p103000010026"></a>默认值：None。</p>
<p id="zh-cn_topic_0000002600000301_p103000010027"><a name="zh-cn_topic_0000002600000301_p103000010027"></a><a name="zh-cn_topic_0000002600000301_p103000010027"></a>数据类型：int 或 None。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002600000301_row103000010007"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000301_p103000010028"><a name="zh-cn_topic_0000002600000301_p103000010028"></a><a name="zh-cn_topic_0000002600000301_p103000010028"></a>evaluator</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000301_p103000010029"><a name="zh-cn_topic_0000002600000301_p103000010029"></a><a name="zh-cn_topic_0000002600000301_p103000010029"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000301_p103000010030"><a name="zh-cn_topic_0000002600000301_p103000010030"></a><a name="zh-cn_topic_0000002600000301_p103000010030"></a>含义：精度来源：Callable[[model], float] 或 AutoCalibrationEvaluatorBase（与量化共用）；为 None 时用默认 top-1 保真度。</p>
<p id="zh-cn_topic_0000002600000301_p103000010031"><a name="zh-cn_topic_0000002600000301_p103000010031"></a><a name="zh-cn_topic_0000002600000301_p103000010031"></a>默认值：None。</p>
<p id="zh-cn_topic_0000002600000301_p103000010032"><a name="zh-cn_topic_0000002600000301_p103000010032"></a><a name="zh-cn_topic_0000002600000301_p103000010032"></a>数据类型：Callable / AutoCalibrationEvaluatorBase。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002600000301_row103000010008"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000301_p103000010033"><a name="zh-cn_topic_0000002600000301_p103000010033"></a><a name="zh-cn_topic_0000002600000301_p103000010033"></a>eval_iterations</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000301_p103000010034"><a name="zh-cn_topic_0000002600000301_p103000010034"></a><a name="zh-cn_topic_0000002600000301_p103000010034"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000301_p103000010035"><a name="zh-cn_topic_0000002600000301_p103000010035"></a><a name="zh-cn_topic_0000002600000301_p103000010035"></a>含义：传给 evaluator.evaluate 的迭代次数。</p>
<p id="zh-cn_topic_0000002600000301_p103000010036"><a name="zh-cn_topic_0000002600000301_p103000010036"></a><a name="zh-cn_topic_0000002600000301_p103000010036"></a>默认值：1。</p>
<p id="zh-cn_topic_0000002600000301_p103000010037"><a name="zh-cn_topic_0000002600000301_p103000010037"></a><a name="zh-cn_topic_0000002600000301_p103000010037"></a>数据类型：int。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002600000301_row103000010009"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000301_p103000010038"><a name="zh-cn_topic_0000002600000301_p103000010038"></a><a name="zh-cn_topic_0000002600000301_p103000010038"></a>eval_data</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000301_p103000010039"><a name="zh-cn_topic_0000002600000301_p103000010039"></a><a name="zh-cn_topic_0000002600000301_p103000010039"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000301_p103000010040"><a name="zh-cn_topic_0000002600000301_p103000010040"></a><a name="zh-cn_topic_0000002600000301_p103000010040"></a>含义：默认保真度指标使用的评估批次；缺省时回退使用 data。</p>
<p id="zh-cn_topic_0000002600000301_p103000010041"><a name="zh-cn_topic_0000002600000301_p103000010041"></a><a name="zh-cn_topic_0000002600000301_p103000010041"></a>默认值：None。</p>
<p id="zh-cn_topic_0000002600000301_p103000010042"><a name="zh-cn_topic_0000002600000301_p103000010042"></a><a name="zh-cn_topic_0000002600000301_p103000010042"></a>数据类型：Sequence 或 None。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002600000301_row103000010010"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000301_p103000010043"><a name="zh-cn_topic_0000002600000301_p103000010043"></a><a name="zh-cn_topic_0000002600000301_p103000010043"></a>ratio_grid</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000301_p103000010044"><a name="zh-cn_topic_0000002600000301_p103000010044"></a><a name="zh-cn_topic_0000002600000301_p103000010044"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000301_p103000010045"><a name="zh-cn_topic_0000002600000301_p103000010045"></a><a name="zh-cn_topic_0000002600000301_p103000010045"></a>含义：候选剪枝率网格（升序排列）。</p>
<p id="zh-cn_topic_0000002600000301_p103000010046"><a name="zh-cn_topic_0000002600000301_p103000010046"></a><a name="zh-cn_topic_0000002600000301_p103000010046"></a>默认值：DEFAULT_RATIO_GRID。</p>
<p id="zh-cn_topic_0000002600000301_p103000010047"><a name="zh-cn_topic_0000002600000301_p103000010047"></a><a name="zh-cn_topic_0000002600000301_p103000010047"></a>数据类型：Sequence[float]。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002600000301_row103000010011"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000301_p103000010048"><a name="zh-cn_topic_0000002600000301_p103000010048"></a><a name="zh-cn_topic_0000002600000301_p103000010048"></a>batch_adapter</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000301_p103000010049"><a name="zh-cn_topic_0000002600000301_p103000010049"></a><a name="zh-cn_topic_0000002600000301_p103000010049"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000301_p103000010050"><a name="zh-cn_topic_0000002600000301_p103000010050"></a><a name="zh-cn_topic_0000002600000301_p103000010050"></a>含义：将数据批次映射为模型前向所需的 (args, kwargs)。</p>
<p id="zh-cn_topic_0000002600000301_p103000010051"><a name="zh-cn_topic_0000002600000301_p103000010051"></a><a name="zh-cn_topic_0000002600000301_p103000010051"></a>默认值：None。</p>
<p id="zh-cn_topic_0000002600000301_p103000010052"><a name="zh-cn_topic_0000002600000301_p103000010052"></a><a name="zh-cn_topic_0000002600000301_p103000010052"></a>数据类型：BatchAdapter 或 None。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002600000301_row103000010012"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000301_p103000010053"><a name="zh-cn_topic_0000002600000301_p103000010053"></a><a name="zh-cn_topic_0000002600000301_p103000010053"></a>safe_skip_attention</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000301_p103000010054"><a name="zh-cn_topic_0000002600000301_p103000010054"></a><a name="zh-cn_topic_0000002600000301_p103000010054"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000301_p103000010055"><a name="zh-cn_topic_0000002600000301_p103000010055"></a><a name="zh-cn_topic_0000002600000301_p103000010055"></a>含义：True 时将注意力子模块并入 skip_layers，不参与剪枝。</p>
<p id="zh-cn_topic_0000002600000301_p103000010056"><a name="zh-cn_topic_0000002600000301_p103000010056"></a><a name="zh-cn_topic_0000002600000301_p103000010056"></a>默认值：True。</p>
<p id="zh-cn_topic_0000002600000301_p103000010057"><a name="zh-cn_topic_0000002600000301_p103000010057"></a><a name="zh-cn_topic_0000002600000301_p103000010057"></a>数据类型：bool。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002600000301_row103000010013"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000301_p103000010058"><a name="zh-cn_topic_0000002600000301_p103000010058"></a><a name="zh-cn_topic_0000002600000301_p103000010058"></a>finetune_fn</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000301_p103000010059"><a name="zh-cn_topic_0000002600000301_p103000010059"></a><a name="zh-cn_topic_0000002600000301_p103000010059"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000301_p103000010060"><a name="zh-cn_topic_0000002600000301_p103000010060"></a><a name="zh-cn_topic_0000002600000301_p103000010060"></a>含义：可选的剪枝后恢复回调，在评估前对每个剪枝副本执行。</p>
<p id="zh-cn_topic_0000002600000301_p103000010061"><a name="zh-cn_topic_0000002600000301_p103000010061"></a><a name="zh-cn_topic_0000002600000301_p103000010061"></a>默认值：None。</p>
<p id="zh-cn_topic_0000002600000301_p103000010062"><a name="zh-cn_topic_0000002600000301_p103000010062"></a><a name="zh-cn_topic_0000002600000301_p103000010062"></a>数据类型：Callable[[torch.nn.Module], Any] 或 None。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002600000301_row103000010014"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000301_p103000010063"><a name="zh-cn_topic_0000002600000301_p103000010063"></a><a name="zh-cn_topic_0000002600000301_p103000010063"></a>quant_fn</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000301_p103000010064"><a name="zh-cn_topic_0000002600000301_p103000010064"></a><a name="zh-cn_topic_0000002600000301_p103000010064"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000301_p103000010065"><a name="zh-cn_topic_0000002600000301_p103000010065"></a><a name="zh-cn_topic_0000002600000301_p103000010065"></a>含义：可选的量化感知回调，在评估前应用，使搜索在量化状态下决策。</p>
<p id="zh-cn_topic_0000002600000301_p103000010066"><a name="zh-cn_topic_0000002600000301_p103000010066"></a><a name="zh-cn_topic_0000002600000301_p103000010066"></a>默认值：None。</p>
<p id="zh-cn_topic_0000002600000301_p103000010067"><a name="zh-cn_topic_0000002600000301_p103000010067"></a><a name="zh-cn_topic_0000002600000301_p103000010067"></a>数据类型：Callable[[torch.nn.Module], Any] 或 None。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002600000301_row103000010015"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000301_p103000010068"><a name="zh-cn_topic_0000002600000301_p103000010068"></a><a name="zh-cn_topic_0000002600000301_p103000010068"></a>apply</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000301_p103000010069"><a name="zh-cn_topic_0000002600000301_p103000010069"></a><a name="zh-cn_topic_0000002600000301_p103000010069"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000301_p103000010070"><a name="zh-cn_topic_0000002600000301_p103000010070"></a><a name="zh-cn_topic_0000002600000301_p103000010070"></a>含义：True 时将选定的剪枝率原地应用于模型；False 时仅搜索不应用。</p>
<p id="zh-cn_topic_0000002600000301_p103000010071"><a name="zh-cn_topic_0000002600000301_p103000010071"></a><a name="zh-cn_topic_0000002600000301_p103000010071"></a>默认值：True。</p>
<p id="zh-cn_topic_0000002600000301_p103000010072"><a name="zh-cn_topic_0000002600000301_p103000010072"></a><a name="zh-cn_topic_0000002600000301_p103000010072"></a>数据类型：bool。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002600000301_row103000010016"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000301_p103000010073"><a name="zh-cn_topic_0000002600000301_p103000010073"></a><a name="zh-cn_topic_0000002600000301_p103000010073"></a>budget_scope</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000301_p103000010074"><a name="zh-cn_topic_0000002600000301_p103000010074"></a><a name="zh-cn_topic_0000002600000301_p103000010074"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000301_p103000010075"><a name="zh-cn_topic_0000002600000301_p103000010075"></a><a name="zh-cn_topic_0000002600000301_p103000010075"></a>含义：预算口径。'total'（默认，预算覆盖全部参数）或 'prunable'（预算仅覆盖可剪枝质量，排除 embedding 等）；当预算低于不可剪枝质量下界时告警并记录 budget_unreachable=True，不抛异常。</p>
<p id="zh-cn_topic_0000002600000301_p103000010076"><a name="zh-cn_topic_0000002600000301_p103000010076"></a><a name="zh-cn_topic_0000002600000301_p103000010076"></a>默认值："total"。</p>
<p id="zh-cn_topic_0000002600000301_p103000010077"><a name="zh-cn_topic_0000002600000301_p103000010077"></a><a name="zh-cn_topic_0000002600000301_p103000010077"></a>数据类型：str。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000002600000301_section103000010007"></a>

AutoTuneResult。其中 weight_reduction 为实际剪掉的比例，quality_drop 为该尺寸下实测的精度损失，tolerance 固定为 -1（非容差模式），chosen_ratio 在无任何剪枝率满足预算时为 None；budget_unreachable 标记预算是否低于不可剪枝质量下界，prunable_fraction 为可剪枝质量占总参数的比例。

## 调用示例<a name="zh-cn_topic_0000002600000301_section103000010008"></a>

```python
import amct_pytorch as amct
from amct_pytorch.pruning import size_budget_prune

# 准备待剪枝模型与校准数据
# model = ...
# calib = ...

# 按保留 70% 参数的目标预算原地剪枝
result = size_budget_prune(model, data=calib, target_keep_ratio=0.7)
print(result.summary())
```
