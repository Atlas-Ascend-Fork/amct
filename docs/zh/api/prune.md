# prune<a name="ZH-CN_TOPIC_0000002600000201"></a>

## 产品支持情况<a name="zh-cn_topic_0000002600000201_section102000010001"></a>

<a name="zh-cn_topic_0000002600000201_table102000010002"></a>

| 产品                                        | 是否支持 |
| ------------------------------------------- | -------- |
| Ascend 950PR/Ascend 950DT                   | √        |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √        |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √        |



## 功能说明<a name="zh-cn_topic_0000002600000201_section102000010003"></a>

结构化剪枝主入口，对 torch.nn.Module 原地剪枝（dense FFN 中间维 / CNN 通道 / MoE 专家），并同步改写模型 `config` 维度（intermediate_size / num_experts）。传入 tolerance 时按精度损失上界自动搜索最大剪枝率；传入 size_budget 时按尺寸预算剪枝，二者互斥。

## 函数原型<a name="zh-cn_topic_0000002600000201_section102000010004"></a>

```python
prune(model, config=None, data=None, batch_adapter=None, tolerance=None,
      evaluator=None, eval_iterations=1, eval_data=None, ratio_grid=None,
      size_budget=None, finetune_fn=None, quant_fn=None, report=None)
```

## 参数说明<a name="zh-cn_topic_0000002600000201_section102000010005"></a>

<a name="zh-cn_topic_0000002600000201_table102000010006"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002600000201_row102000010001"><th class="cellrowborder" valign="top" width="13.78%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000002600000201_p102000010001"><a name="zh-cn_topic_0000002600000201_p102000010001"></a><a name="zh-cn_topic_0000002600000201_p102000010001"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="8.01%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000002600000201_p102000010002"><a name="zh-cn_topic_0000002600000201_p102000010002"></a><a name="zh-cn_topic_0000002600000201_p102000010002"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="78.21%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000002600000201_p102000010003"><a name="zh-cn_topic_0000002600000201_p102000010003"></a><a name="zh-cn_topic_0000002600000201_p102000010003"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002600000201_row102000010002"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000201_p102000010004"><a name="zh-cn_topic_0000002600000201_p102000010004"></a><a name="zh-cn_topic_0000002600000201_p102000010004"></a>model</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000201_p102000010005"><a name="zh-cn_topic_0000002600000201_p102000010005"></a><a name="zh-cn_topic_0000002600000201_p102000010005"></a>输入/输出</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000201_p102000010006"><a name="zh-cn_topic_0000002600000201_p102000010006"></a><a name="zh-cn_topic_0000002600000201_p102000010006"></a>含义：待剪枝模型，调用方负责实例化（如 from_pretrained）；原地剪枝。</p>
<p id="zh-cn_topic_0000002600000201_p102000010007"><a name="zh-cn_topic_0000002600000201_p102000010007"></a><a name="zh-cn_topic_0000002600000201_p102000010007"></a>数据类型：torch.nn.Module。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002600000201_row102000010003"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000201_p102000010008"><a name="zh-cn_topic_0000002600000201_p102000010008"></a><a name="zh-cn_topic_0000002600000201_p102000010008"></a>config</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000201_p102000010009"><a name="zh-cn_topic_0000002600000201_p102000010009"></a><a name="zh-cn_topic_0000002600000201_p102000010009"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000201_p102000010010"><a name="zh-cn_topic_0000002600000201_p102000010010"></a><a name="zh-cn_topic_0000002600000201_p102000010010"></a>含义：剪枝配置，覆盖各域（cnn/dense/moe）默认剪枝率与方法；None 时使用默认 PruneConfig。</p>
<p id="zh-cn_topic_0000002600000201_p102000010011"><a name="zh-cn_topic_0000002600000201_p102000010011"></a><a name="zh-cn_topic_0000002600000201_p102000010011"></a>默认值：None。</p>
<p id="zh-cn_topic_0000002600000201_p102000010012"><a name="zh-cn_topic_0000002600000201_p102000010012"></a><a name="zh-cn_topic_0000002600000201_p102000010012"></a>数据类型：dict / PruneConfig / None。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002600000201_row102000010004"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000201_p102000010013"><a name="zh-cn_topic_0000002600000201_p102000010013"></a><a name="zh-cn_topic_0000002600000201_p102000010013"></a>data</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000201_p102000010014"><a name="zh-cn_topic_0000002600000201_p102000010014"></a><a name="zh-cn_topic_0000002600000201_p102000010014"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000201_p102000010015"><a name="zh-cn_topic_0000002600000201_p102000010015"></a><a name="zh-cn_topic_0000002600000201_p102000010015"></a>含义：校准数据（DataLoader / tensor / dict），方差类方法必需，也作默认评估集。</p>
<p id="zh-cn_topic_0000002600000201_p102000010016"><a name="zh-cn_topic_0000002600000201_p102000010016"></a><a name="zh-cn_topic_0000002600000201_p102000010016"></a>默认值：None。</p>
<p id="zh-cn_topic_0000002600000201_p102000010017"><a name="zh-cn_topic_0000002600000201_p102000010017"></a><a name="zh-cn_topic_0000002600000201_p102000010017"></a>数据类型：Any。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002600000201_row102000010005"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000201_p102000010018"><a name="zh-cn_topic_0000002600000201_p102000010018"></a><a name="zh-cn_topic_0000002600000201_p102000010018"></a>batch_adapter</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000201_p102000010019"><a name="zh-cn_topic_0000002600000201_p102000010019"></a><a name="zh-cn_topic_0000002600000201_p102000010019"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000201_p102000010020"><a name="zh-cn_topic_0000002600000201_p102000010020"></a><a name="zh-cn_topic_0000002600000201_p102000010020"></a>含义：将自定义 batch 解包为前向 args、kwargs 的回调。</p>
<p id="zh-cn_topic_0000002600000201_p102000010021"><a name="zh-cn_topic_0000002600000201_p102000010021"></a><a name="zh-cn_topic_0000002600000201_p102000010021"></a>默认值：None。</p>
<p id="zh-cn_topic_0000002600000201_p102000010022"><a name="zh-cn_topic_0000002600000201_p102000010022"></a><a name="zh-cn_topic_0000002600000201_p102000010022"></a>数据类型：Callable / None。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002600000201_row102000010006"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000201_p102000010023"><a name="zh-cn_topic_0000002600000201_p102000010023"></a><a name="zh-cn_topic_0000002600000201_p102000010023"></a>tolerance</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000201_p102000010024"><a name="zh-cn_topic_0000002600000201_p102000010024"></a><a name="zh-cn_topic_0000002600000201_p102000010024"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000201_p102000010025"><a name="zh-cn_topic_0000002600000201_p102000010025"></a><a name="zh-cn_topic_0000002600000201_p102000010025"></a>含义：可接受的精度损失上界；给定时开启精度驱动自动剪枝，在 ratio_grid 上搜索容差内的最大剪枝率。</p>
<p id="zh-cn_topic_0000002600000201_p102000010026"><a name="zh-cn_topic_0000002600000201_p102000010026"></a><a name="zh-cn_topic_0000002600000201_p102000010026"></a>默认值：None（固定剪枝率模式）。</p>
<p id="zh-cn_topic_0000002600000201_p102000010027"><a name="zh-cn_topic_0000002600000201_p102000010027"></a><a name="zh-cn_topic_0000002600000201_p102000010027"></a>数据类型：float / None。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002600000201_row102000010007"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000201_p102000010028"><a name="zh-cn_topic_0000002600000201_p102000010028"></a><a name="zh-cn_topic_0000002600000201_p102000010028"></a>evaluator</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000201_p102000010029"><a name="zh-cn_topic_0000002600000201_p102000010029"></a><a name="zh-cn_topic_0000002600000201_p102000010029"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000201_p102000010030"><a name="zh-cn_topic_0000002600000201_p102000010030"></a><a name="zh-cn_topic_0000002600000201_p102000010030"></a>含义：精度来源，与量化共用 —— Callable[[model], float] 或任何暴露 evaluate(model) 并返回标量的对象；None 时使用默认 top-1 保真度评估。</p>
<p id="zh-cn_topic_0000002600000201_p102000010031"><a name="zh-cn_topic_0000002600000201_p102000010031"></a><a name="zh-cn_topic_0000002600000201_p102000010031"></a>默认值：None。</p>
<p id="zh-cn_topic_0000002600000201_p102000010032"><a name="zh-cn_topic_0000002600000201_p102000010032"></a><a name="zh-cn_topic_0000002600000201_p102000010032"></a>数据类型：Callable / 暴露 evaluate(model) 的对象 / None。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002600000201_row102000010008"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000201_p102000010033"><a name="zh-cn_topic_0000002600000201_p102000010033"></a><a name="zh-cn_topic_0000002600000201_p102000010033"></a>eval_iterations</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000201_p102000010034"><a name="zh-cn_topic_0000002600000201_p102000010034"></a><a name="zh-cn_topic_0000002600000201_p102000010034"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000201_p102000010035"><a name="zh-cn_topic_0000002600000201_p102000010035"></a><a name="zh-cn_topic_0000002600000201_p102000010035"></a>含义：容差模式下的评估迭代次数。</p>
<p id="zh-cn_topic_0000002600000201_p102000010036"><a name="zh-cn_topic_0000002600000201_p102000010036"></a><a name="zh-cn_topic_0000002600000201_p102000010036"></a>默认值：1。</p>
<p id="zh-cn_topic_0000002600000201_p102000010037"><a name="zh-cn_topic_0000002600000201_p102000010037"></a><a name="zh-cn_topic_0000002600000201_p102000010037"></a>数据类型：int。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002600000201_row102000010009"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000201_p102000010038"><a name="zh-cn_topic_0000002600000201_p102000010038"></a><a name="zh-cn_topic_0000002600000201_p102000010038"></a>eval_data</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000201_p102000010039"><a name="zh-cn_topic_0000002600000201_p102000010039"></a><a name="zh-cn_topic_0000002600000201_p102000010039"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000201_p102000010040"><a name="zh-cn_topic_0000002600000201_p102000010040"></a><a name="zh-cn_topic_0000002600000201_p102000010040"></a>含义：容差模式下传入自动剪枝的评估数据；默认回退到 data。</p>
<p id="zh-cn_topic_0000002600000201_p102000010041"><a name="zh-cn_topic_0000002600000201_p102000010041"></a><a name="zh-cn_topic_0000002600000201_p102000010041"></a>默认值：None。</p>
<p id="zh-cn_topic_0000002600000201_p102000010042"><a name="zh-cn_topic_0000002600000201_p102000010042"></a><a name="zh-cn_topic_0000002600000201_p102000010042"></a>数据类型：Any。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002600000201_row102000010010"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000201_p102000010043"><a name="zh-cn_topic_0000002600000201_p102000010043"></a><a name="zh-cn_topic_0000002600000201_p102000010043"></a>ratio_grid</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000201_p102000010044"><a name="zh-cn_topic_0000002600000201_p102000010044"></a><a name="zh-cn_topic_0000002600000201_p102000010044"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000201_p102000010045"><a name="zh-cn_topic_0000002600000201_p102000010045"></a><a name="zh-cn_topic_0000002600000201_p102000010045"></a>含义：容差模式下的候选剪枝率（升序）；None 时使用默认网格。</p>
<p id="zh-cn_topic_0000002600000201_p102000010046"><a name="zh-cn_topic_0000002600000201_p102000010046"></a><a name="zh-cn_topic_0000002600000201_p102000010046"></a>默认值：None。</p>
<p id="zh-cn_topic_0000002600000201_p102000010047"><a name="zh-cn_topic_0000002600000201_p102000010047"></a><a name="zh-cn_topic_0000002600000201_p102000010047"></a>数据类型：Any / None。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002600000201_row102000010011"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000201_p102000010048"><a name="zh-cn_topic_0000002600000201_p102000010048"></a><a name="zh-cn_topic_0000002600000201_p102000010048"></a>size_budget</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000201_p102000010049"><a name="zh-cn_topic_0000002600000201_p102000010049"></a><a name="zh-cn_topic_0000002600000201_p102000010049"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000201_p102000010050"><a name="zh-cn_topic_0000002600000201_p102000010050"></a><a name="zh-cn_topic_0000002600000201_p102000010050"></a>含义：给定时（保留比例，(0,1]）按尺寸预算剪枝；与 tolerance 互斥。</p>
<p id="zh-cn_topic_0000002600000201_p102000010051"><a name="zh-cn_topic_0000002600000201_p102000010051"></a><a name="zh-cn_topic_0000002600000201_p102000010051"></a>默认值：None。</p>
<p id="zh-cn_topic_0000002600000201_p102000010052"><a name="zh-cn_topic_0000002600000201_p102000010052"></a><a name="zh-cn_topic_0000002600000201_p102000010052"></a>数据类型：float / None。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002600000201_row102000010012"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000201_p102000010053"><a name="zh-cn_topic_0000002600000201_p102000010053"></a><a name="zh-cn_topic_0000002600000201_p102000010053"></a>finetune_fn</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000201_p102000010054"><a name="zh-cn_topic_0000002600000201_p102000010054"></a><a name="zh-cn_topic_0000002600000201_p102000010054"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000201_p102000010055"><a name="zh-cn_topic_0000002600000201_p102000010055"></a><a name="zh-cn_topic_0000002600000201_p102000010055"></a>含义：容差模式下对每个候选率剪枝后做（轻量）微调恢复再评估的回调，使搜索能选出更激进的可用率。</p>
<p id="zh-cn_topic_0000002600000201_p102000010056"><a name="zh-cn_topic_0000002600000201_p102000010056"></a><a name="zh-cn_topic_0000002600000201_p102000010056"></a>默认值：None。</p>
<p id="zh-cn_topic_0000002600000201_p102000010057"><a name="zh-cn_topic_0000002600000201_p102000010057"></a><a name="zh-cn_topic_0000002600000201_p102000010057"></a>数据类型：Callable[[nn.Module], Any] / None。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002600000201_row102000010013"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000201_p102000010058"><a name="zh-cn_topic_0000002600000201_p102000010058"></a><a name="zh-cn_topic_0000002600000201_p102000010058"></a>quant_fn</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000201_p102000010059"><a name="zh-cn_topic_0000002600000201_p102000010059"></a><a name="zh-cn_topic_0000002600000201_p102000010059"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000201_p102000010060"><a name="zh-cn_topic_0000002600000201_p102000010060"></a><a name="zh-cn_topic_0000002600000201_p102000010060"></a>含义：容差搜索过程中对模型做量化的回调（量化感知搜索）。</p>
<p id="zh-cn_topic_0000002600000201_p102000010061"><a name="zh-cn_topic_0000002600000201_p102000010061"></a><a name="zh-cn_topic_0000002600000201_p102000010061"></a>默认值：None。</p>
<p id="zh-cn_topic_0000002600000201_p102000010062"><a name="zh-cn_topic_0000002600000201_p102000010062"></a><a name="zh-cn_topic_0000002600000201_p102000010062"></a>数据类型：Callable[[nn.Module], Any] / None。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002600000201_row102000010014"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000201_p102000010063"><a name="zh-cn_topic_0000002600000201_p102000010063"></a><a name="zh-cn_topic_0000002600000201_p102000010063"></a>report</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000201_p102000010064"><a name="zh-cn_topic_0000002600000201_p102000010064"></a><a name="zh-cn_topic_0000002600000201_p102000010064"></a>输入/输出</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000201_p102000010065"><a name="zh-cn_topic_0000002600000201_p102000010065"></a><a name="zh-cn_topic_0000002600000201_p102000010065"></a>含义：调用方提供的统计 sink；传入后剪枝统计写入其中，事后通过 <code>report.as_dict()</code> 读取（无进程级全局状态）。</p>
<p id="zh-cn_topic_0000002600000201_p102000010066"><a name="zh-cn_topic_0000002600000201_p102000010066"></a><a name="zh-cn_topic_0000002600000201_p102000010066"></a>默认值：None。</p>
<p id="zh-cn_topic_0000002600000201_p102000010067"><a name="zh-cn_topic_0000002600000201_p102000010067"></a><a name="zh-cn_topic_0000002600000201_p102000010067"></a>数据类型：PruneReport（from amct_pytorch.pruning import）/ None。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000002600000201_section102000010007"></a>

None。剪枝为原地操作。剪枝统计通过调用方提供的 `report=PruneReport()` sink 获取（事后读 `report.as_dict()`，含 backend、params_before、params_after、per_layer_sparsity、warnings、budget_unreachable 等字段）（无进程级全局状态）。

## 支持的剪枝结构及约束<a name="zh-cn_topic_0000002600000201_section102000010009"></a>

仅当 producer<->consumer 接口可被验证时才剪枝，其余结构跳过。

<a name="zh-cn_topic_0000002600000201_table102000010010"></a>

<table><thead align="left"><tr id="row102001010001"><th class="cellrowborder" valign="top" width="13%" id="mcps1.2.4.1.1"><p id="p102001010001"><a name="p102001010001"></a><a name="p102001010001"></a>域</p>
</th>
<th class="cellrowborder" valign="top" width="42%" id="mcps1.2.4.1.2"><p id="p102001010002"><a name="p102001010002"></a><a name="p102001010002"></a>支持的结构</p>
</th>
<th class="cellrowborder" valign="top" width="45%" id="mcps1.2.4.1.3"><p id="p102001010003"><a name="p102001010003"></a><a name="p102001010003"></a>约束/自动排除</p>
</th>
</tr>
</thead>
<tbody><tr id="row102001010004"><td class="cellrowborder" valign="top" width="13%" headers="mcps1.2.4.1.1 "><p id="p102001010004"><a name="p102001010004"></a><a name="p102001010004"></a>dense</p>
</td>
<td class="cellrowborder" valign="top" width="42%" headers="mcps1.2.4.1.2 "><p id="p102001010005"><a name="p102001010005"></a><a name="p102001010005"></a>三 Linear 结构 gate/up/down_proj；融合 gate_up_proj（Phi-3/GLM-4）；相邻两 Linear/Conv1D（含 Bloom 风格）。Llama/Qwen2/Mistral/Qwen3 无需手动 skip_layers 即可剪枝。</p>
</td>
<td class="cellrowborder" valign="top" width="45%" headers="mcps1.2.4.1.3 "><p id="p102001010006"><a name="p102001010006"></a><a name="p102001010006"></a>仅剪中间维（gate/up.out_features + down.in_features），hidden/residual 宽度保持不变；注意力投影 q/k/v/o 自动排除。</p>
</td>
</tr>
<tr id="row102001010007"><td class="cellrowborder" valign="top" width="13%" headers="mcps1.2.4.1.1 "><p id="p102001010007"><a name="p102001010007"></a><a name="p102001010007"></a>cnn</p>
</td>
<td class="cellrowborder" valign="top" width="42%" headers="mcps1.2.4.1.2 "><p id="p102001010008"><a name="p102001010008"></a><a name="p102001010008"></a>producer Conv2d（groups=1）-> 可选 BatchNorm2d -> 通道匹配的 consumer Conv2d/Linear，同步缩放。</p>
</td>
<td class="cellrowborder" valign="top" width="45%" headers="mcps1.2.4.1.3 "><p id="p102001010009"><a name="p102001010009"></a><a name="p102001010009"></a>residual add 直接馈入的 conv 自动排除；Concat consumer、grouped/depthwise 卷积不支持剪枝（0 targets）。</p>
</td>
</tr>
<tr id="row102001010010"><td class="cellrowborder" valign="top" width="13%" headers="mcps1.2.4.1.1 "><p id="p102001010010"><a name="p102001010010"></a><a name="p102001010010"></a>moe</p>
</td>
<td class="cellrowborder" valign="top" width="42%" headers="mcps1.2.4.1.2 "><p id="p102001010011"><a name="p102001010011"></a><a name="p102001010011"></a>nn.ModuleList experts + nn.Linear gate；融合批量 experts（MixtralExperts/Qwen3MoeExperts + *TopKRouter）；grouped router（n_group/topk_group）；共享专家 + sigmoid 路由（noaux_tc）；同级两 tensor 融合 experts（GraniteMoE）；嵌套 router-bias（Ernie4.5）。</p>
</td>
<td class="cellrowborder" valign="top" width="45%" headers="mcps1.2.4.1.3 "><p id="p102001010012"><a name="p102001010012"></a><a name="p102001010012"></a>整体移除路由专家，hidden in/out 不变；shared（常驻）专家自动排除（其 FFN 仍属 dense 域，只剪 MoE 时需显式将 dense 固定为 prune_ratio: 0.0）。</p>
</td>
</tr>
</tbody>
</table>

## 调用示例<a name="zh-cn_topic_0000002600000201_section102000010008"></a>

```python
import amct_pytorch as amct
from amct_pytorch.pruning import PruneReport

# 固定剪枝率：dense 域中间维剪 50%（自动避开注意力投影）
cfg = {"methods": {"dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.5}}}}
rep = PruneReport()
amct.prune(model, cfg, data=calib, report=rep)
print(rep.as_dict())

# 容差驱动：最多损失 2%，自动搜索并应用满足容差的最大剪枝率
rep2 = PruneReport()
amct.prune(model, data=calib, tolerance=0.02, report=rep2)
print(rep2.as_dict())
```

## 预定义配置<a name="zh-cn_topic_0000002600000201_section102000010010"></a>

开箱即用的预设配置（dict），直接作为 `config` 参数传入。各预设针对不同网络结构（CNN、Dense、MoE）封装了对应的剪枝方法与超参。下表中的常量均在 `amct_pytorch` 顶层导出，可直接 `amct.<常量名>` 引用。

<a name="zh-cn_topic_0000002600000201_table102000010011"></a>

| 配置常量 | 适用域 | 关键 kwargs（默认值） | 说明 |
| -------- | ------ | -------------------- | ---- |
| CNN_RECONSTRUCT_PRUNE_CFG | cnn | prune_ratio=0.30 | CNN 卷积通道重建剪枝，剪枝后对下游权重做最小二乘重建。 |
| CNN_VARIANCE_PRUNE_CFG | cnn | prune_ratio=0.30 | CNN 卷积通道方差剪枝，按激活方差移除信息量最低的通道。 |
| DENSE_LOWVAR_PRUNE_CFG | dense | prune_ratio=0.50 | Dense（全连接）低方差剪枝，移除方差最低的神经元。 |
| MOE_MASSVAR_PRUNE_CFG | moe | prune_ratio=0.50, boundary=10 | MoE 质量-方差（mass_variance）剪枝，结合权重质量与方差，boundary 控制边界。 |
| MOE_OUTPUT_MERGE_PRUNE_CFG | moe | keep_ratio=0.50, selector="calib_nll" | MoE 输出空间专家合并，以 calib-NLL 选择器在合并与 drop 之间择优。 |
| SENSITIVITY_ALLOC_PRUNE_CFG | dense | prune_ratio=0.50；allocation: strategy="sensitivity", ref_ratio=0.50, min_ratio=0.05, max_ratio=0.90, guard="calib_nll" | 按逐层敏感度分配剪枝率（稳健层多剪、敏感层少剪），以 calib_nll 守卫回退到均匀分配；需传入 data=。 |
| FULL_STRUCTURED_PRUNE_CFG | cnn + dense + moe | cnn: prune_ratio=0.30；dense: prune_ratio=0.50；moe: prune_ratio=0.50, boundary=10 | 一次性对 CNN、Dense、MoE 三种结构同时做结构化剪枝的组合预设。 |

> 说明：表中「适用域」即该预置在 `methods` 里列出的域，其余域自动固定为 `prune_ratio: 0.0`，不会被顺带剪枝。以上常量均开启 `"missing_data_policy": "warn_skip"`（缺校准数据时告警并跳过对应方法）。`MOE_VARIANCE_MENU_CFG`、`DENSE_RECOVERY_MENU_CFG`、`CNN_RECOVERY_MENU_CFG` 为守卫式菜单模式（menu）专用配置：`prune` 检测到配置带 `menu` 即改走菜单择优，在 `eval_data` 上实测各候选后择优应用，无需额外开关。三者均从 `amct_pytorch.pruning` 导入，不在顶层 `amct_pytorch` 导出。

```python
# 直接传入预定义剪枝配置作为第二个参数
amct.prune(model, amct.DENSE_LOWVAR_PRUNE_CFG, data=calib)
```
