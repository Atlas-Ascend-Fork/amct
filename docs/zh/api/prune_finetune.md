# prune\_finetune<a name="ZH-CN_TOPIC_0000002600000401"></a>

## 产品支持情况<a name="zh-cn_topic_0000002600000401_section104000010001"></a>

<a name="zh-cn_topic_0000002600000401_table104000010002"></a>

| 产品                                        | 是否支持 |
| ------------------------------------------- | -------- |
| Ascend 950PR/Ascend 950DT                   | √        |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √        |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √        |



## 功能说明<a name="zh-cn_topic_0000002600000401_section104000010003"></a>

对模型做就地（in-place）的少量梯度步微调，以恢复结构化剪枝损失的精度。可直接调用，也可包装为 `finetune_fn=lambda m: prune_finetune(m, train_data, ...)` 传给 `amct.prune` / 自动剪枝接口。

## 函数原型<a name="zh-cn_topic_0000002600000401_section104000010004"></a>

```python
prune_finetune(model, data, loss_fn=None, steps=200, lr=1e-4, weight_decay=0.0, warmup=0, optimizer=None, batch_adapter=None, device=None, grad_clip=1.0, log_every=0)
```

## 参数说明<a name="zh-cn_topic_0000002600000401_section104000010005"></a>

<a name="zh-cn_topic_0000002600000401_table104000010006"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002600000401_row104000010001"><th class="cellrowborder" valign="top" width="13.78%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000002600000401_p104000010001"><a name="zh-cn_topic_0000002600000401_p104000010001"></a><a name="zh-cn_topic_0000002600000401_p104000010001"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="8.01%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000002600000401_p104000010002"><a name="zh-cn_topic_0000002600000401_p104000010002"></a><a name="zh-cn_topic_0000002600000401_p104000010002"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="78.21%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000002600000401_p104000010003"><a name="zh-cn_topic_0000002600000401_p104000010003"></a><a name="zh-cn_topic_0000002600000401_p104000010003"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002600000401_row104000010002"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000401_p104000010004"><a name="zh-cn_topic_0000002600000401_p104000010004"></a><a name="zh-cn_topic_0000002600000401_p104000010004"></a>model</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000401_p104000010005"><a name="zh-cn_topic_0000002600000401_p104000010005"></a><a name="zh-cn_topic_0000002600000401_p104000010005"></a>输入/输出</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000401_p104000010006"><a name="zh-cn_topic_0000002600000401_p104000010006"></a><a name="zh-cn_topic_0000002600000401_p104000010006"></a>含义：待微调的模型，权重就地（in-place）更新。</p>
<p id="zh-cn_topic_0000002600000401_p104000010007"><a name="zh-cn_topic_0000002600000401_p104000010007"></a><a name="zh-cn_topic_0000002600000401_p104000010007"></a>数据类型：torch.nn.Module。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002600000401_row104000010003"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000401_p104000010008"><a name="zh-cn_topic_0000002600000401_p104000010008"></a><a name="zh-cn_topic_0000002600000401_p104000010008"></a>data</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000401_p104000010009"><a name="zh-cn_topic_0000002600000401_p104000010009"></a><a name="zh-cn_topic_0000002600000401_p104000010009"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000401_p104000010010"><a name="zh-cn_topic_0000002600000401_p104000010010"></a><a name="zh-cn_topic_0000002600000401_p104000010010"></a>含义：训练批序列，循环复用直至达到 <code>steps</code> 指定的步数。</p>
<p id="zh-cn_topic_0000002600000401_p104000010011"><a name="zh-cn_topic_0000002600000401_p104000010011"></a><a name="zh-cn_topic_0000002600000401_p104000010011"></a>数据类型：Sequence[Any]。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002600000401_row104000010004"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000401_p104000010012"><a name="zh-cn_topic_0000002600000401_p104000010012"></a><a name="zh-cn_topic_0000002600000401_p104000010012"></a>loss_fn</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000401_p104000010013"><a name="zh-cn_topic_0000002600000401_p104000010013"></a><a name="zh-cn_topic_0000002600000401_p104000010013"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000401_p104000010014"><a name="zh-cn_topic_0000002600000401_p104000010014"></a><a name="zh-cn_topic_0000002600000401_p104000010014"></a>含义：自定义损失函数 <code>loss_fn(model, batch) -> Tensor</code>；为 None 时使用内置默认损失（dict 含 <code>input_ids</code> 走因果 LM 自监督损失，<code>(x, y)</code> 二元组走交叉熵分类损失）。</p>
<p id="zh-cn_topic_0000002600000401_p104000010015"><a name="zh-cn_topic_0000002600000401_p104000010015"></a><a name="zh-cn_topic_0000002600000401_p104000010015"></a>默认值：None。</p>
<p id="zh-cn_topic_0000002600000401_p104000010016"><a name="zh-cn_topic_0000002600000401_p104000010016"></a><a name="zh-cn_topic_0000002600000401_p104000010016"></a>数据类型：Callable[[nn.Module, Any], torch.Tensor] 或 None。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002600000401_row104000010005"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000401_p104000010017"><a name="zh-cn_topic_0000002600000401_p104000010017"></a><a name="zh-cn_topic_0000002600000401_p104000010017"></a>steps</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000401_p104000010018"><a name="zh-cn_topic_0000002600000401_p104000010018"></a><a name="zh-cn_topic_0000002600000401_p104000010018"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000401_p104000010019"><a name="zh-cn_topic_0000002600000401_p104000010019"></a><a name="zh-cn_topic_0000002600000401_p104000010019"></a>含义：梯度更新步数；≤0 时直接返回空结果不做训练。</p>
<p id="zh-cn_topic_0000002600000401_p104000010020"><a name="zh-cn_topic_0000002600000401_p104000010020"></a><a name="zh-cn_topic_0000002600000401_p104000010020"></a>默认值：200。</p>
<p id="zh-cn_topic_0000002600000401_p104000010021"><a name="zh-cn_topic_0000002600000401_p104000010021"></a><a name="zh-cn_topic_0000002600000401_p104000010021"></a>数据类型：int。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002600000401_row104000010006"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000401_p104000010022"><a name="zh-cn_topic_0000002600000401_p104000010022"></a><a name="zh-cn_topic_0000002600000401_p104000010022"></a>lr</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000401_p104000010023"><a name="zh-cn_topic_0000002600000401_p104000010023"></a><a name="zh-cn_topic_0000002600000401_p104000010023"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000401_p104000010024"><a name="zh-cn_topic_0000002600000401_p104000010024"></a><a name="zh-cn_topic_0000002600000401_p104000010024"></a>含义：默认 AdamW 优化器的学习率。</p>
<p id="zh-cn_topic_0000002600000401_p104000010025"><a name="zh-cn_topic_0000002600000401_p104000010025"></a><a name="zh-cn_topic_0000002600000401_p104000010025"></a>默认值：1e-4。</p>
<p id="zh-cn_topic_0000002600000401_p104000010026"><a name="zh-cn_topic_0000002600000401_p104000010026"></a><a name="zh-cn_topic_0000002600000401_p104000010026"></a>数据类型：float。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002600000401_row104000010007"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000401_p104000010027"><a name="zh-cn_topic_0000002600000401_p104000010027"></a><a name="zh-cn_topic_0000002600000401_p104000010027"></a>weight_decay</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000401_p104000010028"><a name="zh-cn_topic_0000002600000401_p104000010028"></a><a name="zh-cn_topic_0000002600000401_p104000010028"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000401_p104000010029"><a name="zh-cn_topic_0000002600000401_p104000010029"></a><a name="zh-cn_topic_0000002600000401_p104000010029"></a>含义：默认 AdamW 优化器的权重衰减。</p>
<p id="zh-cn_topic_0000002600000401_p104000010030"><a name="zh-cn_topic_0000002600000401_p104000010030"></a><a name="zh-cn_topic_0000002600000401_p104000010030"></a>默认值：0.0。</p>
<p id="zh-cn_topic_0000002600000401_p104000010031"><a name="zh-cn_topic_0000002600000401_p104000010031"></a><a name="zh-cn_topic_0000002600000401_p104000010031"></a>数据类型：float。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002600000401_row104000010008"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000401_p104000010032"><a name="zh-cn_topic_0000002600000401_p104000010032"></a><a name="zh-cn_topic_0000002600000401_p104000010032"></a>warmup</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000401_p104000010033"><a name="zh-cn_topic_0000002600000401_p104000010033"></a><a name="zh-cn_topic_0000002600000401_p104000010033"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000401_p104000010034"><a name="zh-cn_topic_0000002600000401_p104000010034"></a><a name="zh-cn_topic_0000002600000401_p104000010034"></a>含义：线性 warmup 步数，前 <code>warmup</code> 步内学习率从 0 线性升至 <code>lr</code>。</p>
<p id="zh-cn_topic_0000002600000401_p104000010035"><a name="zh-cn_topic_0000002600000401_p104000010035"></a><a name="zh-cn_topic_0000002600000401_p104000010035"></a>默认值：0。</p>
<p id="zh-cn_topic_0000002600000401_p104000010036"><a name="zh-cn_topic_0000002600000401_p104000010036"></a><a name="zh-cn_topic_0000002600000401_p104000010036"></a>数据类型：int。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002600000401_row104000010009"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000401_p104000010037"><a name="zh-cn_topic_0000002600000401_p104000010037"></a><a name="zh-cn_topic_0000002600000401_p104000010037"></a>optimizer</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000401_p104000010038"><a name="zh-cn_topic_0000002600000401_p104000010038"></a><a name="zh-cn_topic_0000002600000401_p104000010038"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000401_p104000010039"><a name="zh-cn_topic_0000002600000401_p104000010039"></a><a name="zh-cn_topic_0000002600000401_p104000010039"></a>含义：自定义 torch 优化器；为 None 时基于可训练参数自动构建 AdamW。</p>
<p id="zh-cn_topic_0000002600000401_p104000010040"><a name="zh-cn_topic_0000002600000401_p104000010040"></a><a name="zh-cn_topic_0000002600000401_p104000010040"></a>默认值：None。</p>
<p id="zh-cn_topic_0000002600000401_p104000010041"><a name="zh-cn_topic_0000002600000401_p104000010041"></a><a name="zh-cn_topic_0000002600000401_p104000010041"></a>数据类型：torch.optim.Optimizer 或 None。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002600000401_row104000010010"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000401_p104000010042"><a name="zh-cn_topic_0000002600000401_p104000010042"></a><a name="zh-cn_topic_0000002600000401_p104000010042"></a>batch_adapter</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000401_p104000010043"><a name="zh-cn_topic_0000002600000401_p104000010043"></a><a name="zh-cn_topic_0000002600000401_p104000010043"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000401_p104000010044"><a name="zh-cn_topic_0000002600000401_p104000010044"></a><a name="zh-cn_topic_0000002600000401_p104000010044"></a>含义：批适配器，将一个 batch 映射为 <code>(args, kwargs)</code> 供模型前向；默认损失会用其输出的 <code>.loss</code>。</p>
<p id="zh-cn_topic_0000002600000401_p104000010045"><a name="zh-cn_topic_0000002600000401_p104000010045"></a><a name="zh-cn_topic_0000002600000401_p104000010045"></a>默认值：None。</p>
<p id="zh-cn_topic_0000002600000401_p104000010046"><a name="zh-cn_topic_0000002600000401_p104000010046"></a><a name="zh-cn_topic_0000002600000401_p104000010046"></a>数据类型：BatchAdapter 或 None。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002600000401_row104000010011"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000401_p104000010047"><a name="zh-cn_topic_0000002600000401_p104000010047"></a><a name="zh-cn_topic_0000002600000401_p104000010047"></a>device</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000401_p104000010048"><a name="zh-cn_topic_0000002600000401_p104000010048"></a><a name="zh-cn_topic_0000002600000401_p104000010048"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000401_p104000010049"><a name="zh-cn_topic_0000002600000401_p104000010049"></a><a name="zh-cn_topic_0000002600000401_p104000010049"></a>含义：目标设备；为 None 时从模型自动推断。</p>
<p id="zh-cn_topic_0000002600000401_p104000010050"><a name="zh-cn_topic_0000002600000401_p104000010050"></a><a name="zh-cn_topic_0000002600000401_p104000010050"></a>默认值：None。</p>
<p id="zh-cn_topic_0000002600000401_p104000010051"><a name="zh-cn_topic_0000002600000401_p104000010051"></a><a name="zh-cn_topic_0000002600000401_p104000010051"></a>数据类型：torch.device / str 或 None。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002600000401_row104000010012"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000401_p104000010052"><a name="zh-cn_topic_0000002600000401_p104000010052"></a><a name="zh-cn_topic_0000002600000401_p104000010052"></a>grad_clip</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000401_p104000010053"><a name="zh-cn_topic_0000002600000401_p104000010053"></a><a name="zh-cn_topic_0000002600000401_p104000010053"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000401_p104000010054"><a name="zh-cn_topic_0000002600000401_p104000010054"></a><a name="zh-cn_topic_0000002600000401_p104000010054"></a>含义：梯度裁剪的最大范数；为假值（None/0）时关闭裁剪。</p>
<p id="zh-cn_topic_0000002600000401_p104000010055"><a name="zh-cn_topic_0000002600000401_p104000010055"></a><a name="zh-cn_topic_0000002600000401_p104000010055"></a>默认值：1.0。</p>
<p id="zh-cn_topic_0000002600000401_p104000010056"><a name="zh-cn_topic_0000002600000401_p104000010056"></a><a name="zh-cn_topic_0000002600000401_p104000010056"></a>数据类型：float 或 None。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002600000401_row104000010013"><td class="cellrowborder" valign="top" width="13.78%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002600000401_p104000010057"><a name="zh-cn_topic_0000002600000401_p104000010057"></a><a name="zh-cn_topic_0000002600000401_p104000010057"></a>log_every</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002600000401_p104000010058"><a name="zh-cn_topic_0000002600000401_p104000010058"></a><a name="zh-cn_topic_0000002600000401_p104000010058"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002600000401_p104000010059"><a name="zh-cn_topic_0000002600000401_p104000010059"></a><a name="zh-cn_topic_0000002600000401_p104000010059"></a>含义：>0 时对返回的 <code>loss_history</code> 做下采样（每步 loss 始终记录）。</p>
<p id="zh-cn_topic_0000002600000401_p104000010060"><a name="zh-cn_topic_0000002600000401_p104000010060"></a><a name="zh-cn_topic_0000002600000401_p104000010060"></a>默认值：0。</p>
<p id="zh-cn_topic_0000002600000401_p104000010061"><a name="zh-cn_topic_0000002600000401_p104000010061"></a><a name="zh-cn_topic_0000002600000401_p104000010061"></a>数据类型：int。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000002600000401_section104000010007"></a>

返回一个 dict，包含以下键：

- `steps`：实际执行的梯度步数（int）。
- `initial_loss`：首步损失，无步数时为 None。
- `final_loss`：末步损失，无步数时为 None。
- `loss_history`：逐步损失列表（`log_every > 0` 时为下采样后的列表）。

模型权重为就地更新，无需接收返回的模型对象。

## 调用示例<a name="zh-cn_topic_0000002600000401_section104000010008"></a>

```python
import amct_pytorch as amct
from amct_pytorch.pruning import prune_finetune

# model 为用户已剪枝的 torch.nn.Module；calib 为校准/训练批序列
result = prune_finetune(model, calib, steps=300, lr=1e-5, warmup=20)
print(result["initial_loss"], "->", result["final_loss"])
```
