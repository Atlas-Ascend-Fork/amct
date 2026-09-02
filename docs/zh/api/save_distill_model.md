# save\_distill\_model<a name="ZH-CN_TOPIC_0000002517188690"></a>

## 产品支持情况<a name="section185612964420"></a>

<a name="zh-cn_topic_0000002517188794_table38301303189"></a>

| 产品                                        | 是否支持 |
| ------------------------------------------- | -------- |
| Ascend 950PR/Ascend 950DT                   | √        |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √        |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √        |


## 功能说明<a name="zh-cn_topic_0240187365_section15406195619561"></a>

蒸馏接口，根据用户最终的蒸馏好的模型，生成最终量化精度仿真模型以及量化部署模型。

## 函数原型<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_section428121323411"></a>

```python
save_distill_model(model, save_path, input_data, record_file=None, input_names=None, output_names=None, dynamic_axes=None)
```

## 参数说明<a name="section73811524135618"></a>

<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="9.84%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="7.66%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0240187365_p1769255516412"><a name="zh-cn_topic_0240187365_p1769255516412"></a><a name="zh-cn_topic_0240187365_p1769255516412"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="82.5%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0240187365_p15231205416325"><a name="zh-cn_topic_0240187365_p15231205416325"></a><a name="zh-cn_topic_0240187365_p15231205416325"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row148501718185117"><td class="cellrowborder" valign="top" width="9.84%" headers="mcps1.1.4.1.1 "><p id="p15703125017513"><a name="p15703125017513"></a><a name="p15703125017513"></a>model</p>
</td>
<td class="cellrowborder" valign="top" width="7.66%" headers="mcps1.1.4.1.2 "><p id="p970311508516"><a name="p970311508516"></a><a name="p970311508516"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="82.5%" headers="mcps1.1.4.1.3 "><p id="p1246265654716"><a name="p1246265654716"></a><a name="p1246265654716"></a>含义：已进行蒸馏后的量化模型。</p>
<p id="p157031950195111"><a name="p157031950195111"></a><a name="p157031950195111"></a>数据类型：torch.nn.Module</p>
</td>
</tr>
<tr id="row588872319519"><td class="cellrowborder" valign="top" width="9.84%" headers="mcps1.1.4.1.1 "><p id="p8703550155111"><a name="p8703550155111"></a><a name="p8703550155111"></a>save_path</p>
</td>
<td class="cellrowborder" valign="top" width="7.66%" headers="mcps1.1.4.1.2 "><p id="p177031550185118"><a name="p177031550185118"></a><a name="p177031550185118"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="82.5%" headers="mcps1.1.4.1.3 "><p id="p660915344817"><a name="p660915344817"></a><a name="p660915344817"></a>含义：蒸馏后的量化模型存放路径。该路径需要包含模型名前缀，例如./quantized_model/*model。</p>
<p id="p87041950175120"><a name="p87041950175120"></a><a name="p87041950175120"></a>数据类型：string</p>
</td>
</tr>
<tr id="row1739023214510"><td class="cellrowborder" valign="top" width="9.84%" headers="mcps1.1.4.1.1 "><p id="p8704105085113"><a name="p8704105085113"></a><a name="p8704105085113"></a>input_data</p>
</td>
<td class="cellrowborder" valign="top" width="7.66%" headers="mcps1.1.4.1.2 "><p id="p3704135016518"><a name="p3704135016518"></a><a name="p3704135016518"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="82.5%" headers="mcps1.1.4.1.3 "><p id="p12260524816"><a name="p12260524816"></a><a name="p12260524816"></a>含义：模型的输入数据。一个<span>torch.tensor</span><span>会被等价为tuple（torch.tensor）</span>。</p>
<p id="p37045500510"><a name="p37045500510"></a><a name="p37045500510"></a>数据类型：tuple</p>
</td>
</tr>
<tr id="row14467165613613"><td class="cellrowborder" valign="top" width="9.84%" headers="mcps1.1.4.1.1 "><p id="p372671579"><a name="p372671579"></a><a name="p372671579"></a>record_file</p>
</td>
<td class="cellrowborder" valign="top" width="7.66%" headers="mcps1.1.4.1.2 "><p id="p157267118720"><a name="p157267118720"></a><a name="p157267118720"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="82.5%" headers="mcps1.1.4.1.3 "><p id="p111303704818"><a name="p111303704818"></a><a name="p111303704818"></a>含义：量化因子记录文件路径及名称。</p>
<p id="p167321191674"><a name="p167321191674"></a><a name="p167321191674"></a>默认值：None</p>
<p id="p12726411675"><a name="p12726411675"></a><a name="p12726411675"></a>数据类型：string</p>
<p id="p17762111782"><a name="p17762111782"></a><a name="p17762111782"></a>使用约束：传入值为None的情况下量化因子记录文件存放在amct_log文件夹中。</p>
</td>
</tr>
<tr id="row539163214513"><td class="cellrowborder" valign="top" width="9.84%" headers="mcps1.1.4.1.1 "><p id="p37041850185117"><a name="p37041850185117"></a><a name="p37041850185117"></a>input_names</p>
</td>
<td class="cellrowborder" valign="top" width="7.66%" headers="mcps1.1.4.1.2 "><p id="p117041506519"><a name="p117041506519"></a><a name="p117041506519"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="82.5%" headers="mcps1.1.4.1.3 "><p id="p1715710924811"><a name="p1715710924811"></a><a name="p1715710924811"></a>含义：模型输入节点的名称，用于在保存的量化onnx模型中显示。</p>
<p id="p4704195095118"><a name="p4704195095118"></a><a name="p4704195095118"></a>默认值：None</p>
<p id="p14704850115116"><a name="p14704850115116"></a><a name="p14704850115116"></a>数据类型：list(string)</p>
</td>
</tr>
<tr id="row1391532115114"><td class="cellrowborder" valign="top" width="9.84%" headers="mcps1.1.4.1.1 "><p id="p107040501510"><a name="p107040501510"></a><a name="p107040501510"></a>output_names</p>
</td>
<td class="cellrowborder" valign="top" width="7.66%" headers="mcps1.1.4.1.2 "><p id="p16705195017512"><a name="p16705195017512"></a><a name="p16705195017512"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="82.5%" headers="mcps1.1.4.1.3 "><p id="p9720410144818"><a name="p9720410144818"></a><a name="p9720410144818"></a>含义：模型输出节点的名称，用于在保存的量化onnx模型中显示。</p>
<p id="p147058501513"><a name="p147058501513"></a><a name="p147058501513"></a>默认值：None</p>
<p id="p170525018516"><a name="p170525018516"></a><a name="p170525018516"></a>数据类型：list(string)</p>
</td>
</tr>
<tr id="row1021713282517"><td class="cellrowborder" valign="top" width="9.84%" headers="mcps1.1.4.1.1 "><p id="p1705195075116"><a name="p1705195075116"></a><a name="p1705195075116"></a>dynamic_axes</p>
</td>
<td class="cellrowborder" valign="top" width="7.66%" headers="mcps1.1.4.1.2 "><p id="p117054506516"><a name="p117054506516"></a><a name="p117054506516"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="82.5%" headers="mcps1.1.4.1.3 "><p id="p172491012114819"><a name="p172491012114819"></a><a name="p172491012114819"></a>含义：对模型输入输出动态轴的指定，例如对于输入inputs（NCHW），N、H、W为不确定大小，输出outputs（NL），N为不确定大小，则指定形式为：{"inputs": [0,2,3], "outputs": [0]}，其中0,2,3分别表示N，H，W所在位置的索引。</p>
<p id="p17705165016517"><a name="p17705165016517"></a><a name="p17705165016517"></a>默认值：None</p>
<p id="p270525085117"><a name="p270525085117"></a><a name="p270525085117"></a>数据类型：dict&lt;string, dict&lt;python:int, string&gt;&gt; or dict&lt;string, list(int)<em id="i17705135016515"><a name="i17705135016515"></a><a name="i17705135016515"></a>&gt;</em></p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0240188739_zh-cn_topic_0122830089_section293415513458"></a>

无

## 调用示例<a name="zh-cn_topic_0240188739_section64231658994"></a>

```python
import amct_pytorch as amct
# 建立待进行蒸馏量化的网络图结构
model = build_model()
model.load_state_dict(torch.load(state_dict_path))
input_data = tuple([torch.randn(input_shape)])
 
# 插入蒸馏API，将蒸馏的模型存为onnx文件
amct.save_distill_model(
               model, 
               "./model/distilled",
               input_data,
               record_file="./results/records.txt",
               input_names=['input'],
               output_names=['output'],
               dynamic_axes={'input':{0: 'batch_size'},
                             'output':{0: 'batch_size'}})
```

落盘文件说明：

-   精度仿真模型文件：ONNX格式的模型文件，模型名中包含fake\_quant，可以在ONNX Runtime环境进行精度仿真。

-   部署模型文件：ONNX格式的模型文件，模型名中包含deploy，经过ATC转换工具转换后可部署到AI处理器。

重新执行蒸馏时，该接口输出的上述文件将会被覆盖。

