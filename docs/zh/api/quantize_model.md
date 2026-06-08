# quantize\_model<a name="ZH-CN_TOPIC_0000002517028742"></a>

## 产品支持情况<a name="section185612964420"></a>

<a name="zh-cn_topic_0000002517188794_table38301303189"></a>

| 产品                                        | 是否支持 |
| ------------------------------------------- | -------- |
| Ascend 950PR/Ascend 950DT                   | √        |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √        |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √        |


## 功能说明<a name="zh-cn_topic_0240187994_section15406195619561"></a>

训练后量化接口，将输入的待量化的图结构按照给定的量化配置文件进行量化处理，在传入的图结构中插入权重量化、数据量化相关的算子，生成量化因子记录文件record\_file，返回修改后的torch.nn.Module校准模型。

## 函数原型<a name="zh-cn_topic_0240187994_zh-cn_topic_0122830089_section428121323411"></a>

```python
calibration_model = quantize_model(config_file, modfied_onnx_file, record_file, model, input_data, input_names=None, output_names=None, dynamic_axes=None)
```

## 参数说明<a name="zh-cn_topic_0240187994_zh-cn_topic_0122830089_section795991810344"></a>

<a name="zh-cn_topic_0240187994_zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0240187994_zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="12.17%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0240187994_zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0240187994_zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0240187994_zh-cn_topic_0122830089_p1438834363520"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="7.51%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0240187994_p1769255516412"><a name="zh-cn_topic_0240187994_p1769255516412"></a><a name="zh-cn_topic_0240187994_p1769255516412"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="80.32000000000001%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0240187994_p15231205416325"><a name="zh-cn_topic_0240187994_p15231205416325"></a><a name="zh-cn_topic_0240187994_p15231205416325"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0240187994_row32991154514"><td class="cellrowborder" valign="top" width="12.17%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0240187994_p11214202165418"><a name="zh-cn_topic_0240187994_p11214202165418"></a><a name="zh-cn_topic_0240187994_p11214202165418"></a>config_file</p>
</td>
<td class="cellrowborder" valign="top" width="7.51%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0240187994_p8693185517417"><a name="zh-cn_topic_0240187994_p8693185517417"></a><a name="zh-cn_topic_0240187994_p8693185517417"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="80.32000000000001%" headers="mcps1.1.4.1.3 "><p id="p20120649205617"><a name="p20120649205617"></a><a name="p20120649205617"></a>含义：用户生成的量化配置文件，用于指定模型network中量化层的配置情况。</p>
<p id="zh-cn_topic_0240187994_p131301850191420"><a name="zh-cn_topic_0240187994_p131301850191420"></a><a name="zh-cn_topic_0240187994_p131301850191420"></a>数据类型：string</p>
</td>
</tr>
<tr id="row87261172575"><td class="cellrowborder" valign="top" width="12.17%" headers="mcps1.1.4.1.1 "><p id="p67261617135720"><a name="p67261617135720"></a><a name="p67261617135720"></a>modfied_onnx_file</p>
</td>
<td class="cellrowborder" valign="top" width="7.51%" headers="mcps1.1.4.1.2 "><p id="p67266176576"><a name="p67266176576"></a><a name="p67266176576"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="80.32000000000001%" headers="mcps1.1.4.1.3 "><p id="p1780475820567"><a name="p1780475820567"></a><a name="p1780475820567"></a>含义：文件名，用于存储融合后模型的onnx格式。</p>
<p id="p1072601710573"><a name="p1072601710573"></a><a name="p1072601710573"></a>数据类型：string</p>
</td>
</tr>
<tr id="zh-cn_topic_0240187994_row4537142419283"><td class="cellrowborder" valign="top" width="12.17%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0240187994_p1353832472811"><a name="zh-cn_topic_0240187994_p1353832472811"></a><a name="zh-cn_topic_0240187994_p1353832472811"></a>record_file</p>
</td>
<td class="cellrowborder" valign="top" width="7.51%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0240187994_p1153815249285"><a name="zh-cn_topic_0240187994_p1153815249285"></a><a name="zh-cn_topic_0240187994_p1153815249285"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="80.32000000000001%" headers="mcps1.1.4.1.3 "><p id="p1095310012573"><a name="p1095310012573"></a><a name="p1095310012573"></a>含义：量化因子记录文件路径及名称。</p>
<p id="zh-cn_topic_0240187994_p12538152410286"><a name="zh-cn_topic_0240187994_p12538152410286"></a><a name="zh-cn_topic_0240187994_p12538152410286"></a>数据类型：string</p>
</td>
</tr>
<tr id="row198701324594"><td class="cellrowborder" valign="top" width="12.17%" headers="mcps1.1.4.1.1 "><p id="p10871153295919"><a name="p10871153295919"></a><a name="p10871153295919"></a>model</p>
</td>
<td class="cellrowborder" valign="top" width="7.51%" headers="mcps1.1.4.1.2 "><p id="p832794819212"><a name="p832794819212"></a><a name="p832794819212"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="80.32000000000001%" headers="mcps1.1.4.1.3 "><p id="p236973105714"><a name="p236973105714"></a><a name="p236973105714"></a>含义：待量化的模型，已加载权重。</p>
<p id="p88711432155917"><a name="p88711432155917"></a><a name="p88711432155917"></a>数据类型：torch.nn.Module</p>
</td>
</tr>
<tr id="row349137105914"><td class="cellrowborder" valign="top" width="12.17%" headers="mcps1.1.4.1.1 "><p id="p349123714599"><a name="p349123714599"></a><a name="p349123714599"></a>input_data</p>
</td>
<td class="cellrowborder" valign="top" width="7.51%" headers="mcps1.1.4.1.2 "><p id="p18331648329"><a name="p18331648329"></a><a name="p18331648329"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="80.32000000000001%" headers="mcps1.1.4.1.3 "><p id="p138580418576"><a name="p138580418576"></a><a name="p138580418576"></a>含义：模型的输入数据。一个<span>torch.tensor</span><span>会被等价为</span><span>tuple</span><span>（</span><span>torch.tensor</span><span>）</span>。</p>
<p id="p7501837155915"><a name="p7501837155915"></a><a name="p7501837155915"></a>数据类型：tuple</p>
</td>
</tr>
<tr id="row952183595911"><td class="cellrowborder" valign="top" width="12.17%" headers="mcps1.1.4.1.1 "><p id="p5521035195919"><a name="p5521035195919"></a><a name="p5521035195919"></a>input_names</p>
</td>
<td class="cellrowborder" valign="top" width="7.51%" headers="mcps1.1.4.1.2 "><p id="p20334184811218"><a name="p20334184811218"></a><a name="p20334184811218"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="80.32000000000001%" headers="mcps1.1.4.1.3 "><p id="p18924137185717"><a name="p18924137185717"></a><a name="p18924137185717"></a>含义：模型的输入的名称，用于modfied_onnx_file中显示。</p>
<p id="p196913313314"><a name="p196913313314"></a><a name="p196913313314"></a>默认值：None</p>
<p id="p15526359598"><a name="p15526359598"></a><a name="p15526359598"></a>数据类型：list(string)</p>
</td>
</tr>
<tr id="row17781840115919"><td class="cellrowborder" valign="top" width="12.17%" headers="mcps1.1.4.1.1 "><p id="p177916405597"><a name="p177916405597"></a><a name="p177916405597"></a>output_names</p>
</td>
<td class="cellrowborder" valign="top" width="7.51%" headers="mcps1.1.4.1.2 "><p id="p1233617486215"><a name="p1233617486215"></a><a name="p1233617486215"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="80.32000000000001%" headers="mcps1.1.4.1.3 "><p id="p151924106575"><a name="p151924106575"></a><a name="p151924106575"></a>含义：模型的输出的名称，用于modfied_onnx_file中显示。</p>
<p id="p14974491771"><a name="p14974491771"></a><a name="p14974491771"></a>默认值：None</p>
<p id="p19779164013599"><a name="p19779164013599"></a><a name="p19779164013599"></a>数据类型：list(string)</p>
</td>
</tr>
<tr id="row786318131519"><td class="cellrowborder" valign="top" width="12.17%" headers="mcps1.1.4.1.1 "><p id="p88637131116"><a name="p88637131116"></a><a name="p88637131116"></a>dynamic_axes</p>
</td>
<td class="cellrowborder" valign="top" width="7.51%" headers="mcps1.1.4.1.2 "><p id="p933914814211"><a name="p933914814211"></a><a name="p933914814211"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="80.32000000000001%" headers="mcps1.1.4.1.3 "><p id="p6899514145712"><a name="p6899514145712"></a><a name="p6899514145712"></a>含义：对模型输入输出动态轴的指定，例如对于输入inputs（NCHW），N、H、W为不确定大小，输出outputs（NL），N为不确定大小，则dynamic_axes={"inputs": [0,2,3], "outputs": [0]}。</p>
<p id="p58938131378"><a name="p58938131378"></a><a name="p58938131378"></a>默认值：None</p>
<p id="p1486351312114"><a name="p1486351312114"></a><a name="p1486351312114"></a>数据类型：dict&lt;string, dict&lt;python:int, string&gt;&gt; or dict&lt;string, list(int)<em id="i82727557411"><a name="i82727557411"></a><a name="i82727557411"></a>&gt;</em></p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0240187994_zh-cn_topic_0122830089_section293415513458"></a>

返回修改后的torch.nn.Module校准模型。

## 调用示例<a name="zh-cn_topic_0240187994_section542843616112"></a>

```python
import amct_pytorch as amct
# 建立待量化的网络图结构
model = build_model()
model.load_state_dict(torch.load(state_dict_path))
input_data = tuple([torch.randn(input_shape)])

scale_offset_record_file = os.path.join(TMP, 'scale_offset_record.txt')
modfied_model = os.path.join(TMP, 'modfied_model.onnx')
# 插入量化API
calibration_model = amct.quantize_model(config_json_file,
                                        modfied_model,
                                        scale_offset_record_file,
                                        model,
                                        input_data,
                                        input_names=['input'],
                                        output_names=['output'],
                                        dynamic_axes={'input':{0: 'batch_size'},
                                                      'output':{0: 'batch_size'}})
```

