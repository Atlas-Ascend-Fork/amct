# quantize\_preprocess<a name="ZH-CN_TOPIC_0000002517028800"></a>

## 产品支持情况<a name="section185612964420"></a>

<a name="zh-cn_topic_0000002517188794_table38301303189"></a>

| 产品                                        | 是否支持 |
| ------------------------------------------- | -------- |
| Ascend 950PR/Ascend 950DT                   | √        |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √        |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √        |


## 功能说明<a name="zh-cn_topic_0240187365_section15406195619561"></a>

量化数据均衡预处理接口，将输入的待量化的图结构按照给定的量化配置文件进行量化处理，在传入的图结构中插入均衡量化相关的算子，生成均衡量化因子记录文件record\_file，返回修改后的torch.nn.Module校准模型。

## 函数原型<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_section428121323411"></a>

```python
calibration_model = quantize_preprocess(config_file, record_file, model, input_data)
```

## 参数说明<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_section795991810344"></a>

<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="10.13%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="9.6%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0240187365_p1769255516412"><a name="zh-cn_topic_0240187365_p1769255516412"></a><a name="zh-cn_topic_0240187365_p1769255516412"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="80.27%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0240187365_p15231205416325"><a name="zh-cn_topic_0240187365_p15231205416325"></a><a name="zh-cn_topic_0240187365_p15231205416325"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="10.13%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0240187994_p11214202165418"><a name="zh-cn_topic_0240187994_p11214202165418"></a><a name="zh-cn_topic_0240187994_p11214202165418"></a>config_file</p>
</td>
<td class="cellrowborder" valign="top" width="9.6%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0240187994_p8693185517417"><a name="zh-cn_topic_0240187994_p8693185517417"></a><a name="zh-cn_topic_0240187994_p8693185517417"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="80.27%" headers="mcps1.1.4.1.3 "><p id="p195971352101120"><a name="p195971352101120"></a><a name="p195971352101120"></a>含义：用户生成的量化配置文件，用于指定模型network中量化层的配置情况。</p>
<p id="zh-cn_topic_0240187994_p131301850191420"><a name="zh-cn_topic_0240187994_p131301850191420"></a><a name="zh-cn_topic_0240187994_p131301850191420"></a>数据类型：string</p>
</td>
</tr>
<tr id="zh-cn_topic_0240187365_row2997141123911"><td class="cellrowborder" valign="top" width="10.13%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0240187994_p1353832472811"><a name="zh-cn_topic_0240187994_p1353832472811"></a><a name="zh-cn_topic_0240187994_p1353832472811"></a>record_file</p>
</td>
<td class="cellrowborder" valign="top" width="9.6%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0240187994_p1153815249285"><a name="zh-cn_topic_0240187994_p1153815249285"></a><a name="zh-cn_topic_0240187994_p1153815249285"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="80.27%" headers="mcps1.1.4.1.3 "><p id="p20262125871111"><a name="p20262125871111"></a><a name="p20262125871111"></a>含义：均衡量化因子记录文件路径及名称。</p>
<p id="zh-cn_topic_0240187994_p12538152410286"><a name="zh-cn_topic_0240187994_p12538152410286"></a><a name="zh-cn_topic_0240187994_p12538152410286"></a>数据类型：string</p>
</td>
</tr>
<tr id="zh-cn_topic_0240187365_row13558046183914"><td class="cellrowborder" valign="top" width="10.13%" headers="mcps1.1.4.1.1 "><p id="p10871153295919"><a name="p10871153295919"></a><a name="p10871153295919"></a>model</p>
</td>
<td class="cellrowborder" valign="top" width="9.6%" headers="mcps1.1.4.1.2 "><p id="p832794819212"><a name="p832794819212"></a><a name="p832794819212"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="80.27%" headers="mcps1.1.4.1.3 "><p id="p37210595113"><a name="p37210595113"></a><a name="p37210595113"></a>含义：待量化的模型，已加载权重。</p>
<p id="p88711432155917"><a name="p88711432155917"></a><a name="p88711432155917"></a>数据类型：torch.nn.Module</p>
</td>
</tr>
<tr id="zh-cn_topic_0240187365_row473005916274"><td class="cellrowborder" valign="top" width="10.13%" headers="mcps1.1.4.1.1 "><p id="p349123714599"><a name="p349123714599"></a><a name="p349123714599"></a>input_data</p>
</td>
<td class="cellrowborder" valign="top" width="9.6%" headers="mcps1.1.4.1.2 "><p id="p18331648329"><a name="p18331648329"></a><a name="p18331648329"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="80.27%" headers="mcps1.1.4.1.3 "><p id="p1265301111213"><a name="p1265301111213"></a><a name="p1265301111213"></a>含义：模型的输入数据。一个<span>torch.tensor</span><span>会被等价为</span><span>tuple</span><span>（</span><span>torch.tensor</span><span>）</span>。</p>
<p id="p7501837155915"><a name="p7501837155915"></a><a name="p7501837155915"></a>数据类型：tuple</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_section293415513458"></a>

返回修改后的torch.nn.Module校准模型。

## 调用示例<a name="zh-cn_topic_0240187365_section64231658994"></a>

```python
import amct_pytorch as amct
# 建立待量化的网络图结构
model = build_model()
model.load_state_dict(torch.load(state_dict_path))
input_data = tuple([torch.randn(input_shape)])

tensor_balance_factor_record_file = os.path.join(TMP, 'tensor_balance_factor_record.txt')
modified_model = os.path.join(TMP, 'modified_model.onnx')
# 插入量化API
calibration_model = amct.quantize_preprocess(config_json_file,
                                             tensor_balance_factor_record_file,
                                             model,
                                             input_data)

```

