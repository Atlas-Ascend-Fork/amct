# create\_distill\_model<a name="ZH-CN_TOPIC_0000002517028774"></a>

## 产品支持情况<a name="section185612964420"></a>

<a name="zh-cn_topic_0000002517188794_table38301303189"></a>

| 产品                                        | 是否支持 |
| ------------------------------------------- | -------- |
| Ascend 950PR/Ascend 950DT                   | √        |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √        |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √        |


## 功能说明<a name="zh-cn_topic_0240187365_section15406195619561"></a>

蒸馏接口，将输入的待量化压缩的图结构按照给定的蒸馏量化配置文件进行量化处理，在传入的图结构中插入量化相关的算子（数据和权重的蒸馏量化层以及找N的层），返回修改后可用于蒸馏的torch.nn.Module模型。

## 函数原型<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_section428121323411"></a>

```python
compress_model = create_distill_model(config_file, model, input_data)
```

## 参数说明<a name="section73811524135618"></a>

<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="10.901090109010902%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="7.4007400740074%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0240187365_p1769255516412"><a name="zh-cn_topic_0240187365_p1769255516412"></a><a name="zh-cn_topic_0240187365_p1769255516412"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="81.69816981698169%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0240187365_p15231205416325"><a name="zh-cn_topic_0240187365_p15231205416325"></a><a name="zh-cn_topic_0240187365_p15231205416325"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="10.901090109010902%" headers="mcps1.1.4.1.1 "><p id="p31941230154616"><a name="p31941230154616"></a><a name="p31941230154616"></a>config_file</p>
</td>
<td class="cellrowborder" valign="top" width="7.4007400740074%" headers="mcps1.1.4.1.2 "><p id="p1219493024610"><a name="p1219493024610"></a><a name="p1219493024610"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="81.69816981698169%" headers="mcps1.1.4.1.3 "><p id="p2828559124411"><a name="p2828559124411"></a><a name="p2828559124411"></a>含义：用户生成的蒸馏量化配置文件，用于指定模型network中量化层的配置情况和蒸馏结构。</p>
<p id="p7194123094612"><a name="p7194123094612"></a><a name="p7194123094612"></a>数据类型：string</p>
<p id="p9725102314288"><a name="p9725102314288"></a><a name="p9725102314288"></a>使用约束：该接口输入的config.json必须和<a href="create_distill_config.md">create_distill_config</a>接口输入的config.json一致</p>
</td>
</tr>
<tr id="row4280131024613"><td class="cellrowborder" valign="top" width="10.901090109010902%" headers="mcps1.1.4.1.1 "><p id="p4194143084610"><a name="p4194143084610"></a><a name="p4194143084610"></a>model</p>
</td>
<td class="cellrowborder" valign="top" width="7.4007400740074%" headers="mcps1.1.4.1.2 "><p id="p14194430114612"><a name="p14194430114612"></a><a name="p14194430114612"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="81.69816981698169%" headers="mcps1.1.4.1.3 "><p id="p1980219674518"><a name="p1980219674518"></a><a name="p1980219674518"></a>含义：待进行蒸馏量化的原始浮点模型，已加载权重。</p>
<p id="p19194123013463"><a name="p19194123013463"></a><a name="p19194123013463"></a>数据类型：torch.nn.Module</p>
</td>
</tr>
<tr id="row156638166467"><td class="cellrowborder" valign="top" width="10.901090109010902%" headers="mcps1.1.4.1.1 "><p id="p1619443010466"><a name="p1619443010466"></a><a name="p1619443010466"></a>input_data</p>
</td>
<td class="cellrowborder" valign="top" width="7.4007400740074%" headers="mcps1.1.4.1.2 "><p id="p919503013466"><a name="p919503013466"></a><a name="p919503013466"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="81.69816981698169%" headers="mcps1.1.4.1.3 "><p id="p54624844511"><a name="p54624844511"></a><a name="p54624844511"></a>含义：模型的输入数据。一个<span>torch.tensor</span><span>会被等价为tuple(torch.tensor)</span>。</p>
<p id="p1195430114611"><a name="p1195430114611"></a><a name="p1195430114611"></a>数据类型：tuple</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0240188739_zh-cn_topic_0122830089_section293415513458"></a>

修改后可用于蒸馏的torch.nn.Module模型。

## 调用示例<a name="zh-cn_topic_0240188739_section64231658994"></a>

```python
import amct_pytorch as amct
# 建立待进行蒸馏量化的网络图结构
model = build_model()
model.load_state_dict(torch.load(state_dict_path))
input_data = tuple([torch.randn(input_shape)])

# 生成压缩模型
compress_model = amct.create_distill_model(
                 config_json_file,
                 model,
                 input_data)
```

