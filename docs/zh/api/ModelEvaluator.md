# ModelEvaluator<a name="ZH-CN_TOPIC_0000002517188794"></a>

## 产品支持情况<a name="section197451857688"></a>

<a name="zh-cn_topic_0000002517188794_table38301303189"></a>

| 产品                                        | 是否支持 |
| ------------------------------------------- | -------- |
| Ascend 950PR/Ascend 950DT                   | √        |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √        |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √        |


## 功能说明<a name="zh-cn_topic_0240187994_section15406195619561"></a>

针对某一个模型，根据模型的bin类型输入数据，提供一个Python实例，可对该模型执行校准和推理的评估器。

## 函数原型<a name="zh-cn_topic_0240187994_zh-cn_topic_0122830089_section428121323411"></a>

```python
class ModelEvaluator(AutoCalibrationEvaluatorBase):
def __init__(self, data_dir, input_shape, data_types):
```

## 参数说明<a name="zh-cn_topic_0240187994_zh-cn_topic_0122830089_section795991810344"></a>

<a name="zh-cn_topic_0240187994_zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0240187994_zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="9.69%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0240187994_zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0240187994_zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0240187994_zh-cn_topic_0122830089_p1438834363520"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="6.819999999999999%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0240187994_p1769255516412"><a name="zh-cn_topic_0240187994_p1769255516412"></a><a name="zh-cn_topic_0240187994_p1769255516412"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="83.49%" id="mcps1.1.4.1.3"><p id="p19911124613711"><a name="p19911124613711"></a><a name="p19911124613711"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row8680426943"><td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.1.4.1.1 "><p id="p12818830740"><a name="p12818830740"></a><a name="p12818830740"></a>data_dir</p>
</td>
<td class="cellrowborder" valign="top" width="6.819999999999999%" headers="mcps1.1.4.1.2 "><p id="p128185303416"><a name="p128185303416"></a><a name="p128185303416"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="83.49%" headers="mcps1.1.4.1.3 "><p id="p7193212111914"><a name="p7193212111914"></a><a name="p7193212111914"></a>含义：与模型匹配的bin格式数据集路径。</p>
<p id="p9912946779"><a name="p9912946779"></a><a name="p9912946779"></a>数据类型：string</p>
<p id="p10881253527"><a name="p10881253527"></a><a name="p10881253527"></a>参数值格式："data/input1/;data/input2/"</p>
<p id="p1241012377217"><a name="p1241012377217"></a><a name="p1241012377217"></a>使用约束：</p>
<a name="zh-cn_topic_0000001135587500_ul14146154111109"></a><a name="zh-cn_topic_0000001135587500_ul14146154111109"></a><ul id="zh-cn_topic_0000001135587500_ul14146154111109"><li>路径支持大小写字母（a-z，A-Z）、数字（0-9）、下划线（_）、中划线（-）、句点（.）、中文字符。</li><li>若模型有多个输入，且每个输入有多个batch数据，则不同的输入数据必须存储在不同的目录中，目录中文件的名称必须按照升序排序。所有的输入数据路径必须放在双引号中，节点中间使用英文分号分隔。</li><li>单个bin文件中存储的数组shape需要和input_shape中输入的shape相匹配，例如：单张图片bin存储的数组shape为1x224x224x3，则input_shape中输入的必须为1x224x224x3；如需多个bin做量化，则可通过调整batch_num取值实现。</li></ul>
</td>
</tr>
<tr id="zh-cn_topic_0240187994_row32991154514"><td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0240187994_p1254338193919"><a name="zh-cn_topic_0240187994_p1254338193919"></a><a name="zh-cn_topic_0240187994_p1254338193919"></a>input_shape</p>
</td>
<td class="cellrowborder" valign="top" width="6.819999999999999%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0240187994_p1255438123917"><a name="zh-cn_topic_0240187994_p1255438123917"></a><a name="zh-cn_topic_0240187994_p1255438123917"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="83.49%" headers="mcps1.1.4.1.3 "><p id="p6500142751916"><a name="p6500142751916"></a><a name="p6500142751916"></a>含义：模型输入的shape信息。</p>
<p id="p9912144617713"><a name="p9912144617713"></a><a name="p9912144617713"></a>数据类型：string</p>
<p id="p44111151944"><a name="p44111151944"></a><a name="p44111151944"></a>参数值格式："input_name1:n1,c1,h1,w1;input_name2:n2,c2,h2,w2"。</p>
<p id="p18687175419410"><a name="p18687175419410"></a><a name="p18687175419410"></a>使用约束：指定的节点必须放在双引号中，节点中间使用英文分号分隔。</p>
</td>
</tr>
<tr id="zh-cn_topic_0240187994_row4537142419283"><td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.1.4.1.1 "><p id="p196984129618"><a name="p196984129618"></a><a name="p196984129618"></a>data_types</p>
</td>
<td class="cellrowborder" valign="top" width="6.819999999999999%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0240187994_p1153815249285"><a name="zh-cn_topic_0240187994_p1153815249285"></a><a name="zh-cn_topic_0240187994_p1153815249285"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="83.49%" headers="mcps1.1.4.1.3 "><p id="p4355124118192"><a name="p4355124118192"></a><a name="p4355124118192"></a>含义：输入数据的类型。</p>
<p id="p6912174617711"><a name="p6912174617711"></a><a name="p6912174617711"></a>数据类型：string</p>
<p id="p21059815512"><a name="p21059815512"></a><a name="p21059815512"></a>参数值格式："float32;float64"</p>
<p id="p26158361957"><a name="p26158361957"></a><a name="p26158361957"></a>使用约束：若模型有多个输入，且数据类型不同，则需要分别指定不同输入的数据类型，指定的输入数据类型必须按照输入节点顺序依次放在双引号中，所有的输入数据类型必须放在双引号中，中间使用英文分号分隔。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0240187994_zh-cn_topic_0122830089_section293415513458"></a>

一个Python实例。

## 调用示例<a name="zh-cn_topic_0240187994_section542843616112"></a>

```python
import amct_pytorch as amct

evaluator = amct.ModelEvaluator(
    data_dir="./data/input_bin/", 
    input_shape="input:32,3,224,224", 
    data_types="float32")
```

