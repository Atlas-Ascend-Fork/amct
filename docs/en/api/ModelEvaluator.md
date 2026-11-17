# ModelEvaluator<a name="ZH-CN_TOPIC_0000002517188794"></a>

<!-- md-trans-meta sourceCommit=9d54d17e0942052193c827180dc43312c580acc7 translatedAt=2026-10-16T08:27:42.730Z pushedAt=2026-11-05T01:12:03.418Z nl=2 -->

## Supported Products<a name="section197451857688"></a>

<a name="zh-cn_topic_0000002517188794_table38301303189"></a>

| Product | Supported |
| ------------------------------------------- | -------- |
| Ascend 950PR&950DT products | √ |
| Atlas A3 products | √ |
| Atlas A2 products | √ |

## Description<a name="zh-cn_topic_0240187994_section15406195619561"></a>

Provides a Python instance that serves as an evaluator for performing calibration and inference on a specific model based on the bin-type input data of the model.

## Prototype<a name="zh-cn_topic_0240187994_zh-cn_topic_0122830089_section428121323411"></a>

```python
class ModelEvaluator(AutoCalibrationEvaluatorBase):
    def __init__(self, data_dir, input_shape, data_types):
        ...
```

## Parameters<a name="zh-cn_topic_0240187994_zh-cn_topic_0122830089_section795991810344"></a>

<a name="zh-cn_topic_0240187994_zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0240187994_zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="9.69%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0240187994_zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0240187994_zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0240187994_zh-cn_topic_0122830089_p1438834363520"></a>Parameter</p></th>
<th class="cellrowborder" valign="top" width="6.819999999999999%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0240187994_p1769255516412"><a name="zh-cn_topic_0240187994_p1769255516412"></a><a name="zh-cn_topic_0240187994_p1769255516412"></a>Input/Output</p></th>
<th class="cellrowborder" valign="top" width="83.49%" id="mcps1.1.4.1.3"><p id="p19911124613711"><a name="p19911124613711"></a><a name="p19911124613711"></a>Description</p></th>
</tr>
</thead>
<tbody><tr id="row8680426943"><td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.1.4.1.1 "><p id="p12818830740"><a name="p12818830740"></a><a name="p12818830740"></a>data_dir</p></td>
<td class="cellrowborder" valign="top" width="6.819999999999999%" headers="mcps1.1.4.1.2 "><p id="p128185303416"><a name="p128185303416"></a><a name="p128185303416"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="83.49%" headers="mcps1.1.4.1.3 "><p id="p7193212111914"><a name="p7193212111914"></a><a name="p7193212111914"></a>Meaning: Path of the bin-type dataset that matches the model.</p>
<p id="p9912946779"><a name="p9912946779"></a><a name="p9912946779"></a>Data type: string</p>
<p id="p10881253527"><a name="p10881253527"></a><a name="p10881253527"></a>Parameter value format: "data/input1/;data/input2/"</p>
<p id="p1241012377217"><a name="p1241012377217"></a><a name="p1241012377217"></a>Usage constraints:</p>
<a name="zh-cn_topic_0000001135587500_ul14146154111109"></a><a name="zh-cn_topic_0000001135587500_ul14146154111109"></a><ul id="zh-cn_topic_0000001135587500_ul14146154111109"><li>The path supports uppercase and lowercase letters (a-z, A-Z), digits (0-9), underscores (_), hyphens (-), periods (.), and Chinese characters.</li><li>If the model has multiple inputs and each input has multiple batches of data, different input data must be stored in different directories, and the file names in each directory must be sorted in ascending order. All input data paths must be placed in double quotes, and the nodes must be separated by English semicolons.</li><li>The array shape stored in a single bin file must match the shape entered in input_shape. For example, if the array shape stored in a single-image bin is 1x224x224x3, the value entered in input_shape must be 1x224x224x3. To quantize multiple bins, adjust the batch_num value.</li></ul></td>
</tr>
<tr id="zh-cn_topic_0240187994_row32991154514"><td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0240187994_p1254338193919"><a name="zh-cn_topic_0240187994_p1254338193919"></a><a name="zh-cn_topic_0240187994_p1254338193919"></a>input_shape</p></td>
<td class="cellrowborder" valign="top" width="6.819999999999999%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0240187994_p1255438123917"><a name="zh-cn_topic_0240187994_p1255438123917"></a><a name="zh-cn_topic_0240187994_p1255438123917"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="83.49%" headers="mcps1.1.4.1.3 "><p id="p6500142751916"><a name="p6500142751916"></a><a name="p6500142751916"></a>Meaning: Shape information of the model input.</p>
<p id="p9912144617713"><a name="p9912144617713"></a><a name="p9912144617713"></a>Data type: string</p>
<p id="p44111151944"><a name="p44111151944"></a><a name="p44111151944"></a>Parameter value format: "input_name1:n1,c1,h1,w1;input_name2:n2,c2,h2,w2".</p>
<p id="p18687175419410"><a name="p18687175419410"></a><a name="p18687175419410"></a>Usage constraints: The specified nodes must be placed in double quotes, and nodes are separated by English semicolons.</p></td>
</tr>
<tr id="zh-cn_topic_0240187994_row4537142419283"><td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.1.4.1.1 "><p id="p196984129618"><a name="p196984129618"></a><a name="p196984129618"></a>data_types</p></td>
<td class="cellrowborder" valign="top" width="6.819999999999999%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0240187994_p1153815249285"><a name="zh-cn_topic_0240187994_p1153815249285"></a><a name="zh-cn_topic_0240187994_p1153815249285"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="83.49%" headers="mcps1.1.4.1.3 "><p id="p4355124118192"><a name="p4355124118192"></a><a name="p4355124118192"></a>Meaning: Type of the input data.</p>
<p id="p6912174617711"><a name="p6912174617711"></a><a name="p6912174617711"></a>Data type: string</p>
<p id="p21059815512"><a name="p21059815512"></a><a name="p21059815512"></a>Parameter value format: "float32;float64"</p>
<p id="p26158361957"><a name="p26158361957"></a><a name="p26158361957"></a>Usage constraints: If the model has multiple inputs with different data types, the data type of each input must be specified separately. The specified input data types must be placed in double quotes in the order of the input nodes, and all input data types must be placed in double quotes, separated by English semicolons.</p></td>
</tr>
</tbody>
</table>

## Return Value<a name="zh-cn_topic_0240187994_zh-cn_topic_0122830089_section293415513458"></a>

A Python instance.

## Example<a name="zh-cn_topic_0240187994_section542843616112"></a>

```python
import amct_pytorch as amct

evaluator = amct.ModelEvaluator(
    data_dir="./data/input_bin/", 
    input_shape="input:32,3,224,224", 
    data_types="float32")
```
