# create\_quant\_cali\_config<a name="ZH-CN_TOPIC_0000002517188700"></a>

<!-- md-trans-meta sourceCommit=4a47814eed18048e444f9a2acc42d9864ddc477f translatedAt=2026-10-16T08:20:46.379Z pushedAt=2026-11-05T01:12:03.410Z nl=2 -->

## Supported Products<a name="section197451857688"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>Product</span></p></th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>Supported</p></th>
</tr>
</thead>
<tbody><tr id="row574891710101"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p177491117171015"><a name="p177491117171015"></a><a name="p177491117171015"></a><span id="ph2272194216543"><a name="ph2272194216543"></a><a name="ph2272194216543"></a>Ascend 950PR&950DT products</span></p></td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p14226338117"><a name="p14226338117"></a><a name="p14226338117"></a>√</p></td>
</tr>
<tr id="row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p48327011813"><a name="p48327011813"></a><a name="p48327011813"></a><span id="ph583230201815"><a name="ph583230201815"></a><a name="ph583230201815"></a><term id="zh-cn_topic_0000001312391781_term1253731311225">Atlas A3 products</term></span></p></td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p108715341013"><a name="p108715341013"></a><a name="p108715341013"></a>√</p></td>
</tr>
<tr id="row173226882415"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p14832120181815"><a name="p14832120181815"></a><a name="p14832120181815"></a><span id="ph1483216010188"><a name="ph1483216010188"></a><a name="ph1483216010188"></a><term id="zh-cn_topic_0000001312391781_term11962195213215">Atlas A2 products</term></span></p></td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p19948143911820"><a name="p19948143911820"></a><a name="p19948143911820"></a>√</p></td>
</tr>
</tbody>
</table>

**Note: For products marked with "x", calling the API does not report an error, but no performance gain is obtained.**

## Description<a name="section14397134012517"></a>

The **KV-cache** quantization API generates detailed quantization configuration for each layer based on the user-provided model, quantization layer information, and quantization configuration information.

## Prototype<a name="section5661182817510"></a>

```python
create_quant_cali_config(config_file,model,quant_layers=None,config_defination=None)
```

## Parameters<a name="section73811524135618"></a>

<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="12.898710128987101%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a>Parameter</p></th>
<th class="cellrowborder" valign="top" width="8.18918108189181%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0240187365_p1769255516412"><a name="zh-cn_topic_0240187365_p1769255516412"></a><a name="zh-cn_topic_0240187365_p1769255516412"></a>Input/Output</p></th>
<th class="cellrowborder" valign="top" width="78.91210878912109%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0240187365_p15231205416325"><a name="zh-cn_topic_0240187365_p15231205416325"></a><a name="zh-cn_topic_0240187365_p15231205416325"></a>Description</p></th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="12.898710128987101%" headers="mcps1.1.4.1.1 "><p id="p3829146134310"><a name="p3829146134310"></a><a name="p3829146134310"></a>config_file</p></td>
<td class="cellrowborder" valign="top" width="8.18918108189181%" headers="mcps1.1.4.1.2 "><p id="p11274056174319"><a name="p11274056174319"></a><a name="p11274056174319"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.91210878912109%" headers="mcps1.1.4.1.3 "><p id="p1227053374911"><a name="p1227053374911"></a><a name="p1227053374911"></a>Meaning: Path and name of the quantization configuration file to be generated. The file is in JSON format and contains the quantization configuration information of each KV Cache quantization layer. If a file with the same name already exists in the path, calling this API overwrites the existing file.</p>
<p id="p1727418563433"><a name="p1727418563433"></a><a name="p1727418563433"></a>Data type: string</p></td>
</tr>
<tr id="row10475143684319"><td class="cellrowborder" valign="top" width="12.898710128987101%" headers="mcps1.1.4.1.1 "><p id="p128298463436"><a name="p128298463436"></a><a name="p128298463436"></a>model</p></td>
<td class="cellrowborder" valign="top" width="8.18918108189181%" headers="mcps1.1.4.1.2 "><p id="p8274556154316"><a name="p8274556154316"></a><a name="p8274556154316"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.91210878912109%" headers="mcps1.1.4.1.3 "><p id="p13700164115496"><a name="p13700164115496"></a><a name="p13700164115496"></a>Meaning: User-provided model to be quantized.</p>
<p id="p17274105617434"><a name="p17274105617434"></a><a name="p17274105617434"></a>Data type: torch.nn.Module</p></td>
</tr>
<tr id="row11737244204314"><td class="cellrowborder" valign="top" width="12.898710128987101%" headers="mcps1.1.4.1.1 "><p id="p18829134610436"><a name="p18829134610436"></a><a name="p18829134610436"></a>quant_layers</p></td>
<td class="cellrowborder" valign="top" width="8.18918108189181%" headers="mcps1.1.4.1.2 "><p id="p18274156124317"><a name="p18274156124317"></a><a name="p18274156124317"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.91210878912109%" headers="mcps1.1.4.1.3 "><p id="p661214394919"><a name="p661214394919"></a><a name="p661214394919"></a>Meaning: Quantization layer information, represented by a dictionary. If an easy-to-configure quantization configuration file is passed in, the configuration file takes precedence.</p>
<p id="p1477212427156"><a name="p1477212427156"></a><a name="p1477212427156"></a>KV-cache quantization example:</p>
<pre class="screen" id="screen660818309166"><a name="screen660818309166"></a><a name="screen660818309166"></a>{'kv_cache_quant_layers': ['MatMul_1']}</pre>
<p id="p026696131716"><a name="p026696131716"></a><a name="p026696131716"></a>Default value: None</p>
<p id="p927405684319"><a name="p927405684319"></a><a name="p927405684319"></a>Data type: dict</p>
<p id="p6591173620174"><a name="p6591173620174"></a><a name="p6591173620174"></a>Usage constraints: quant_layers can be specified either in the parameter or in the easy-to-configure configuration file. When the value is None, the parameter takes precedence; otherwise, the easy-to-configure configuration file takes precedence.</p></td>
</tr>
<tr id="row41919427437"><td class="cellrowborder" valign="top" width="12.898710128987101%" headers="mcps1.1.4.1.1 "><p id="p158301146144319"><a name="p158301146144319"></a><a name="p158301146144319"></a>config_defination</p></td>
<td class="cellrowborder" valign="top" width="8.18918108189181%" headers="mcps1.1.4.1.2 "><p id="p1027410568437"><a name="p1027410568437"></a><a name="p1027410568437"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.91210878912109%" headers="mcps1.1.4.1.3 "><p id="p55493456496"><a name="p55493456496"></a><a name="p55493456496"></a>Meaning: Easy-to-configure quantization configuration file.</p>
<p id="p827435611439"><a name="p827435611439"></a><a name="p827435611439"></a>The easy-to-configure file quant.cfg is generated based on quant_calibration_config_pytorch.proto. The path of the *.proto file is: <em id="i12274856104314"><a name="i12274856104314"></a><a name="i12274856104314"></a><span id="ph244511481409"><a name="ph244511481409"></a><a name="ph244511481409"></a>AMCT</span> installation directory</em>/amct_pytorch/proto/. For the parameter description of the *.proto file and the sample of the generated quant.cfg easy-to-configure quantization configuration file, see <a href="../context/kv_cache_config.md">KV Cache Easy-to-Configure Quantization Configuration File</a>.</p>
<p id="p1275756114320"><a name="p1275756114320"></a><a name="p1275756114320"></a>Default value: None</p>
<p id="p3275175644316"><a name="p3275175644316"></a><a name="p3275175644316"></a>Data type: string</p></td>
</tr>
</tbody>
</table>

## Return Value<a name="zh-cn_topic_0240188739_zh-cn_topic_0122830089_section293415513458"></a>

None

## Example<a name="zh-cn_topic_0240188739_section64231658994"></a>

```python
import amct_pytorch as amct
# Build the network graph structure to be quantized.
model = build_model()
model.load_state_dict(torch.load(state_dict_path))
input_data = tuple([torch.randn(input_shape)])

# Generate the quantization configuration file.
amct.create_quant_cali_config(config_file="./configs/config.json",
                             model=model,
                             quant_layers=None,
                             config_defination="./configs/quant.cfg")
```

Description of the generated file:

A quantization configuration file in JSON format is generated (when quantization is re-executed, the configuration file output by this API will be overwritten). A sample is as follows:

```json
{
    "batch_num":1,
    "activation_offset":true,
    "matmul1":{
        "kv_data_quant_config":{
            "act_algo":"hfmg",
            "num_of_bins":4096,
            "quant_granularity":0
        }
    },
    "matmul2":{
        "kv_data_quant_config":{
            "act_algo":"hfmg",
            "num_of_bins":4096,
            "quant_granularity":0
        }
    },
    "matmul3":{
        "kv_data_quant_config":{
            "act_algo":"ifmr",
            "max_percentile":0.999999,
            "min_percentile":0.999999,
            "search_range":[
                0.7,
                1.3
            ],
            "search_step":0.01,
            "quant_granularity":0
        }
    }
}
```
