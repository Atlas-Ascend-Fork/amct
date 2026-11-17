# create\_distill\_config<a name="ZH-CN_TOPIC_0000002517028788"></a>

<!-- md-trans-meta sourceCommit=9d54d17e0942052193c827180dc43312c580acc7 translatedAt=2026-10-16T08:16:21.695Z pushedAt=2026-11-05T01:12:03.408Z nl=2 -->

## Supported Products<a name="section185612964420"></a>

<a name="zh-cn_topic_0000002517188794_table38301303189"></a>

| Product | Supported |
| ------------------------------------------- | -------- |
| Ascend 950PR&950DT products | √ |
| Atlas A3 products | √ |
| Atlas A2 products | √ |

## Description<a name="section14397134012517"></a>

Finds all distillable quantization layers and distillable quantization structures based on the graph structure, automatically generates a distillation quantization configuration file, and writes the quantization config of the distillable quantization layers and the distillation structure into the configuration file.

## Prototype<a name="section5661182817510"></a>

```python
create_distill_config(config_file, model, input_data, config_defination=None)
```

## Parameters<a name="section73811524135618"></a>

<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="12.93%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a>Parameter</p></th>
<th class="cellrowborder" valign="top" width="7.57%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0240187365_p1769255516412"><a name="zh-cn_topic_0240187365_p1769255516412"></a><a name="zh-cn_topic_0240187365_p1769255516412"></a>Input/Output</p></th>
<th class="cellrowborder" valign="top" width="79.5%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0240187365_p15231205416325"><a name="zh-cn_topic_0240187365_p15231205416325"></a><a name="zh-cn_topic_0240187365_p15231205416325"></a>Description</p></th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="12.93%" headers="mcps1.1.4.1.1 "><p id="p3829146134310"><a name="p3829146134310"></a><a name="p3829146134310"></a>config_file</p></td>
<td class="cellrowborder" valign="top" width="7.57%" headers="mcps1.1.4.1.2 "><p id="p11274056174319"><a name="p11274056174319"></a><a name="p11274056174319"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="79.5%" headers="mcps1.1.4.1.3 "><p id="p1835396124311"><a name="p1835396124311"></a><a name="p1835396124311"></a>Meaning: Path and name of the distillation quantization configuration file to be generated. If a file with the same name already exists in the path, calling this API overwrites the existing file.</p>
<p id="p1727418563433"><a name="p1727418563433"></a><a name="p1727418563433"></a>Data type: string</p></td>
</tr>
<tr id="row10475143684319"><td class="cellrowborder" valign="top" width="12.93%" headers="mcps1.1.4.1.1 "><p id="p128298463436"><a name="p128298463436"></a><a name="p128298463436"></a>model</p></td>
<td class="cellrowborder" valign="top" width="7.57%" headers="mcps1.1.4.1.2 "><p id="p8274556154316"><a name="p8274556154316"></a><a name="p8274556154316"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="79.5%" headers="mcps1.1.4.1.3 "><p id="p13317181214313"><a name="p13317181214313"></a><a name="p13317181214313"></a>Meaning: Original floating-point model to be distilled and quantized, with loaded weights.</p>
<p id="p17274105617434"><a name="p17274105617434"></a><a name="p17274105617434"></a>Data type: torch.nn.Module</p></td>
</tr>
<tr id="row11737244204314"><td class="cellrowborder" valign="top" width="12.93%" headers="mcps1.1.4.1.1 "><p id="p18829134610436"><a name="p18829134610436"></a><a name="p18829134610436"></a>input_data</p></td>
<td class="cellrowborder" valign="top" width="7.57%" headers="mcps1.1.4.1.2 "><p id="p18274156124317"><a name="p18274156124317"></a><a name="p18274156124317"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="79.5%" headers="mcps1.1.4.1.3 "><p id="p20817101314320"><a name="p20817101314320"></a><a name="p20817101314320"></a>Meaning: Input data of the model. A torch.tensor is equivalent to tuple(torch.tensor).</p>
<p id="p927405684319"><a name="p927405684319"></a><a name="p927405684319"></a>Data type: tuple</p></td>
</tr>
<tr id="row41919427437"><td class="cellrowborder" valign="top" width="12.93%" headers="mcps1.1.4.1.1 "><p id="p158301146144319"><a name="p158301146144319"></a><a name="p158301146144319"></a>config_defination</p></td>
<td class="cellrowborder" valign="top" width="7.57%" headers="mcps1.1.4.1.2 "><p id="p1027410568437"><a name="p1027410568437"></a><a name="p1027410568437"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="79.5%" headers="mcps1.1.4.1.3 "><p id="p14527715174319"><a name="p14527715174319"></a><a name="p14527715174319"></a>Meaning: Simple configuration file. The simple configuration file distill.cfg is generated based on the distill_config_pytorch.proto file. The path of the *.proto file is: <em id="i12274856104314"><a name="i12274856104314"></a><a name="i12274856104314"></a><span id="ph244511481409"><a name="ph244511481409"></a><a name="ph244511481409"></a>AMCT</span> installation directory</em>/amct_pytorch/proto/. For the parameter description of the *.proto file and the sample simple quantization configuration file <em id="i3473122313913"><a name="i3473122313913"></a><a name="i3473122313913"></a>distill</em>.cfg, see <a href="../context/distill_config.md">Distillation Simple Configuration File</a>.</p>
<p id="p1275756114320"><a name="p1275756114320"></a><a name="p1275756114320"></a>Default value: None.</p>
<p id="p3275175644316"><a name="p3275175644316"></a><a name="p3275175644316"></a>Data type: string</p></td>
</tr>
</tbody>
</table>

## Return Value<a name="zh-cn_topic_0240188739_zh-cn_topic_0122830089_section293415513458"></a>

None

## Example<a name="zh-cn_topic_0240188739_section64231658994"></a>

```python
import amct_pytorch as amct
# Build the network graph structure to be distilled and quantized.
model = build_model()
model.load_state_dict(torch.load(state_dict_path))
input_data = tuple([torch.randn(input_shape)])

# Generate the distillation configuration file.
amct.create_distill_config(config_file="./configs/config.json",
                           model=model,
                           input_data=input_data,
                           config_defination="./configs/distill.cfg")
```

Description of the generated file:

A distillation quantization configuration file in JSON format is generated. A sample is as follows (when distillation is re-executed, the configuration file output by this API will be overwritten; the following is the configuration file in the INT8 quantization scenario):

```json
{
    "version":1,
    "batch_num":1,
    "group_size":1,
    "data_dump":false,
    "distill_group":[
        [
            "conv1",
            "bn",
            "relu"
        ],
        [
            "conv2",
            "bn2",
            "relu2"
        ]
    ],
    "conv1":{
        "quant_enable":true,
        "distill_data_config":{
            "algo":"ulq_quantize",
            "dst_type":"INT8"
        },
        "distill_weight_config":{
            "algo":"arq_distill",
            "channel_wise":true,
            "dst_type":"INT8"
        }
    },
    "conv2":{
        "quant_enable":true,
        "distill_data_config":{
            "algo":"ulq_quantize",
            "dst_type":"INT8"
        },
        "distill_weight_config":{
            "algo":"arq_distill",
            "channel_wise":true,
            "dst_type":"INT8"
        }
    }
}
```
