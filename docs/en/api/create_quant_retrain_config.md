# create\_quant\_retrain\_config<a name="ZH-CN_TOPIC_0000002517028880"></a>

<!-- md-trans-meta sourceCommit=4a47814eed18048e444f9a2acc42d9864ddc477f translatedAt=2026-10-16T08:24:17.966Z pushedAt=2026-11-05T01:12:03.414Z nl=2 -->

## Supported Products<a name="section197451857688"></a>

| Product | Supported |
| ------------------------------------------- | -------- |
| Ascend 950PR&950DT products | √ |
| Atlas A3 products | √ |
| Atlas A2 products | √ |

## Description<a name="zh-cn_topic_0240187365_section15406195619561"></a>

This is a quantization-aware training API. It finds all quantizable layers based on the graph structure, automatically generates a quantization configuration file, and writes the quantization configuration information of the quantizable layers into the configuration file.

## Prototype<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_section428121323411"></a>

```python
create_quant_retrain_config(config_file, model, input_data, config_defination=None)
```

## Parameters<a name="section73811524135618"></a>

<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="12.370000000000001%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a>Parameter</p></th>
<th class="cellrowborder" valign="top" width="7.290000000000001%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0240187365_p1769255516412"><a name="zh-cn_topic_0240187365_p1769255516412"></a><a name="zh-cn_topic_0240187365_p1769255516412"></a>Input/Output</p></th>
<th class="cellrowborder" valign="top" width="80.34%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0240187365_p15231205416325"><a name="zh-cn_topic_0240187365_p15231205416325"></a><a name="zh-cn_topic_0240187365_p15231205416325"></a>Description</p></th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="12.370000000000001%" headers="mcps1.1.4.1.1 "><p id="p3829146134310"><a name="p3829146134310"></a><a name="p3829146134310"></a>config_file</p></td>
<td class="cellrowborder" valign="top" width="7.290000000000001%" headers="mcps1.1.4.1.2 "><p id="p11274056174319"><a name="p11274056174319"></a><a name="p11274056174319"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="80.34%" headers="mcps1.1.4.1.3 "><p id="p20591103915201"><a name="p20591103915201"></a><a name="p20591103915201"></a>Meaning: Path and name of the quantization-aware training configuration file to be generated. If a file with the same name already exists in the specified path, calling this API overwrites the existing file.</p>
<p id="p1727418563433"><a name="p1727418563433"></a><a name="p1727418563433"></a>Data type: string</p></td>
</tr>
<tr id="row10475143684319"><td class="cellrowborder" valign="top" width="12.370000000000001%" headers="mcps1.1.4.1.1 "><p id="p128298463436"><a name="p128298463436"></a><a name="p128298463436"></a>model</p></td>
<td class="cellrowborder" valign="top" width="7.290000000000001%" headers="mcps1.1.4.1.2 "><p id="p8274556154316"><a name="p8274556154316"></a><a name="p8274556154316"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="80.34%" headers="mcps1.1.4.1.3 "><p id="p121542444203"><a name="p121542444203"></a><a name="p121542444203"></a>Meaning: Model to undergo quantization-aware training, with weights already loaded.</p>
<p id="p17274105617434"><a name="p17274105617434"></a><a name="p17274105617434"></a>Data type: torch.nn.Module</p></td>
</tr>
<tr id="row11737244204314"><td class="cellrowborder" valign="top" width="12.370000000000001%" headers="mcps1.1.4.1.1 "><p id="p18829134610436"><a name="p18829134610436"></a><a name="p18829134610436"></a>input_data</p></td>
<td class="cellrowborder" valign="top" width="7.290000000000001%" headers="mcps1.1.4.1.2 "><p id="p18274156124317"><a name="p18274156124317"></a><a name="p18274156124317"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="80.34%" headers="mcps1.1.4.1.3 "><p id="p1970764572015"><a name="p1970764572015"></a><a name="p1970764572015"></a>Meaning: Input data of the model. A torch.tensor is treated as equivalent to tuple(torch.tensor).</p>
<p id="p927405684319"><a name="p927405684319"></a><a name="p927405684319"></a>Data type: tuple</p></td>
</tr>
<tr id="row41919427437"><td class="cellrowborder" valign="top" width="12.370000000000001%" headers="mcps1.1.4.1.1 "><p id="p158301146144319"><a name="p158301146144319"></a><a name="p158301146144319"></a>config_defination</p></td>
<td class="cellrowborder" valign="top" width="7.290000000000001%" headers="mcps1.1.4.1.2 "><p id="p1027410568437"><a name="p1027410568437"></a><a name="p1027410568437"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="80.34%" headers="mcps1.1.4.1.3 "><p id="p17143647122018"><a name="p17143647122018"></a><a name="p17143647122018"></a>Meaning: Simple configuration file.</p>
<p id="p827435611439"><a name="p827435611439"></a><a name="p827435611439"></a>Simple configuration file quant.cfg generated based on the retrain_config_pytorch.proto file. The path of the *.proto file is: <em id="i12274856104314"><a name="i12274856104314"></a><a name="i12274856104314"></a><span id="ph244511481409"><a name="ph244511481409"></a><a name="ph244511481409"></a>AMCT</span> installation directory</em>/amct_pytorch/proto/. For the parameter description of the *.proto file and a sample of the generated <em id="i1127455694315"><a name="i1127455694315"></a><a name="i1127455694315"></a>quant</em>.cfg simple quantization configuration file, see <a href="../context/qat_config.md">Quantization-Aware Training Simple Configuration File</a>.</p>
<p id="p1275756114320"><a name="p1275756114320"></a><a name="p1275756114320"></a>Default value: None.</p>
<p id="p3275175644316"><a name="p3275175644316"></a><a name="p3275175644316"></a>Data type: string</p></td>
</tr>
</tbody>
</table>

## Quantization-Supported Layers and Constraints

<a name="table206799319355"></a>

<table><thead align="left"><tr id="row26791539351"><th class="cellrowborder" valign="top" width="19.521952195219523%" id="mcps1.2.4.1.1"><p id="p1921472241319"><a name="p1921472241319"></a><a name="p1921472241319"></a>Supported Layer Type</p></th>
<th class="cellrowborder" valign="top" width="51.93519351935193%" id="mcps1.2.4.1.2"><p id="p16791313517"><a name="p16791313517"></a><a name="p16791313517"></a>Constraint</p></th>
<th class="cellrowborder" valign="top" width="28.542854285428543%" id="mcps1.2.4.1.3"><p id="p86801434351"><a name="p86801434351"></a><a name="p86801434351"></a>Remarks</p></th>
</tr>
</thead>
<tbody><tr id="row168043183517"><td class="cellrowborder" valign="top" width="19.521952195219523%" headers="mcps1.2.4.1.1 "><p id="p46801332355"><a name="p46801332355"></a><a name="p46801332355"></a>torch.nn.Linear</p></td>
<td class="cellrowborder" valign="top" width="51.93519351935193%" headers="mcps1.2.4.1.2 "><p id="p136801833354"><a name="p136801833354"></a><a name="p136801833354"></a>-</p></td>
<td class="cellrowborder" rowspan="3" valign="top" width="28.542854285428543%" headers="mcps1.2.4.1.3 "><p id="p15127478257"><a name="p15127478257"></a><a name="p15127478257"></a>Reused layers (sharing weight and bias parameters) do not support quantization.</p></td>
</tr>
<tr id="row6680153123510"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1868011363517"><a name="p1868011363517"></a><a name="p1868011363517"></a>torch.nn.Conv2d</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><a name="ul16547311918"></a><a name="ul16547311918"></a><ul id="ul16547311918"><li>padding_mode is zeros.</li><li>Due to hardware constraints, quantization-aware training is not recommended when the input channel count Cin of the original model is less than or equal to 16; otherwise, the inference accuracy of the quantized deployment model may degrade.</li><li>Only the input data shape (N, Cin, Hin, Win) is supported.</li></ul></td>
</tr>
<tr id="row24121633571"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p04138331776"><a name="p04138331776"></a><a name="p04138331776"></a>torch.nn.ConvTranspose2d</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><a name="ul4861040154317"></a><a name="ul4861040154317"></a><ul id="ul4861040154317"><li>padding_mode is zeros.</li><li>Due to hardware constraints, quantization-aware training is not recommended when the number of input channels Cin in the original model is less than or equal to 16; otherwise, the inference accuracy of the quantized deployment model may degrade.</li><li>Only the input data shape (N, Cin, Hin, Win) is supported.</li></ul></td>
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
amct.create_quant_retrain_config(config_file="./configs/config.json",
                            model=model,
                            input_data=input_data)
```

Description of the generated file: A quantization-aware training configuration file in JSON format is generated. A sample is as follows (when quantization-aware training is re-executed, the configuration file output by this API will be overwritten). For parameter descriptions, see [Quantization-Aware Training Configuration Parameters](../context/qat_config_param.md).

```json
{
    "version":1,
    "batch_num":1,
    "conv1":{
        "retrain_enable":true,
        "retrain_data_config":{
            "algo":"ulq_quantize",
            "dst_type":"INT8"
        },
        "retrain_weight_config":{
            "algo":"arq_retrain",
            "channel_wise":true,
            "dst_type":"INT8"
        }
    },
    "layer1.0.conv1":{
        "retrain_enable":true,
        "retrain_data_config":{
            "algo":"ulq_quantize",
            "dst_type":"INT8"
        },
        "retrain_weight_config":{
            "algo":"arq_retrain",
            "channel_wise":true,
            "dst_type":"INT8"
        }
    },
    "fc":{
        "retrain_enable":true,
        "retrain_data_config":{
            "algo":"ulq_quantize",
            "dst_type":"INT8"
        },
        "retrain_weight_config":{
            "algo":"arq_retrain",
            "channel_wise":false,
            "dst_type":"INT8"
        }
    }
}
```
