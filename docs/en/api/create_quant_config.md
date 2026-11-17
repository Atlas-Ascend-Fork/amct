# create\_quant\_config

<!-- md-trans-meta sourceCommit=4a47814eed18048e444f9a2acc42d9864ddc477f translatedAt=2026-10-16T08:23:29.198Z pushedAt=2026-11-05T01:18:02.568Z nl=2 -->

## Supported Products

| Product | Supported |
| ------------------------------------------- | -------- |
| Ascend 950PR&950DT products | √ |
| Atlas A3 products | √ |
| Atlas A2 products | √ |

## Description<a name="zh-cn_topic_0240187365_section15406195619561"></a>

This post-training quantization API finds all quantizable layers based on the graph structure, automatically generates a quantization configuration file, and writes the quantization configuration information of the quantizable layers to the file.

## Prototype<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_section428121323411"></a>

```
create_quant_config(config_file, model, input_data, skip_layers=None, batch_num=1, activation_offset=True, config_defination=None)
```

## Parameters<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_section795991810344"></a>

<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="10.9%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a>Parameter</p></th>
<th class="cellrowborder" valign="top" width="7.090000000000001%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0240187365_p1769255516412"><a name="zh-cn_topic_0240187365_p1769255516412"></a><a name="zh-cn_topic_0240187365_p1769255516412"></a>Input/Output</p></th>
<th class="cellrowborder" valign="top" width="82.01%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0240187365_p15231205416325"><a name="zh-cn_topic_0240187365_p15231205416325"></a><a name="zh-cn_topic_0240187365_p15231205416325"></a>Description</p></th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="10.9%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0240187365_p8886819105313"><a name="zh-cn_topic_0240187365_p8886819105313"></a><a name="zh-cn_topic_0240187365_p8886819105313"></a>config_file</p></td>
<td class="cellrowborder" valign="top" width="7.090000000000001%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0240187365_p8693185517417"><a name="zh-cn_topic_0240187365_p8693185517417"></a><a name="zh-cn_topic_0240187365_p8693185517417"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="82.01%" headers="mcps1.1.4.1.3 "><p id="p14786121019275"><a name="p14786121019275"></a><a name="p14786121019275"></a>Meaning: Path and name of the quantization configuration file to be generated. If a file with the same name already exists in the specified path, calling this API overwrites the existing file.</p>
<p id="zh-cn_topic_0240187365_p11225740182619"><a name="zh-cn_topic_0240187365_p11225740182619"></a><a name="zh-cn_topic_0240187365_p11225740182619"></a>Data type: string</p></td>
</tr>
<tr id="row550615345319"><td class="cellrowborder" valign="top" width="10.9%" headers="mcps1.1.4.1.1 "><p id="p6506175375310"><a name="p6506175375310"></a><a name="p6506175375310"></a>model</p></td>
<td class="cellrowborder" valign="top" width="7.090000000000001%" headers="mcps1.1.4.1.2 "><p id="p150610539536"><a name="p150610539536"></a><a name="p150610539536"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="82.01%" headers="mcps1.1.4.1.3 "><p id="p417791616274"><a name="p417791616274"></a><a name="p417791616274"></a>Meaning: Model to be quantized, with weights already loaded.</p>
<p id="p1550745314533"><a name="p1550745314533"></a><a name="p1550745314533"></a>Data type: torch.nn.Module</p></td>
</tr>
<tr id="zh-cn_topic_0240187365_row16254133823918"><td class="cellrowborder" valign="top" width="10.9%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0240187365_p1254338193919"><a name="zh-cn_topic_0240187365_p1254338193919"></a><a name="zh-cn_topic_0240187365_p1254338193919"></a>input_data</p></td>
<td class="cellrowborder" valign="top" width="7.090000000000001%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0240187365_p1255438123917"><a name="zh-cn_topic_0240187365_p1255438123917"></a><a name="zh-cn_topic_0240187365_p1255438123917"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="82.01%" headers="mcps1.1.4.1.3 "><p id="p32005409277"><a name="p32005409277"></a><a name="p32005409277"></a>Meaning: Input data of the model. A <span>torch.tensor</span><span> is equivalent to a tuple (</span><span>torch.tensor</span><span>).</span></p>
<p id="zh-cn_topic_0240187365_p182551638143919"><a name="zh-cn_topic_0240187365_p182551638143919"></a><a name="zh-cn_topic_0240187365_p182551638143919"></a>Data type: tuple</p></td>
</tr>
<tr id="zh-cn_topic_0240187365_row2997141123911"><td class="cellrowborder" valign="top" width="10.9%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0240187365_p99979413397"><a name="zh-cn_topic_0240187365_p99979413397"></a><a name="zh-cn_topic_0240187365_p99979413397"></a>skip_layers</p></td>
<td class="cellrowborder" valign="top" width="7.090000000000001%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0240187365_p10997194193915"><a name="zh-cn_topic_0240187365_p10997194193915"></a><a name="zh-cn_topic_0240187365_p10997194193915"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="82.01%" headers="mcps1.1.4.1.3 "><p id="p773474815272"><a name="p773474815272"></a><a name="p773474815272"></a>Meaning: Names of layers that can be quantized but do not need to be quantized.</p>
<p id="zh-cn_topic_0240187365_p163711335174212"><a name="zh-cn_topic_0240187365_p163711335174212"></a><a name="zh-cn_topic_0240187365_p163711335174212"></a>Default value: None</p>
<p id="zh-cn_topic_0240187365_p355615248166"><a name="zh-cn_topic_0240187365_p355615248166"></a><a name="zh-cn_topic_0240187365_p355615248166"></a>Data type: list, with elements of type string</p>
<p id="p167685264181"><a name="p167685264181"></a><a name="p167685264181"></a>Usage constraints: If a simple configuration file is used as the input parameter, this parameter must be set in the simple configuration file, in which case the configuration of this parameter in the input parameters does not take effect.</p></td>
</tr>
<tr id="zh-cn_topic_0240187365_row13558046183914"><td class="cellrowborder" valign="top" width="10.9%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0240187365_p455834616397"><a name="zh-cn_topic_0240187365_p455834616397"></a><a name="zh-cn_topic_0240187365_p455834616397"></a>batch_num</p></td>
<td class="cellrowborder" valign="top" width="7.090000000000001%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0240187365_p855954603911"><a name="zh-cn_topic_0240187365_p855954603911"></a><a name="zh-cn_topic_0240187365_p855954603911"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="82.01%" headers="mcps1.1.4.1.3 "><p id="p6650185119275"><a name="p6650185119275"></a><a name="p6650185119275"></a>Meaning: Number of batches used for quantization, that is, the number of batches of data used to generate quantization factors.</p>
<p id="zh-cn_topic_0240187365_p3913143317168"><a name="zh-cn_topic_0240187365_p3913143317168"></a><a name="zh-cn_topic_0240187365_p3913143317168"></a>Data type: int</p>
<p id="zh-cn_topic_0240187365_p12301331144210"><a name="zh-cn_topic_0240187365_p12301331144210"></a><a name="zh-cn_topic_0240187365_p12301331144210"></a>Value range: integer greater than 0, with a default value of 1.</p>
<p id="p175532467564"><a name="p175532467564"></a><a name="p175532467564"></a>Usage constraints:</p>
<a name="ul205257489562"></a><a name="ul205257489562"></a><ul id="ul205257489562"><li>batch_num should not be too large. The product of batch_num and batch_size is the number of images used in the quantization process, and too many images consume a large amount of memory.</li><li>If a simple configuration file is used as the input parameter, this parameter must be set in the simple configuration file, in which case the configuration of this parameter in the input parameters does not take effect.</li></ul></td>
</tr>
<tr id="zh-cn_topic_0240187365_row473005916274"><td class="cellrowborder" valign="top" width="10.9%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0240187365_p146233804215"><a name="zh-cn_topic_0240187365_p146233804215"></a><a name="zh-cn_topic_0240187365_p146233804215"></a>activation_offset</p></td>
<td class="cellrowborder" valign="top" width="7.090000000000001%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0240187365_p1546220381429"><a name="zh-cn_topic_0240187365_p1546220381429"></a><a name="zh-cn_topic_0240187365_p1546220381429"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="82.01%" headers="mcps1.1.4.1.3 "><p id="p13171222162812"><a name="p13171222162812"></a><a name="p13171222162812"></a>Meaning: Whether data quantization carries an offset.</p>
<p id="zh-cn_topic_0240187365_p6462103818429"><a name="zh-cn_topic_0240187365_p6462103818429"></a><a name="zh-cn_topic_0240187365_p6462103818429"></a>Default value: True</p>
<p id="zh-cn_topic_0240187365_p1588701864411"><a name="zh-cn_topic_0240187365_p1588701864411"></a><a name="zh-cn_topic_0240187365_p1588701864411"></a>Data type: bool</p>
<p id="p36813346566"><a name="p36813346566"></a><a name="p36813346566"></a>Usage constraints: If a simple configuration file is used as the input parameter, this parameter must be set in the simple configuration file, in which case the configuration of this parameter in the input parameters does not take effect.</p></td>
</tr>
<tr id="row8672528631"><td class="cellrowborder" valign="top" width="10.9%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0240188739_p12612155914458"><a name="zh-cn_topic_0240188739_p12612155914458"></a><a name="zh-cn_topic_0240188739_p12612155914458"></a>config_defination</p></td>
<td class="cellrowborder" valign="top" width="7.090000000000001%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0240188739_p46121659184520"><a name="zh-cn_topic_0240188739_p46121659184520"></a><a name="zh-cn_topic_0240188739_p46121659184520"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="82.01%" headers="mcps1.1.4.1.3 "><p id="p77792024182811"><a name="p77792024182811"></a><a name="p77792024182811"></a>Meaning: The simple quantization configuration file <em id="zh-cn_topic_0240188739_i04991385212"><a name="zh-cn_topic_0240188739_i04991385212"></a><a name="zh-cn_topic_0240188739_i04991385212"></a>quant</em>.cfg generated based on the calibration_config_pytorch.proto file. The path of the *.proto file is: <em id="zh-cn_topic_0240188739_i13405105118558"><a name="zh-cn_topic_0240188739_i13405105118558"></a><a name="zh-cn_topic_0240188739_i13405105118558"></a><span id="ph244511481409"><a name="ph244511481409"></a><a name="ph244511481409"></a>AMCT</span> installation directory</em>/amct_pytorch/proto/.</p>
<p id="zh-cn_topic_0240188739_p153825102371"><a name="zh-cn_topic_0240188739_p153825102371"></a><a name="zh-cn_topic_0240188739_p153825102371"></a>For the parameter description of the *.proto file and the sample of the generated <em id="zh-cn_topic_0240188739_i8781171513315"><a name="zh-cn_topic_0240188739_i8781171513315"></a><a name="zh-cn_topic_0240188739_i8781171513315"></a>quant</em>.cfg simple quantization configuration file, see <a href="../context/ptq_config.md">Post-training Quantization Simple Configuration File</a>.</p>
<p id="zh-cn_topic_0240188739_p1944053764718"><a name="zh-cn_topic_0240188739_p1944053764718"></a><a name="zh-cn_topic_0240188739_p1944053764718"></a>Default value: None</p>
<p id="zh-cn_topic_0240188739_p86121059144513"><a name="zh-cn_topic_0240188739_p86121059144513"></a><a name="zh-cn_topic_0240188739_p86121059144513"></a>Data type: string</p>
<p id="zh-cn_topic_0240188739_p337494234812"><a name="zh-cn_topic_0240188739_p337494234812"></a><a name="zh-cn_topic_0240188739_p337494234812"></a>Usage constraints: When the value is None, the configuration file is generated using the input parameters; otherwise, the other quantization parameters (skip_layers, batch_num, activation_offset) in the input are ignored, and a configuration file in JSON format is generated based on the simple quantization configuration file parameter config_defination.</p></td>
</tr>
</tbody>
</table>

## Layers and Constraints Supported by Post-Training Quantization<a id="section001"></a>

<table><thead align="left"><tr id="row9214202215138"><th class="cellrowborder" valign="top" width="20.32%" id="mcps1.2.4.1.1"><p id="p1921472241319"><a name="p1921472241319"></a><a name="p1921472241319"></a>Supported Layer Type</p></th>
<th class="cellrowborder" valign="top" width="41.17%" id="mcps1.2.4.1.2"><p id="p122141822101318"><a name="p122141822101318"></a><a name="p122141822101318"></a>Constraint</p></th>
<th class="cellrowborder" valign="top" width="38.51%" id="mcps1.2.4.1.3"><p id="p421411228138"><a name="p421411228138"></a><a name="p421411228138"></a>Remarks</p></th>
</tr>
</thead>
<tbody><tr id="row7214182271318"><td class="cellrowborder" valign="top" width="20.32%" headers="mcps1.2.4.1.1 "><p id="p32146222139"><a name="p32146222139"></a><a name="p32146222139"></a>torch.nn.Linear</p></td>
<td class="cellrowborder" valign="top" width="41.17%" headers="mcps1.2.4.1.2 "><p id="p162141722111314"><a name="p162141722111314"></a><a name="p162141722111314"></a>-</p></td>
<td class="cellrowborder" rowspan="4" valign="top" width="38.51%" headers="mcps1.2.4.1.3 "><p id="p381615103334"><a name="p381615103334"></a><a name="p381615103334"></a>Reused layers (sharing weight and bias parameters) do not support quantization.</p></td>
</tr>
<tr id="row1521462251317"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1021414226138"><a name="p1021414226138"></a><a name="p1021414226138"></a>torch.nn.Conv2d</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><a name="ul04571458192118"></a><a name="ul04571458192118"></a><ul id="ul04571458192118"><li>padding_mode is zeros</li><li>Only supports input data with the shape (N, Cin, Hin, Win)</li></ul></td>
</tr>
<tr id="row1551419103911"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p731502015364"><a name="p731502015364"></a><a name="p731502015364"></a>torch.nn.Conv3d</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><a name="ul17271930142215"></a><a name="ul17271930142215"></a><ul id="ul17271930142215"><li>dilation_d is 1, and dilation_h/dilation_w &gt;= 1.</li><li>Only supports input data with the shape (N, Cin, Din, Hin, Win).</li></ul></td>
</tr>
<tr id="row6215102221319"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1421510225139"><a name="p1421510225139"></a><a name="p1421510225139"></a>torch.nn.ConvTranspose2d</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><a name="ul460311315239"></a><a name="ul460311315239"></a><ul id="ul460311315239"><li>padding_mode is zeros.</li><li>Only supports input data with the shape (N, Cin, Hin, Win).</li></ul></td>
</tr>
<tr id="row163051438934"><td class="cellrowborder" valign="top" width="20.32%" headers="mcps1.2.4.1.1 "><p id="p103051638333"><a name="p103051638333"></a><a name="p103051638333"></a>torch.nn.AvgPool2d</p></td>
<td class="cellrowborder" valign="top" width="41.17%" headers="mcps1.2.4.1.2 "><p id="p4305738238"><a name="p4305738238"></a><a name="p4305738238"></a>-</p></td>
<td class="cellrowborder" valign="top" width="38.51%" headers="mcps1.2.4.1.3 "><p id="p33051338039"><a name="p33051338039"></a><a name="p33051338039"></a>-</p></td>
</tr>
</tbody>
</table>

## Layers and Constraints Supported by Balanced Quantization Preprocessing

<a name="table32131022111318"></a>

<table><thead align="left"><tr id="row9214202215138"><th class="cellrowborder" valign="top" width="23.242324232423243%" id="mcps1.2.4.1.1"><p id="p1921472241319"><a name="p1921472241319"></a><a name="p1921472241319"></a>Supported Layer Type</p></th>
<th class="cellrowborder" valign="top" width="31.91319131913191%" id="mcps1.2.4.1.2"><p id="p122141822101318"><a name="p122141822101318"></a><a name="p122141822101318"></a>Constraint</p></th>
<th class="cellrowborder" valign="top" width="44.844484448444845%" id="mcps1.2.4.1.3"><p id="p421411228138"><a name="p421411228138"></a><a name="p421411228138"></a>Remarks</p></th>
</tr>
</thead>
<tbody><tr id="row7214182271318"><td class="cellrowborder" valign="top" width="23.242324232423243%" headers="mcps1.2.4.1.1 "><p id="p32146222139"><a name="p32146222139"></a><a name="p32146222139"></a>torch.nn.Linear</p></td>
<td class="cellrowborder" valign="top" width="31.91319131913191%" headers="mcps1.2.4.1.2 "><p id="p162141722111314"><a name="p162141722111314"></a><a name="p162141722111314"></a>-</p></td>
<td class="cellrowborder" rowspan="4" valign="top" width="44.844484448444845%" headers="mcps1.2.4.1.3 "><p id="p381615103334"><a name="p381615103334"></a><a name="p381615103334"></a>Reused layers (sharing weight and bias parameters) do not support quantization.</p></td>
</tr>
<tr id="row1521462251317"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1021414226138"><a name="p1021414226138"></a><a name="p1021414226138"></a>torch.nn.Conv2d</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p8417519168"><a name="p8417519168"></a><a name="p8417519168"></a>padding_mode is zeros</p></td>
</tr>
<tr id="row1551419103911"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p731502015364"><a name="p731502015364"></a><a name="p731502015364"></a>torch.nn.Conv3d</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p05616359369"><a name="p05616359369"></a><a name="p05616359369"></a>dilation_d is 1, dilation_h/dilation_w &gt;= 1.</p>
<p id="p72994182514"><a name="p72994182514"></a><a name="p72994182514"></a>padding_mode is zeros.</p></td>
</tr>
<tr id="row6215102221319"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1421510225139"><a name="p1421510225139"></a><a name="p1421510225139"></a>torch.nn.ConvTranspose2d</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p924818415338"><a name="p924818415338"></a><a name="p924818415338"></a>padding_mode is zeros.</p></td>
</tr>
</tbody>
</table>

## Return Value<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_section293415513458"></a>

None

## Example<a name="zh-cn_topic_0240187365_section64231658994"></a>

```
import amct_pytorch as amct
# Build the network graph structure to be quantized.
model = build_model()
model.load_state_dict(torch.load(state_dict_path))
input_data = tuple([torch.randn(input_shape)])
model.eval()

# Generate the quantization configuration file.
amct.create_quant_config(config_file="./configs/config.json",
                         model=model,
                         input_data=input_data,
                         skip_layers=None,
                         batch_num=1,
                         activation_offset=True)
```

Description of the file written to disk: A quantization configuration file in JSON format is generated. A sample is as follows (when quantization is re-executed, the quantization configuration file generated by this API will be overwritten). For parameter description, see [Post-training Quantization Configuration Parameters](../context/ptq_config_param.md).

- Post-training quantization configuration file (data quantization uses the [IFMR data quantization algorithm](../algorithm_brief.md))

    ```
    {
        "version":1,
        "batch_num":2,
        "activation_offset":true,
        "do_fusion":true,
        "skip_fusion_layers":[],
        "conv1":{
            "quant_enable":true,
            "dmq_balancer_param":0.5,
            "activation_quant_params":{
                "num_bits":8,
                "max_percentile":0.999999,
                "min_percentile":0.999999,
                "search_range":[
                    0.7,
                    1.3
                ],
                "search_step":0.01,
                "act_algo":"ifmr",
                "asymmetric":false
            },
            "weight_quant_params":{
                "num_bits":8,
                "wts_algo":"arq_quantize",
                "channel_wise":true
            }
        },
        "fc":{
            "quant_enable":true,
            "dmq_balancer_param":0.5,
            "activation_quant_params":{
                "num_bits":8,
                "max_percentile":0.999999,
                "min_percentile":0.999999,
                "search_range":[
                    0.7,
                    1.3
                ],
                "search_step":0.01,
                "act_algo":"ifmr",
                "asymmetric":false
            },
            "weight_quant_params":{
                "num_bits":8,
                "wts_algo":"arq_quantize",
                "channel_wise":false
            }
        }
    }
    ```

- Post-training quantization configuration file (data quantization uses the [HFMG data quantization algorithm](../algorithm_brief.md))

    ```
    {
        "version":1,
        "batch_num":2,
        "activation_offset":true,
        "do_fusion":true,
        "skip_fusion_layers":[],
        "conv1":{
            "quant_enable":true,
            "dmq_balancer_param":0.5,
            "activation_quant_params":{
                "num_bits":8,
                "act_algo":"hfmg",
                "num_of_bins":4096,
                "asymmetric":false
            },
            "weight_quant_params":{
                "num_bits":8,
                "wts_algo":"arq_quantize",
                "channel_wise":true
            }
        }
    }
    ```

- Adaptive rounding quantization simple configuration file (weight quantization uses the [ADA weight quantization algorithm](../algorithm_brief.md))

    ```
    "layer_name1":{
            "quant_enable":true,
            "weight_quant_params":{
                "wts_algo":"ada_quantize",
                "num_iteration":10000,
                "reg_param":0.1,
                "beta_range":[20,2], 
      "warm_start":0.2,
      "num_bits":8,
      "channel_wise":true
            }
        }
    ```
