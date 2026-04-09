# create\_quant\_config

## 产品支持情况

<a name="zh-cn_topic_0000002517188794_table38301303189"></a>

| 产品                                        | 是否支持 |
| ------------------------------------------- | -------- |
| Ascend 950PR/Ascend 950DT                   | √        |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √        |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √        |



## 功能说明<a name="zh-cn_topic_0240187365_section15406195619561"></a>

训练后量化接口，根据图的结构找到所有可量化的层，自动生成量化配置文件，并将可量化层的量化配置信息写入文件。

## 函数原型<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_section428121323411"></a>

```
create_quant_config(config_file, model, input_data, skip_layers=None, batch_num=1, activation_offset=True, config_defination=None)
```

## 参数说明<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_section795991810344"></a>

<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="10.9%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="7.090000000000001%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0240187365_p1769255516412"><a name="zh-cn_topic_0240187365_p1769255516412"></a><a name="zh-cn_topic_0240187365_p1769255516412"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="82.01%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0240187365_p15231205416325"><a name="zh-cn_topic_0240187365_p15231205416325"></a><a name="zh-cn_topic_0240187365_p15231205416325"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="10.9%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0240187365_p8886819105313"><a name="zh-cn_topic_0240187365_p8886819105313"></a><a name="zh-cn_topic_0240187365_p8886819105313"></a>config_file</p>
</td>
<td class="cellrowborder" valign="top" width="7.090000000000001%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0240187365_p8693185517417"><a name="zh-cn_topic_0240187365_p8693185517417"></a><a name="zh-cn_topic_0240187365_p8693185517417"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="82.01%" headers="mcps1.1.4.1.3 "><p id="p14786121019275"><a name="p14786121019275"></a><a name="p14786121019275"></a>含义：待生成的量化配置文件存放路径及名称。如果存放路径下已经存在该文件，则调用该接口时会覆盖已有文件。</p>
<p id="zh-cn_topic_0240187365_p11225740182619"><a name="zh-cn_topic_0240187365_p11225740182619"></a><a name="zh-cn_topic_0240187365_p11225740182619"></a>数据类型：string</p>
</td>
</tr>
<tr id="row550615345319"><td class="cellrowborder" valign="top" width="10.9%" headers="mcps1.1.4.1.1 "><p id="p6506175375310"><a name="p6506175375310"></a><a name="p6506175375310"></a>model</p>
</td>
<td class="cellrowborder" valign="top" width="7.090000000000001%" headers="mcps1.1.4.1.2 "><p id="p150610539536"><a name="p150610539536"></a><a name="p150610539536"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="82.01%" headers="mcps1.1.4.1.3 "><p id="p417791616274"><a name="p417791616274"></a><a name="p417791616274"></a>含义：待量化的模型，已加载权重。</p>
<p id="p1550745314533"><a name="p1550745314533"></a><a name="p1550745314533"></a>数据类型：torch.nn.Module</p>
</td>
</tr>
<tr id="zh-cn_topic_0240187365_row16254133823918"><td class="cellrowborder" valign="top" width="10.9%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0240187365_p1254338193919"><a name="zh-cn_topic_0240187365_p1254338193919"></a><a name="zh-cn_topic_0240187365_p1254338193919"></a>input_data</p>
</td>
<td class="cellrowborder" valign="top" width="7.090000000000001%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0240187365_p1255438123917"><a name="zh-cn_topic_0240187365_p1255438123917"></a><a name="zh-cn_topic_0240187365_p1255438123917"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="82.01%" headers="mcps1.1.4.1.3 "><p id="p32005409277"><a name="p32005409277"></a><a name="p32005409277"></a>含义：模型的输入数据。一个<span>torch.tensor</span><span>会被等价为tuple（</span><span>torch.tensor</span><span>）</span>。</p>
<p id="zh-cn_topic_0240187365_p182551638143919"><a name="zh-cn_topic_0240187365_p182551638143919"></a><a name="zh-cn_topic_0240187365_p182551638143919"></a>数据类型：tuple</p>
</td>
</tr>
<tr id="zh-cn_topic_0240187365_row2997141123911"><td class="cellrowborder" valign="top" width="10.9%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0240187365_p99979413397"><a name="zh-cn_topic_0240187365_p99979413397"></a><a name="zh-cn_topic_0240187365_p99979413397"></a>skip_layers</p>
</td>
<td class="cellrowborder" valign="top" width="7.090000000000001%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0240187365_p10997194193915"><a name="zh-cn_topic_0240187365_p10997194193915"></a><a name="zh-cn_topic_0240187365_p10997194193915"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="82.01%" headers="mcps1.1.4.1.3 "><p id="p773474815272"><a name="p773474815272"></a><a name="p773474815272"></a>含义：可量化但不需要量化的层名。</p>
<p id="zh-cn_topic_0240187365_p163711335174212"><a name="zh-cn_topic_0240187365_p163711335174212"></a><a name="zh-cn_topic_0240187365_p163711335174212"></a>默认值：None</p>
<p id="zh-cn_topic_0240187365_p355615248166"><a name="zh-cn_topic_0240187365_p355615248166"></a><a name="zh-cn_topic_0240187365_p355615248166"></a>数据类型：list，列表中元素类型为string</p>
<p id="p167685264181"><a name="p167685264181"></a><a name="p167685264181"></a>使用约束：如果使用简易配置文件作为入参，则该参数需要在简易配置文件中设置，此时输入参数中该参数配置不生效。</p>
</td>
</tr>
<tr id="zh-cn_topic_0240187365_row13558046183914"><td class="cellrowborder" valign="top" width="10.9%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0240187365_p455834616397"><a name="zh-cn_topic_0240187365_p455834616397"></a><a name="zh-cn_topic_0240187365_p455834616397"></a>batch_num</p>
</td>
<td class="cellrowborder" valign="top" width="7.090000000000001%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0240187365_p855954603911"><a name="zh-cn_topic_0240187365_p855954603911"></a><a name="zh-cn_topic_0240187365_p855954603911"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="82.01%" headers="mcps1.1.4.1.3 "><p id="p6650185119275"><a name="p6650185119275"></a><a name="p6650185119275"></a>含义：量化使用的batch数量，即使用多少个batch的数据生成量化因子。</p>
<p id="zh-cn_topic_0240187365_p3913143317168"><a name="zh-cn_topic_0240187365_p3913143317168"></a><a name="zh-cn_topic_0240187365_p3913143317168"></a>数据类型：int</p>
<p id="zh-cn_topic_0240187365_p12301331144210"><a name="zh-cn_topic_0240187365_p12301331144210"></a><a name="zh-cn_topic_0240187365_p12301331144210"></a>取值范围：大于0的整数，默认值为1。</p>
<p id="p175532467564"><a name="p175532467564"></a><a name="p175532467564"></a>使用约束：</p>
<a name="ul205257489562"></a><a name="ul205257489562"></a><ul id="ul205257489562"><li>batch_num不宜过大，batch_num与batch_size的乘积为量化过程中使用的图片数量，过多的图片会占用较大的内存。</li><li>如果使用简易配置文件作为入参，则该参数需要在简易配置文件中设置，此时输入参数中该参数配置不生效。</li></ul>
</td>
</tr>
<tr id="zh-cn_topic_0240187365_row473005916274"><td class="cellrowborder" valign="top" width="10.9%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0240187365_p146233804215"><a name="zh-cn_topic_0240187365_p146233804215"></a><a name="zh-cn_topic_0240187365_p146233804215"></a>activation_offset</p>
</td>
<td class="cellrowborder" valign="top" width="7.090000000000001%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0240187365_p1546220381429"><a name="zh-cn_topic_0240187365_p1546220381429"></a><a name="zh-cn_topic_0240187365_p1546220381429"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="82.01%" headers="mcps1.1.4.1.3 "><p id="p13171222162812"><a name="p13171222162812"></a><a name="p13171222162812"></a>含义：数据量化是否带offset。</p>
<p id="zh-cn_topic_0240187365_p6462103818429"><a name="zh-cn_topic_0240187365_p6462103818429"></a><a name="zh-cn_topic_0240187365_p6462103818429"></a>默认值：True</p>
<p id="zh-cn_topic_0240187365_p1588701864411"><a name="zh-cn_topic_0240187365_p1588701864411"></a><a name="zh-cn_topic_0240187365_p1588701864411"></a>数据类型：bool</p>
<p id="p36813346566"><a name="p36813346566"></a><a name="p36813346566"></a>使用约束：如果使用简易配置文件作为入参，则该参数需要在简易配置文件中设置，此时输入参数中该参数配置不生效。</p>
</td>
</tr>
<tr id="row8672528631"><td class="cellrowborder" valign="top" width="10.9%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0240188739_p12612155914458"><a name="zh-cn_topic_0240188739_p12612155914458"></a><a name="zh-cn_topic_0240188739_p12612155914458"></a>config_defination</p>
</td>
<td class="cellrowborder" valign="top" width="7.090000000000001%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0240188739_p46121659184520"><a name="zh-cn_topic_0240188739_p46121659184520"></a><a name="zh-cn_topic_0240188739_p46121659184520"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="82.01%" headers="mcps1.1.4.1.3 "><p id="p77792024182811"><a name="p77792024182811"></a><a name="p77792024182811"></a>含义：基于calibration_config_pytorch.proto文件生成的简易量化配置文件<em id="zh-cn_topic_0240188739_i04991385212"><a name="zh-cn_topic_0240188739_i04991385212"></a><a name="zh-cn_topic_0240188739_i04991385212"></a>quant</em>.cfg，*.proto文件所在路径为：<em id="zh-cn_topic_0240188739_i13405105118558"><a name="zh-cn_topic_0240188739_i13405105118558"></a><a name="zh-cn_topic_0240188739_i13405105118558"></a><span id="ph244511481409"><a name="ph244511481409"></a><a name="ph244511481409"></a>AMCT</span>安装目录</em>/amct_pytorch/proto/。</p>
<p id="zh-cn_topic_0240188739_p153825102371"><a name="zh-cn_topic_0240188739_p153825102371"></a><a name="zh-cn_topic_0240188739_p153825102371"></a>*.proto文件参数解释以及生成的<em id="zh-cn_topic_0240188739_i8781171513315"><a name="zh-cn_topic_0240188739_i8781171513315"></a><a name="zh-cn_topic_0240188739_i8781171513315"></a>quant</em>.cfg简易量化配置文件样例请参见<a href="../context/训练后量化简易配置文件.md">训练后量化简易配置文件</a>。</p>
<p id="zh-cn_topic_0240188739_p1944053764718"><a name="zh-cn_topic_0240188739_p1944053764718"></a><a name="zh-cn_topic_0240188739_p1944053764718"></a>默认值：None</p>
<p id="zh-cn_topic_0240188739_p86121059144513"><a name="zh-cn_topic_0240188739_p86121059144513"></a><a name="zh-cn_topic_0240188739_p86121059144513"></a>数据类型：string</p>
<p id="zh-cn_topic_0240188739_p337494234812"><a name="zh-cn_topic_0240188739_p337494234812"></a><a name="zh-cn_topic_0240188739_p337494234812"></a>使用约束：当取值为None时，使用输入参数生成配置文件；否则，忽略输入的其他量化参数（skip_layers，batch_num，activation_offset），根据简易量化配置文件参数config_defination生成JSON格式的配置文件。</p>
</td>
</tr>
</tbody>
</table>

## 训练后量化支持的层及约束<a id="section001"></a>

<table><thead align="left"><tr id="row9214202215138"><th class="cellrowborder" valign="top" width="20.32%" id="mcps1.2.4.1.1"><p id="p1921472241319"><a name="p1921472241319"></a><a name="p1921472241319"></a>支持的层类型</p>
</th>
<th class="cellrowborder" valign="top" width="41.17%" id="mcps1.2.4.1.2"><p id="p122141822101318"><a name="p122141822101318"></a><a name="p122141822101318"></a>约束</p>
</th>
<th class="cellrowborder" valign="top" width="38.51%" id="mcps1.2.4.1.3"><p id="p421411228138"><a name="p421411228138"></a><a name="p421411228138"></a>备注</p>
</th>
</tr>
</thead>
<tbody><tr id="row7214182271318"><td class="cellrowborder" valign="top" width="20.32%" headers="mcps1.2.4.1.1 "><p id="p32146222139"><a name="p32146222139"></a><a name="p32146222139"></a>torch.nn.Linear</p>
</td>
<td class="cellrowborder" valign="top" width="41.17%" headers="mcps1.2.4.1.2 "><p id="p162141722111314"><a name="p162141722111314"></a><a name="p162141722111314"></a>-</p>
</td>
<td class="cellrowborder" rowspan="4" valign="top" width="38.51%" headers="mcps1.2.4.1.3 "><p id="p381615103334"><a name="p381615103334"></a><a name="p381615103334"></a>复用层（共用weight和bias参数）不支持量化。</p>
</td>
</tr>
<tr id="row1521462251317"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1021414226138"><a name="p1021414226138"></a><a name="p1021414226138"></a>torch.nn.Conv2d</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><a name="ul04571458192118"></a><a name="ul04571458192118"></a><ul id="ul04571458192118"><li>padding_mode为zeros</li><li>只支持input data的shape为(N, Cin, Hin, Win)</li></ul>
</td>
</tr>
<tr id="row1551419103911"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p731502015364"><a name="p731502015364"></a><a name="p731502015364"></a>torch.nn.Conv3d</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><a name="ul17271930142215"></a><a name="ul17271930142215"></a><ul id="ul17271930142215"><li>dilation_d为1，dilation_h/dilation_w &gt;= 1</li><li>只支持input data的shape为(N, Cin, Din, Hin, Win)</li></ul>
</td>
</tr>
<tr id="row6215102221319"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1421510225139"><a name="p1421510225139"></a><a name="p1421510225139"></a>torch.nn.ConvTranspose2d</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><a name="ul460311315239"></a><a name="ul460311315239"></a><ul id="ul460311315239"><li>padding_mode为zeros</li><li>只支持input data的shape为(N, Cin, Hin, Win)</li></ul>
</td>
</tr>
<tr id="row163051438934"><td class="cellrowborder" valign="top" width="20.32%" headers="mcps1.2.4.1.1 "><p id="p103051638333"><a name="p103051638333"></a><a name="p103051638333"></a>torch.nn.AvgPool2d</p>
</td>
<td class="cellrowborder" valign="top" width="41.17%" headers="mcps1.2.4.1.2 "><p id="p4305738238"><a name="p4305738238"></a><a name="p4305738238"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="38.51%" headers="mcps1.2.4.1.3 "><p id="p33051338039"><a name="p33051338039"></a><a name="p33051338039"></a>-</p>
</td>
</tr>
</tbody>
</table>


## 量化均衡预处理支持的层及约束

<a name="table32131022111318"></a>

<table><thead align="left"><tr id="row9214202215138"><th class="cellrowborder" valign="top" width="23.242324232423243%" id="mcps1.2.4.1.1"><p id="p1921472241319"><a name="p1921472241319"></a><a name="p1921472241319"></a>支持的层类型</p>
</th>
<th class="cellrowborder" valign="top" width="31.91319131913191%" id="mcps1.2.4.1.2"><p id="p122141822101318"><a name="p122141822101318"></a><a name="p122141822101318"></a>约束</p>
</th>
<th class="cellrowborder" valign="top" width="44.844484448444845%" id="mcps1.2.4.1.3"><p id="p421411228138"><a name="p421411228138"></a><a name="p421411228138"></a>备注</p>
</th>
</tr>
</thead>
<tbody><tr id="row7214182271318"><td class="cellrowborder" valign="top" width="23.242324232423243%" headers="mcps1.2.4.1.1 "><p id="p32146222139"><a name="p32146222139"></a><a name="p32146222139"></a>torch.nn.Linear</p>
</td>
<td class="cellrowborder" valign="top" width="31.91319131913191%" headers="mcps1.2.4.1.2 "><p id="p162141722111314"><a name="p162141722111314"></a><a name="p162141722111314"></a>-</p>
</td>
<td class="cellrowborder" rowspan="4" valign="top" width="44.844484448444845%" headers="mcps1.2.4.1.3 "><p id="p381615103334"><a name="p381615103334"></a><a name="p381615103334"></a>复用层（共用weight和bias参数）不支持量化。</p>
</td>
</tr>
<tr id="row1521462251317"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1021414226138"><a name="p1021414226138"></a><a name="p1021414226138"></a>torch.nn.Conv2d</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p8417519168"><a name="p8417519168"></a><a name="p8417519168"></a>padding_mode为zeros</p>
</td>
</tr>
<tr id="row1551419103911"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p731502015364"><a name="p731502015364"></a><a name="p731502015364"></a>torch.nn.Conv3d</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p05616359369"><a name="p05616359369"></a><a name="p05616359369"></a>dilation_d为1，dilation_h/dilation_w &gt;= 1</p>
<p id="p72994182514"><a name="p72994182514"></a><a name="p72994182514"></a>padding_mode为zeros</p>
</td>
</tr>
<tr id="row6215102221319"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1421510225139"><a name="p1421510225139"></a><a name="p1421510225139"></a>torch.nn.ConvTranspose2d</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p924818415338"><a name="p924818415338"></a><a name="p924818415338"></a>padding_mode为zeros</p>
</td>
</tr>
</tbody>
</table>



## 返回值说明<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_section293415513458"></a>

无

## 调用示例<a name="zh-cn_topic_0240187365_section64231658994"></a>

```
import amct_pytorch as amct
# 建立待量化的网络图结构
model = build_model()
model.load_state_dict(torch.load(state_dict_path))
input_data = tuple([torch.randn(input_shape)])
model.eval()

# 生成量化配置文件
amct.create_quant_config(config_file="./configs/config.json",
                         model=model,
                         input_data=input_data,
                         skip_layers=None,
                         batch_num=1,
                         activation_offset=True)
```

落盘文件说明：生成JSON格式的量化配置文件，样例如下（重新执行量化时，该接口生成的量化配置文件将会被覆盖），参数解释请参见[训练后量化配置参数](../context/训练后量化配置参数.md).

-   训练后量化配置文件（数据量化使用[IFMR数据量化算法](../算法介绍.md)）

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

-   训练后量化配置文件（数据量化使用[HFMG数据量化算法](../算法介绍.md)）

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

-   自适应舍入量化简易配置文件（权重量化使用[ADA权重量化算法](../算法介绍.md)）

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

