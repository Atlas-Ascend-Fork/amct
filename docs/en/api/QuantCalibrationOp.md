# QuantCalibrationOp<a name="ZH-CN_TOPIC_0000002548668565"></a>

<!-- md-trans-meta sourceCommit=4a47814eed18048e444f9a2acc42d9864ddc477f translatedAt=2026-10-16T08:30:55.694Z pushedAt=2026-11-05T01:12:03.421Z nl=2 -->

## Supported Products<a name="section1610172518544"></a>

| Product | Supported |
| ------------------------------------------- | -------- |
| Ascend 950PR&950DT products | √ |
| Atlas A3 products | √ |
| Atlas A2 products | √ |

**Note: For products marked with "x", calling the API does not report an error, but no performance gain is obtained.**

## Description<a name="zh-cn_topic_0240187365_section15406195619561"></a>

The KV Cache quantization API is used for user graph construction. During forward propagation, it calls the IFMR/HFMG quantization algorithm based on the user's quantization algorithm configuration to calibrate the output. After calibration, it writes the quantization factors to the layer names specified in the **record\_file** file in the corresponding format.

During forward propagation, the operator passes through the original output without modifying the activation input information:

- If the number of currently passed data items is less than **batch\_num**, the data accumulation method in the IFMR/HFMG operator is used to save the dataset.
- If the number of currently passed data items equals **batch\_num**, the IFMR/HFMG algorithm is called to compute the quantization factors. The format of the written quantization factors is distinguished based on the **quant\_method** parameter, and they are written to the corresponding record file in that format. **quant\_method** currently supports only the **kv\_cache\_quant** method.

When writing, the record file is written incrementally. If an overwrite occurs, a prompt is displayed indicating which layers and which parameters have been overwritten.

## Prototype<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_section428121323411"></a>

```python
QuantCalibrationOp (record_file,quant_algo_params, quant_method)
```

## Parameters<a name="section73811524135618"></a>

<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="13.658634136586342%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a>Parameter</p></th>
<th class="cellrowborder" valign="top" width="7.939206079392061%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0240187365_p1769255516412"><a name="zh-cn_topic_0240187365_p1769255516412"></a><a name="zh-cn_topic_0240187365_p1769255516412"></a>Input/Output</p></th>
<th class="cellrowborder" valign="top" width="78.40215978402159%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0240187365_p15231205416325"><a name="zh-cn_topic_0240187365_p15231205416325"></a><a name="zh-cn_topic_0240187365_p15231205416325"></a>Description</p></th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="13.658634136586342%" headers="mcps1.1.4.1.1 "><p id="p816684420110"><a name="p816684420110"></a><a name="p816684420110"></a>record_file</p></td>
<td class="cellrowborder" valign="top" width="7.939206079392061%" headers="mcps1.1.4.1.2 "><p id="p1219493024610"><a name="p1219493024610"></a><a name="p1219493024610"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.40215978402159%" headers="mcps1.1.4.1.3 "><p id="p132525675114"><a name="p132525675114"></a><a name="p132525675114"></a>Meaning: Path of the record file that stores the quantization factors.</p>
<p id="p7194123094612"><a name="p7194123094612"></a><a name="p7194123094612"></a>Data type: string</p></td>
</tr>
<tr id="row36995816444"><td class="cellrowborder" valign="top" width="13.658634136586342%" headers="mcps1.1.4.1.1 "><p id="p269918812448"><a name="p269918812448"></a><a name="p269918812448"></a>quant_algo_params</p></td>
<td class="cellrowborder" valign="top" width="7.939206079392061%" headers="mcps1.1.4.1.2 "><p id="p14699985449"><a name="p14699985449"></a><a name="p14699985449"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.40215978402159%" headers="mcps1.1.4.1.3 "><p id="p113721429525"><a name="p113721429525"></a><a name="p113721429525"></a>Meaning: Specifies the quantization algorithm used for quantization and adds the corresponding quantization configuration. The format is as follows:</p>
<pre class="screen" id="screen1285281775716"><a name="screen1285281775716"></a><a name="screen1285281775716"></a>{
  "act_algo": "hfmg",
  "num_bits": 8,
  "quant_granularity": "1",
  "with_offset": true,
  "batch_num": 1
}</pre>
<p id="p13330105451319"><a name="p13330105451319"></a><a name="p13330105451319"></a>Different algorithms configured for act_algo support different fields. The fields supported by each algorithm and their explanations are shown in <a href="#table1778775810151">Table 1</a>.</p>
<p id="p66998816443"><a name="p66998816443"></a><a name="p66998816443"></a>Data type: dict</p></td>
</tr>
<tr id="row4280131024613"><td class="cellrowborder" valign="top" width="13.658634136586342%" headers="mcps1.1.4.1.1 "><p id="p4194143084610"><a name="p4194143084610"></a><a name="p4194143084610"></a>quant_method</p></td>
<td class="cellrowborder" valign="top" width="7.939206079392061%" headers="mcps1.1.4.1.2 "><p id="p14194430114612"><a name="p14194430114612"></a><a name="p14194430114612"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.40215978402159%" headers="mcps1.1.4.1.3 "><p id="p121312414525"><a name="p121312414525"></a><a name="p121312414525"></a>Meaning: Quantization method. The default is 'kv_cache_quant', which specifies the format of the output quantization factor.</p>
<p id="p19194123013463"><a name="p19194123013463"></a><a name="p19194123013463"></a>Data type: string</p></td>
</tr>
</tbody>
</table>

**Table 1**  quant\_algo\_params fields supported for configuration by the parameter

<a name="table1778775810151"></a>
<table><thead align="left"><tr id="row4855358111517"><th class="cellrowborder" valign="top" width="12.68%" id="mcps1.2.5.1.1"><p id="p1855105821518"><a name="p1855105821518"></a><a name="p1855105821518"></a>Field</p></th>
<th class="cellrowborder" valign="top" width="60.06%" id="mcps1.2.5.1.2"><p id="p18604114031612"><a name="p18604114031612"></a><a name="p18604114031612"></a>Meaning</p></th>
<th class="cellrowborder" valign="top" width="13.320000000000002%" id="mcps1.2.5.1.3"><p id="p94109382310"><a name="p94109382310"></a><a name="p94109382310"></a>Fields Supported for Configuration by the IFMR Algorithm</p></th>
<th class="cellrowborder" valign="top" width="13.94%" id="mcps1.2.5.1.4"><p id="p19813143213316"><a name="p19813143213316"></a><a name="p19813143213316"></a>Fields Supported for Configuration by the HFMG Algorithm</p></th>
</tr>
</thead>
<tbody><tr id="row6855358141517"><td class="cellrowborder" valign="top" width="12.68%" headers="mcps1.2.5.1.1 "><p id="p985595819155"><a name="p985595819155"></a><a name="p985595819155"></a>act_algo</p></td>
<td class="cellrowborder" valign="top" width="60.06%" headers="mcps1.2.5.1.2 "><p id="p196046400166"><a name="p196046400166"></a><a name="p196046400166"></a>Data quantization algorithm. The following two algorithms are supported:</p>
<a name="ul121219224249"></a><a name="ul121219224249"></a><ul id="ul121219224249"><li>IFMR data quantization algorithm: ifmr, which is the default.</li><li>HFMG data quantization algorithm: hfmg.</li></ul></td>
<td class="cellrowborder" valign="top" width="13.320000000000002%" headers="mcps1.2.5.1.3 "><p id="p1841013381932"><a name="p1841013381932"></a><a name="p1841013381932"></a>-</p></td>
<td class="cellrowborder" valign="top" width="13.94%" headers="mcps1.2.5.1.4 "><p id="p208131032735"><a name="p208131032735"></a><a name="p208131032735"></a>-</p></td>
</tr>
<tr id="row1223511354619"><td class="cellrowborder" valign="top" width="12.68%" headers="mcps1.2.5.1.1 "><p id="p14235191324612"><a name="p14235191324612"></a><a name="p14235191324612"></a>num_bits</p></td>
<td class="cellrowborder" valign="top" width="60.06%" headers="mcps1.2.5.1.2 "><p id="p32352139461"><a name="p32352139461"></a><a name="p32352139461"></a>Quantization bit width. Currently, only 8 is supported, indicating the INT8 quantization bit width.</p>
<p id="p18486171216"><a name="p18486171216"></a><a name="p18486171216"></a>Both the IFMR and HFMG algorithms support it.</p></td>
<td class="cellrowborder" valign="top" width="13.320000000000002%" headers="mcps1.2.5.1.3 "><p id="p1062425516499"><a name="p1062425516499"></a><a name="p1062425516499"></a>Supported</p></td>
<td class="cellrowborder" valign="top" width="13.94%" headers="mcps1.2.5.1.4 "><p id="p156244556497"><a name="p156244556497"></a><a name="p156244556497"></a>Supported</p></td>
</tr>
<tr id="row18518612396"><td class="cellrowborder" valign="top" width="12.68%" headers="mcps1.2.5.1.1 "><p id="p2051810121995"><a name="p2051810121995"></a><a name="p2051810121995"></a>with_offset</p></td>
<td class="cellrowborder" valign="top" width="60.06%" headers="mcps1.2.5.1.2 "><p id="p551817121196"><a name="p551817121196"></a><a name="p551817121196"></a>Controls whether data quantization is symmetric or asymmetric. This is a global configuration parameter.</p>
<a name="ul11492132182611"></a><a name="ul11492132182611"></a><ul id="ul11492132182611"><li>true: Asymmetric quantization is used for data quantization. This is the default.</li><li>false: Symmetric quantization is used for data quantization.</li></ul>
<p id="p144123051913"><a name="p144123051913"></a><a name="p144123051913"></a>If both the with_offset and asymmetric parameters exist in the configuration file, the asymmetric parameter takes precedence over the with_offset parameter.</p>
<p id="p11149132022117"><a name="p11149132022117"></a><a name="p11149132022117"></a>Both the IFMR and HFMG algorithms support it.</p></td>
<td class="cellrowborder" valign="top" width="13.320000000000002%" headers="mcps1.2.5.1.3 "><p id="p16493233599"><a name="p16493233599"></a><a name="p16493233599"></a>Supported</p></td>
<td class="cellrowborder" valign="top" width="13.94%" headers="mcps1.2.5.1.4 "><p id="p174933331293"><a name="p174933331293"></a><a name="p174933331293"></a>Supported</p></td>
</tr>
<tr id="row688319149917"><td class="cellrowborder" valign="top" width="12.68%" headers="mcps1.2.5.1.1 "><p id="p13883111418919"><a name="p13883111418919"></a><a name="p13883111418919"></a>batch_num</p></td>
<td class="cellrowborder" valign="top" width="60.06%" headers="mcps1.2.5.1.2 "><p id="p148831014795"><a name="p148831014795"></a><a name="p148831014795"></a>Controls the number of batches of data used for quantization. The value range is greater than 0, and the default is 1.</p>
<p id="p281411218216"><a name="p281411218216"></a><a name="p281411218216"></a>Both the IFMR and HFMG algorithms support it.</p></td>
<td class="cellrowborder" valign="top" width="13.320000000000002%" headers="mcps1.2.5.1.3 "><p id="p11501834096"><a name="p11501834096"></a><a name="p11501834096"></a>Supported</p></td>
<td class="cellrowborder" valign="top" width="13.94%" headers="mcps1.2.5.1.4 "><p id="p2510345917"><a name="p2510345917"></a><a name="p2510345917"></a>Supported</p></td>
</tr>
<tr id="row6856058121519"><td class="cellrowborder" valign="top" width="12.68%" headers="mcps1.2.5.1.1 "><p id="p7856958201516"><a name="p7856958201516"></a><a name="p7856958201516"></a>asymmetric</p></td>
<td class="cellrowborder" valign="top" width="60.06%" headers="mcps1.2.5.1.2 "><p id="p18570132515213"><a name="p18570132515213"></a><a name="p18570132515213"></a>Controls whether data quantization is symmetric or asymmetric, and controls the selection of the per-layer quantization algorithm.</p>
<a name="ul31744286216"></a><a name="ul31744286216"></a><ul id="ul31744286216"><li>true: Asymmetric quantization, and the default is true.</li><li>false: Symmetric quantization.</li></ul>
<p id="p1617482812118"><a name="p1617482812118"></a><a name="p1617482812118"></a>Both the IFMR and HFMG algorithms support it.</p></td>
<td class="cellrowborder" valign="top" width="13.320000000000002%" headers="mcps1.2.5.1.3 "><p id="p74101838535"><a name="p74101838535"></a><a name="p74101838535"></a>Supported</p></td>
<td class="cellrowborder" valign="top" width="13.94%" headers="mcps1.2.5.1.4 "><p id="p1881319321313"><a name="p1881319321313"></a><a name="p1881319321313"></a>Supported</p></td>
</tr>
<tr id="row785695841518"><td class="cellrowborder" valign="top" width="12.68%" headers="mcps1.2.5.1.1 "><p id="p2085675811512"><a name="p2085675811512"></a><a name="p2085675811512"></a>quant_granularity</p></td>
<td class="cellrowborder" valign="top" width="60.06%" headers="mcps1.2.5.1.2 "><p id="p15112312194416"><a name="p15112312194416"></a><a name="p15112312194416"></a>Quantization granularity, which supports the following two methods:</p>
<a name="ul14112141214414"></a><a name="ul14112141214414"></a><ul id="ul14112141214414"><li>0: per_tensor, and the default is per_tensor.</li><li>1: per_channel.</li></ul>
<p id="p171183588210"><a name="p171183588210"></a><a name="p171183588210"></a>Both the IFMR and HFMG algorithms support it.</p></td>
<td class="cellrowborder" valign="top" width="13.320000000000002%" headers="mcps1.2.5.1.3 "><p id="p14107381231"><a name="p14107381231"></a><a name="p14107381231"></a>Supported</p></td>
<td class="cellrowborder" valign="top" width="13.94%" headers="mcps1.2.5.1.4 "><p id="p138135321133"><a name="p138135321133"></a><a name="p138135321133"></a>Supported</p></td>
</tr>
<tr id="row28561658121519"><td class="cellrowborder" valign="top" width="12.68%" headers="mcps1.2.5.1.1 "><p id="p14856058121514"><a name="p14856058121514"></a><a name="p14856058121514"></a>max_percentile</p></td>
<td class="cellrowborder" valign="top" width="60.06%" headers="mcps1.2.5.1.2 "><p id="p1360434016169"><a name="p1360434016169"></a><a name="p1360434016169"></a><span>Maximum value search position parameter in the IFMR data quantization algorithm.</span></p>
<p id="p16646181010261"><a name="p16646181010261"></a><a name="p16646181010261"></a>The value range is (0.5,1], and the default is 0.999999.</p>
<p id="p1521163372613"><a name="p1521163372613"></a><a name="p1521163372613"></a>Only the <span>IFMR</span> algorithm supports it.</p></td>
<td class="cellrowborder" valign="top" width="13.320000000000002%" headers="mcps1.2.5.1.3 "><p id="p4410113820316"><a name="p4410113820316"></a><a name="p4410113820316"></a>Supported</p></td>
<td class="cellrowborder" valign="top" width="13.94%" headers="mcps1.2.5.1.4 "><p id="p116912417610"><a name="p116912417610"></a><a name="p116912417610"></a>Not supported</p></td>
</tr>
<tr id="row128578581150"><td class="cellrowborder" valign="top" width="12.68%" headers="mcps1.2.5.1.1 "><p id="p48572583153"><a name="p48572583153"></a><a name="p48572583153"></a>min_percentile</p></td>
<td class="cellrowborder" valign="top" width="60.06%" headers="mcps1.2.5.1.2 "><p id="p1460424014164"><a name="p1460424014164"></a><a name="p1460424014164"></a><span>Minimum value search position</span> parameter in the IFMR data quantization algorithm.</p>
<p id="p3898204416267"><a name="p3898204416267"></a><a name="p3898204416267"></a>The value range is (0.5,1], and the default is 0.999999.</p>
<p id="p12980511172711"><a name="p12980511172711"></a><a name="p12980511172711"></a>Only the <span>IFMR</span> algorithm supports it.</p></td>
<td class="cellrowborder" valign="top" width="13.320000000000002%" headers="mcps1.2.5.1.3 "><p id="p114101938234"><a name="p114101938234"></a><a name="p114101938234"></a>Supported</p></td>
<td class="cellrowborder" valign="top" width="13.94%" headers="mcps1.2.5.1.4 "><p id="p4749413620"><a name="p4749413620"></a><a name="p4749413620"></a>Not supported</p></td>
</tr>
<tr id="row128571058101518"><td class="cellrowborder" valign="top" width="12.68%" headers="mcps1.2.5.1.1 "><p id="p685718581155"><a name="p685718581155"></a><a name="p685718581155"></a>search_range</p></td>
<td class="cellrowborder" valign="top" width="60.06%" headers="mcps1.2.5.1.2 "><p id="p18605174011612"><a name="p18605174011612"></a><a name="p18605174011612"></a>In the IFMR data quantization algorithm, controls the search range [search_range_start, search_range_end] of the quantization factor.</p>
<p id="p455153310277"><a name="p455153310277"></a><a name="p455153310277"></a>The value range is 0&lt;search_range_start&lt;search_range_end, and the recommended configuration is [0.7, 1.3].</p>
<p id="p1121528132818"><a name="p1121528132818"></a><a name="p1121528132818"></a>Only the <span>IFMR</span> algorithm supports this.</p></td>
<td class="cellrowborder" valign="top" width="13.320000000000002%" headers="mcps1.2.5.1.3 "><p id="p841017380317"><a name="p841017380317"></a><a name="p841017380317"></a>Supported</p></td>
<td class="cellrowborder" valign="top" width="13.94%" headers="mcps1.2.5.1.4 "><p id="p37854113618"><a name="p37854113618"></a><a name="p37854113618"></a>Not supported</p></td>
</tr>
<tr id="row085725816155"><td class="cellrowborder" valign="top" width="12.68%" headers="mcps1.2.5.1.1 "><p id="p6857195861517"><a name="p6857195861517"></a><a name="p6857195861517"></a>search_step</p></td>
<td class="cellrowborder" valign="top" width="60.06%" headers="mcps1.2.5.1.2 "><p id="zh-cn_topic_0240188002_p111814492426"><a name="zh-cn_topic_0240188002_p111814492426"></a><a name="zh-cn_topic_0240188002_p111814492426"></a>In the IFMR data quantization algorithm, controls the search step of the quantization factor.</p>
<p id="p14906729132817"><a name="p14906729132817"></a><a name="p14906729132817"></a>The value range is (0, (search_range_end-search_range_start)], and the default is 0.01.</p>
<p id="p14430123052817"><a name="p14430123052817"></a><a name="p14430123052817"></a>Only the <span>IFMR</span> algorithm supports this.</p></td>
<td class="cellrowborder" valign="top" width="13.320000000000002%" headers="mcps1.2.5.1.3 "><p id="p124101038834"><a name="p124101038834"></a><a name="p124101038834"></a>Supported</p></td>
<td class="cellrowborder" valign="top" width="13.94%" headers="mcps1.2.5.1.4 "><p id="p118218415618"><a name="p118218415618"></a><a name="p118218415618"></a>Not supported</p></td>
</tr>
<tr id="row6857175813158"><td class="cellrowborder" valign="top" width="12.68%" headers="mcps1.2.5.1.1 "><p id="p1185715589158"><a name="p1185715589158"></a><a name="p1185715589158"></a>num_of_bins</p></td>
<td class="cellrowborder" valign="top" width="60.06%" headers="mcps1.2.5.1.2 "><p id="p14190312023"><a name="p14190312023"></a><a name="p14190312023"></a>Number of bins (the smallest unit in a histogram) in the histogram. The supported range is {1024, 2048, 4096, 8192}. The default value is 4096.</p>
<p id="p5348173411224"><a name="p5348173411224"></a><a name="p5348173411224"></a>Supported only by the HFMG algorithm.</p></td>
<td class="cellrowborder" valign="top" width="13.320000000000002%" headers="mcps1.2.5.1.3 "><p id="p1341083816319"><a name="p1341083816319"></a><a name="p1341083816319"></a>Not supported</p></td>
<td class="cellrowborder" valign="top" width="13.94%" headers="mcps1.2.5.1.4 "><p id="p581383211313"><a name="p581383211313"></a><a name="p581383211313"></a>Supported</p></td>
</tr>
</tbody>
</table>

## Return Value<a name="zh-cn_topic_0240188739_zh-cn_topic_0122830089_section293415513458"></a>

None

## Example<a name="zh-cn_topic_0240188739_section64231658994"></a>

```python
import amct_pytorch as amct
from amct_pytorch.nn.module.quantization.quant_calibration_op import QuantCalibrationOp

class LinearNet(nn.Module):
    def __init__(self, quant_algo_params):

        super(LinearNet,self).__init__()
        self.quant_algo_params = quant_algo_params
        self.layer1 = nn.Linear(28, 1024, bias=False)
        self.ptq_1 = QuantCalibrationOp(record_file, quant_algo_params=self.quant_algo_params, quant_method="kv_cache_quant")

    def forward(self, layer_name, x):
        x = self.layer1(x)
        x = self.ptq_1(layer_name, x)

# Save the record file path for the quantization factors.
temp_folder = "./"
record_file = os.path.join(temp_folder, 'kv_cache.txt')
input_data = torch.randn((2, 2, 28, 28))

quant_algo_params = {"act_algo": "hfmg"}
model = LinearNet(quant_algo_params).to(torch.device("cpu"))
model.eval()

ans_2 = model("qat_1", input_data)
```
