# create\_compressed\_retrain\_model<a name="ZH-CN_TOPIC_0000002517188752"></a>

<!-- md-trans-meta sourceCommit=4a47814eed18048e444f9a2acc42d9864ddc477f translatedAt=2026-10-16T08:15:31.350Z pushedAt=2026-11-05T01:12:03.407Z nl=2 -->

## Supported Products<a name="section197451857688"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="47.92%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>Product</span></p></th>
<th class="cellrowborder" align="left" valign="top" width="52.080000000000005%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>Supported</p></th>
</tr>
</thead>
<tbody><tr id="row574891710101"><td class="cellrowborder" valign="top" width="47.92%" headers="mcps1.1.3.1.1 "><p id="p177491117171015"><a name="p177491117171015"></a><a name="p177491117171015"></a><span id="ph2272194216543"><a name="ph2272194216543"></a><a name="ph2272194216543"></a>Ascend 950PR&950DT products</span></p></td>
<td class="cellrowborder" align="left" valign="top" width="52.080000000000005%" headers="mcps1.1.3.1.2 "><a name="ul1367712433612"></a><a name="ul1367712433612"></a><ul id="ul1367712433612"><li>Quantization-aware training: <a name="ul5366163611332"></a><a name="ul5366163611332"></a><ul id="ul5366163611332"><li>INT8 quantization: √</li><li>INT4 quantization: x</li></ul>
</li><li>Channel sparsity: √</li><li>4:2 structured sparsity: x</li></ul></td>
</tr>
<tr id="row220181016240"><td class="cellrowborder" valign="top" width="47.92%" headers="mcps1.1.3.1.1 "><p id="p48327011813"><a name="p48327011813"></a><a name="p48327011813"></a><span id="ph583230201815"><a name="ph583230201815"></a><a name="ph583230201815"></a><term id="zh-cn_topic_0000001312391781_term1253731311225">Atlas A3 products</term></span></p></td>
<td class="cellrowborder" align="left" valign="top" width="52.080000000000005%" headers="mcps1.1.3.1.2 "><a name="ul66191179379"></a><a name="ul66191179379"></a><ul id="ul66191179379"><li>Quantization-aware training: <a name="ul7274174217362"></a><a name="ul7274174217362"></a><ul id="ul7274174217362"><li>INT8 quantization: √</li><li>INT4 quantization: x</li></ul>
</li><li>Channel sparsity: √</li><li>4:2 structured sparsity: √</li></ul></td>
</tr>
<tr id="row173226882415"><td class="cellrowborder" valign="top" width="47.92%" headers="mcps1.1.3.1.1 "><p id="p14832120181815"><a name="p14832120181815"></a><a name="p14832120181815"></a><span id="ph1483216010188"><a name="ph1483216010188"></a><a name="ph1483216010188"></a><term id="zh-cn_topic_0000001312391781_term11962195213215">Atlas A2 products</term></span></p></td>
<td class="cellrowborder" align="left" valign="top" width="52.080000000000005%" headers="mcps1.1.3.1.2 "><a name="ul19623147153713"></a><a name="ul19623147153713"></a><ul id="ul19623147153713"><li>Quantization-aware training: <a name="ul0992647173610"></a><a name="ul0992647173610"></a><ul id="ul0992647173610"><li>INT8 quantization: √</li><li>INT4 quantization: x</li></ul>
</li><li>Channel sparsity: √</li><li>4:2 structured sparsity: √</li></ul></td>
</tr>
</tbody>
</table>

**Note: For products marked with "x" in the feature, calling the API does not report an error, but no performance benefit can be obtained.**

## Description<a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_section15406195619561"></a>

Compresses the input model to be statically combined and compressed according to the given combination compression configuration file. That is, the input model is first sparsified (channel sparsity or 4:2 structured sparsity, choose one), and then quantization-related operators are inserted into the model (quantization-aware training layers for data and weights, as well as the searchN layer), generating sparsity and quantization factor record files **record\_file** (if the configuration exists), and returning the modified **torch.nn.Module** model.

## Prototype<a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_section428121323411"></a>

```python
compressed_retrain_model = create_compressed_retrain_model(model, input_data, config_defination, record_file)
```

## Parameters<a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_section795991810344"></a>

<a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="12.411241124112411%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_p1438834363520"></a>Parameter</p></th>
<th class="cellrowborder" valign="top" width="7.490749074907491%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p1769255516412"><a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p1769255516412"></a><a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p1769255516412"></a>Input/Output</p></th>
<th class="cellrowborder" valign="top" width="80.0980098009801%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p15231205416325"><a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p15231205416325"></a><a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p15231205416325"></a>Description</p></th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="12.411241124112411%" headers="mcps1.1.4.1.1 "><p id="p166458380213"><a name="p166458380213"></a><a name="p166458380213"></a>model</p></td>
<td class="cellrowborder" valign="top" width="7.490749074907491%" headers="mcps1.1.4.1.2 "><p id="p168681026184710"><a name="p168681026184710"></a><a name="p168681026184710"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="80.0980098009801%" headers="mcps1.1.4.1.3 "><p id="p15959183616327"><a name="p15959183616327"></a><a name="p15959183616327"></a>Meaning: PyTorch model.</p>
<p id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p11225740182619"><a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p11225740182619"></a><a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p11225740182619"></a>Data type: torch.nn.Module</p></td>
</tr>
<tr id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_row21251438195"><td class="cellrowborder" valign="top" width="12.411241124112411%" headers="mcps1.1.4.1.1 "><p id="p17645193813215"><a name="p17645193813215"></a><a name="p17645193813215"></a>input_data</p></td>
<td class="cellrowborder" valign="top" width="7.490749074907491%" headers="mcps1.1.4.1.2 "><p id="p13482131324113"><a name="p13482131324113"></a><a name="p13482131324113"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="80.0980098009801%" headers="mcps1.1.4.1.3 "><p id="p14176174617320"><a name="p14176174617320"></a><a name="p14176174617320"></a>Meaning: Input data of the model.</p>
<p id="p12421175414419"><a name="p12421175414419"></a><a name="p12421175414419"></a>Data type: tuple</p></td>
</tr>
<tr id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_row16254133823918"><td class="cellrowborder" valign="top" width="12.411241124112411%" headers="mcps1.1.4.1.1 "><p id="p164543814218"><a name="p164543814218"></a><a name="p164543814218"></a>config_defination</p></td>
<td class="cellrowborder" valign="top" width="7.490749074907491%" headers="mcps1.1.4.1.2 "><p id="p18491181320411"><a name="p18491181320411"></a><a name="p18491181320411"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="80.0980098009801%" headers="mcps1.1.4.1.3 "><p id="p17101548133217"><a name="p17101548133217"></a><a name="p17101548133217"></a>Meaning: Simple configuration file for static combination compression.</p>
<p id="p827435611439"><a name="p827435611439"></a><a name="p827435611439"></a>Simple configuration file <em id="i497145210579"><a name="i497145210579"></a><a name="i497145210579"></a>compressed</em>.cfg generated based on the retrain_config_pytorch.proto file. The path of the *.proto file is: <em id="i12274856104314"><a name="i12274856104314"></a><a name="i12274856104314"></a><span id="ph244511481409"><a name="ph244511481409"></a><a name="ph244511481409"></a>AMCT</span> installation directory</em>/amct_pytorch/proto/. For the parameter description of the *.proto file and the sample of the generated <em id="i2675712185810"><a name="i2675712185810"></a><a name="i2675712185810"></a>compressed</em>.cfg simple configuration file, see <a href="../context/qat_config.md">Quantization-Aware Training Simple Configuration File</a>.</p>
<p id="p64303542418"><a name="p64303542418"></a><a name="p64303542418"></a>Data type: string</p></td>
</tr>
<tr id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_row2997141123911"><td class="cellrowborder" valign="top" width="12.411241124112411%" headers="mcps1.1.4.1.1 "><p id="p264514389215"><a name="p264514389215"></a><a name="p264514389215"></a>record_file</p></td>
<td class="cellrowborder" valign="top" width="7.490749074907491%" headers="mcps1.1.4.1.2 "><p id="p175011134411"><a name="p175011134411"></a><a name="p175011134411"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="80.0980098009801%" headers="mcps1.1.4.1.3 "><p id="p1872594993211"><a name="p1872594993211"></a><a name="p1872594993211"></a>Meaning: Path and name of the file for recording sparsity and quantization factors.</p>
<p id="p1543455424113"><a name="p1543455424113"></a><a name="p1543455424113"></a>Data type: string</p></td>
</tr>
</tbody>
</table>

## Return Value<a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_section293415513458"></a>

Returns a **torch.nn.Module** static combination compressed model that has been sparsified according to the configuration file (if sparsity is configured) and has quantization-related layers inserted (if quantization is configured).

## Constraints<a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_section1443392021419"></a>

The combination compression configuration file must contain at least one configuration: a sparsity configuration or a quantization configuration.

## Example<a name="zh-cn_topic_0254208276_section213682915617"></a>

```python
import amct_pytorch as amct
# Build the network to be statically combined and compressed.
model = build_model()
input_data = tuple([torch.randn(input_shape)])

# Call the static combination compression API.
record_file = os.path.join(TMP, 'compressed_record.txt')
config_defination = './compressed_cfg.cfg'

compressed_retrain_model = amct.create_compressed_retrain_model(
                                model,
                                input_data,
                                config_defination,
                                record_file)
```

Description of the files written to disk:

If the simple configuration file contains a sparsity configuration, after this function completes, the saved static combination compression record file **record\_file** contains the sparsity record information.
