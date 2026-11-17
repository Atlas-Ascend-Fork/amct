# restore\_compressed\_retrain\_model<a name="ZH-CN_TOPIC_0000002548668611"></a>

<!-- md-trans-meta sourceCommit=9d54d17e0942052193c827180dc43312c580acc7 translatedAt=2026-10-16T08:34:03.709Z pushedAt=2026-11-05T01:12:03.423Z nl=2 -->

## Supported Products<a name="section329012164010"></a>

<a name="zh-cn_topic_0000002517188752_table38301303189"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002517188752_row20831180131817"><th class="cellrowborder" valign="top" width="47.92%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000002517188752_p1883113061818"><a name="zh-cn_topic_0000002517188752_p1883113061818"></a><a name="zh-cn_topic_0000002517188752_p1883113061818"></a><span id="zh-cn_topic_0000002517188752_ph20833205312295"><a name="zh-cn_topic_0000002517188752_ph20833205312295"></a><a name="zh-cn_topic_0000002517188752_ph20833205312295"></a>Product</span></p></th>
<th class="cellrowborder" align="left" valign="top" width="52.080000000000005%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000002517188752_p783113012187"><a name="zh-cn_topic_0000002517188752_p783113012187"></a><a name="zh-cn_topic_0000002517188752_p783113012187"></a>Supported</p></th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002517188752_row574891710101"><td class="cellrowborder" valign="top" width="47.92%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002517188752_p177491117171015"><a name="zh-cn_topic_0000002517188752_p177491117171015"></a><a name="zh-cn_topic_0000002517188752_p177491117171015"></a><span id="zh-cn_topic_0000002517188752_ph2272194216543"><a name="zh-cn_topic_0000002517188752_ph2272194216543"></a><a name="zh-cn_topic_0000002517188752_ph2272194216543"></a>Ascend 950PR&950DT products</span></p></td>
<td class="cellrowborder" align="left" valign="top" width="52.080000000000005%" headers="mcps1.1.3.1.2 "><a name="zh-cn_topic_0000002517188752_ul1367712433612"></a><a name="zh-cn_topic_0000002517188752_ul1367712433612"></a><ul id="zh-cn_topic_0000002517188752_ul1367712433612"><li>Quantization-aware training: <a name="zh-cn_topic_0000002517188752_ul5366163611332"></a><a name="zh-cn_topic_0000002517188752_ul5366163611332"></a><ul id="zh-cn_topic_0000002517188752_ul5366163611332"><li>INT8 quantization: √</li><li>INT4 quantization: x</li></ul>
</li><li>Channel sparsity: √</li><li>4:2 structured sparsity: x</li></ul></td>
</tr>
<tr id="zh-cn_topic_0000002517188752_row220181016240"><td class="cellrowborder" valign="top" width="47.92%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002517188752_p48327011813"><a name="zh-cn_topic_0000002517188752_p48327011813"></a><a name="zh-cn_topic_0000002517188752_p48327011813"></a><span id="zh-cn_topic_0000002517188752_ph583230201815"><a name="zh-cn_topic_0000002517188752_ph583230201815"></a><a name="zh-cn_topic_0000002517188752_ph583230201815"></a><term id="zh-cn_topic_0000002517188752_zh-cn_topic_0000001312391781_term1253731311225">Atlas A3 products</term></span></p></td>
<td class="cellrowborder" align="left" valign="top" width="52.080000000000005%" headers="mcps1.1.3.1.2 "><a name="zh-cn_topic_0000002517188752_ul66191179379"></a><a name="zh-cn_topic_0000002517188752_ul66191179379"></a><ul id="zh-cn_topic_0000002517188752_ul66191179379"><li>Quantization-aware training: <a name="zh-cn_topic_0000002517188752_ul7274174217362"></a><a name="zh-cn_topic_0000002517188752_ul7274174217362"></a><ul id="zh-cn_topic_0000002517188752_ul7274174217362"><li>INT8 quantization: √</li><li>INT4 quantization: x</li></ul>
</li><li>Channel sparsity: √</li><li>4:2 structured sparsity: √</li></ul></td>
</tr>
<tr id="zh-cn_topic_0000002517188752_row173226882415"><td class="cellrowborder" valign="top" width="47.92%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002517188752_p14832120181815"><a name="zh-cn_topic_0000002517188752_p14832120181815"></a><a name="zh-cn_topic_0000002517188752_p14832120181815"></a><span id="zh-cn_topic_0000002517188752_ph1483216010188"><a name="zh-cn_topic_0000002517188752_ph1483216010188"></a><a name="zh-cn_topic_0000002517188752_ph1483216010188"></a><term id="zh-cn_topic_0000002517188752_zh-cn_topic_0000001312391781_term11962195213215">Atlas A2 products</term></span></p></td>
<td class="cellrowborder" align="left" valign="top" width="52.080000000000005%" headers="mcps1.1.3.1.2 "><a name="zh-cn_topic_0000002517188752_ul19623147153713"></a><a name="zh-cn_topic_0000002517188752_ul19623147153713"></a><ul id="zh-cn_topic_0000002517188752_ul19623147153713"><li>Quantization-aware training: <a name="zh-cn_topic_0000002517188752_ul0992647173610"></a><a name="zh-cn_topic_0000002517188752_ul0992647173610"></a><ul id="zh-cn_topic_0000002517188752_ul0992647173610"><li>INT8 quantization: √</li><li>INT4 quantization: x</li></ul>
</li><li>Channel sparsity: √</li><li>4:2 structured sparsity: √</li></ul></td>
</tr>
</tbody>
</table>

**Note: For products marked with "x" in the feature, calling the API does not report an error, but no performance gain is obtained.**

## Description<a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_section15406195619561"></a>

Static combination compression training API. It compresses the input model to be statically combined and compressed according to the given combination compression configuration file and record file (sparsification first, then quantization), and loads the saved weights. The input model is sparsified according to the sparsity records in the given **record\_file**, and then quantization-related operators (quantization-aware training layers for data and weights, as well as the searchN layer) are inserted into the model. The checkpoint weight parameters saved during training are loaded, and the modified torch.nn.Module model is returned.

## Prototype<a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_section428121323411"></a>

```python
compressed_retrain_model = restore_compressed_retrain_model(model, input_data, config_defination, record_file, pth_file, state_dict_name=None)
```

## Parameters<a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_section795991810344"></a>

<a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="12.950000000000001%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_p1438834363520"></a>Parameter</p></th>
<th class="cellrowborder" valign="top" width="5.949999999999999%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p1769255516412"><a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p1769255516412"></a><a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p1769255516412"></a>Input/Output</p></th>
<th class="cellrowborder" valign="top" width="81.10000000000001%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p15231205416325"><a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p15231205416325"></a><a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p15231205416325"></a>Description</p></th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="12.950000000000001%" headers="mcps1.1.4.1.1 "><p id="p202571341976"><a name="p202571341976"></a><a name="p202571341976"></a>model</p></td>
<td class="cellrowborder" valign="top" width="5.949999999999999%" headers="mcps1.1.4.1.2 "><p id="p168681026184710"><a name="p168681026184710"></a><a name="p168681026184710"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="81.10000000000001%" headers="mcps1.1.4.1.3 "><p id="p128631143163419"><a name="p128631143163419"></a><a name="p128631143163419"></a>Meaning: Torch model.</p>
<p id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p11225740182619"><a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p11225740182619"></a><a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p11225740182619"></a>Data type: torch.nn.Module</p></td>
</tr>
<tr id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_row21251438195"><td class="cellrowborder" valign="top" width="12.950000000000001%" headers="mcps1.1.4.1.1 "><p id="p16256941670"><a name="p16256941670"></a><a name="p16256941670"></a>input_data</p></td>
<td class="cellrowborder" valign="top" width="5.949999999999999%" headers="mcps1.1.4.1.2 "><p id="p13482131324113"><a name="p13482131324113"></a><a name="p13482131324113"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="81.10000000000001%" headers="mcps1.1.4.1.3 "><p id="p6795104910343"><a name="p6795104910343"></a><a name="p6795104910343"></a>Meaning: Input data of the model. A torch.tensor is equivalent to tuple(torch.tensor).</p>
<p id="p12421175414419"><a name="p12421175414419"></a><a name="p12421175414419"></a>Data type: tuple</p></td>
</tr>
<tr id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_row16254133823918"><td class="cellrowborder" valign="top" width="12.950000000000001%" headers="mcps1.1.4.1.1 "><p id="p325314411272"><a name="p325314411272"></a><a name="p325314411272"></a>config_defination</p></td>
<td class="cellrowborder" valign="top" width="5.949999999999999%" headers="mcps1.1.4.1.2 "><p id="p18491181320411"><a name="p18491181320411"></a><a name="p18491181320411"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="81.10000000000001%" headers="mcps1.1.4.1.3 "><p id="p1896615017345"><a name="p1896615017345"></a><a name="p1896615017345"></a>Meaning: Simple configuration file for static combination compression.</p>
<p id="p827435611439"><a name="p827435611439"></a><a name="p827435611439"></a>The simple configuration file <em id="i497145210579"><a name="i497145210579"></a><a name="i497145210579"></a>compressed</em>.cfg is generated based on the retrain_config_pytorch.proto file. The path of the *.proto file is: <em id="i12274856104314"><a name="i12274856104314"></a><a name="i12274856104314"></a><span id="ph244511481409"><a name="ph244511481409"></a><a name="ph244511481409"></a>AMCT</span> installation directory</em>/amct_pytorch/proto/. For the parameter description of the *.proto file and a sample of the generated <em id="i2675712185810"><a name="i2675712185810"></a><a name="i2675712185810"></a>compressed</em>.cfg simple quantization configuration file, see <a href="../context/qat_config.md">Quantization-Aware Training Simple Configuration File</a>.</p>
<p id="p64303542418"><a name="p64303542418"></a><a name="p64303542418"></a>Data type: string</p></td>
</tr>
<tr id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_row2997141123911"><td class="cellrowborder" valign="top" width="12.950000000000001%" headers="mcps1.1.4.1.1 "><p id="p524715411979"><a name="p524715411979"></a><a name="p524715411979"></a>record_file</p></td>
<td class="cellrowborder" valign="top" width="5.949999999999999%" headers="mcps1.1.4.1.2 "><p id="p175011134411"><a name="p175011134411"></a><a name="p175011134411"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="81.10000000000001%" headers="mcps1.1.4.1.3 "><p id="p945375373414"><a name="p945375373414"></a><a name="p945375373414"></a>Meaning: File that records the sparsification and quantization factors.</p>
<p id="p1543455424113"><a name="p1543455424113"></a><a name="p1543455424113"></a>Data type: string</p></td>
</tr>
<tr id="row198861208188"><td class="cellrowborder" valign="top" width="12.950000000000001%" headers="mcps1.1.4.1.1 "><p id="p98861007189"><a name="p98861007189"></a><a name="p98861007189"></a>pth_file</p></td>
<td class="cellrowborder" valign="top" width="5.949999999999999%" headers="mcps1.1.4.1.2 "><p id="p11886100141815"><a name="p11886100141815"></a><a name="p11886100141815"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="81.10000000000001%" headers="mcps1.1.4.1.3 "><p id="p97281755163418"><a name="p97281755163418"></a><a name="p97281755163418"></a>Meaning: Weight file saved during training.</p>
<p id="p2886100161815"><a name="p2886100161815"></a><a name="p2886100161815"></a>Data type: string</p></td>
</tr>
<tr id="row573523151818"><td class="cellrowborder" valign="top" width="12.950000000000001%" headers="mcps1.1.4.1.1 "><p id="p87352315184"><a name="p87352315184"></a><a name="p87352315184"></a>state_dict_name</p></td>
<td class="cellrowborder" valign="top" width="5.949999999999999%" headers="mcps1.1.4.1.2 "><p id="p107352035189"><a name="p107352035189"></a><a name="p107352035189"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="81.10000000000001%" headers="mcps1.1.4.1.3 "><p id="p18239165793412"><a name="p18239165793412"></a><a name="p18239165793412"></a>Meaning: Key value corresponding to the weight in the weight file.</p>
<p id="p7244115232112"><a name="p7244115232112"></a><a name="p7244115232112"></a>Default value: None</p>
<p id="p187185713210"><a name="p187185713210"></a><a name="p187185713210"></a>Data type: string</p></td>
</tr>
</tbody>
</table>

## Return Value<a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_section293415513458"></a>

Returns a torch.nn.Module static combination compressed retraining model that has been sparsified according to the sparsity relationship in **record\_file**, has quantization-related layers inserted, and has the weight file loaded.

## Constraints<a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_section1443392021419"></a>

The combined compression configuration file must contain at least one configuration: a sparsification configuration or a quantization configuration.

## Example<a name="zh-cn_topic_0254208276_section213682915617"></a>

```python
import amct_pytorch as amct
# Build the network graph structure for combined compression.
model = build_model()
input_data = tuple(torch.randn(input_shape))
save_pth_path = "/your/path/to/save/tmp.pth"

record_file = os.path.join(TMP, 'compressed_record.txt')
config_defination = './compressed_cfg.cfg'

torch.save({'state_dict': model.state_dict()}, save_pth_path)
compressed_retrain_model = amct.restore_compressed_retrain_model(
                                model,
                                input_data,
                                config_defination,
                                record_file,
                                save_pth_path,
                                'state_dict')
```
