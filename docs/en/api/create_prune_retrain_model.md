# create\_prune\_retrain\_model<a name="ZH-CN_TOPIC_0000002548668559"></a>

<!-- md-trans-meta sourceCommit=4a47814eed18048e444f9a2acc42d9864ddc477f translatedAt=2026-10-16T08:19:49.154Z pushedAt=2026-11-05T01:12:03.409Z nl=2 -->

## Supported Products<a name="section197451857688"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>Product</span></p></th>
<th class="cellrowborder" align="left" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>Supported</p></th>
</tr>
</thead>
<tbody><tr id="row574891710101"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p177491117171015"><a name="p177491117171015"></a><a name="p177491117171015"></a><span id="ph2272194216543"><a name="ph2272194216543"></a><a name="ph2272194216543"></a>Ascend 950PR&950DT products</span></p></td>
<td class="cellrowborder" align="left" valign="top" width="42%" headers="mcps1.1.3.1.2 "><a name="ul1367712433612"></a><a name="ul1367712433612"></a><ul id="ul1367712433612"><li>Channel sparsity: √</li><li>4:2 structured sparsity API: x</li></ul></td>
</tr>
<tr id="row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p48327011813"><a name="p48327011813"></a><a name="p48327011813"></a><span id="ph583230201815"><a name="ph583230201815"></a><a name="ph583230201815"></a><term id="zh-cn_topic_0000001312391781_term1253731311225">Atlas A3 products</term></span></p></td>
<td class="cellrowborder" align="left" valign="top" width="42%" headers="mcps1.1.3.1.2 "><a name="ul66191179379"></a><a name="ul66191179379"></a><ul id="ul66191179379"><li>Channel sparsity: √</li><li>4:2 structured sparsity API: √</li></ul></td>
</tr>
<tr id="row173226882415"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p14832120181815"><a name="p14832120181815"></a><a name="p14832120181815"></a><span id="ph1483216010188"><a name="ph1483216010188"></a><a name="ph1483216010188"></a><term id="zh-cn_topic_0000001312391781_term11962195213215">Atlas A2 products</term></span></p></td>
<td class="cellrowborder" align="left" valign="top" width="42%" headers="mcps1.1.3.1.2 "><a name="ul19623147153713"></a><a name="ul19623147153713"></a><ul id="ul19623147153713"><li>Channel sparsity: √</li><li>4:2 structured sparsity API: √</li></ul></td>
</tr>
</tbody>
</table>

**Note: For the 4:2 structured sparsity feature described above, calling the API on products marked with "x" does not report an error, but no performance gain is obtained.**

## Description<a name="zh-cn_topic_0240187365_section15406195619561"></a>

Performs sparsification on the input graph structure to be sparsified according to the given sparsity configuration file, inserts or replaces related operators in the input graph structure, generates the **record\_file** that records sparsity information, and returns the modified torch.nn.Module model that can be used for post-sparsity training. Only one of channel sparsity and 4:2 structured sparsity APIs can be enabled at a time.

## Prototype<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_section428121323411"></a>

```python
prune_retrain_model = create_prune_retrain_model (model, input_data, config_defination, record_file)
```

## Parameters<a name="section73811524135618"></a>

<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="13.780000000000001%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a>Parameter</p></th>
<th class="cellrowborder" valign="top" width="8.01%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0240187365_p1769255516412"><a name="zh-cn_topic_0240187365_p1769255516412"></a><a name="zh-cn_topic_0240187365_p1769255516412"></a>Input/Output</p></th>
<th class="cellrowborder" valign="top" width="78.21000000000001%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0240187365_p15231205416325"><a name="zh-cn_topic_0240187365_p15231205416325"></a><a name="zh-cn_topic_0240187365_p15231205416325"></a>Description</p></th>
</tr>
</thead>
<tbody><tr id="row4280131024613"><td class="cellrowborder" valign="top" width="13.780000000000001%" headers="mcps1.1.4.1.1 "><p id="p4194143084610"><a name="p4194143084610"></a><a name="p4194143084610"></a>model</p></td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="p14194430114612"><a name="p14194430114612"></a><a name="p14194430114612"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.21000000000001%" headers="mcps1.1.4.1.3 "><p id="p661419384017"><a name="p661419384017"></a><a name="p661419384017"></a>Meaning: Model to be sparsified, with weights loaded.</p>
<p id="p19194123013463"><a name="p19194123013463"></a><a name="p19194123013463"></a>Data type: torch.nn.Module</p></td>
</tr>
<tr id="row156638166467"><td class="cellrowborder" valign="top" width="13.780000000000001%" headers="mcps1.1.4.1.1 "><p id="p1619443010466"><a name="p1619443010466"></a><a name="p1619443010466"></a>input_data</p></td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="p919503013466"><a name="p919503013466"></a><a name="p919503013466"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.21000000000001%" headers="mcps1.1.4.1.3 "><p id="p1750084516018"><a name="p1750084516018"></a><a name="p1750084516018"></a>Meaning: Input data of the model. A <span>torch.tensor</span><span> is equivalent to tuple(torch.tensor)</span>.</p>
<p id="p1195430114611"><a name="p1195430114611"></a><a name="p1195430114611"></a>Data type: tuple</p></td>
</tr>
<tr id="row135351841171711"><td class="cellrowborder" valign="top" width="13.780000000000001%" headers="mcps1.1.4.1.1 "><p id="p158301146144319"><a name="p158301146144319"></a><a name="p158301146144319"></a>config_defination</p></td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="p1027410568437"><a name="p1027410568437"></a><a name="p1027410568437"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.21000000000001%" headers="mcps1.1.4.1.3 "><p id="p8827154613010"><a name="p8827154613010"></a><a name="p8827154613010"></a>Meaning: Simplified configuration file.</p>
<p id="p827435611439"><a name="p827435611439"></a><a name="p827435611439"></a>The simplified configuration file <em id="i37264185718"><a name="i37264185718"></a><a name="i37264185718"></a>prune</em>.cfg is generated based on the retrain_config_pytorch.proto file. The *.proto file is located in: <em id="i12274856104314"><a name="i12274856104314"></a><a name="i12274856104314"></a><span id="ph244511481409"><a name="ph244511481409"></a><a name="ph244511481409"></a>AMCT</span> installation directory</em>/amct_pytorch/proto/. For the parameter description of the *.proto file and a sample of the generated <em id="i12381525154418"><a name="i12381525154418"></a><a name="i12381525154418"></a>prune</em>.cfg simplified quantization configuration file, see <a href="../context/qat_config.md">Simplified Quantization-Aware Training Configuration File</a>.</p>
<p id="p3275175644316"><a name="p3275175644316"></a><a name="p3275175644316"></a>Data type: string</p></td>
</tr>
<tr id="row10181104881712"><td class="cellrowborder" valign="top" width="13.780000000000001%" headers="mcps1.1.4.1.1 "><p id="p19194113044616"><a name="p19194113044616"></a><a name="p19194113044616"></a>record_file</p></td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="p1519433016468"><a name="p1519433016468"></a><a name="p1519433016468"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.21000000000001%" headers="mcps1.1.4.1.3 "><p id="p1335544810020"><a name="p1335544810020"></a><a name="p1335544810020"></a>Meaning: File path and name for recording sparsity information, which records the cascade relationships between channel sparsity nodes or the nodes of 4:2 structured sparsity.</p>
<p id="p31945306466"><a name="p31945306466"></a><a name="p31945306466"></a>Data type: string</p></td>
</tr>
</tbody>
</table>

## Supported Layers and Constraints for Channel Sparsity

**Table 1**  Supported layers and constraints for channel sparsity

<a name="table151175131181"></a>

<table><thead align="left"><tr id="row61178136184"><th class="cellrowborder" valign="top" width="7.5200000000000005%" id="mcps1.2.4.1.1"><p id="p185862080479"><a name="p185862080479"></a><a name="p185862080479"></a>Optimization Method</p></th>
<th class="cellrowborder" valign="top" width="16.42%" id="mcps1.2.4.1.2"><p id="p1921472241319"><a name="p1921472241319"></a><a name="p1921472241319"></a>Supported Layer Type</p></th>
<th class="cellrowborder" valign="top" width="76.06%" id="mcps1.2.4.1.3"><p id="p9117111341815"><a name="p9117111341815"></a><a name="p9117111341815"></a>Constraint</p></th>
</tr>
</thead>
<tbody><tr id="row4117613121819"><td class="cellrowborder" rowspan="2" valign="top" width="7.5200000000000005%" headers="mcps1.2.4.1.1 "><p id="p14586087476"><a name="p14586087476"></a><a name="p14586087476"></a>Channel sparsity</p></td>
<td class="cellrowborder" valign="top" width="16.42%" headers="mcps1.2.4.1.2 "><p id="p46801332355"><a name="p46801332355"></a><a name="p46801332355"></a>torch.nn.Linear: fully connected layer</p></td>
<td class="cellrowborder" valign="top" width="76.06%" headers="mcps1.2.4.1.3 "><p id="p257843204111"><a name="p257843204111"></a><a name="p257843204111"></a>Reused layers (sharing weight and bias parameters) do not support sparsity.</p></td>
</tr>
<tr id="row13118181318183"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1868011363517"><a name="p1868011363517"></a><a name="p1868011363517"></a>torch.nn.Conv2d: convolution layer</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><a name="ul4262123921520"></a><a name="ul4262123921520"></a><ul id="ul4262123921520"><li>Reused layers (sharing weight and bias parameters) do not support sparsity.</li><li>Depthwise layers can only be passively sparsified (groups=in_channels) and cannot be actively sparsified.</li><li>Only input data with the shape (N, Cin, Hin, Win) is supported.</li></ul></td>
</tr>
</tbody>
</table>

## Supported Layers and Constraints for 4:2 Structured Sparsity

**Table 1**  Supported layers and constraints

<a name="table151175131181"></a>

<table><thead align="left"><tr id="row61178136184"><th class="cellrowborder" valign="top" width="7.5200000000000005%" id="mcps1.2.4.1.1"><p id="p185862080479"><a name="p185862080479"></a><a name="p185862080479"></a>Optimization Method</p></th>
<th class="cellrowborder" valign="top" width="27.61%" id="mcps1.2.4.1.2"><p id="p1921472241319"><a name="p1921472241319"></a><a name="p1921472241319"></a>Supported Layer Type</p></th>
<th class="cellrowborder" valign="top" width="64.87%" id="mcps1.2.4.1.3"><p id="p9117111341815"><a name="p9117111341815"></a><a name="p9117111341815"></a>Constraint</p></th>
</tr>
</thead>
<tbody><tr id="row4117613121819"><td class="cellrowborder" rowspan="3" valign="top" width="7.5200000000000005%" headers="mcps1.2.4.1.1 "><p id="p14586087476"><a name="p14586087476"></a><a name="p14586087476"></a>4:2 structured sparsity</p>
<p id="p112635116333"><a name="p112635116333"></a><a name="p112635116333"></a></p></td>
<td class="cellrowborder" valign="top" width="27.61%" headers="mcps1.2.4.1.2 "><p id="p46801332355"><a name="p46801332355"></a><a name="p46801332355"></a>torch.nn.Linear: fully connected layer</p></td>
<td class="cellrowborder" valign="top" width="64.87%" headers="mcps1.2.4.1.3 "><p id="p1670925185513"><a name="p1670925185513"></a><a name="p1670925185513"></a>Reused layers (sharing weight) do not support sparsity.</p></td>
</tr>
<tr id="row13118181318183"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1868011363517"><a name="p1868011363517"></a><a name="p1868011363517"></a>torch.nn.Conv2d: convolution layer</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><a name="ul19441153635116"></a><a name="ul19441153635116"></a><ul id="ul19441153635116"><li>Reused layers (sharing weight) do not support sparsity.</li><li>Only the input data shape (N, Cin, Hin, Win) is supported.</li></ul></td>
</tr>
<tr id="row1312619511332"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p91261851123316"><a name="p91261851123316"></a><a name="p91261851123316"></a>torch.nn.ConvTranspose2d: deconvolution layer</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><a name="ul1382854365314"></a><a name="ul1382854365314"></a><ul id="ul1382854365314"><li>Reused layers (sharing weight) do not support sparsity.</li><li>Only the input data shape (N, Cin, Hin, Win) is supported.</li></ul></td>
</tr>
</tbody>
</table>

## Return Value<a name="zh-cn_topic_0240188739_zh-cn_topic_0122830089_section293415513458"></a>

Returns the modified torch.nn.Module model that can be used for post-sparsity training.

## Example<a name="zh-cn_topic_0240188739_section64231658994"></a>

```python
import amct_pytorch as amct
# Build the network graph structure to be sparsified.
model = build_model()
model.load_state_dict(torch.load(state_dict_path))
input_data = tuple([torch.randn(input_shape)])

# Call the sparsity model API.
record_file = os.path.join(TMP, 'scale_offset_record.txt')
cfg_file = './prune_config.cfg'
prune_retrain_model = amct.create_prune_retrain_model(
               model,
               input_data,
               cfg_file,
               record_file)
```
