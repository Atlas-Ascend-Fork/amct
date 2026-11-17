# restore\_prune\_retrain\_model<a name="ZH-CN_TOPIC_0000002548668541"></a>

<!-- md-trans-meta sourceCommit=9d54d17e0942052193c827180dc43312c580acc7 translatedAt=2026-10-16T08:35:40.001Z pushedAt=2026-11-05T01:12:03.424Z nl=2 -->

## Supported Products<a name="section18210163319189"></a>

<a name="zh-cn_topic_0000002548668559_table38301303189"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002548668559_row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000002548668559_p1883113061818"><a name="zh-cn_topic_0000002548668559_p1883113061818"></a><a name="zh-cn_topic_0000002548668559_p1883113061818"></a><span id="zh-cn_topic_0000002548668559_ph20833205312295"><a name="zh-cn_topic_0000002548668559_ph20833205312295"></a><a name="zh-cn_topic_0000002548668559_ph20833205312295"></a>Product</span></p></th>
<th class="cellrowborder" align="left" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000002548668559_p783113012187"><a name="zh-cn_topic_0000002548668559_p783113012187"></a><a name="zh-cn_topic_0000002548668559_p783113012187"></a>Supported</p></th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002548668559_row574891710101"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002548668559_p177491117171015"><a name="zh-cn_topic_0000002548668559_p177491117171015"></a><a name="zh-cn_topic_0000002548668559_p177491117171015"></a><span id="zh-cn_topic_0000002548668559_ph2272194216543"><a name="zh-cn_topic_0000002548668559_ph2272194216543"></a><a name="zh-cn_topic_0000002548668559_ph2272194216543"></a>Ascend 950PR&950DT products</span></p></td>
<td class="cellrowborder" align="left" valign="top" width="42%" headers="mcps1.1.3.1.2 "><a name="zh-cn_topic_0000002548668559_ul1367712433612"></a><a name="zh-cn_topic_0000002548668559_ul1367712433612"></a><ul id="zh-cn_topic_0000002548668559_ul1367712433612"><li>Channel sparsity: √</li><li>4:2 structured sparsity API: x</li></ul></td>
</tr>
<tr id="zh-cn_topic_0000002548668559_row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002548668559_p48327011813"><a name="zh-cn_topic_0000002548668559_p48327011813"></a><a name="zh-cn_topic_0000002548668559_p48327011813"></a><span id="zh-cn_topic_0000002548668559_ph583230201815"><a name="zh-cn_topic_0000002548668559_ph583230201815"></a><a name="zh-cn_topic_0000002548668559_ph583230201815"></a><term id="zh-cn_topic_0000002548668559_zh-cn_topic_0000001312391781_term1253731311225">Atlas A3 products</term></span></p></td>
<td class="cellrowborder" align="left" valign="top" width="42%" headers="mcps1.1.3.1.2 "><a name="zh-cn_topic_0000002548668559_ul66191179379"></a><a name="zh-cn_topic_0000002548668559_ul66191179379"></a><ul id="zh-cn_topic_0000002548668559_ul66191179379"><li>Channel sparsity: √</li><li>4:2 structured sparsity API: √</li></ul></td>
</tr>
<tr id="zh-cn_topic_0000002548668559_row173226882415"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002548668559_p14832120181815"><a name="zh-cn_topic_0000002548668559_p14832120181815"></a><a name="zh-cn_topic_0000002548668559_p14832120181815"></a><span id="zh-cn_topic_0000002548668559_ph1483216010188"><a name="zh-cn_topic_0000002548668559_ph1483216010188"></a><a name="zh-cn_topic_0000002548668559_ph1483216010188"></a><term id="zh-cn_topic_0000002548668559_zh-cn_topic_0000001312391781_term11962195213215">Atlas A2 products</term></span></p></td>
<td class="cellrowborder" align="left" valign="top" width="42%" headers="mcps1.1.3.1.2 "><a name="zh-cn_topic_0000002548668559_ul19623147153713"></a><a name="zh-cn_topic_0000002548668559_ul19623147153713"></a><ul id="zh-cn_topic_0000002548668559_ul19623147153713"><li>Channel sparsity: √</li><li>4:2 structured sparsity API: √</li></ul></td>
</tr>
</tbody>
</table>

**Note: For the 4:2 structured sparsity feature described above, calling the API on products marked with "x" does not report an error, but no performance gain can be obtained.**

## Description<a name="zh-cn_topic_0240187365_section15406195619561"></a>

Sparsifies the input graph structure to be pruned according to the sparsity records in **record\_file**, and returns the modified torch.nn.Module model that can be used for post-sparsity retraining. Only one of channel sparsity and 4:2 structured sparsity APIs can be enabled at a time.

## Prototype<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_section428121323411"></a>

```python
prune_retrain_model = restore_prune_retrain_model (model, input_data, record_file, config_defination, pth_file, state_dict_name=None)
```

## Parameters<a name="section73811524135618"></a>

<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="12.690000000000001%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a>Parameter</p></th>
<th class="cellrowborder" valign="top" width="8.82%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0240187365_p1769255516412"><a name="zh-cn_topic_0240187365_p1769255516412"></a><a name="zh-cn_topic_0240187365_p1769255516412"></a>Input/Output</p></th>
<th class="cellrowborder" valign="top" width="78.49000000000001%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0240187365_p15231205416325"><a name="zh-cn_topic_0240187365_p15231205416325"></a><a name="zh-cn_topic_0240187365_p15231205416325"></a>Description</p></th>
</tr>
</thead>
<tbody><tr id="row4280131024613"><td class="cellrowborder" valign="top" width="12.690000000000001%" headers="mcps1.1.4.1.1 "><p id="p4194143084610"><a name="p4194143084610"></a><a name="p4194143084610"></a>model</p></td>
<td class="cellrowborder" valign="top" width="8.82%" headers="mcps1.1.4.1.2 "><p id="p14194430114612"><a name="p14194430114612"></a><a name="p14194430114612"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.49000000000001%" headers="mcps1.1.4.1.3 "><p id="p11108588311"><a name="p11108588311"></a><a name="p11108588311"></a>Meaning: Model to be sparsified, with weights not loaded.</p>
<p id="p19194123013463"><a name="p19194123013463"></a><a name="p19194123013463"></a>Data type: torch.nn.Module</p></td>
</tr>
<tr id="row156638166467"><td class="cellrowborder" valign="top" width="12.690000000000001%" headers="mcps1.1.4.1.1 "><p id="p1619443010466"><a name="p1619443010466"></a><a name="p1619443010466"></a>input_data</p></td>
<td class="cellrowborder" valign="top" width="8.82%" headers="mcps1.1.4.1.2 "><p id="p919503013466"><a name="p919503013466"></a><a name="p919503013466"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.49000000000001%" headers="mcps1.1.4.1.3 "><p id="p97411513932"><a name="p97411513932"></a><a name="p97411513932"></a>Meaning: Input data of the model. A <span>torch.tensor</span><span> is equivalent to tuple(torch.tensor)</span>.</p>
<p id="p1195430114611"><a name="p1195430114611"></a><a name="p1195430114611"></a>Data type: tuple</p></td>
</tr>
<tr id="row10181104881712"><td class="cellrowborder" valign="top" width="12.690000000000001%" headers="mcps1.1.4.1.1 "><p id="p19194113044616"><a name="p19194113044616"></a><a name="p19194113044616"></a>record_file</p></td>
<td class="cellrowborder" valign="top" width="8.82%" headers="mcps1.1.4.1.2 "><p id="p1519433016468"><a name="p1519433016468"></a><a name="p1519433016468"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.49000000000001%" headers="mcps1.1.4.1.3 "><p id="p199961411311"><a name="p199961411311"></a><a name="p199961411311"></a>Meaning: File path and name that records the sparse information, generated by <a href="create_prune_retrain_model.md">create_prune_retrain_model</a>. Ensure that the models generated by the two APIs are consistent.</p>
<p id="p31945306466"><a name="p31945306466"></a><a name="p31945306466"></a>Data type: string</p></td>
</tr>
<tr id="row1944910420618"><td class="cellrowborder" valign="top" width="12.690000000000001%" headers="mcps1.1.4.1.1 "><p id="p9475183920481"><a name="p9475183920481"></a><a name="p9475183920481"></a>config_defination</p></td>
<td class="cellrowborder" valign="top" width="8.82%" headers="mcps1.1.4.1.2 "><p id="p14475173934819"><a name="p14475173934819"></a><a name="p14475173934819"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.49000000000001%" headers="mcps1.1.4.1.3 "><p id="p102701016133"><a name="p102701016133"></a><a name="p102701016133"></a>Meaning: Simplified configuration file. The simplified configuration file <em id="i37264185718"><a name="i37264185718"></a><a name="i37264185718"></a>prune</em>.cfg is generated based on the retrain_config_pytorch.proto file. The path of the *.proto file is: <em id="i12274856104314"><a name="i12274856104314"></a><a name="i12274856104314"></a><span id="ph244511481409"><a name="ph244511481409"></a><a name="ph244511481409"></a>AMCT</span> installation directory</em>/amct_pytorch/proto/. For the parameter description of the *.proto file and the sample of the generated <em id="i12381525154418"><a name="i12381525154418"></a><a name="i12381525154418"></a>prune</em>.cfg simplified quantization configuration file, see <a href="../context/qat_config.md">Simplified Configuration File for Quantization-Aware Training</a>.</p>
<p id="p174753395482"><a name="p174753395482"></a><a name="p174753395482"></a>Data type: string</p></td>
</tr>
<tr id="row191946331432"><td class="cellrowborder" valign="top" width="12.690000000000001%" headers="mcps1.1.4.1.1 "><p id="p19477173954813"><a name="p19477173954813"></a><a name="p19477173954813"></a>pth_file</p></td>
<td class="cellrowborder" valign="top" width="8.82%" headers="mcps1.1.4.1.2 "><p id="p184774396482"><a name="p184774396482"></a><a name="p184774396482"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.49000000000001%" headers="mcps1.1.4.1.3 "><p id="p271119171436"><a name="p271119171436"></a><a name="p271119171436"></a>Meaning: Weight file saved during training.</p>
<p id="p1247716394489"><a name="p1247716394489"></a><a name="p1247716394489"></a>Data type: string</p></td>
</tr>
<tr id="row1744420354437"><td class="cellrowborder" valign="top" width="12.690000000000001%" headers="mcps1.1.4.1.1 "><p id="p154772395483"><a name="p154772395483"></a><a name="p154772395483"></a>state_dict_name</p></td>
<td class="cellrowborder" valign="top" width="8.82%" headers="mcps1.1.4.1.2 "><p id="p20477939124816"><a name="p20477939124816"></a><a name="p20477939124816"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.49000000000001%" headers="mcps1.1.4.1.3 "><p id="p1717319539"><a name="p1717319539"></a><a name="p1717319539"></a>Meaning: Key corresponding to the weight in the weight file.</p>
<p id="p225724119501"><a name="p225724119501"></a><a name="p225724119501"></a>Default value: None</p>
<p id="p94771439144818"><a name="p94771439144818"></a><a name="p94771439144818"></a>Data type: string</p></td>
</tr>
</tbody>
</table>

## Return Value<a name="zh-cn_topic_0240188739_zh-cn_topic_0122830089_section293415513458"></a>

Returns the modified torch.nn.Module model that can be used for post-sparsity retraining.

## Example<a name="zh-cn_topic_0240188739_section64231658994"></a>

```python
import amct_pytorch as amct
# Build the network graph structure to be sparsified.
config_defination = './prune_cfg.cfg'
model = build_model()
input_data = tuple([torch.randn(input_shape)])
save_pth_path = "/your/path/to/save/tmp.pth"
model.load_state_dict(torch.load(state_dict_path))

# Call the sparse model API.
record_file = os.path.join(TMP, 'scale_offset_record.txt')
prune_retrain_model = amct.restore_prune_retrain_model(
               model,
               input_data,
               record_file,
               config_defination,
               save_pth_path,
              'state_dict')
```
