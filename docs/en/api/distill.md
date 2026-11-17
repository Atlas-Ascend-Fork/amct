# distill<a name="ZH-CN_TOPIC_0000002517028792"></a>

<!-- md-trans-meta sourceCommit=9d54d17e0942052193c827180dc43312c580acc7 translatedAt=2026-10-16T08:27:19.460Z pushedAt=2026-11-05T01:12:03.417Z nl=2 -->

## Supported Products<a name="section185612964420"></a>

<a name="zh-cn_topic_0000002517188794_table38301303189"></a>

| Product | Supported |
| ------------------------------------------- | -------- |
| Ascend 950PR&950DT products | √ |
| Atlas A3 products | √ |
| Atlas A2 products | √ |

## Description<a name="zh-cn_topic_0240187365_section15406195619561"></a>

Distills the input graph structure to be distilled according to the given distillation quantization configuration file, and returns the modified torch.nn.Module distillation model.

## Prototype<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_section428121323411"></a>

```python
distill_model = distill(model, compress_model, config_file, train_loader, epochs=1, lr=1e-3, sample_instance=None, loss=None, optimizer=None)
```

## Parameters<a name="section73811524135618"></a>

<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="12.208779122087792%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a>Parameter</p></th>
<th class="cellrowborder" valign="top" width="9.27907209279072%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0240187365_p1769255516412"><a name="zh-cn_topic_0240187365_p1769255516412"></a><a name="zh-cn_topic_0240187365_p1769255516412"></a>Input/Output</p></th>
<th class="cellrowborder" valign="top" width="78.5121487851215%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0240187365_p15231205416325"><a name="zh-cn_topic_0240187365_p15231205416325"></a><a name="zh-cn_topic_0240187365_p15231205416325"></a>Description</p></th>
</tr>
</thead>
<tbody><tr id="row26703438259"><td class="cellrowborder" valign="top" width="12.208779122087792%" headers="mcps1.1.4.1.1 "><p id="p17670243192514"><a name="p17670243192514"></a><a name="p17670243192514"></a>model</p></td>
<td class="cellrowborder" valign="top" width="9.27907209279072%" headers="mcps1.1.4.1.2 "><p id="p1167011431259"><a name="p1167011431259"></a><a name="p1167011431259"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.5121487851215%" headers="mcps1.1.4.1.3 "><p id="p18594105994512"><a name="p18594105994512"></a><a name="p18594105994512"></a>Meaning: The original floating-point model to be distilled and quantized, with weights already loaded.</p>
<p id="p1367084319253"><a name="p1367084319253"></a><a name="p1367084319253"></a>Data type: torch.nn.Module</p></td>
</tr>
<tr id="row7958203817251"><td class="cellrowborder" valign="top" width="12.208779122087792%" headers="mcps1.1.4.1.1 "><p id="p3959163822513"><a name="p3959163822513"></a><a name="p3959163822513"></a>compress_model</p></td>
<td class="cellrowborder" valign="top" width="9.27907209279072%" headers="mcps1.1.4.1.2 "><p id="p696093882516"><a name="p696093882516"></a><a name="p696093882516"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.5121487851215%" headers="mcps1.1.4.1.3 "><p id="p744919615469"><a name="p744919615469"></a><a name="p744919615469"></a>Meaning: Modified torch.nn.Module model that can be used for distillation.</p>
<p id="p105551238122714"><a name="p105551238122714"></a><a name="p105551238122714"></a>Data type: torch.nn.Module</p>
<p id="p1127454492712"><a name="p1127454492712"></a><a name="p1127454492712"></a>Usage constraints: The model input to this API must be a quantized compressed model.</p></td>
</tr>
<tr id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="12.208779122087792%" headers="mcps1.1.4.1.1 "><p id="p31941230154616"><a name="p31941230154616"></a><a name="p31941230154616"></a>config_file</p></td>
<td class="cellrowborder" valign="top" width="9.27907209279072%" headers="mcps1.1.4.1.2 "><p id="p1219493024610"><a name="p1219493024610"></a><a name="p1219493024610"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.5121487851215%" headers="mcps1.1.4.1.3 "><p id="p165807816468"><a name="p165807816468"></a><a name="p165807816468"></a>Meaning: User-generated distillation quantization configuration file, used to specify the configuration of quantization layers and the distillation structure in the model network.</p>
<p id="p7194123094612"><a name="p7194123094612"></a><a name="p7194123094612"></a>Data type: string</p>
<p id="p9725102314288"><a name="p9725102314288"></a><a name="p9725102314288"></a>Usage constraints: The config.json passed to this API must be consistent with the config.json passed to the <a href="create_distill_config.md">create_distill_config</a> API.</p></td>
</tr>
<tr id="row4280131024613"><td class="cellrowborder" valign="top" width="12.208779122087792%" headers="mcps1.1.4.1.1 "><p id="p4194143084610"><a name="p4194143084610"></a><a name="p4194143084610"></a>train_loader</p></td>
<td class="cellrowborder" valign="top" width="9.27907209279072%" headers="mcps1.1.4.1.2 "><p id="p14194430114612"><a name="p14194430114612"></a><a name="p14194430114612"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.5121487851215%" headers="mcps1.1.4.1.3 "><p id="p81391110194614"><a name="p81391110194614"></a><a name="p81391110194614"></a>Meaning: Training dataset.</p>
<p id="p19194123013463"><a name="p19194123013463"></a><a name="p19194123013463"></a>Data type: torch.utils.data.DataLoader</p>
<p id="p3677131516495"><a name="p3677131516495"></a><a name="p3677131516495"></a>Usage constraints: Must match the model input size.</p></td>
</tr>
<tr id="row16306625164620"><td class="cellrowborder" valign="top" width="12.208779122087792%" headers="mcps1.1.4.1.1 "><p id="p19194113044616"><a name="p19194113044616"></a><a name="p19194113044616"></a>epochs</p></td>
<td class="cellrowborder" valign="top" width="9.27907209279072%" headers="mcps1.1.4.1.2 "><p id="p1519433016468"><a name="p1519433016468"></a><a name="p1519433016468"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.5121487851215%" headers="mcps1.1.4.1.3 "><p id="p8645131414611"><a name="p8645131414611"></a><a name="p8645131414611"></a>Meaning: Maximum number of iterations.</p>
<p id="p12601625114711"><a name="p12601625114711"></a><a name="p12601625114711"></a>Default value: 1</p>
<p id="p31945306466"><a name="p31945306466"></a><a name="p31945306466"></a>Data type: int</p></td>
</tr>
<tr id="row1653170112112"><td class="cellrowborder" valign="top" width="12.208779122087792%" headers="mcps1.1.4.1.1 "><p id="p5654708217"><a name="p5654708217"></a><a name="p5654708217"></a>lr</p></td>
<td class="cellrowborder" valign="top" width="9.27907209279072%" headers="mcps1.1.4.1.2 "><p id="p965419013216"><a name="p965419013216"></a><a name="p965419013216"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.5121487851215%" headers="mcps1.1.4.1.3 "><p id="p1597721516464"><a name="p1597721516464"></a><a name="p1597721516464"></a>Meaning: Learning rate.</p>
<p id="p1565412010213"><a name="p1565412010213"></a><a name="p1565412010213"></a>Default value: 1e-3</p>
<p id="p58971631182118"><a name="p58971631182118"></a><a name="p58971631182118"></a>Data type: float</p></td>
</tr>
<tr id="row93651958172014"><td class="cellrowborder" valign="top" width="12.208779122087792%" headers="mcps1.1.4.1.1 "><p id="p20365135822020"><a name="p20365135822020"></a><a name="p20365135822020"></a>sample_instance</p></td>
<td class="cellrowborder" valign="top" width="9.27907209279072%" headers="mcps1.1.4.1.2 "><p id="p636615582207"><a name="p636615582207"></a><a name="p636615582207"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.5121487851215%" headers="mcps1.1.4.1.3 "><p id="p99191117194610"><a name="p99191117194610"></a><a name="p99191117194610"></a>Meaning: Instantiated object of the method provided by the user for obtaining model input data.</p>
<p id="p7366185814202"><a name="p7366185814202"></a><a name="p7366185814202"></a>Default value: None</p>
<p id="p48401250112119"><a name="p48401250112119"></a><a name="p48401250112119"></a>Data type: DistillSampleBase</p>
<p id="p1871951718229"><a name="p1871951718229"></a><a name="p1871951718229"></a>Usage constraints: Must inherit from the DistillSampleBase class and implement the get_model_input_data method. See the <em id="i12274856104314"><a name="i12274856104314"></a><a name="i12274856104314"></a><span id="ph244511481409"><a name="ph244511481409"></a><a name="ph244511481409"></a>AMCT</span> installation directory</em>/amct_pytorch/distill/distill_sample.py file.</p></td>
</tr>
<tr id="row156638166467"><td class="cellrowborder" valign="top" width="12.208779122087792%" headers="mcps1.1.4.1.1 "><p id="p85601757113913"><a name="p85601757113913"></a><a name="p85601757113913"></a>loss</p></td>
<td class="cellrowborder" valign="top" width="9.27907209279072%" headers="mcps1.1.4.1.2 "><p id="p919503013466"><a name="p919503013466"></a><a name="p919503013466"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.5121487851215%" headers="mcps1.1.4.1.3 "><p id="p16891172110462"><a name="p16891172110462"></a><a name="p16891172110462"></a>Meaning: Instantiated object used to compute the loss.</p>
<p id="p12576154584710"><a name="p12576154584710"></a><a name="p12576154584710"></a>Default value: None</p>
<p id="p1195430114611"><a name="p1195430114611"></a><a name="p1195430114611"></a>Data type: torch.nn.Modules.loss._Loss</p></td>
</tr>
<tr id="row5663171654613"><td class="cellrowborder" valign="top" width="12.208779122087792%" headers="mcps1.1.4.1.1 "><p id="p1661158114011"><a name="p1661158114011"></a><a name="p1661158114011"></a>optimizer</p></td>
<td class="cellrowborder" valign="top" width="9.27907209279072%" headers="mcps1.1.4.1.2 "><p id="p2195330174616"><a name="p2195330174616"></a><a name="p2195330174616"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.5121487851215%" headers="mcps1.1.4.1.3 "><p id="p152467230463"><a name="p152467230463"></a><a name="p152467230463"></a>Meaning: Instantiated object of the optimizer.</p>
<p id="p15195153013461"><a name="p15195153013461"></a><a name="p15195153013461"></a>Default value: None</p>
<p id="p10195230194612"><a name="p10195230194612"></a><a name="p10195230194612"></a>Data type: torch.optim.Optimizer</p></td>
</tr>
</tbody>
</table>

## Return Value<a name="zh-cn_topic_0240188739_zh-cn_topic_0122830089_section293415513458"></a>

Modified torch.nn.Module distillation model.

## Example<a name="zh-cn_topic_0240188739_section64231658994"></a>

```python
import amct_pytorch as amct
# Build the network graph structure to be distilled and quantized.
model = build_model()
model.load_state_dict(torch.load(state_dict_path))
compress_model = compress(model)
input_data = tuple([torch.randn(input_shape)])
train_loader = torch.utils.data.DataLoader(input_data)
loss = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(compress_model.parameters(), lr=0.1)

# Distill.
distill_model = amct.distill(
                model,
                compress_model,
                config_json_file,
                train_loader,
                epochs=1,
                lr=1e-3,
                sample_instance=None, 
                loss=loss,
                optimizer=optimizer)
```
