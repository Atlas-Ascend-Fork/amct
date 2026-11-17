# auto\_channel\_prune\_search<a name="ZH-CN_TOPIC_0000002517188676"></a>

<!-- md-trans-meta sourceCommit=4a47814eed18048e444f9a2acc42d9864ddc477f translatedAt=2026-10-16T08:05:53.489Z pushedAt=2026-11-05T01:12:03.403Z nl=2 -->

## Supported Products<a name="section185612964420"></a>

<a name="zh-cn_topic_0000002517188794_table38301303189"></a>

| Product | Supported |
| ------------------------------------------- | -------- |
| Ascend 950PR&950DT products | √ |
| Atlas A3 products | √ |
| Atlas A2 products | √ |

## Description<a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_section15406195619561"></a>

The automatic channel sparsity API computes the sparsity sensitivity (which affects accuracy) and sparsity gain (which affects performance) of each channel based on the user model. The search strategy then uses this input to search for the optimal per-layer channel sparsity ratio, balancing accuracy and performance. Finally, a configuration file is output.

## Prototype<a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_section428121323411"></a>

```python
auto_channel_prune_search(model, config, input_data, output_cfg, sensitivity, search_alg)
```

## Parameters<a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_section795991810344"></a>

<a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="12.61%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_p1438834363520"></a>Parameter</p></th>
<th class="cellrowborder" valign="top" width="8.35%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p1769255516412"><a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p1769255516412"></a><a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p1769255516412"></a>Input/Output</p></th>
<th class="cellrowborder" valign="top" width="79.03999999999999%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p15231205416325"><a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p15231205416325"></a><a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p15231205416325"></a>Description</p></th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="12.61%" headers="mcps1.1.4.1.1 "><p id="p1586914264474"><a name="p1586914264474"></a><a name="p1586914264474"></a>model</p></td>
<td class="cellrowborder" valign="top" width="8.35%" headers="mcps1.1.4.1.2 "><p id="p168681026184710"><a name="p168681026184710"></a><a name="p168681026184710"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="79.03999999999999%" headers="mcps1.1.4.1.3 "><p id="p034616965"><a name="p034616965"></a><a name="p034616965"></a>Meaning: PyTorch model to be sparsified.</p>
<p id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p11225740182619"><a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p11225740182619"></a><a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p11225740182619"></a>Data type: torch.nn.Module</p></td>
</tr>
<tr id="row13348191611917"><td class="cellrowborder" valign="top" width="12.61%" headers="mcps1.1.4.1.1 "><p id="p1785212266471"><a name="p1785212266471"></a><a name="p1785212266471"></a>config</p></td>
<td class="cellrowborder" valign="top" width="8.35%" headers="mcps1.1.4.1.2 "><p id="p175011134411"><a name="p175011134411"></a><a name="p175011134411"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="79.03999999999999%" headers="mcps1.1.4.1.3 "><p id="p7677132119613"><a name="p7677132119613"></a><a name="p7677132119613"></a>Meaning: Path of the automatic channel sparsity configuration file.</p>
<p id="p4435183911710"><a name="p4435183911710"></a><a name="p4435183911710"></a>A simplified configuration file generated based on AutoChannelPruneConfig in the basic_info.proto file. The path of the *.proto file is: <em id="i54361539151713"><a name="i54361539151713"></a><a name="i54361539151713"></a><span id="ph2538725141110"><a name="ph2538725141110"></a><a name="ph2538725141110"></a>AMCT</span> installation directory</em>/amct_pytorch/proto/.</p>
<p id="p4232851666"><a name="p4232851666"></a><a name="p4232851666"></a>For the parameter description of the *.proto file and the generated example of the automatic channel sparsity search configuration file, see <a href="../context/auto_channel_prune_search_config.md">automatic channel sparsity search simplified configuration file</a>.</p>
<p id="p1543455424113"><a name="p1543455424113"></a><a name="p1543455424113"></a>Data type: string</p></td>
</tr>
<tr id="row56401431115"><td class="cellrowborder" valign="top" width="12.61%" headers="mcps1.1.4.1.1 "><p id="p17641123201113"><a name="p17641123201113"></a><a name="p17641123201113"></a>input_data</p></td>
<td class="cellrowborder" valign="top" width="8.35%" headers="mcps1.1.4.1.2 "><p id="p76411936117"><a name="p76411936117"></a><a name="p76411936117"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="79.03999999999999%" headers="mcps1.1.4.1.3 "><p id="p5913202219613"><a name="p5913202219613"></a><a name="p5913202219613"></a>Meaning: User-provided input data (including labels).</p>
<p id="p08321173268"><a name="p08321173268"></a><a name="p08321173268"></a>Data type: list[data,label], where the data type of the list elements is torch.tensor.</p></td>
</tr>
<tr id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_row21251438195"><td class="cellrowborder" valign="top" width="12.61%" headers="mcps1.1.4.1.1 "><p id="p487821312913"><a name="p487821312913"></a><a name="p487821312913"></a>output_cfg</p></td>
<td class="cellrowborder" valign="top" width="8.35%" headers="mcps1.1.4.1.2 "><p id="p810515371917"><a name="p810515371917"></a><a name="p810515371917"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="79.03999999999999%" headers="mcps1.1.4.1.3 "><p id="p128965295620"><a name="p128965295620"></a><a name="p128965295620"></a>Meaning: Path of the final output channel sparsity configuration file.</p>
<p id="p89686200106"><a name="p89686200106"></a><a name="p89686200106"></a>Data type: string</p></td>
</tr>
<tr id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_row16254133823918"><td class="cellrowborder" valign="top" width="12.61%" headers="mcps1.1.4.1.1 "><p id="p285812654718"><a name="p285812654718"></a><a name="p285812654718"></a>sensitivity</p></td>
<td class="cellrowborder" valign="top" width="8.35%" headers="mcps1.1.4.1.2 "><p id="p18491181320411"><a name="p18491181320411"></a><a name="p18491181320411"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="79.03999999999999%" headers="mcps1.1.4.1.3 "><p id="p12644431963"><a name="p12644431963"></a><a name="p12644431963"></a>Meaning: Sensitivity calculation method.</p>
<p id="p64303542418"><a name="p64303542418"></a><a name="p64303542418"></a>Data type: string or a subclass of SensitivityBase. string refers to the methods already available in <span id="ph6261819124911"><a name="ph6261819124911"></a><a name="ph6261819124911"></a>AMCT</span>, and currently 'TaylorLossSensitivity' is available. A subclass of SensitivityBase can be instantiated and defined by the user through inheritance.</p></td>
</tr>
<tr id="row22610479916"><td class="cellrowborder" valign="top" width="12.61%" headers="mcps1.1.4.1.1 "><p id="p526217471690"><a name="p526217471690"></a><a name="p526217471690"></a>search_alg</p></td>
<td class="cellrowborder" valign="top" width="8.35%" headers="mcps1.1.4.1.2 "><p id="p149701928101817"><a name="p149701928101817"></a><a name="p149701928101817"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="79.03999999999999%" headers="mcps1.1.4.1.3 "><p id="p151474331619"><a name="p151474331619"></a><a name="p151474331619"></a>Meaning: Channel search method for the channels to be sparsified.</p>
<p id="p613275510219"><a name="p613275510219"></a><a name="p613275510219"></a>Data type: string or a subclass of SearchChannelBase. string refers to the existing methods in <span id="ph3713131922414"><a name="ph3713131922414"></a><a name="ph3713131922414"></a>AMCT</span>, with 'GreedySearch' currently available; a subclass of SearchChannelBase can be instantiated and defined by users through inheritance.</p></td>
</tr>
</tbody>
</table>

## Return Value<a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_section293415513458"></a>

None

## Example<a name="zh-cn_topic_0254208276_section213682915617"></a>

```python
import amct_pytorch as amct
# Construct the input data input_data.
input_data = torch.randn(input_shape)
model.eval()
output = model.forward(input_data)
labels = torch.randn(output.size())
data = [input_data,labels]

amct.auto_channel_prune_search(
     model=model,
     config='./tmp/sample.cfg',
     input_data=data,
     output_cfg='./tmp/output.cfg',
     sensitivity='TaylorLossSensitivity',
     search_alg='GreedySearch')
```

Description of the persisted files:

The saved automatic channel sparsity configuration file must be passed to the channel sparsity API to complete subsequent operations.
