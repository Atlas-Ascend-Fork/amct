# auto\_decomposition<a name="ZH-CN_TOPIC_0000002548788637"></a>

<!-- md-trans-meta sourceCommit=4a47814eed18048e444f9a2acc42d9864ddc477f translatedAt=2026-10-16T08:09:22.714Z pushedAt=2026-11-05T01:12:03.403Z nl=2 -->

## Supported Products<a name="section185612964420"></a>

<a name="zh-cn_topic_0000002517188794_table38301303189"></a>

| Product | Supported |
| ------------------------------------------- | -------- |
| Ascend 950PR&950DT products | √ |
| Atlas A3 products | √ |
| Atlas A2 products | √ |

## Description<a name="zh-cn_topic_0240188739_section15406195619561"></a>

Performs tensor decomposition on the user-input PyTorch model object to obtain the decomposed model object and the corresponding layer names before and after decomposition, and optionally saves the decomposition information file.

## Prototype<a name="zh-cn_topic_0240188739_zh-cn_topic_0122830089_section428121323411"></a>

```python
model, changes = auto_decomposition(model, decompose_info_path=None)
```

## Parameters<a name="zh-cn_topic_0240188739_zh-cn_topic_0122830089_section795991810344"></a>

<a name="zh-cn_topic_0240188739_zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0240188739_zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="14.45%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0240188739_zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0240188739_zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0240188739_zh-cn_topic_0122830089_p1438834363520"></a>Parameter</p></th>
<th class="cellrowborder" valign="top" width="7.8100000000000005%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0240188739_p1769255516412"><a name="zh-cn_topic_0240188739_p1769255516412"></a><a name="zh-cn_topic_0240188739_p1769255516412"></a>Input/Output</p></th>
<th class="cellrowborder" valign="top" width="77.74%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0240188739_p15231205416325"><a name="zh-cn_topic_0240188739_p15231205416325"></a><a name="zh-cn_topic_0240188739_p15231205416325"></a>Usage Restrictions</p></th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0240188739_zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="14.45%" headers="mcps1.1.4.1.1 "><p id="p163296363467"><a name="p163296363467"></a><a name="p163296363467"></a>model</p></td>
<td class="cellrowborder" valign="top" width="7.8100000000000005%" headers="mcps1.1.4.1.2 "><p id="p16326936194620"><a name="p16326936194620"></a><a name="p16326936194620"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="77.74%" headers="mcps1.1.4.1.3 "><p id="p3490152934016"><a name="p3490152934016"></a><a name="p3490152934016"></a>Meaning: PyTorch model object to be decomposed that contains pretrained weights. When calling this API, it is recommended to place the model on the CPU to prevent insufficient memory during decomposition.</p>
<p id="zh-cn_topic_0240188739_p11225740182619"><a name="zh-cn_topic_0240188739_p11225740182619"></a><a name="zh-cn_topic_0240188739_p11225740182619"></a>Data type: torch.nn.Module</p></td>
</tr>
<tr id="zh-cn_topic_0240188739_row21251438195"><td class="cellrowborder" valign="top" width="14.45%" headers="mcps1.1.4.1.1 "><p id="p679084764614"><a name="p679084764614"></a><a name="p679084764614"></a>decompose_info_path</p></td>
<td class="cellrowborder" valign="top" width="7.8100000000000005%" headers="mcps1.1.4.1.2 "><p id="p6690203419464"><a name="p6690203419464"></a><a name="p6690203419464"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="77.74%" headers="mcps1.1.4.1.3 "><p id="p11521032134013"><a name="p11521032134013"></a><a name="p11521032134013"></a>Meaning: Path for saving the decomposition information file. The file is stored in JSON format, so a .json extension is recommended. When set to None, the decomposition information file is not saved (default).</p>
<p id="p16659115502"><a name="p16659115502"></a><a name="p16659115502"></a>Data type: string</p>
<p id="p1920518526403"><a name="p1920518526403"></a><a name="p1920518526403"></a>Default value: None</p></td>
</tr>
</tbody>
</table>

## Return Value<a name="zh-cn_topic_0240188739_zh-cn_topic_0122830089_section293415513458"></a>

- The model object after tensor decomposition, of the type **torch.nn.Module**.
- A dictionary composed of the corresponding layer names before and after tensor decomposition, of the type **dict**, in the form of \{'conv1': \['conv1.0', 'conv1.1'\], 'conv2': \['conv2.0', 'conv2.1'\], ...\}.

## Constraints<a name="zh-cn_topic_0240188739_section1443392021419"></a>

- The model input by the user must be an object of the **torch.nn.Module** type.
- This function only decomposes convolutions built through **torch.nn.Conv2d\(\)**.
- When the user calls this API, the function automatically decomposes convolutional layers that meet the decomposition conditions. For constraints, see [Decomposition Constraints](../../README.md).

## Example<a name="section179052217494"></a>

```python
from amct_pytorch.tensor_decompose import auto_decomposition
net = Net()                                                    # Build the user model object.
net.load_state_dict(torch.load("src_path/weights.pth"))        # Load the model weights.
net, changes = auto_decomposition(                             # Perform tensor decomposition.
    model=net,
    decompose_info_path="decomposed_path/decompose_info.json"
)
```

> [!NOTE] Note
>
>1. When model training is involved, this API must be called before the model parameters are passed to the optimizer. If **torch.nn.parallel.DistributedDataParallel** \(DDP\) is used, this API must also be called before the model is passed to DDP.
>2. This API modifies the passed-in model object in place. That is, after decomposition, the model object passed in by the user is changed (exception: if the passed-in model is a **torch.nn.Conv2d** object, this API does not modify it; if decomposition occurs, the returned model is a newly constructed **torch.nn.Module** object).
