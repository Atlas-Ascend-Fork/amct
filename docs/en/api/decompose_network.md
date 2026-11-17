# decompose\_network<a name="ZH-CN_TOPIC_0000002548788617"></a>

<!-- md-trans-meta sourceCommit=4a47814eed18048e444f9a2acc42d9864ddc477f translatedAt=2026-10-16T08:25:40.818Z pushedAt=2026-11-05T01:12:03.416Z nl=2 -->

## Supported Products<a name="section185612964420"></a>

<a name="zh-cn_topic_0000002517188794_table38301303189"></a>

| Product | Supported |
| ------------------------------------------- | -------- |
| Ascend 950PR&950DT products | √ |
| Atlas A3 products | √ |
| Atlas A2 products | √ |

## Description<a name="zh-cn_topic_0240188739_section15406195619561"></a>

The user inputs a PyTorch model object and the decomposition information file saved through [auto\_decomposition](./auto_decomposition.md). Based on the decomposition information file, the model object is transformed into a tensor-decomposed structure, yielding the decomposed model object and the corresponding names of the layers before and after decomposition.

## Prototype<a name="zh-cn_topic_0240188739_zh-cn_topic_0122830089_section428121323411"></a>

```
model, changes = decompose_network(model, decompose_info_path)
```

## Parameters<a name="zh-cn_topic_0240188739_zh-cn_topic_0122830089_section795991810344"></a>

<a name="zh-cn_topic_0240188739_zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0240188739_zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="13.819999999999999%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0240188739_zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0240188739_zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0240188739_zh-cn_topic_0122830089_p1438834363520"></a>Parameter</p></th>
<th class="cellrowborder" valign="top" width="7.53%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0240188739_p1769255516412"><a name="zh-cn_topic_0240188739_p1769255516412"></a><a name="zh-cn_topic_0240188739_p1769255516412"></a>Input/Output</p></th>
<th class="cellrowborder" valign="top" width="78.64999999999999%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0240188739_p15231205416325"><a name="zh-cn_topic_0240188739_p15231205416325"></a><a name="zh-cn_topic_0240188739_p15231205416325"></a>Description</p></th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0240188739_zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="13.819999999999999%" headers="mcps1.1.4.1.1 "><p id="p163296363467"><a name="p163296363467"></a><a name="p163296363467"></a>model</p></td>
<td class="cellrowborder" valign="top" width="7.53%" headers="mcps1.1.4.1.2 "><p id="p16326936194620"><a name="p16326936194620"></a><a name="p16326936194620"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.64999999999999%" headers="mcps1.1.4.1.3 "><p id="p196581825114215"><a name="p196581825114215"></a><a name="p196581825114215"></a>Meaning: PyTorch model object to be decomposed. When calling this API, it is recommended to place the model on the CPU to prevent insufficient video memory during decomposition.</p>
<p id="zh-cn_topic_0240188739_p11225740182619"><a name="zh-cn_topic_0240188739_p11225740182619"></a><a name="zh-cn_topic_0240188739_p11225740182619"></a>Data type: torch.nn.Module</p></td>
</tr>
<tr id="zh-cn_topic_0240188739_row21251438195"><td class="cellrowborder" valign="top" width="13.819999999999999%" headers="mcps1.1.4.1.1 "><p id="p679084764614"><a name="p679084764614"></a><a name="p679084764614"></a>decompose_info_path</p></td>
<td class="cellrowborder" valign="top" width="7.53%" headers="mcps1.1.4.1.2 "><p id="p6690203419464"><a name="p6690203419464"></a><a name="p6690203419464"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.64999999999999%" headers="mcps1.1.4.1.3 "><p id="p971414288427"><a name="p971414288427"></a><a name="p971414288427"></a>Meaning: Path of the decomposition information file, which is obtained through <a href="auto_decomposition.md">auto_decomposition</a>.</p>
<p id="p16659115502"><a name="p16659115502"></a><a name="p16659115502"></a>Data type: string</p></td>
</tr>
</tbody>
</table>

## Return Value<a name="zh-cn_topic_0240188739_zh-cn_topic_0122830089_section293415513458"></a>

- Model object changed to the tensor-decomposed structure, of the type **torch.nn.Module**.
- Dictionary composed of the corresponding layer names before and after tensor decomposition, of the type **dict**, in the form of \{'conv1': \['conv1.0', 'conv1.1'\], 'conv2': \['conv2.0', 'conv2.1'\], ...\}.

## Constraints<a name="zh-cn_topic_0240188739_section1443392021419"></a>

- The user input model must be an object of the **torch.nn.Module** type.
- This API function only supports modifying the structure of convolutions built through **torch.nn.Conv2d\(\)**.
- The user input model structure must be consistent with the model structure used when calling [auto\_decomposition](./auto_decomposition.md) to obtain the decomposition information file. The decomposition information file must be used together with this model structure.

## Example<a name="section179052217494"></a>

```python
from amct_pytorch.tensor_decompose import decompose_network
net = Net()                                                      # Build the user model object.
net, changes = decompose_network(                                # Load the decomposition information file and modify the model structure to the tensor-decomposed structure.
    model=net,
    decompose_info_path="decomposed_path/decompose_info.json"    # Path of the decomposition info file saved by auto_decomposition.
)
```

> [!NOTE] Note:
>
>1. When model training is involved, this API must be called before the model parameters are passed to the optimizer. If **torch.nn.parallel.DistributedDataParallel** \(DDP\) is used, this API must also be called before the model is passed to DDP.
>2. This API modifies the passed-in model object in place, meaning that after decomposition, the model object passed in by the user is changed (exception: if the passed-in model is a **torch.nn.Conv2d** object, this API does not modify it, and the returned decomposed model is a newly constructed **torch.nn.Module** object).
>3. This API only modifies the model structure and does not update the decomposed convolution weights, whose values are the default values constructed by torch.nn.Conv2d\(\). If fine-tuning is required, save the decomposed model weights after calling **auto\_decomposition**, load these weights after calling this API, and then perform fine-tuning.
