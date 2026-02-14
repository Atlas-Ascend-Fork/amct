# decompose\_network<a name="ZH-CN_TOPIC_0000002548788617"></a>

## 产品支持情况<a name="section185612964420"></a>

<a name="zh-cn_topic_0000002517188794_table38301303189"></a>

| 产品                                        | 是否支持 |
| ------------------------------------------- | -------- |
| Ascend 950PR/Ascend 950DT                   | √        |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √        |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √        |


## 功能说明<a name="zh-cn_topic_0240188739_section15406195619561"></a>

用户输入PyTorch模型对象和通过[auto\_decomposition](./auto_decomposition.md)保存的分解信息文件，根据分解信息文件将模型对象改变为张量分解后的结构，得到分解后的模型对象和分解前后层的对应名称。

## 函数原型<a name="zh-cn_topic_0240188739_zh-cn_topic_0122830089_section428121323411"></a>

```
model, changes = decompose_network(model, decompose_info_path)
```

## 参数说明<a name="zh-cn_topic_0240188739_zh-cn_topic_0122830089_section795991810344"></a>

<a name="zh-cn_topic_0240188739_zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0240188739_zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="13.819999999999999%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0240188739_zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0240188739_zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0240188739_zh-cn_topic_0122830089_p1438834363520"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="7.53%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0240188739_p1769255516412"><a name="zh-cn_topic_0240188739_p1769255516412"></a><a name="zh-cn_topic_0240188739_p1769255516412"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="78.64999999999999%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0240188739_p15231205416325"><a name="zh-cn_topic_0240188739_p15231205416325"></a><a name="zh-cn_topic_0240188739_p15231205416325"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0240188739_zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="13.819999999999999%" headers="mcps1.1.4.1.1 "><p id="p163296363467"><a name="p163296363467"></a><a name="p163296363467"></a>model</p>
</td>
<td class="cellrowborder" valign="top" width="7.53%" headers="mcps1.1.4.1.2 "><p id="p16326936194620"><a name="p16326936194620"></a><a name="p16326936194620"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.64999999999999%" headers="mcps1.1.4.1.3 "><p id="p196581825114215"><a name="p196581825114215"></a><a name="p196581825114215"></a>含义：待分解的PyTorch模型对象。在调用该接口时建议将模型放置于CPU而不是GPU上，以防分解时显存不足。</p>
<p id="zh-cn_topic_0240188739_p11225740182619"><a name="zh-cn_topic_0240188739_p11225740182619"></a><a name="zh-cn_topic_0240188739_p11225740182619"></a>数据类型：torch.nn.Module</p>
</td>
</tr>
<tr id="zh-cn_topic_0240188739_row21251438195"><td class="cellrowborder" valign="top" width="13.819999999999999%" headers="mcps1.1.4.1.1 "><p id="p679084764614"><a name="p679084764614"></a><a name="p679084764614"></a>decompose_info_path</p>
</td>
<td class="cellrowborder" valign="top" width="7.53%" headers="mcps1.1.4.1.2 "><p id="p6690203419464"><a name="p6690203419464"></a><a name="p6690203419464"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.64999999999999%" headers="mcps1.1.4.1.3 "><p id="p971414288427"><a name="p971414288427"></a><a name="p971414288427"></a>含义：分解信息文件路径，该文件通过<a href="auto_decomposition.md">auto_decomposition</a>获得。</p>
<p id="p16659115502"><a name="p16659115502"></a><a name="p16659115502"></a>数据类型：string</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0240188739_zh-cn_topic_0122830089_section293415513458"></a>

-   改变为张量分解后结构的模型对象，类型为torch.nn.Module。
-   张量分解前后的对应层名构成的字典，类型为dict，形如\{'conv1': \['conv1.0', 'conv1.1'\], 'conv2': \['conv2.0', 'conv2.1'\], ...\}。

## 约束说明<a name="zh-cn_topic_0240188739_section1443392021419"></a>

-   用户输入的模型需为torch.nn.Module类型的对象。
-   本接口函数仅支持对通过torch.nn.Conv2d\(\)构建的卷积的结构修改。
-   用户输入的模型结构需与调用[auto\_decomposition](./auto_decomposition.md)获取分解信息文件时的模型结构一致，分解信息文件要与该模型结构配套使用。

## 调用示例<a name="section179052217494"></a>

```python
from amct_pytorch.tensor_decompose import decompose_network
net = Net()                                                      # 构建用户模型对象
net, changes = decompose_network(                                # 加载分解信息文件，将模型结构修改为张量分解后的结构
    model=net,
    decompose_info_path="decomposed_path/decompose_info.json"    # 由auto_decomposition保存的分解信息文件路径
)
```

> [!NOTE]说明：
>1.  当涉及模型训练时，本接口的调用需在将模型参数传递给优化器之前；如使用了torch.nn.parallel.DistributedDataParallel \(DDP\)，则本接口的调用也需在将模型传递给DDP之前。
>2.  本接口将原地修改传入的模型对象，即分解后会改变用户传入的模型对象本身（例外：传入的模型是一个torch.nn.Conv2d对象，该情况下本接口不会对其进行修改，返回的分解后模型是新构建的torch.nn.Module对象）。
>3.  本接口仅对模型结构进行修改，不会更新分解后的卷积权重，权重的值为torch.nn.Conv2d\(\)构建的默认值。如需fine-tune，请在调用auto\_decomposition后将分解后的模型权重保存下来，在调用本接口之后加载该权重，再进行fine-tune。

