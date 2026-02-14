# auto\_decomposition<a name="ZH-CN_TOPIC_0000002548788637"></a>

## 产品支持情况<a name="section185612964420"></a>

<a name="zh-cn_topic_0000002517188794_table38301303189"></a>

| 产品                                        | 是否支持 |
| ------------------------------------------- | -------- |
| Ascend 950PR/Ascend 950DT                   | √        |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √        |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √        |




## 功能说明<a name="zh-cn_topic_0240188739_section15406195619561"></a>

对用户输入的PyTorch模型对象进行张量分解，得到分解后的模型对象和分解前后层的对应名称，并保存分解信息文件（可选）。

## 函数原型<a name="zh-cn_topic_0240188739_zh-cn_topic_0122830089_section428121323411"></a>

```python
model, changes = auto_decomposition(model, decompose_info_path=None)
```

## 参数说明<a name="zh-cn_topic_0240188739_zh-cn_topic_0122830089_section795991810344"></a>

<a name="zh-cn_topic_0240188739_zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0240188739_zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="14.45%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0240188739_zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0240188739_zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0240188739_zh-cn_topic_0122830089_p1438834363520"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="7.8100000000000005%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0240188739_p1769255516412"><a name="zh-cn_topic_0240188739_p1769255516412"></a><a name="zh-cn_topic_0240188739_p1769255516412"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="77.74%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0240188739_p15231205416325"><a name="zh-cn_topic_0240188739_p15231205416325"></a><a name="zh-cn_topic_0240188739_p15231205416325"></a>使用限制</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0240188739_zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="14.45%" headers="mcps1.1.4.1.1 "><p id="p163296363467"><a name="p163296363467"></a><a name="p163296363467"></a>model</p>
</td>
<td class="cellrowborder" valign="top" width="7.8100000000000005%" headers="mcps1.1.4.1.2 "><p id="p16326936194620"><a name="p16326936194620"></a><a name="p16326936194620"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="77.74%" headers="mcps1.1.4.1.3 "><p id="p3490152934016"><a name="p3490152934016"></a><a name="p3490152934016"></a>含义：待分解的含有预训练权重的PyTorch模型对象。在调用该接口时建议将模型放置于CPU而不是GPU上，以防分解时显存不足。</p>
<p id="zh-cn_topic_0240188739_p11225740182619"><a name="zh-cn_topic_0240188739_p11225740182619"></a><a name="zh-cn_topic_0240188739_p11225740182619"></a>数据类型：torch.nn.Module</p>
</td>
</tr>
<tr id="zh-cn_topic_0240188739_row21251438195"><td class="cellrowborder" valign="top" width="14.45%" headers="mcps1.1.4.1.1 "><p id="p679084764614"><a name="p679084764614"></a><a name="p679084764614"></a>decompose_info_path</p>
</td>
<td class="cellrowborder" valign="top" width="7.8100000000000005%" headers="mcps1.1.4.1.2 "><p id="p6690203419464"><a name="p6690203419464"></a><a name="p6690203419464"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="77.74%" headers="mcps1.1.4.1.3 "><p id="p11521032134013"><a name="p11521032134013"></a><a name="p11521032134013"></a>含义：分解信息文件保存路径。将以JSON格式存储，因此建议使用.json扩展名。为None时不保存分解信息文件（默认）。</p>
<p id="p16659115502"><a name="p16659115502"></a><a name="p16659115502"></a>数据类型：string</p>
<p id="p1920518526403"><a name="p1920518526403"></a><a name="p1920518526403"></a>默认值：None</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0240188739_zh-cn_topic_0122830089_section293415513458"></a>

-   张量分解后的模型对象，类型为torch.nn.Module。
-   张量分解前后的对应层名构成的字典，类型为dict，形如\{'conv1': \['conv1.0', 'conv1.1'\], 'conv2': \['conv2.0', 'conv2.1'\], ...\}。

## 约束说明<a name="zh-cn_topic_0240188739_section1443392021419"></a>

-   用户输入的模型需为torch.nn.Module类型的对象。
-   本接口函数仅对通过torch.nn.Conv2d\(\)构建的卷积进行分解。
-   用户调用本接口函数，接口函数对符合分解条件的卷积层进行自动分解，约束请参见[分解约束](../README.md)。

## 调用示例<a name="section179052217494"></a>

```python
from amct_pytorch.tensor_decompose import auto_decomposition
net = Net()                                                    # 构建用户模型对象
net.load_state_dict(torch.load("src_path/weights.pth"))        # 加载模型权重
net, changes = auto_decomposition(                             # 执行张量分解
    model=net,
    decompose_info_path="decomposed_path/decompose_info.json"
)
```

> [!NOTE]说明 
>
>1.  当涉及模型训练时，本接口的调用需在将模型参数传递给优化器之前；如使用了torch.nn.parallel.DistributedDataParallel \(DDP\)，则本接口的调用也需在将模型传递给DDP之前。
>2.  本接口将原地修改传入的模型对象，即分解后会改变用户传入的模型对象本身（例外：传入的模型是一个torch.nn.Conv2d对象，该情况下本接口不会对其进行修改，如发生分解，则返回的模型是新构建的torch.nn.Module对象）。

