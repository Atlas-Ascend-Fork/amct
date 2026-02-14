# auto\_channel\_prune\_search<a name="ZH-CN_TOPIC_0000002517188676"></a>

## 产品支持情况<a name="section185612964420"></a>

<a name="zh-cn_topic_0000002517188794_table38301303189"></a>

| 产品                                        | 是否支持 |
| ------------------------------------------- | -------- |
| Ascend 950PR/Ascend 950DT                   | √        |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √        |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √        |




## 功能说明<a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_section15406195619561"></a>

自动通道稀疏接口，根据用户模型来计算各通道的稀疏敏感度（影响精度）以及稀疏收益（影响性能），然后搜索策略依据该输入来搜索最优的逐层通道稀疏率，以平衡精度和性能。最终输出一个配置文件。

## 函数原型<a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_section428121323411"></a>

```python
auto_channel_prune_search(model, config, input_data, output_cfg, sensitivity, search_alg)
```

## 参数说明<a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_section795991810344"></a>

<a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="12.61%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_p1438834363520"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="8.35%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p1769255516412"><a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p1769255516412"></a><a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p1769255516412"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="79.03999999999999%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p15231205416325"><a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p15231205416325"></a><a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p15231205416325"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="12.61%" headers="mcps1.1.4.1.1 "><p id="p1586914264474"><a name="p1586914264474"></a><a name="p1586914264474"></a>model</p>
</td>
<td class="cellrowborder" valign="top" width="8.35%" headers="mcps1.1.4.1.2 "><p id="p168681026184710"><a name="p168681026184710"></a><a name="p168681026184710"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="79.03999999999999%" headers="mcps1.1.4.1.3 "><p id="p034616965"><a name="p034616965"></a><a name="p034616965"></a>含义：待稀疏的PyTorch模型。</p>
<p id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p11225740182619"><a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p11225740182619"></a><a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p11225740182619"></a>数据类型：torch.nn.Module</p>
</td>
</tr>
<tr id="row13348191611917"><td class="cellrowborder" valign="top" width="12.61%" headers="mcps1.1.4.1.1 "><p id="p1785212266471"><a name="p1785212266471"></a><a name="p1785212266471"></a>config</p>
</td>
<td class="cellrowborder" valign="top" width="8.35%" headers="mcps1.1.4.1.2 "><p id="p175011134411"><a name="p175011134411"></a><a name="p175011134411"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="79.03999999999999%" headers="mcps1.1.4.1.3 "><p id="p7677132119613"><a name="p7677132119613"></a><a name="p7677132119613"></a>含义：自动通道稀疏配置文件路径。</p>
<p id="p4435183911710"><a name="p4435183911710"></a><a name="p4435183911710"></a>基于basic_info.proto文件中的AutoChannelPruneConfig生成的简易配置文件，*.proto文件所在路径为：<em id="i54361539151713"><a name="i54361539151713"></a><a name="i54361539151713"></a><span id="ph2538725141110"><a name="ph2538725141110"></a><a name="ph2538725141110"></a>AMCT</span>安装目录</em>/amct_pytorch/proto/。</p>
<p id="p4232851666"><a name="p4232851666"></a><a name="p4232851666"></a>*.proto文件参数解释以及生成的自动通道稀疏搜索配置文件样例请参见<a href="../context/自动通道稀疏搜索简易配置文件.md">自动通道稀疏搜索简易配置文件</a>。</p>
<p id="p1543455424113"><a name="p1543455424113"></a><a name="p1543455424113"></a>数据类型：string</p>
</td>
</tr>
<tr id="row56401431115"><td class="cellrowborder" valign="top" width="12.61%" headers="mcps1.1.4.1.1 "><p id="p17641123201113"><a name="p17641123201113"></a><a name="p17641123201113"></a>input_data</p>
</td>
<td class="cellrowborder" valign="top" width="8.35%" headers="mcps1.1.4.1.2 "><p id="p76411936117"><a name="p76411936117"></a><a name="p76411936117"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="79.03999999999999%" headers="mcps1.1.4.1.3 "><p id="p5913202219613"><a name="p5913202219613"></a><a name="p5913202219613"></a>含义：用户提供获取输入数据（含label）。</p>
<p id="p08321173268"><a name="p08321173268"></a><a name="p08321173268"></a>数据类型：list[data,label]，列表元素数据类型为torch.tensor。</p>
</td>
</tr>
<tr id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_row21251438195"><td class="cellrowborder" valign="top" width="12.61%" headers="mcps1.1.4.1.1 "><p id="p487821312913"><a name="p487821312913"></a><a name="p487821312913"></a>output_cfg</p>
</td>
<td class="cellrowborder" valign="top" width="8.35%" headers="mcps1.1.4.1.2 "><p id="p810515371917"><a name="p810515371917"></a><a name="p810515371917"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="79.03999999999999%" headers="mcps1.1.4.1.3 "><p id="p128965295620"><a name="p128965295620"></a><a name="p128965295620"></a>含义：输出的最终的通道稀疏配置文件路径。</p>
<p id="p89686200106"><a name="p89686200106"></a><a name="p89686200106"></a>数据类型：string</p>
</td>
</tr>
<tr id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_row16254133823918"><td class="cellrowborder" valign="top" width="12.61%" headers="mcps1.1.4.1.1 "><p id="p285812654718"><a name="p285812654718"></a><a name="p285812654718"></a>sensitivity</p>
</td>
<td class="cellrowborder" valign="top" width="8.35%" headers="mcps1.1.4.1.2 "><p id="p18491181320411"><a name="p18491181320411"></a><a name="p18491181320411"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="79.03999999999999%" headers="mcps1.1.4.1.3 "><p id="p12644431963"><a name="p12644431963"></a><a name="p12644431963"></a>含义：敏感度计算方法。</p>
<p id="p64303542418"><a name="p64303542418"></a><a name="p64303542418"></a>数据类型：string或SensitivityBase的子类，string为<span id="ph6261819124911"><a name="ph6261819124911"></a><a name="ph6261819124911"></a>AMCT</span>已有的方法，目前可选为'TaylorLossSensitivity'；SensitivityBase的子类实例化，可由用户来继承定义。</p>
</td>
</tr>
<tr id="row22610479916"><td class="cellrowborder" valign="top" width="12.61%" headers="mcps1.1.4.1.1 "><p id="p526217471690"><a name="p526217471690"></a><a name="p526217471690"></a>search_alg</p>
</td>
<td class="cellrowborder" valign="top" width="8.35%" headers="mcps1.1.4.1.2 "><p id="p149701928101817"><a name="p149701928101817"></a><a name="p149701928101817"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="79.03999999999999%" headers="mcps1.1.4.1.3 "><p id="p151474331619"><a name="p151474331619"></a><a name="p151474331619"></a>含义：待稀疏的通道搜索方法。</p>
<p id="p613275510219"><a name="p613275510219"></a><a name="p613275510219"></a>数据类型：string或SearchChannelBase的子类，string为<span id="ph3713131922414"><a name="ph3713131922414"></a><a name="ph3713131922414"></a>AMCT</span>已有的方法，目前可选为'GreedySearch'；SearchChannelBase的子类实例化，可由用户来继承定义。</p>
</td>
</tr>
</tbody>
</table>


## 返回值说明<a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_section293415513458"></a>

无

## 调用示例<a name="zh-cn_topic_0254208276_section213682915617"></a>

```python
import amct_pytorch as amct
#构造输入数据input_data
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

落盘文件说明：

保存的自动通道稀疏配置文件，需要传给通道稀疏接口完成后续的业务。

