# create\_prune\_retrain\_model<a name="ZH-CN_TOPIC_0000002548668559"></a>

## 产品支持情况<a name="section197451857688"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="left" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row574891710101"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p177491117171015"><a name="p177491117171015"></a><a name="p177491117171015"></a><span id="ph2272194216543"><a name="ph2272194216543"></a><a name="ph2272194216543"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="left" valign="top" width="42%" headers="mcps1.1.3.1.2 "><a name="ul1367712433612"></a><a name="ul1367712433612"></a><ul id="ul1367712433612"><li>通道稀疏：√</li><li>4选2结构化稀疏接口：x</li></ul>
</td>
</tr>
<tr id="row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p48327011813"><a name="p48327011813"></a><a name="p48327011813"></a><span id="ph583230201815"><a name="ph583230201815"></a><a name="ph583230201815"></a><term id="zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term131434243115"><a name="zh-cn_topic_0000001312391781_term131434243115"></a><a name="zh-cn_topic_0000001312391781_term131434243115"></a>Atlas A3 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="left" valign="top" width="42%" headers="mcps1.1.3.1.2 "><a name="ul66191179379"></a><a name="ul66191179379"></a><ul id="ul66191179379"><li>通道稀疏：√</li><li>4选2结构化稀疏接口：√</li></ul>
</td>
</tr>
<tr id="row173226882415"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p14832120181815"><a name="p14832120181815"></a><a name="p14832120181815"></a><span id="ph1483216010188"><a name="ph1483216010188"></a><a name="ph1483216010188"></a><term id="zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000001312391781_term11962195213215"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term184716139811"><a name="zh-cn_topic_0000001312391781_term184716139811"></a><a name="zh-cn_topic_0000001312391781_term184716139811"></a>Atlas A2 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="left" valign="top" width="42%" headers="mcps1.1.3.1.2 "><a name="ul19623147153713"></a><a name="ul19623147153713"></a><ul id="ul19623147153713"><li>通道稀疏：√</li><li>4选2结构化稀疏接口：√</li></ul>
</td>
</tr>
</tbody>
</table>

**注：上述4选2结构化稀疏特性，标记“x”的产品，调用接口不会报错，但是获取不到性能收益。**

## 功能说明<a name="zh-cn_topic_0240187365_section15406195619561"></a>

通道稀疏或4选2结构化稀疏接口，两种稀疏特性每次只能使能一个：将输入的待稀疏的图结构按照给定的稀疏配置文件进行稀疏处理，在传入的图结构中插入或者替换相关的算子，生成记录稀疏信息的record\_file，返回修改后可用于稀疏后训练的torch.nn.Module模型。

## 函数原型<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_section428121323411"></a>

```python
prune_retrain_model = create_prune_retrain_model (model, input_data, config_defination, record_file)
```

## 参数说明<a name="section73811524135618"></a>

<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="13.780000000000001%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="8.01%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0240187365_p1769255516412"><a name="zh-cn_topic_0240187365_p1769255516412"></a><a name="zh-cn_topic_0240187365_p1769255516412"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="78.21000000000001%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0240187365_p15231205416325"><a name="zh-cn_topic_0240187365_p15231205416325"></a><a name="zh-cn_topic_0240187365_p15231205416325"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row4280131024613"><td class="cellrowborder" valign="top" width="13.780000000000001%" headers="mcps1.1.4.1.1 "><p id="p4194143084610"><a name="p4194143084610"></a><a name="p4194143084610"></a>model</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="p14194430114612"><a name="p14194430114612"></a><a name="p14194430114612"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21000000000001%" headers="mcps1.1.4.1.3 "><p id="p661419384017"><a name="p661419384017"></a><a name="p661419384017"></a>含义：待进行稀疏的模型，已加载权重。</p>
<p id="p19194123013463"><a name="p19194123013463"></a><a name="p19194123013463"></a>数据类型：torch.nn.Module</p>
</td>
</tr>
<tr id="row156638166467"><td class="cellrowborder" valign="top" width="13.780000000000001%" headers="mcps1.1.4.1.1 "><p id="p1619443010466"><a name="p1619443010466"></a><a name="p1619443010466"></a>input_data</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="p919503013466"><a name="p919503013466"></a><a name="p919503013466"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21000000000001%" headers="mcps1.1.4.1.3 "><p id="p1750084516018"><a name="p1750084516018"></a><a name="p1750084516018"></a>含义：模型的输入数据。一个<span>torch.tensor</span><span>会被等价为tuple(torch.tensor)</span>。</p>
<p id="p1195430114611"><a name="p1195430114611"></a><a name="p1195430114611"></a>数据类型：tuple</p>
</td>
</tr>
<tr id="row135351841171711"><td class="cellrowborder" valign="top" width="13.780000000000001%" headers="mcps1.1.4.1.1 "><p id="p158301146144319"><a name="p158301146144319"></a><a name="p158301146144319"></a>config_defination</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="p1027410568437"><a name="p1027410568437"></a><a name="p1027410568437"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21000000000001%" headers="mcps1.1.4.1.3 "><p id="p8827154613010"><a name="p8827154613010"></a><a name="p8827154613010"></a>含义：简易配置文件。</p>
<p id="p827435611439"><a name="p827435611439"></a><a name="p827435611439"></a>基于retrain_config_pytorch.proto文件生成的简易配置文件<em id="i37264185718"><a name="i37264185718"></a><a name="i37264185718"></a>prune</em>.cfg，*.proto文件所在路径为：<em id="i12274856104314"><a name="i12274856104314"></a><a name="i12274856104314"></a><span id="ph244511481409"><a name="ph244511481409"></a><a name="ph244511481409"></a>AMCT</span>安装目录</em>/amct_pytorch/proto/。*.proto文件参数解释以及生成的<em id="i12381525154418"><a name="i12381525154418"></a><a name="i12381525154418"></a>prune</em>.cfg简易量化配置文件样例请参见<a href="../context/量化感知训练简易配置文件.md">量化感知训练简易配置文件</a>。</p>
<p id="p3275175644316"><a name="p3275175644316"></a><a name="p3275175644316"></a>数据类型：string</p>
</td>
</tr>
<tr id="row10181104881712"><td class="cellrowborder" valign="top" width="13.780000000000001%" headers="mcps1.1.4.1.1 "><p id="p19194113044616"><a name="p19194113044616"></a><a name="p19194113044616"></a>record_file</p>
</td>
<td class="cellrowborder" valign="top" width="8.01%" headers="mcps1.1.4.1.2 "><p id="p1519433016468"><a name="p1519433016468"></a><a name="p1519433016468"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.21000000000001%" headers="mcps1.1.4.1.3 "><p id="p1335544810020"><a name="p1335544810020"></a><a name="p1335544810020"></a>含义：记录稀疏信息的文件路径及名称，记录通道稀疏结点间的级联关系或记录4选2稀疏的节点。</p>
<p id="p31945306466"><a name="p31945306466"></a><a name="p31945306466"></a>数据类型：string</p>
</td>
</tr>
</tbody>
</table>



## 通道稀疏支持的层及约束

**表 1**  通道稀疏支持的层以及约束

<a name="table151175131181"></a>

<table><thead align="left"><tr id="row61178136184"><th class="cellrowborder" valign="top" width="7.5200000000000005%" id="mcps1.2.4.1.1"><p id="p185862080479"><a name="p185862080479"></a><a name="p185862080479"></a>优化方式</p>
</th>
<th class="cellrowborder" valign="top" width="16.42%" id="mcps1.2.4.1.2"><p id="p1921472241319"><a name="p1921472241319"></a><a name="p1921472241319"></a>支持的层类型</p>
</th>
<th class="cellrowborder" valign="top" width="76.06%" id="mcps1.2.4.1.3"><p id="p9117111341815"><a name="p9117111341815"></a><a name="p9117111341815"></a>约束</p>
</th>
</tr>
</thead>
<tbody><tr id="row4117613121819"><td class="cellrowborder" rowspan="2" valign="top" width="7.5200000000000005%" headers="mcps1.2.4.1.1 "><p id="p14586087476"><a name="p14586087476"></a><a name="p14586087476"></a>通道稀疏</p>
</td>
<td class="cellrowborder" valign="top" width="16.42%" headers="mcps1.2.4.1.2 "><p id="p46801332355"><a name="p46801332355"></a><a name="p46801332355"></a>torch.nn.Linear：全连接层</p>
</td>
<td class="cellrowborder" valign="top" width="76.06%" headers="mcps1.2.4.1.3 "><p id="p257843204111"><a name="p257843204111"></a><a name="p257843204111"></a>复用层（共用weight和bias参数）不支持稀疏。</p>
</td>
</tr>
<tr id="row13118181318183"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1868011363517"><a name="p1868011363517"></a><a name="p1868011363517"></a>torch.nn.Conv2d：卷积层</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><a name="ul4262123921520"></a><a name="ul4262123921520"></a><ul id="ul4262123921520"><li>复用层（共用weight和bias参数）不支持稀疏。</li><li>depthwise只能被动稀疏(groups=in_channels)，不能主动稀疏。</li><li>只支持input data的shape为(N, Cin, Hin, Win)。</li></ul>
</td>
</tr>
</tbody>
</table>


## 4选2结构化稀疏支持的层及约束

**表 1**  支持的层以及约束

<a name="table151175131181"></a>

<table><thead align="left"><tr id="row61178136184"><th class="cellrowborder" valign="top" width="7.5200000000000005%" id="mcps1.2.4.1.1"><p id="p185862080479"><a name="p185862080479"></a><a name="p185862080479"></a>优化方式</p>
</th>
<th class="cellrowborder" valign="top" width="27.61%" id="mcps1.2.4.1.2"><p id="p1921472241319"><a name="p1921472241319"></a><a name="p1921472241319"></a>支持的层类型</p>
</th>
<th class="cellrowborder" valign="top" width="64.87%" id="mcps1.2.4.1.3"><p id="p9117111341815"><a name="p9117111341815"></a><a name="p9117111341815"></a>约束</p>
</th>
</tr>
</thead>
<tbody><tr id="row4117613121819"><td class="cellrowborder" rowspan="3" valign="top" width="7.5200000000000005%" headers="mcps1.2.4.1.1 "><p id="p14586087476"><a name="p14586087476"></a><a name="p14586087476"></a>4选2结构化稀疏</p>
<p id="p112635116333"><a name="p112635116333"></a><a name="p112635116333"></a></p>
</td>
<td class="cellrowborder" valign="top" width="27.61%" headers="mcps1.2.4.1.2 "><p id="p46801332355"><a name="p46801332355"></a><a name="p46801332355"></a>torch.nn.Linear：全连接层</p>
</td>
<td class="cellrowborder" valign="top" width="64.87%" headers="mcps1.2.4.1.3 "><p id="p1670925185513"><a name="p1670925185513"></a><a name="p1670925185513"></a>复用层（共用weight）不支持稀疏。</p>
</td>
</tr>
<tr id="row13118181318183"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1868011363517"><a name="p1868011363517"></a><a name="p1868011363517"></a>torch.nn.Conv2d：卷积层</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><a name="ul19441153635116"></a><a name="ul19441153635116"></a><ul id="ul19441153635116"><li>复用层（共用weight）不支持稀疏。</li><li>只支持input data的shape为(N, Cin, Hin, Win)。</li></ul>
</td>
</tr>
<tr id="row1312619511332"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p91261851123316"><a name="p91261851123316"></a><a name="p91261851123316"></a>torch.nn.ConvTranspose2d：反卷积层</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><a name="ul1382854365314"></a><a name="ul1382854365314"></a><ul id="ul1382854365314"><li>复用层（共用weight）不支持稀疏。</li><li>只支持input data的shape为(N, Cin, Hin, Win)。</li></ul>
</td>
</tr>
</tbody>
</table>



## 返回值说明<a name="zh-cn_topic_0240188739_zh-cn_topic_0122830089_section293415513458"></a>

返回修改后可用于稀疏后训练的torch.nn.Module模型。

## 调用示例<a name="zh-cn_topic_0240188739_section64231658994"></a>

```python
import amct_pytorch as amct
# 建立待进行稀疏的网络图结构
model = build_model()
model.load_state_dict(torch.load(state_dict_path))
input_data = tuple([torch.randn(input_shape)])
 
# 调用稀疏模型API
record_file = os.path.join(TMP, 'scale_offset_record.txt')
cfg_file = './prune_config.cfg'
prune_retrain_model = amct.create_prune_retrain_model(
               model,
               input_data,
               cfg_file,
               record_file)
```

