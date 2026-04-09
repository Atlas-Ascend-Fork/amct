# create\_compressed\_retrain\_model<a name="ZH-CN_TOPIC_0000002517188752"></a>

## 产品支持情况<a name="section197451857688"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="47.92%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="left" valign="top" width="52.080000000000005%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row574891710101"><td class="cellrowborder" valign="top" width="47.92%" headers="mcps1.1.3.1.1 "><p id="p177491117171015"><a name="p177491117171015"></a><a name="p177491117171015"></a><span id="ph2272194216543"><a name="ph2272194216543"></a><a name="ph2272194216543"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="left" valign="top" width="52.080000000000005%" headers="mcps1.1.3.1.2 "><a name="ul1367712433612"></a><a name="ul1367712433612"></a><ul id="ul1367712433612"><li>量化感知训练：<a name="ul5366163611332"></a><a name="ul5366163611332"></a><ul id="ul5366163611332"><li>INT8量化：√</li><li>INT4量化：x</li></ul>
</li><li>通道稀疏：√</li><li>4选2结构化稀疏：x</li></ul>
</td>
</tr>
<tr id="row220181016240"><td class="cellrowborder" valign="top" width="47.92%" headers="mcps1.1.3.1.1 "><p id="p48327011813"><a name="p48327011813"></a><a name="p48327011813"></a><span id="ph583230201815"><a name="ph583230201815"></a><a name="ph583230201815"></a><term id="zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term131434243115"><a name="zh-cn_topic_0000001312391781_term131434243115"></a><a name="zh-cn_topic_0000001312391781_term131434243115"></a>Atlas A3 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="left" valign="top" width="52.080000000000005%" headers="mcps1.1.3.1.2 "><a name="ul66191179379"></a><a name="ul66191179379"></a><ul id="ul66191179379"><li>量化感知训练：<a name="ul7274174217362"></a><a name="ul7274174217362"></a><ul id="ul7274174217362"><li>INT8量化：√</li><li>INT4量化：x</li></ul>
</li><li>通道稀疏：√</li><li>4选2结构化稀疏：√</li></ul>
</td>
</tr>
<tr id="row173226882415"><td class="cellrowborder" valign="top" width="47.92%" headers="mcps1.1.3.1.1 "><p id="p14832120181815"><a name="p14832120181815"></a><a name="p14832120181815"></a><span id="ph1483216010188"><a name="ph1483216010188"></a><a name="ph1483216010188"></a><term id="zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000001312391781_term11962195213215"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term184716139811"><a name="zh-cn_topic_0000001312391781_term184716139811"></a><a name="zh-cn_topic_0000001312391781_term184716139811"></a>Atlas A2 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="left" valign="top" width="52.080000000000005%" headers="mcps1.1.3.1.2 "><a name="ul19623147153713"></a><a name="ul19623147153713"></a><ul id="ul19623147153713"><li>量化感知训练：<a name="ul0992647173610"></a><a name="ul0992647173610"></a><ul id="ul0992647173610"><li>INT8量化：√</li><li>INT4量化：x</li></ul>
</li><li>通道稀疏：√</li><li>4选2结构化稀疏：√</li></ul>
</td>
</tr>
</tbody>
</table>

**注：特性中标记“x”的产品，调用接口不会报错，但是获取不到性能收益。**

## 功能说明<a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_section15406195619561"></a>

静态组合压缩接口，将输入的待静态组合压缩的模型按照给定的组合压缩配置文件进行压缩处理，即将传入的模型先进行稀疏（通道稀疏或者4选2结构化稀疏，二选一），后对模型插入量化相关的算子（数据和权重的量化感知训练层以及searchN的层），生成稀疏和量化因子记录文件record\_file（如果配置存在），返回修改后的torch.nn.Module模型。

## 函数原型<a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_section428121323411"></a>

```python
compressed_retrain_model = create_compressed_retrain_model(model, input_data, config_defination, record_file)
```

## 参数说明<a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_section795991810344"></a>

<a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="12.411241124112411%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_p1438834363520"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="7.490749074907491%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p1769255516412"><a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p1769255516412"></a><a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p1769255516412"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="80.0980098009801%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p15231205416325"><a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p15231205416325"></a><a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p15231205416325"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="12.411241124112411%" headers="mcps1.1.4.1.1 "><p id="p166458380213"><a name="p166458380213"></a><a name="p166458380213"></a>model</p>
</td>
<td class="cellrowborder" valign="top" width="7.490749074907491%" headers="mcps1.1.4.1.2 "><p id="p168681026184710"><a name="p168681026184710"></a><a name="p168681026184710"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="80.0980098009801%" headers="mcps1.1.4.1.3 "><p id="p15959183616327"><a name="p15959183616327"></a><a name="p15959183616327"></a>含义：PyTorch的model。</p>
<p id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p11225740182619"><a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p11225740182619"></a><a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p11225740182619"></a>数据类型：torch.nn.Module</p>
</td>
</tr>
<tr id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_row21251438195"><td class="cellrowborder" valign="top" width="12.411241124112411%" headers="mcps1.1.4.1.1 "><p id="p17645193813215"><a name="p17645193813215"></a><a name="p17645193813215"></a>input_data</p>
</td>
<td class="cellrowborder" valign="top" width="7.490749074907491%" headers="mcps1.1.4.1.2 "><p id="p13482131324113"><a name="p13482131324113"></a><a name="p13482131324113"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="80.0980098009801%" headers="mcps1.1.4.1.3 "><p id="p14176174617320"><a name="p14176174617320"></a><a name="p14176174617320"></a>含义：模型的输入数据。</p>
<p id="p12421175414419"><a name="p12421175414419"></a><a name="p12421175414419"></a>数据类型：tuple</p>
</td>
</tr>
<tr id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_row16254133823918"><td class="cellrowborder" valign="top" width="12.411241124112411%" headers="mcps1.1.4.1.1 "><p id="p164543814218"><a name="p164543814218"></a><a name="p164543814218"></a>config_defination</p>
</td>
<td class="cellrowborder" valign="top" width="7.490749074907491%" headers="mcps1.1.4.1.2 "><p id="p18491181320411"><a name="p18491181320411"></a><a name="p18491181320411"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="80.0980098009801%" headers="mcps1.1.4.1.3 "><p id="p17101548133217"><a name="p17101548133217"></a><a name="p17101548133217"></a>含义：静态组合压缩简易配置文件。</p>
<p id="p827435611439"><a name="p827435611439"></a><a name="p827435611439"></a>基于retrain_config_pytorch.proto文件生成的简易配置文件<em id="i497145210579"><a name="i497145210579"></a><a name="i497145210579"></a>compressed</em>.cfg，*.proto文件所在路径为：<em id="i12274856104314"><a name="i12274856104314"></a><a name="i12274856104314"></a><span id="ph244511481409"><a name="ph244511481409"></a><a name="ph244511481409"></a>AMCT</span>安装目录</em>/amct_pytorch/proto/。*.proto文件参数解释以及生成的<em id="i2675712185810"><a name="i2675712185810"></a><a name="i2675712185810"></a>compressed</em>.cfg简易配置文件样例请参见<a href="../context/量化感知训练简易配置文件.md">量化感知训练简易配置文件</a>。</p>
<p id="p64303542418"><a name="p64303542418"></a><a name="p64303542418"></a>数据类型：string</p>
</td>
</tr>
<tr id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_row2997141123911"><td class="cellrowborder" valign="top" width="12.411241124112411%" headers="mcps1.1.4.1.1 "><p id="p264514389215"><a name="p264514389215"></a><a name="p264514389215"></a>record_file</p>
</td>
<td class="cellrowborder" valign="top" width="7.490749074907491%" headers="mcps1.1.4.1.2 "><p id="p175011134411"><a name="p175011134411"></a><a name="p175011134411"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="80.0980098009801%" headers="mcps1.1.4.1.3 "><p id="p1872594993211"><a name="p1872594993211"></a><a name="p1872594993211"></a>含义：待记录稀疏和量化因子文件路径及名称。</p>
<p id="p1543455424113"><a name="p1543455424113"></a><a name="p1543455424113"></a>数据类型：string</p>
</td>
</tr>
</tbody>
</table>


## 返回值说明<a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_section293415513458"></a>

根据配置文件进行稀疏后（如果配置稀疏），且插入量化相关层（如果配置量化）的torch.nn.Module静态组合压缩模型。

## 约束说明<a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_section1443392021419"></a>

组合压缩配置文件至少存在一个配置：稀疏配置或者量化配置。

## 调用示例<a name="zh-cn_topic_0254208276_section213682915617"></a>

```python
import amct_pytorch as amct
# 建立待进行静态组合压缩的网络
model = build_model()
input_data = tuple([torch.randn(input_shape)])

# 调用静态组合压缩API
record_file = os.path.join(TMP, 'compressed_record.txt')
config_defination = './compressed_cfg.cfg'

compressed_retrain_model = amct.create_compressed_retrain_model(
                                model,
                                input_data,
                                config_defination,
                                record_file)
```

落盘文件说明：

保存的静态组合压缩记录文件record\_file，如果简易配置文件中含有稀疏配置，则在该函数完成后，record\_file中含有稀疏记录信息。

