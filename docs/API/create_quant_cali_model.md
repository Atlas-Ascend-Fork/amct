# create\_quant\_cali\_model<a name="ZH-CN_TOPIC_0000002548668597"></a>

## 产品支持情况<a name="section1610172518544"></a>

<a name="zh-cn_topic_0000002517188700_table38301303189"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002517188700_row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000002517188700_p1883113061818"><a name="zh-cn_topic_0000002517188700_p1883113061818"></a><a name="zh-cn_topic_0000002517188700_p1883113061818"></a><span id="zh-cn_topic_0000002517188700_ph20833205312295"><a name="zh-cn_topic_0000002517188700_ph20833205312295"></a><a name="zh-cn_topic_0000002517188700_ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000002517188700_p783113012187"><a name="zh-cn_topic_0000002517188700_p783113012187"></a><a name="zh-cn_topic_0000002517188700_p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002517188700_row574891710101"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002517188700_p177491117171015"><a name="zh-cn_topic_0000002517188700_p177491117171015"></a><a name="zh-cn_topic_0000002517188700_p177491117171015"></a><span id="zh-cn_topic_0000002517188700_ph2272194216543"><a name="zh-cn_topic_0000002517188700_ph2272194216543"></a><a name="zh-cn_topic_0000002517188700_ph2272194216543"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002517188700_p14226338117"><a name="zh-cn_topic_0000002517188700_p14226338117"></a><a name="zh-cn_topic_0000002517188700_p14226338117"></a>√</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002517188700_row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002517188700_p48327011813"><a name="zh-cn_topic_0000002517188700_p48327011813"></a><a name="zh-cn_topic_0000002517188700_p48327011813"></a><span id="zh-cn_topic_0000002517188700_ph583230201815"><a name="zh-cn_topic_0000002517188700_ph583230201815"></a><a name="zh-cn_topic_0000002517188700_ph583230201815"></a><term id="zh-cn_topic_0000002517188700_zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000002517188700_zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000002517188700_zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000002517188700_zh-cn_topic_0000001312391781_term131434243115"><a name="zh-cn_topic_0000002517188700_zh-cn_topic_0000001312391781_term131434243115"></a><a name="zh-cn_topic_0000002517188700_zh-cn_topic_0000001312391781_term131434243115"></a>Atlas A3 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002517188700_p108715341013"><a name="zh-cn_topic_0000002517188700_p108715341013"></a><a name="zh-cn_topic_0000002517188700_p108715341013"></a>√</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002517188700_row173226882415"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002517188700_p14832120181815"><a name="zh-cn_topic_0000002517188700_p14832120181815"></a><a name="zh-cn_topic_0000002517188700_p14832120181815"></a><span id="zh-cn_topic_0000002517188700_ph1483216010188"><a name="zh-cn_topic_0000002517188700_ph1483216010188"></a><a name="zh-cn_topic_0000002517188700_ph1483216010188"></a><term id="zh-cn_topic_0000002517188700_zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000002517188700_zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000002517188700_zh-cn_topic_0000001312391781_term11962195213215"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000002517188700_zh-cn_topic_0000001312391781_term184716139811"><a name="zh-cn_topic_0000002517188700_zh-cn_topic_0000001312391781_term184716139811"></a><a name="zh-cn_topic_0000002517188700_zh-cn_topic_0000001312391781_term184716139811"></a>Atlas A2 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002517188700_p19948143911820"><a name="zh-cn_topic_0000002517188700_p19948143911820"></a><a name="zh-cn_topic_0000002517188700_p19948143911820"></a>√</p>
</td>
</tr>
</tbody>
</table>

**注：标记“x”的产品，调用接口不会报错，但是获取不到性能收益。**

## 功能说明<a name="zh-cn_topic_0240187365_section15406195619561"></a>

KV-cache量化接口，根据模型和量化详细配置，对用户模型进行改图，将待量化Linear算子替换为输出后进行IFMR/HFMG量化的量化算子，后续用户拿到模型后进行在线校准，校准后生成量化因子保存在record\_file中。

## 函数原型<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_section428121323411"></a>

```python
calibration_model = create_quant_cali_model(config_file, record_file, model)
```

## 参数说明<a name="section73811524135618"></a>

<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="9.09%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="7.35%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0240187365_p1769255516412"><a name="zh-cn_topic_0240187365_p1769255516412"></a><a name="zh-cn_topic_0240187365_p1769255516412"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="83.56%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0240187365_p15231205416325"><a name="zh-cn_topic_0240187365_p15231205416325"></a><a name="zh-cn_topic_0240187365_p15231205416325"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="9.09%" headers="mcps1.1.4.1.1 "><p id="p31941230154616"><a name="p31941230154616"></a><a name="p31941230154616"></a>config_file</p>
</td>
<td class="cellrowborder" valign="top" width="7.35%" headers="mcps1.1.4.1.2 "><p id="p1219493024610"><a name="p1219493024610"></a><a name="p1219493024610"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.1.4.1.3 "><p id="p859151319514"><a name="p859151319514"></a><a name="p859151319514"></a>含义：生成的量化配置文件路径，配置文件为JSON格式。</p>
<p id="p7194123094612"><a name="p7194123094612"></a><a name="p7194123094612"></a>数据类型：string</p>
<p id="p9725102314288"><a name="p9725102314288"></a><a name="p9725102314288"></a>使用约束：该接口输入的config.json必须和<a href="create_quant_cali_config.md">create_quant_cali_config</a>接口输入的config.json一致</p>
</td>
</tr>
<tr id="row36995816444"><td class="cellrowborder" valign="top" width="9.09%" headers="mcps1.1.4.1.1 "><p id="p269918812448"><a name="p269918812448"></a><a name="p269918812448"></a>record_file</p>
</td>
<td class="cellrowborder" valign="top" width="7.35%" headers="mcps1.1.4.1.2 "><p id="p14699985449"><a name="p14699985449"></a><a name="p14699985449"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.1.4.1.3 "><p id="p179501119125112"><a name="p179501119125112"></a><a name="p179501119125112"></a>含义：在线校准量化因子保存的路径及文件名称。</p>
<p id="p66998816443"><a name="p66998816443"></a><a name="p66998816443"></a>数据类型：string</p>
</td>
</tr>
<tr id="row4280131024613"><td class="cellrowborder" valign="top" width="9.09%" headers="mcps1.1.4.1.1 "><p id="p4194143084610"><a name="p4194143084610"></a><a name="p4194143084610"></a>model</p>
</td>
<td class="cellrowborder" valign="top" width="7.35%" headers="mcps1.1.4.1.2 "><p id="p14194430114612"><a name="p14194430114612"></a><a name="p14194430114612"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.1.4.1.3 "><p id="p03091421185117"><a name="p03091421185117"></a><a name="p03091421185117"></a>含义：用户提供的待量化模型。</p>
<p id="p19194123013463"><a name="p19194123013463"></a><a name="p19194123013463"></a>数据类型：torch.nn.Module</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0240188739_zh-cn_topic_0122830089_section293415513458"></a>

替换为校准算子的量化校准模型。

## 调用示例<a name="zh-cn_topic_0240188739_section64231658994"></a>

```python
import amct_pytorch as amct
# 建立待进行量化的网络图结构
model = build_model()
model.load_state_dict(torch.load(state_dict_path))

record_file = os.path.join(TMP, 'kv_cache.txt')
# 插入量化API，生成量化校准模型
calibration_model = amct.create_quant_cali_model(
                    config_file="./configs/config.json",  # 生成的量化因子记录文件
                    record_file,                   
                    model)
```

