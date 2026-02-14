# QuantCalibrationOp<a name="ZH-CN_TOPIC_0000002548668565"></a>

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

KV Cache量化接口，用于用户构图，在前向传播时，根据用户的量化算法配置调用IFMR/HFMG量化算法对输出做校准，校准后，将量化因子依据对应格式输出到record\_file文件指定层名中。

在进行前向传播时，算子对原始输出会做透传，不修改activation输入信息：

-   若当前传入数据个数小于batch\_num时，使用IFMR/HFMG算子中的积攒数据方法，将数据集进行保存。
-   若当前传入数据个数等于batch\_num时，会调用IFMR/HFMG算法计算量化因子，根据quant\_method参数进行区分写出来的量化因子格式，按照格式写入对应record文件；quant\_method目前仅支持“kv\_cache\_quant”方式。

写入时，对record文件进行增量写入，如果进行了覆盖写入，则会提示哪个层哪些参数被覆盖。

## 函数原型<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_section428121323411"></a>

```python
QuantCalibrationOp (record_file,quant_algo_params, quant_method)
```

## 参数说明<a name="section73811524135618"></a>

<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="13.658634136586342%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="7.939206079392061%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0240187365_p1769255516412"><a name="zh-cn_topic_0240187365_p1769255516412"></a><a name="zh-cn_topic_0240187365_p1769255516412"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="78.40215978402159%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0240187365_p15231205416325"><a name="zh-cn_topic_0240187365_p15231205416325"></a><a name="zh-cn_topic_0240187365_p15231205416325"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="13.658634136586342%" headers="mcps1.1.4.1.1 "><p id="p816684420110"><a name="p816684420110"></a><a name="p816684420110"></a>record_file</p>
</td>
<td class="cellrowborder" valign="top" width="7.939206079392061%" headers="mcps1.1.4.1.2 "><p id="p1219493024610"><a name="p1219493024610"></a><a name="p1219493024610"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.40215978402159%" headers="mcps1.1.4.1.3 "><p id="p132525675114"><a name="p132525675114"></a><a name="p132525675114"></a>含义：保存量化因子的record文件路径。</p>
<p id="p7194123094612"><a name="p7194123094612"></a><a name="p7194123094612"></a>数据类型：string</p>
</td>
</tr>
<tr id="row36995816444"><td class="cellrowborder" valign="top" width="13.658634136586342%" headers="mcps1.1.4.1.1 "><p id="p269918812448"><a name="p269918812448"></a><a name="p269918812448"></a>quant_algo_params</p>
</td>
<td class="cellrowborder" valign="top" width="7.939206079392061%" headers="mcps1.1.4.1.2 "><p id="p14699985449"><a name="p14699985449"></a><a name="p14699985449"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.40215978402159%" headers="mcps1.1.4.1.3 "><p id="p113721429525"><a name="p113721429525"></a><a name="p113721429525"></a>含义：指定量化使用的量化算法，以及添加对应量化配置，格式如下：</p>
<pre class="screen" id="screen1285281775716"><a name="screen1285281775716"></a><a name="screen1285281775716"></a>{
  "act_algo": "hfmg",
  "num_bits": 8,
  "quant_granularity": "1",
  "with_offset": true,
  "batch_num": 1
}</pre>
<p id="p13330105451319"><a name="p13330105451319"></a><a name="p13330105451319"></a>act_algo配置的算法不同，配置的字段不同，算法支持字段和解释如<a href="#table1778775810151">表1</a>所示。</p>
<p id="p66998816443"><a name="p66998816443"></a><a name="p66998816443"></a>数据类型：dict</p>
</td>
</tr>
<tr id="row4280131024613"><td class="cellrowborder" valign="top" width="13.658634136586342%" headers="mcps1.1.4.1.1 "><p id="p4194143084610"><a name="p4194143084610"></a><a name="p4194143084610"></a>quant_method</p>
</td>
<td class="cellrowborder" valign="top" width="7.939206079392061%" headers="mcps1.1.4.1.2 "><p id="p14194430114612"><a name="p14194430114612"></a><a name="p14194430114612"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.40215978402159%" headers="mcps1.1.4.1.3 "><p id="p121312414525"><a name="p121312414525"></a><a name="p121312414525"></a>含义：量化方式。默认为'kv_cache_quant'，用于指定输出量化因子的格式。</p>
<p id="p19194123013463"><a name="p19194123013463"></a><a name="p19194123013463"></a>数据类型：string</p>
</td>
</tr>
</tbody>
</table>

**表 1**  quant\_algo\_params参数支持配置的字段

<a name="table1778775810151"></a>
<table><thead align="left"><tr id="row4855358111517"><th class="cellrowborder" valign="top" width="12.68%" id="mcps1.2.5.1.1"><p id="p1855105821518"><a name="p1855105821518"></a><a name="p1855105821518"></a>字段</p>
</th>
<th class="cellrowborder" valign="top" width="60.06%" id="mcps1.2.5.1.2"><p id="p18604114031612"><a name="p18604114031612"></a><a name="p18604114031612"></a>含义</p>
</th>
<th class="cellrowborder" valign="top" width="13.320000000000002%" id="mcps1.2.5.1.3"><p id="p94109382310"><a name="p94109382310"></a><a name="p94109382310"></a>IFMR算法支持配置的字段</p>
</th>
<th class="cellrowborder" valign="top" width="13.94%" id="mcps1.2.5.1.4"><p id="p19813143213316"><a name="p19813143213316"></a><a name="p19813143213316"></a>HFMG算法支持配置的字段</p>
</th>
</tr>
</thead>
<tbody><tr id="row6855358141517"><td class="cellrowborder" valign="top" width="12.68%" headers="mcps1.2.5.1.1 "><p id="p985595819155"><a name="p985595819155"></a><a name="p985595819155"></a>act_algo</p>
</td>
<td class="cellrowborder" valign="top" width="60.06%" headers="mcps1.2.5.1.2 "><p id="p196046400166"><a name="p196046400166"></a><a name="p196046400166"></a>数据量化算法，支持如下两种：</p>
<a name="ul121219224249"></a><a name="ul121219224249"></a><ul id="ul121219224249"><li>IFMR数据量化算法：ifmr，默认为ifmr。</li><li>HFMG数据量化算法：hfmg。</li></ul>
</td>
<td class="cellrowborder" valign="top" width="13.320000000000002%" headers="mcps1.2.5.1.3 "><p id="p1841013381932"><a name="p1841013381932"></a><a name="p1841013381932"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="13.94%" headers="mcps1.2.5.1.4 "><p id="p208131032735"><a name="p208131032735"></a><a name="p208131032735"></a>-</p>
</td>
</tr>
<tr id="row1223511354619"><td class="cellrowborder" valign="top" width="12.68%" headers="mcps1.2.5.1.1 "><p id="p14235191324612"><a name="p14235191324612"></a><a name="p14235191324612"></a>num_bits</p>
</td>
<td class="cellrowborder" valign="top" width="60.06%" headers="mcps1.2.5.1.2 "><p id="p32352139461"><a name="p32352139461"></a><a name="p32352139461"></a>量化位宽，当前仅支持配置为8，表示采用INT8量化位宽。</p>
<p id="p18486171216"><a name="p18486171216"></a><a name="p18486171216"></a>IFMR/HFMG两种算法都支持。</p>
</td>
<td class="cellrowborder" valign="top" width="13.320000000000002%" headers="mcps1.2.5.1.3 "><p id="p1062425516499"><a name="p1062425516499"></a><a name="p1062425516499"></a>支持</p>
</td>
<td class="cellrowborder" valign="top" width="13.94%" headers="mcps1.2.5.1.4 "><p id="p156244556497"><a name="p156244556497"></a><a name="p156244556497"></a>支持</p>
</td>
</tr>
<tr id="row18518612396"><td class="cellrowborder" valign="top" width="12.68%" headers="mcps1.2.5.1.1 "><p id="p2051810121995"><a name="p2051810121995"></a><a name="p2051810121995"></a>with_offset</p>
</td>
<td class="cellrowborder" valign="top" width="60.06%" headers="mcps1.2.5.1.2 "><p id="p551817121196"><a name="p551817121196"></a><a name="p551817121196"></a>控制数据量化是对称量化还是非对称量化，全局配置参数。</p>
<a name="ul11492132182611"></a><a name="ul11492132182611"></a><ul id="ul11492132182611"><li>true：数据量化时为非对称量化，默认为true.</li><li>false：数据量化时为对称量化。</li></ul>
<p id="p144123051913"><a name="p144123051913"></a><a name="p144123051913"></a>若配置文件中同时存在with_offset和asymmetric参数，asymmetric参数优先级&gt;with_offset参数。</p>
<p id="p11149132022117"><a name="p11149132022117"></a><a name="p11149132022117"></a>IFMR/HFMG两种算法都支持。</p>
</td>
<td class="cellrowborder" valign="top" width="13.320000000000002%" headers="mcps1.2.5.1.3 "><p id="p16493233599"><a name="p16493233599"></a><a name="p16493233599"></a>支持</p>
</td>
<td class="cellrowborder" valign="top" width="13.94%" headers="mcps1.2.5.1.4 "><p id="p174933331293"><a name="p174933331293"></a><a name="p174933331293"></a>支持</p>
</td>
</tr>
<tr id="row688319149917"><td class="cellrowborder" valign="top" width="12.68%" headers="mcps1.2.5.1.1 "><p id="p13883111418919"><a name="p13883111418919"></a><a name="p13883111418919"></a>batch_num</p>
</td>
<td class="cellrowborder" valign="top" width="60.06%" headers="mcps1.2.5.1.2 "><p id="p148831014795"><a name="p148831014795"></a><a name="p148831014795"></a>控制量化使用多少个batch的数据。取值范围大于0，默认为1。</p>
<p id="p281411218216"><a name="p281411218216"></a><a name="p281411218216"></a>IFMR/HFMG两种算法都支持。</p>
</td>
<td class="cellrowborder" valign="top" width="13.320000000000002%" headers="mcps1.2.5.1.3 "><p id="p11501834096"><a name="p11501834096"></a><a name="p11501834096"></a>支持</p>
</td>
<td class="cellrowborder" valign="top" width="13.94%" headers="mcps1.2.5.1.4 "><p id="p2510345917"><a name="p2510345917"></a><a name="p2510345917"></a>支持</p>
</td>
</tr>
<tr id="row6856058121519"><td class="cellrowborder" valign="top" width="12.68%" headers="mcps1.2.5.1.1 "><p id="p7856958201516"><a name="p7856958201516"></a><a name="p7856958201516"></a>asymmetric</p>
</td>
<td class="cellrowborder" valign="top" width="60.06%" headers="mcps1.2.5.1.2 "><p id="p18570132515213"><a name="p18570132515213"></a><a name="p18570132515213"></a>控制数据量化是对称量化还是非对称量化，用于控制逐层量化算法的选择。</p>
<a name="ul31744286216"></a><a name="ul31744286216"></a><ul id="ul31744286216"><li>true：非对称量化，默认为true。</li><li>false：对称量化。</li></ul>
<p id="p1617482812118"><a name="p1617482812118"></a><a name="p1617482812118"></a>IFMR/HFMG两种算法都支持。</p>
</td>
<td class="cellrowborder" valign="top" width="13.320000000000002%" headers="mcps1.2.5.1.3 "><p id="p74101838535"><a name="p74101838535"></a><a name="p74101838535"></a>支持</p>
</td>
<td class="cellrowborder" valign="top" width="13.94%" headers="mcps1.2.5.1.4 "><p id="p1881319321313"><a name="p1881319321313"></a><a name="p1881319321313"></a>支持</p>
</td>
</tr>
<tr id="row785695841518"><td class="cellrowborder" valign="top" width="12.68%" headers="mcps1.2.5.1.1 "><p id="p2085675811512"><a name="p2085675811512"></a><a name="p2085675811512"></a>quant_granularity</p>
</td>
<td class="cellrowborder" valign="top" width="60.06%" headers="mcps1.2.5.1.2 "><p id="p15112312194416"><a name="p15112312194416"></a><a name="p15112312194416"></a>量化粒度，支持如下两种方式：</p>
<a name="ul14112141214414"></a><a name="ul14112141214414"></a><ul id="ul14112141214414"><li>0：per_tensor，默认为per_tensor。</li><li>1：per_channel。</li></ul>
<p id="p171183588210"><a name="p171183588210"></a><a name="p171183588210"></a>IFMR/HFMG两种算法都支持。</p>
</td>
<td class="cellrowborder" valign="top" width="13.320000000000002%" headers="mcps1.2.5.1.3 "><p id="p14107381231"><a name="p14107381231"></a><a name="p14107381231"></a>支持</p>
</td>
<td class="cellrowborder" valign="top" width="13.94%" headers="mcps1.2.5.1.4 "><p id="p138135321133"><a name="p138135321133"></a><a name="p138135321133"></a>支持</p>
</td>
</tr>
<tr id="row28561658121519"><td class="cellrowborder" valign="top" width="12.68%" headers="mcps1.2.5.1.1 "><p id="p14856058121514"><a name="p14856058121514"></a><a name="p14856058121514"></a>max_percentile</p>
</td>
<td class="cellrowborder" valign="top" width="60.06%" headers="mcps1.2.5.1.2 "><p id="p1360434016169"><a name="p1360434016169"></a><a name="p1360434016169"></a><span>IFMR数据量化算法中，最大值搜索位置参数。</span></p>
<p id="p16646181010261"><a name="p16646181010261"></a><a name="p16646181010261"></a>取值范围为(0.5,1]，默认为0.999999。</p>
<p id="p1521163372613"><a name="p1521163372613"></a><a name="p1521163372613"></a>仅<span>IFMR</span>算法支持。</p>
</td>
<td class="cellrowborder" valign="top" width="13.320000000000002%" headers="mcps1.2.5.1.3 "><p id="p4410113820316"><a name="p4410113820316"></a><a name="p4410113820316"></a>支持</p>
</td>
<td class="cellrowborder" valign="top" width="13.94%" headers="mcps1.2.5.1.4 "><p id="p116912417610"><a name="p116912417610"></a><a name="p116912417610"></a>不支持</p>
</td>
</tr>
<tr id="row128578581150"><td class="cellrowborder" valign="top" width="12.68%" headers="mcps1.2.5.1.1 "><p id="p48572583153"><a name="p48572583153"></a><a name="p48572583153"></a>min_percentile</p>
</td>
<td class="cellrowborder" valign="top" width="60.06%" headers="mcps1.2.5.1.2 "><p id="p1460424014164"><a name="p1460424014164"></a><a name="p1460424014164"></a><span>IFMR数据量化算法中，最小值搜索位置</span>参数。</p>
<p id="p3898204416267"><a name="p3898204416267"></a><a name="p3898204416267"></a>取值范围为(0.5,1]，默认为0.999999。</p>
<p id="p12980511172711"><a name="p12980511172711"></a><a name="p12980511172711"></a>仅<span>IFMR</span>算法支持。</p>
</td>
<td class="cellrowborder" valign="top" width="13.320000000000002%" headers="mcps1.2.5.1.3 "><p id="p114101938234"><a name="p114101938234"></a><a name="p114101938234"></a>支持</p>
</td>
<td class="cellrowborder" valign="top" width="13.94%" headers="mcps1.2.5.1.4 "><p id="p4749413620"><a name="p4749413620"></a><a name="p4749413620"></a>不支持</p>
</td>
</tr>
<tr id="row128571058101518"><td class="cellrowborder" valign="top" width="12.68%" headers="mcps1.2.5.1.1 "><p id="p685718581155"><a name="p685718581155"></a><a name="p685718581155"></a>search_range</p>
</td>
<td class="cellrowborder" valign="top" width="60.06%" headers="mcps1.2.5.1.2 "><p id="p18605174011612"><a name="p18605174011612"></a><a name="p18605174011612"></a>IFMR数据量化算法中，控制量化因子的搜索范围[search_range_start, search_range_end]。</p>
<p id="p455153310277"><a name="p455153310277"></a><a name="p455153310277"></a>取值范围为0&lt;search_range_start&lt;search_range_end，推荐配置为[0.7,1.3]。</p>
<p id="p1121528132818"><a name="p1121528132818"></a><a name="p1121528132818"></a>仅<span>IFMR</span>算法支持。</p>
</td>
<td class="cellrowborder" valign="top" width="13.320000000000002%" headers="mcps1.2.5.1.3 "><p id="p841017380317"><a name="p841017380317"></a><a name="p841017380317"></a>支持</p>
</td>
<td class="cellrowborder" valign="top" width="13.94%" headers="mcps1.2.5.1.4 "><p id="p37854113618"><a name="p37854113618"></a><a name="p37854113618"></a>不支持</p>
</td>
</tr>
<tr id="row085725816155"><td class="cellrowborder" valign="top" width="12.68%" headers="mcps1.2.5.1.1 "><p id="p6857195861517"><a name="p6857195861517"></a><a name="p6857195861517"></a>search_step</p>
</td>
<td class="cellrowborder" valign="top" width="60.06%" headers="mcps1.2.5.1.2 "><p id="zh-cn_topic_0240188002_p111814492426"><a name="zh-cn_topic_0240188002_p111814492426"></a><a name="zh-cn_topic_0240188002_p111814492426"></a>IFMR数据量化算法中，控制量化因子的搜索步长。</p>
<p id="p14906729132817"><a name="p14906729132817"></a><a name="p14906729132817"></a>取值范围为(0, (search_range_end-search_range_start)]，默认为0.01。</p>
<p id="p14430123052817"><a name="p14430123052817"></a><a name="p14430123052817"></a>仅<span>IFMR</span>算法支持。</p>
</td>
<td class="cellrowborder" valign="top" width="13.320000000000002%" headers="mcps1.2.5.1.3 "><p id="p124101038834"><a name="p124101038834"></a><a name="p124101038834"></a>支持</p>
</td>
<td class="cellrowborder" valign="top" width="13.94%" headers="mcps1.2.5.1.4 "><p id="p118218415618"><a name="p118218415618"></a><a name="p118218415618"></a>不支持</p>
</td>
</tr>
<tr id="row6857175813158"><td class="cellrowborder" valign="top" width="12.68%" headers="mcps1.2.5.1.1 "><p id="p1185715589158"><a name="p1185715589158"></a><a name="p1185715589158"></a>num_of_bins</p>
</td>
<td class="cellrowborder" valign="top" width="60.06%" headers="mcps1.2.5.1.2 "><p id="p14190312023"><a name="p14190312023"></a><a name="p14190312023"></a>直方图的bin（直方图中的一个最小单位直方图形）数目，支持的范围为{1024, 2048, 4096, 8192}。默认值为4096。</p>
<p id="p5348173411224"><a name="p5348173411224"></a><a name="p5348173411224"></a>仅HFMG算法支持。</p>
</td>
<td class="cellrowborder" valign="top" width="13.320000000000002%" headers="mcps1.2.5.1.3 "><p id="p1341083816319"><a name="p1341083816319"></a><a name="p1341083816319"></a>不支持</p>
</td>
<td class="cellrowborder" valign="top" width="13.94%" headers="mcps1.2.5.1.4 "><p id="p581383211313"><a name="p581383211313"></a><a name="p581383211313"></a>支持</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0240188739_zh-cn_topic_0122830089_section293415513458"></a>

无

## 调用示例<a name="zh-cn_topic_0240188739_section64231658994"></a>

```python
import amct_pytorch as amct
from amct_pytorch.nn.module.quantization.quant_calibration_op import QuantCalibrationOp

class LinearNet(nn.Module):
    def __init__(self, quant_algo_params):

        super(LinearNet,self).__init__()
        self.quant_algo_params = quant_algo_params
        self.layer1 = nn.Linear(28, 1024, bias=False)
        self.ptq_1 = QuantCalibrationOp(record_file, quant_algo_params=self.quant_algo_params, quant_method="kv_cache_quant")

    def forward(self, layer_name, x):
        x = self.layer1(x)
        x = self.ptq_1(layer_name, x)

# 保存量化因子的record文件路径
temp_folder = "./"
record_file = os.path.join(temp_folder, 'kv_cache.txt')
input_data = torch.randn((2, 2, 28, 28))

quant_algo_params = {"act_algo": "hfmg"}
model = LinearNet(quant_algo_params).to(torch.device("cpu"))
model.eval()

ans_2 = model("qat_1", input_data)
```

