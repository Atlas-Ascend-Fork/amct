# record记录文件<a name="ZH-CN_TOPIC_0000002517028838"></a>

record文件，为基于protobuf协议的序列化数据结构文件，记录量化场景量化因子scale/offset，稀疏场景各稀疏层间的级联关系等，通过该文件、压缩配置文件以及原始网络模型文件，生成压缩后的模型文件。

## record原型定义<a name="section851092965013"></a>

record文件对应的protobuf原型定义为（或查看_AMCT安装目录_/amct\_pytorch/proto/scale\_offset\_record\_pytorch.proto文件）：

```
syntax = "proto2";
import "amct_pytorch/proto/basic_info.proto";

message SingleLayerRecord {
    optional float scale_d = 1;
    optional int32 offset_d = 2;
    repeated float scale_w = 3;
    repeated int32 offset_w = 4;
    repeated uint32 shift_bit = 5;
    repeated float tensor_balance_factor = 6;
    optional bool skip_fusion = 9 [default = true];
    optional string dst_type = 10 [default = 'INT8'];
    optional string act_type = 11 [default = 'INT8'];
    optional string wts_type = 12 [default = 'INT8'];
}

message SingleLayerKVCacheRecord {
    repeated float scale = 1;
    repeated int32 offset = 2;
}

message MapFiledEntry {
    optional string key = 1;
    optional SingleLayerRecord value = 2;
    optional SingleLayerKVCacheRecord kv_cache_value = 3;

}

message ScaleOffsetRecord {
    repeated MapFiledEntry record = 1;
    repeated PruneRecord prune_record = 2;
}

message PruneRecord {
    repeated PruneNode producer = 1;
    repeated PruneNode consumer = 2;
    optional PruneNode selective_prune = 3;
}

message PruneNode {
    required string name = 1;
    repeated AMCTProto.AttrProto attr = 2;
}
```

参数说明如下：

<a name="zh-cn_topic_0254211904_zh-cn_topic_0240188735_table1225503375617"></a>
<table><thead align="left"><tr id="zh-cn_topic_0254211904_zh-cn_topic_0240188735_row11255153315614"><th class="cellrowborder" valign="top" width="16.878312168783122%" id="mcps1.1.6.1.1"><p id="zh-cn_topic_0254211904_zh-cn_topic_0240188735_p6255193315616"><a name="zh-cn_topic_0254211904_zh-cn_topic_0240188735_p6255193315616"></a><a name="zh-cn_topic_0254211904_zh-cn_topic_0240188735_p6255193315616"></a>消息</p>
</th>
<th class="cellrowborder" valign="top" width="7.3992600739926%" id="mcps1.1.6.1.2"><p id="zh-cn_topic_0254211904_p194851125105014"><a name="zh-cn_topic_0254211904_p194851125105014"></a><a name="zh-cn_topic_0254211904_p194851125105014"></a>是否必填</p>
</th>
<th class="cellrowborder" valign="top" width="11.42885711428857%" id="mcps1.1.6.1.3"><p id="zh-cn_topic_0254211904_p466892519514"><a name="zh-cn_topic_0254211904_p466892519514"></a><a name="zh-cn_topic_0254211904_p466892519514"></a>类型</p>
</th>
<th class="cellrowborder" valign="top" width="12.968703129687032%" id="mcps1.1.6.1.4"><p id="zh-cn_topic_0254211904_zh-cn_topic_0240188735_p499481531910"><a name="zh-cn_topic_0254211904_zh-cn_topic_0240188735_p499481531910"></a><a name="zh-cn_topic_0254211904_zh-cn_topic_0240188735_p499481531910"></a>字段</p>
</th>
<th class="cellrowborder" valign="top" width="51.32486751324868%" id="mcps1.1.6.1.5"><p id="zh-cn_topic_0254211904_zh-cn_topic_0240188735_p122551335563"><a name="zh-cn_topic_0254211904_zh-cn_topic_0240188735_p122551335563"></a><a name="zh-cn_topic_0254211904_zh-cn_topic_0240188735_p122551335563"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0254211904_row173077381744"><td class="cellrowborder" rowspan="11" valign="top" width="16.878312168783122%" headers="mcps1.1.6.1.1 "><p id="zh-cn_topic_0254211904_p1574404011413"><a name="zh-cn_topic_0254211904_p1574404011413"></a><a name="zh-cn_topic_0254211904_p1574404011413"></a>SingleLayerRecord</p>
<p id="p12702171315511"><a name="p12702171315511"></a><a name="p12702171315511"></a></p>
<p id="p20534911145111"><a name="p20534911145111"></a><a name="p20534911145111"></a></p>
</td>
<td class="cellrowborder" valign="top" width="7.3992600739926%" headers="mcps1.1.6.1.2 "><p id="zh-cn_topic_0254211904_p193082381940"><a name="zh-cn_topic_0254211904_p193082381940"></a><a name="zh-cn_topic_0254211904_p193082381940"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="11.42885711428857%" headers="mcps1.1.6.1.3 "><p id="zh-cn_topic_0254211904_p193081538144"><a name="zh-cn_topic_0254211904_p193081538144"></a><a name="zh-cn_topic_0254211904_p193081538144"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="12.968703129687032%" headers="mcps1.1.6.1.4 "><p id="zh-cn_topic_0254211904_p830813384414"><a name="zh-cn_topic_0254211904_p830813384414"></a><a name="zh-cn_topic_0254211904_p830813384414"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="51.32486751324868%" headers="mcps1.1.6.1.5 "><p id="zh-cn_topic_0254211904_p16308193812419"><a name="zh-cn_topic_0254211904_p16308193812419"></a><a name="zh-cn_topic_0254211904_p16308193812419"></a>包含了量化层所需要的所有量化因子记录信息。</p>
</td>
</tr>
<tr id="zh-cn_topic_0254211904_zh-cn_topic_0240188735_row9255123311566"><td class="cellrowborder" valign="top" headers="mcps1.1.6.1.1 "><p id="zh-cn_topic_0254211904_p11249539520"><a name="zh-cn_topic_0254211904_p11249539520"></a><a name="zh-cn_topic_0254211904_p11249539520"></a>optional</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.2 "><p id="zh-cn_topic_0254211904_p19715115505"><a name="zh-cn_topic_0254211904_p19715115505"></a><a name="zh-cn_topic_0254211904_p19715115505"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.3 "><p id="zh-cn_topic_0254211904_p99717119610"><a name="zh-cn_topic_0254211904_p99717119610"></a><a name="zh-cn_topic_0254211904_p99717119610"></a>scale_d</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.4 "><p id="zh-cn_topic_0254211904_zh-cn_topic_0240188735_p32551133175612"><a name="zh-cn_topic_0254211904_zh-cn_topic_0240188735_p32551133175612"></a><a name="zh-cn_topic_0254211904_zh-cn_topic_0240188735_p32551133175612"></a>数据量化scale因子，仅支持对数据进行统一量化。</p>
</td>
</tr>
<tr id="zh-cn_topic_0254211904_zh-cn_topic_0240188735_row11255533205611"><td class="cellrowborder" valign="top" headers="mcps1.1.6.1.1 "><p id="zh-cn_topic_0254211904_p142538635219"><a name="zh-cn_topic_0254211904_p142538635219"></a><a name="zh-cn_topic_0254211904_p142538635219"></a>optional</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.2 "><p id="zh-cn_topic_0254211904_p13170131414515"><a name="zh-cn_topic_0254211904_p13170131414515"></a><a name="zh-cn_topic_0254211904_p13170131414515"></a>int32</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.3 "><p id="zh-cn_topic_0254211904_p1277111618610"><a name="zh-cn_topic_0254211904_p1277111618610"></a><a name="zh-cn_topic_0254211904_p1277111618610"></a>offset_d</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.4 "><p id="zh-cn_topic_0254211904_p1655910371085"><a name="zh-cn_topic_0254211904_p1655910371085"></a><a name="zh-cn_topic_0254211904_p1655910371085"></a>数据量化offset因子，仅支持对数据进行统一量化。</p>
</td>
</tr>
<tr id="zh-cn_topic_0254211904_row1188315112516"><td class="cellrowborder" valign="top" headers="mcps1.1.6.1.1 "><p id="zh-cn_topic_0254211904_p5530513151314"><a name="zh-cn_topic_0254211904_p5530513151314"></a><a name="zh-cn_topic_0254211904_p5530513151314"></a>repeated</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.2 "><p id="zh-cn_topic_0254211904_p1011820184911"><a name="zh-cn_topic_0254211904_p1011820184911"></a><a name="zh-cn_topic_0254211904_p1011820184911"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.3 "><p id="zh-cn_topic_0254211904_p2212018491"><a name="zh-cn_topic_0254211904_p2212018491"></a><a name="zh-cn_topic_0254211904_p2212018491"></a>scale_w</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.4 "><p id="p813434617333"><a name="p813434617333"></a><a name="p813434617333"></a>权重量化scale因子，支持标量（对当前层的权重进行统一量化），向量（对当前层的权重按channel_wise方式进行量化）两种模式，仅支持Conv2d类型进行channel_wise量化模式。</p>
</td>
</tr>
<tr id="zh-cn_topic_0254211904_zh-cn_topic_0240188735_row62551533115615"><td class="cellrowborder" valign="top" headers="mcps1.1.6.1.1 "><p id="zh-cn_topic_0254211904_p1525819172139"><a name="zh-cn_topic_0254211904_p1525819172139"></a><a name="zh-cn_topic_0254211904_p1525819172139"></a>repeated</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.2 "><p id="zh-cn_topic_0254211904_p418281410510"><a name="zh-cn_topic_0254211904_p418281410510"></a><a name="zh-cn_topic_0254211904_p418281410510"></a>int32</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.3 "><p id="zh-cn_topic_0254211904_zh-cn_topic_0240188735_p525553375617"><a name="zh-cn_topic_0254211904_zh-cn_topic_0240188735_p525553375617"></a><a name="zh-cn_topic_0254211904_zh-cn_topic_0240188735_p525553375617"></a>offset_w</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.4 "><p id="zh-cn_topic_0254211904_p3721162210918"><a name="zh-cn_topic_0254211904_p3721162210918"></a><a name="zh-cn_topic_0254211904_p3721162210918"></a>权重量化offset因子，同scale_w一样支持标量和向量两种模式，且需要同scale_w维度一致，当前不支持权重带offset量化模式，offset_w仅支持0。</p>
</td>
</tr>
<tr id="zh-cn_topic_0254211904_zh-cn_topic_0240188735_row1925643313563"><td class="cellrowborder" valign="top" headers="mcps1.1.6.1.1 "><p id="zh-cn_topic_0254211904_p1443019141315"><a name="zh-cn_topic_0254211904_p1443019141315"></a><a name="zh-cn_topic_0254211904_p1443019141315"></a>repeated</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.2 "><p id="zh-cn_topic_0254211904_p17182151405117"><a name="zh-cn_topic_0254211904_p17182151405117"></a><a name="zh-cn_topic_0254211904_p17182151405117"></a>uint32</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.3 "><p id="zh-cn_topic_0254211904_p63418167714"><a name="zh-cn_topic_0254211904_p63418167714"></a><a name="zh-cn_topic_0254211904_p63418167714"></a>shift_bit</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.4 "><p id="zh-cn_topic_0240188735_p12567334561"><a name="zh-cn_topic_0240188735_p12567334561"></a><a name="zh-cn_topic_0240188735_p12567334561"></a>移位因子。保留参数，shift_bit参数不会写入record文件。</p>
</td>
</tr>
<tr id="row1882674045513"><td class="cellrowborder" valign="top" headers="mcps1.1.6.1.1 "><p id="p1662584120719"><a name="p1662584120719"></a><a name="p1662584120719"></a>optional</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.2 "><p id="p76254411871"><a name="p76254411871"></a><a name="p76254411871"></a>bool</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.3 "><p id="p14625104118719"><a name="p14625104118719"></a><a name="p14625104118719"></a>skip_fusion</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.4 "><p id="p1730743410118"><a name="p1730743410118"></a><a name="p1730743410118"></a>配置当前层是否要跳过Conv+BN融合，默认为false，即当前层要做上述融合。</p>
</td>
</tr>
<tr id="row6453123315274"><td class="cellrowborder" valign="top" headers="mcps1.1.6.1.1 "><p id="p11404121824210"><a name="p11404121824210"></a><a name="p11404121824210"></a>optional</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.2 "><p id="p1940431818420"><a name="p1940431818420"></a><a name="p1940431818420"></a>string</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.3 "><p id="p17405151834211"><a name="p17405151834211"></a><a name="p17405151834211"></a>dst_type</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.4 "><p id="p20405018164210"><a name="p20405018164210"></a><a name="p20405018164210"></a>量化位宽，包括INT8和INT4两种量化类型。<strong id="b274541410473"><a name="b274541410473"></a><a name="b274541410473"></a>该字段仅量化感知训练场景使用。</strong></p>
</td>
</tr>
<tr id="row01735569514"><td class="cellrowborder" valign="top" headers="mcps1.1.6.1.1 "><p id="p107512024112318"><a name="p107512024112318"></a><a name="p107512024112318"></a>repeated</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.2 "><p id="p1075172418237"><a name="p1075172418237"></a><a name="p1075172418237"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.3 "><p id="p19751192415238"><a name="p19751192415238"></a><a name="p19751192415238"></a>tensor_balance_factor</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.4 "><p id="p87511624112310"><a name="p87511624112310"></a><a name="p87511624112310"></a>均衡量化因子。<strong id="b3381917194714"><a name="b3381917194714"></a><a name="b3381917194714"></a>该字段仅量化数据均衡预处理场景使用。</strong></p>
</td>
</tr>
<tr id="row207025132519"><td class="cellrowborder" valign="top" headers="mcps1.1.6.1.1 "><p id="p12408102910209"><a name="p12408102910209"></a><a name="p12408102910209"></a>optional</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.2 "><p id="p740812992014"><a name="p740812992014"></a><a name="p740812992014"></a>string</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.3 "><p id="p734414101207"><a name="p734414101207"></a><a name="p734414101207"></a>act_type</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.4 "><p id="p334419103208"><a name="p334419103208"></a><a name="p334419103208"></a>数据量化位宽，包括INT8和INT16两种量化类型。<strong id="b1043485110298"><a name="b1043485110298"></a><a name="b1043485110298"></a>当前版本仅支持INT8量化。</strong></p>
</td>
</tr>
<tr id="row4533161135115"><td class="cellrowborder" valign="top" headers="mcps1.1.6.1.1 "><p id="p4561339192014"><a name="p4561339192014"></a><a name="p4561339192014"></a>optional</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.2 "><p id="p6566394200"><a name="p6566394200"></a><a name="p6566394200"></a>string</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.3 "><p id="p12450151332014"><a name="p12450151332014"></a><a name="p12450151332014"></a>wts_type</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.4 "><p id="p631761122611"><a name="p631761122611"></a><a name="p631761122611"></a>权重量化位宽。</p>
<p id="p10450313192011"><a name="p10450313192011"></a><a name="p10450313192011"></a>当前INT6、INT7量化后的量化因子仍保存为INT8类型。</p>
</td>
</tr>
<tr id="row11121216131510"><td class="cellrowborder" rowspan="3" valign="top" width="16.878312168783122%" headers="mcps1.1.6.1.1 "><p id="p2011231691515"><a name="p2011231691515"></a><a name="p2011231691515"></a>SingleLayerKVCacheRecord</p>
<p id="p157351919121516"><a name="p157351919121516"></a><a name="p157351919121516"></a></p>
</td>
<td class="cellrowborder" valign="top" width="7.3992600739926%" headers="mcps1.1.6.1.2 "><p id="p69771219168"><a name="p69771219168"></a><a name="p69771219168"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="11.42885711428857%" headers="mcps1.1.6.1.3 "><p id="p2097715213161"><a name="p2097715213161"></a><a name="p2097715213161"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="12.968703129687032%" headers="mcps1.1.6.1.4 "><p id="p69773210165"><a name="p69773210165"></a><a name="p69773210165"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="51.32486751324868%" headers="mcps1.1.6.1.5 "><p id="p2011212162158"><a name="p2011212162158"></a><a name="p2011212162158"></a>kv-cache量化因子配置。</p>
</td>
</tr>
<tr id="row12735121917156"><td class="cellrowborder" valign="top" headers="mcps1.1.6.1.1 "><p id="p37351519171510"><a name="p37351519171510"></a><a name="p37351519171510"></a>repeated</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.2 "><p id="p473561951517"><a name="p473561951517"></a><a name="p473561951517"></a>float32</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.3 "><p id="p13580165617152"><a name="p13580165617152"></a><a name="p13580165617152"></a>scale</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.4 "><p id="p1973581910154"><a name="p1973581910154"></a><a name="p1973581910154"></a>scale量化因子。</p>
</td>
</tr>
<tr id="row57811017181514"><td class="cellrowborder" valign="top" headers="mcps1.1.6.1.1 "><p id="p978161761512"><a name="p978161761512"></a><a name="p978161761512"></a>repeated</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.2 "><p id="p278171741513"><a name="p278171741513"></a><a name="p278171741513"></a>int32</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.3 "><p id="p18580456121514"><a name="p18580456121514"></a><a name="p18580456121514"></a>offset</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.4 "><p id="p14438194413491"><a name="p14438194413491"></a><a name="p14438194413491"></a>offset量化因子。</p>
</td>
</tr>
<tr id="row22775011386"><td class="cellrowborder" rowspan="3" valign="top" width="16.878312168783122%" headers="mcps1.1.6.1.1 "><p id="zh-cn_topic_0254211904_zh-cn_topic_0240188735_p1925513338569"><a name="zh-cn_topic_0254211904_zh-cn_topic_0240188735_p1925513338569"></a><a name="zh-cn_topic_0254211904_zh-cn_topic_0240188735_p1925513338569"></a>ScaleOffsetRecord</p>
<p id="p1731316519381"><a name="p1731316519381"></a><a name="p1731316519381"></a></p>
<p id="p188536720170"><a name="p188536720170"></a><a name="p188536720170"></a></p>
</td>
<td class="cellrowborder" valign="top" width="7.3992600739926%" headers="mcps1.1.6.1.2 "><p id="zh-cn_topic_0254211904_p1485162505012"><a name="zh-cn_topic_0254211904_p1485162505012"></a><a name="zh-cn_topic_0254211904_p1485162505012"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="11.42885711428857%" headers="mcps1.1.6.1.3 "><p id="zh-cn_topic_0254211904_p1617031410516"><a name="zh-cn_topic_0254211904_p1617031410516"></a><a name="zh-cn_topic_0254211904_p1617031410516"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="12.968703129687032%" headers="mcps1.1.6.1.4 "><p id="zh-cn_topic_0254211904_zh-cn_topic_0240188735_p2994161514196"><a name="zh-cn_topic_0254211904_zh-cn_topic_0240188735_p2994161514196"></a><a name="zh-cn_topic_0254211904_zh-cn_topic_0240188735_p2994161514196"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="51.32486751324868%" headers="mcps1.1.6.1.5 "><p id="zh-cn_topic_0254211904_p379618813416"><a name="zh-cn_topic_0254211904_p379618813416"></a><a name="zh-cn_topic_0254211904_p379618813416"></a>map结构，为保证兼容性，采用离散的map结构。</p>
</td>
</tr>
<tr id="row968505773713"><td class="cellrowborder" valign="top" headers="mcps1.1.6.1.1 "><p id="zh-cn_topic_0254211904_p12920144517214"><a name="zh-cn_topic_0254211904_p12920144517214"></a><a name="zh-cn_topic_0254211904_p12920144517214"></a>repeated</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.2 "><p id="zh-cn_topic_0254211904_p15971151155016"><a name="zh-cn_topic_0254211904_p15971151155016"></a><a name="zh-cn_topic_0254211904_p15971151155016"></a>MapFiledEntry</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.3 "><p id="zh-cn_topic_0254211904_p178974205413"><a name="zh-cn_topic_0254211904_p178974205413"></a><a name="zh-cn_topic_0254211904_p178974205413"></a>record</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.4 "><p id="zh-cn_topic_0254211904_p171331312158"><a name="zh-cn_topic_0254211904_p171331312158"></a><a name="zh-cn_topic_0254211904_p171331312158"></a>每个record对应一个量化层的量化因子记录；record包括两个成员：</p>
<a name="zh-cn_topic_0254211904_ul1115801910512"></a><a name="zh-cn_topic_0254211904_ul1115801910512"></a><ul id="zh-cn_topic_0254211904_ul1115801910512"><li>key为所记录量化层的layer name。</li><li>value对应SingleLayerRecord定义的具体量化因子。</li></ul>
</td>
</tr>
<tr id="row138531975171"><td class="cellrowborder" valign="top" headers="mcps1.1.6.1.1 "><p id="p28531873175"><a name="p28531873175"></a><a name="p28531873175"></a>repeated</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.2 "><p id="p6853187191717"><a name="p6853187191717"></a><a name="p6853187191717"></a>PruneRecord</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.3 "><p id="p7853977173"><a name="p7853977173"></a><a name="p7853977173"></a>prune_record</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.4 "><p id="p6853197101710"><a name="p6853197101710"></a><a name="p6853197101710"></a>稀疏信息的记录。</p>
</td>
</tr>
<tr id="row3869163719462"><td class="cellrowborder" rowspan="3" valign="top" width="16.878312168783122%" headers="mcps1.1.6.1.1 "><p id="p8869113717464"><a name="p8869113717464"></a><a name="p8869113717464"></a>MapFiledEntry</p>
</td>
<td class="cellrowborder" valign="top" width="7.3992600739926%" headers="mcps1.1.6.1.2 "><p id="p8869237104612"><a name="p8869237104612"></a><a name="p8869237104612"></a>optional</p>
</td>
<td class="cellrowborder" valign="top" width="11.42885711428857%" headers="mcps1.1.6.1.3 "><p id="p198699373461"><a name="p198699373461"></a><a name="p198699373461"></a>string</p>
</td>
<td class="cellrowborder" valign="top" width="12.968703129687032%" headers="mcps1.1.6.1.4 "><p id="p486916373463"><a name="p486916373463"></a><a name="p486916373463"></a>key</p>
</td>
<td class="cellrowborder" valign="top" width="51.32486751324868%" headers="mcps1.1.6.1.5 "><p id="p118698372465"><a name="p118698372465"></a><a name="p118698372465"></a>层名。</p>
</td>
</tr>
<tr id="row111471914463"><td class="cellrowborder" valign="top" headers="mcps1.1.6.1.1 "><p id="p12115819204614"><a name="p12115819204614"></a><a name="p12115819204614"></a>optional</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.2 "><p id="p3115201994619"><a name="p3115201994619"></a><a name="p3115201994619"></a>SingleLayerRecord</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.3 "><p id="p511511917461"><a name="p511511917461"></a><a name="p511511917461"></a>value</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.4 "><p id="p81151019184619"><a name="p81151019184619"></a><a name="p81151019184619"></a>量化因子配置。</p>
</td>
</tr>
<tr id="row5474407911"><td class="cellrowborder" valign="top" headers="mcps1.1.6.1.1 "><p id="p8474102919"><a name="p8474102919"></a><a name="p8474102919"></a>optional</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.2 "><p id="p1247419013913"><a name="p1247419013913"></a><a name="p1247419013913"></a>SingleLayerKVCacheRecord</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.3 "><p id="p18474190493"><a name="p18474190493"></a><a name="p18474190493"></a>kv_cache_value</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.4 "><p id="p647410198"><a name="p647410198"></a><a name="p647410198"></a>kv-cache量化因子配置。</p>
</td>
</tr>
<tr id="row1952913394519"><td class="cellrowborder" rowspan="4" valign="top" width="16.878312168783122%" headers="mcps1.1.6.1.1 "><p id="p1852912364512"><a name="p1852912364512"></a><a name="p1852912364512"></a>PruneRecord</p>
<p id="p779141312223"><a name="p779141312223"></a><a name="p779141312223"></a></p>
</td>
<td class="cellrowborder" valign="top" width="7.3992600739926%" headers="mcps1.1.6.1.2 "><p id="p05291313459"><a name="p05291313459"></a><a name="p05291313459"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="11.42885711428857%" headers="mcps1.1.6.1.3 "><p id="p65295384515"><a name="p65295384515"></a><a name="p65295384515"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="12.968703129687032%" headers="mcps1.1.6.1.4 "><p id="p55291035455"><a name="p55291035455"></a><a name="p55291035455"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="51.32486751324868%" headers="mcps1.1.6.1.5 "><p id="p1529337454"><a name="p1529337454"></a><a name="p1529337454"></a>稀疏信息的记录。</p>
</td>
</tr>
<tr id="row466915710456"><td class="cellrowborder" valign="top" headers="mcps1.1.6.1.1 "><p id="p136709714514"><a name="p136709714514"></a><a name="p136709714514"></a>repeated</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.2 "><p id="p2670197154510"><a name="p2670197154510"></a><a name="p2670197154510"></a>PruneNode</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.3 "><p id="p96708724518"><a name="p96708724518"></a><a name="p96708724518"></a>producer</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.4 "><p id="p12670187144515"><a name="p12670187144515"></a><a name="p12670187144515"></a>稀疏的producer，可稀疏结点间级联关系的根节点。</p>
<p id="p156891641101018"><a name="p156891641101018"></a><a name="p156891641101018"></a>例如conv1&gt;bn&gt;relu&gt;conv2都可以稀疏，且bn、relu、conv2都会受到conv1稀疏的影响，则bn、relu、conv2是conv1的consumer；conv1是bn、relu、conv2的producer。</p>
</td>
</tr>
<tr id="row5922114459"><td class="cellrowborder" valign="top" headers="mcps1.1.6.1.1 "><p id="p1692111184511"><a name="p1692111184511"></a><a name="p1692111184511"></a>repeated</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.2 "><p id="p992711104512"><a name="p992711104512"></a><a name="p992711104512"></a>PruneNode</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.3 "><p id="p11921511124518"><a name="p11921511124518"></a><a name="p11921511124518"></a>consumer</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.4 "><p id="p3923117459"><a name="p3923117459"></a><a name="p3923117459"></a>稀疏的consumer，可稀疏结点间级联关系的下游节点。</p>
<p id="p11521182721312"><a name="p11521182721312"></a><a name="p11521182721312"></a>例如conv1&gt;bn&gt;relu&gt;conv2都可以稀疏，且bn、relu、conv2都会受到conv1稀疏的影响，则bn、relu、conv2是conv1的consumer；conv1是bn、relu、conv2的producer。</p>
</td>
</tr>
<tr id="row8791131311223"><td class="cellrowborder" valign="top" headers="mcps1.1.6.1.1 "><p id="p17398330142713"><a name="p17398330142713"></a><a name="p17398330142713"></a>optional</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.2 "><p id="p18398163010272"><a name="p18398163010272"></a><a name="p18398163010272"></a>PruneNode</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.3 "><p id="p0398153032715"><a name="p0398153032715"></a><a name="p0398153032715"></a>selective_prune</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.4 "><p id="p83981430162717"><a name="p83981430162717"></a><a name="p83981430162717"></a>4选2结构化稀疏节点。</p>
</td>
</tr>
<tr id="row3554132914619"><td class="cellrowborder" rowspan="3" valign="top" width="16.878312168783122%" headers="mcps1.1.6.1.1 "><p id="p14554152984620"><a name="p14554152984620"></a><a name="p14554152984620"></a>PruneNode</p>
</td>
<td class="cellrowborder" valign="top" width="7.3992600739926%" headers="mcps1.1.6.1.2 "><p id="p1455442915464"><a name="p1455442915464"></a><a name="p1455442915464"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="11.42885711428857%" headers="mcps1.1.6.1.3 "><p id="p55542291464"><a name="p55542291464"></a><a name="p55542291464"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="12.968703129687032%" headers="mcps1.1.6.1.4 "><p id="p755402944612"><a name="p755402944612"></a><a name="p755402944612"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="51.32486751324868%" headers="mcps1.1.6.1.5 "><p id="p1255522918468"><a name="p1255522918468"></a><a name="p1255522918468"></a>稀疏的节点。</p>
</td>
</tr>
<tr id="row16365173264616"><td class="cellrowborder" valign="top" headers="mcps1.1.6.1.1 "><p id="p103651232204620"><a name="p103651232204620"></a><a name="p103651232204620"></a>required</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.2 "><p id="p18365113213460"><a name="p18365113213460"></a><a name="p18365113213460"></a>string</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.3 "><p id="p3365123234612"><a name="p3365123234612"></a><a name="p3365123234612"></a>name</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.4 "><p id="p1365932164619"><a name="p1365932164619"></a><a name="p1365932164619"></a>节点名称。</p>
</td>
</tr>
<tr id="row9778193516467"><td class="cellrowborder" valign="top" headers="mcps1.1.6.1.1 "><p id="p12778183544614"><a name="p12778183544614"></a><a name="p12778183544614"></a>repeated</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.2 "><p id="p1877813513462"><a name="p1877813513462"></a><a name="p1877813513462"></a>AMCTProto.AttrProto</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.3 "><p id="p27783351465"><a name="p27783351465"></a><a name="p27783351465"></a>attr</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.4 "><p id="p167789351468"><a name="p167789351468"></a><a name="p167789351468"></a>节点属性。</p>
</td>
</tr>
</tbody>
</table>

**对于optional字段，由于protobuf协议未对重复出现的值报错，而是采用覆盖处理，因此出现重复配置的optional字段内容时会默认保留最后一次配置的值，需要用户自己保证文件的正确性**。

## record记录文件<a name="section1850911716537"></a>

最终生成的record文件格式为_record.txt_，文件内容根据特性不同划分如下。

-   量化特性record文件

    对于一般量化层配置需要包含scale\_d、offset\_d、scale\_w、offset\_w参数，文件内容示例如下：

    ```
    record {
      key: "conv1"
      value {
        scale_d: 0.0798481479
        offset_d: 1
        scale_w: 0.00297622895
        offset_w: 0
        skip_fusion: true
        dst_type: "INT8"
      }
    }
    record {
      key: "layer1.0.conv1"
      value {
        scale_d: 0.00392156886
        offset_d: -128
        scale_w: 0.00106807391
        scale_w: 0.00104224426
        offset_w: 0
        offset_w: 0
        dst_type: "INT4"
      }
    }
    ```

-   量化数据均衡预处理特性record文件，内容示例如下：

    ```
    record {
      key: "linear_1"
      value {
        scale_d: 0.00784554612
        offset_d: -1
        scale_w: 0.00778095098
        offset_w: 0
        tensor_balance_factor: 0.948409557
        tensor_balance_factor: 0.984379828
      }
    }
    record {
      key: "conv_1"
      value {
        scale_d: 0.00759239076
        offset_d: -4
        scale_w: 0.0075149606
        offset_w: 0
        tensor_balance_factor: 1.04744744
        tensor_balance_factor: 1.44586647
      }
    }
    ```

-   通道稀疏record文件记录各稀疏层间的级联关系，文件内容示例如下:

    ```
    prune_record {
      producer {
        name: "conv1"
        attr {
          name: "type"
          type: STRING
          s: "Conv2d"
        }
        attr {
          name: "begin"
          type: INT
          i: 0
        }
        attr {
          name: "end"
          type: INT
          i: 64
        }
      }
      consumer {
        name: "BN_1"
        attr {
          name: "type"
          type: STRING
          s: "FusedBatchNormV3"
        }
        attr {
          name: "begin"
          type: INT
          i: 0
        }
        attr {
          name: "end"
          type: INT
          i: 64
        }
      }
    }
    ```

-   结构化稀疏record文件内容示例如下：

    ```
    prune_record {
      selective_prune {
        name: "conv1"
        attr {
          name: "mask_shape"
          type: INTS
          ints: 3
          ints: 3
          ints: 3
          ints: 32
        }
      }
    }
    ```

