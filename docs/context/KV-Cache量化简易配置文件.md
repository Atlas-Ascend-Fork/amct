# KV Cache量化简易配置文件<a name="ZH-CN_TOPIC_0000002517028718"></a>

quant\_calibration\_config\_pytorch.proto文件参数说明如下表所示，该文件所在目录为：_AMCT安装目录_/amct\_pytorch/proto/。

**表 1**  quant\_calibration\_config\_pytorch.proto参数说明

<a name="zh-cn_topic_0240188735_table1225503375617"></a>
<table><thead align="left"><tr id="zh-cn_topic_0240188735_row11255153315614"><th class="cellrowborder" valign="top" width="11.148885111488852%" id="mcps1.2.6.1.1"><p id="zh-cn_topic_0240188735_p6255193315616"><a name="zh-cn_topic_0240188735_p6255193315616"></a><a name="zh-cn_topic_0240188735_p6255193315616"></a>消息</p>
</th>
<th class="cellrowborder" valign="top" width="6.52934706529347%" id="mcps1.2.6.1.2"><p id="p194851125105014"><a name="p194851125105014"></a><a name="p194851125105014"></a>是否必填</p>
</th>
<th class="cellrowborder" valign="top" width="13.628637136286374%" id="mcps1.2.6.1.3"><p id="p466892519514"><a name="p466892519514"></a><a name="p466892519514"></a>类型</p>
</th>
<th class="cellrowborder" valign="top" width="14.558544145585442%" id="mcps1.2.6.1.4"><p id="zh-cn_topic_0240188735_p499481531910"><a name="zh-cn_topic_0240188735_p499481531910"></a><a name="zh-cn_topic_0240188735_p499481531910"></a>字段</p>
</th>
<th class="cellrowborder" valign="top" width="54.13458654134586%" id="mcps1.2.6.1.5"><p id="zh-cn_topic_0240188735_p122551335563"><a name="zh-cn_topic_0240188735_p122551335563"></a><a name="zh-cn_topic_0240188735_p122551335563"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0240188735_row122551133185610"><td class="cellrowborder" rowspan="5" valign="top" width="11.148885111488852%" headers="mcps1.2.6.1.1 "><p id="zh-cn_topic_0240188735_p1925513338569"><a name="zh-cn_topic_0240188735_p1925513338569"></a><a name="zh-cn_topic_0240188735_p1925513338569"></a>AMCTQuantCaliConfig</p>
</td>
<td class="cellrowborder" valign="top" width="6.52934706529347%" headers="mcps1.2.6.1.2 "><p id="p1485162505012"><a name="p1485162505012"></a><a name="p1485162505012"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="13.628637136286374%" headers="mcps1.2.6.1.3 "><p id="p1617031410516"><a name="p1617031410516"></a><a name="p1617031410516"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="14.558544145585442%" headers="mcps1.2.6.1.4 "><p id="zh-cn_topic_0240188735_p2994161514196"><a name="zh-cn_topic_0240188735_p2994161514196"></a><a name="zh-cn_topic_0240188735_p2994161514196"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="54.13458654134586%" headers="mcps1.2.6.1.5 "><p id="zh-cn_topic_0240188735_p82551633105614"><a name="zh-cn_topic_0240188735_p82551633105614"></a><a name="zh-cn_topic_0240188735_p82551633105614"></a><span id="ph244511481409"><a name="ph244511481409"></a><a name="ph244511481409"></a>AMCT</span> kv-cache量化的简易配置。</p>
</td>
</tr>
<tr id="zh-cn_topic_0240188735_row9255123311566"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p11249539520"><a name="p11249539520"></a><a name="p11249539520"></a>optional</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p19715115505"><a name="p19715115505"></a><a name="p19715115505"></a>uint32</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="zh-cn_topic_0240188735_p11994171511911"><a name="zh-cn_topic_0240188735_p11994171511911"></a><a name="zh-cn_topic_0240188735_p11994171511911"></a>batch_num</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p960514816581"><a name="p960514816581"></a><a name="p960514816581"></a>量化使用的batch数量，用于ifmr/hfmg量化算法积累数据计算量化因子。</p>
</td>
</tr>
<tr id="zh-cn_topic_0240188735_row11255533205611"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p142538635219"><a name="p142538635219"></a><a name="p142538635219"></a>optional</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p13170131414515"><a name="p13170131414515"></a><a name="p13170131414515"></a>bool</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p678714373177"><a name="p678714373177"></a><a name="p678714373177"></a>activation_offset</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p2179152617"><a name="p2179152617"></a><a name="p2179152617"></a>数据量化是否带offset。全局配置参数。</p>
<a name="ul1970216412385"></a><a name="ul1970216412385"></a><ul id="ul1970216412385"><li>true：带offset，数据量化时为非对称量化。</li><li>false：不带offset，数据量化时为对称量化。</li></ul>
</td>
</tr>
<tr id="row410750183617"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p998581133711"><a name="p998581133711"></a><a name="p998581133711"></a>optional</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p7248162531716"><a name="p7248162531716"></a><a name="p7248162531716"></a>CommonCalibrationConfig</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p1324719253175"><a name="p1324719253175"></a><a name="p1324719253175"></a>kv_cache_quant_config</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p2797259134011"><a name="p2797259134011"></a><a name="p2797259134011"></a>通用的kv-cache量化配置，全局量化配置参数。若某层未被override_layer_configs重写，则使用该配置。</p>
<p id="p31455545543"><a name="p31455545543"></a><a name="p31455545543"></a>参数优先级：override_layer_configs&gt;kv_cache_quant_config</p>
</td>
</tr>
<tr id="row116751031143112"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p62711337163112"><a name="p62711337163112"></a><a name="p62711337163112"></a>repeated</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p162461925131713"><a name="p162461925131713"></a><a name="p162461925131713"></a>OverrideLayer</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p22461254170"><a name="p22461254170"></a><a name="p22461254170"></a>override_layers_configs</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p15798105944016"><a name="p15798105944016"></a><a name="p15798105944016"></a>重写某一层的量化配置，即对哪些层进行差异化量化。</p>
<p id="p4394114710214"><a name="p4394114710214"></a><a name="p4394114710214"></a>例如全局量化配置参数配置的量化因子搜索步长为0.01，可以通过该参数对部分层进行差异化量化，可以配置搜索步长为0.02。</p>
<p id="p194543485613"><a name="p194543485613"></a><a name="p194543485613"></a>参数优先级：override_layers_configs&gt;kv_cache_quant_config</p>
</td>
</tr>
<tr id="row344417284313"><td class="cellrowborder" rowspan="3" valign="top" width="11.148885111488852%" headers="mcps1.2.6.1.1 "><p id="p125293127186"><a name="p125293127186"></a><a name="p125293127186"></a>CommonCalibrationConfig</p>
</td>
<td class="cellrowborder" valign="top" width="6.52934706529347%" headers="mcps1.2.6.1.2 "><p id="p992417717189"><a name="p992417717189"></a><a name="p992417717189"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="13.628637136286374%" headers="mcps1.2.6.1.3 "><p id="p8244725171713"><a name="p8244725171713"></a><a name="p8244725171713"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="14.558544145585442%" headers="mcps1.2.6.1.4 "><p id="p62441325171717"><a name="p62441325171717"></a><a name="p62441325171717"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="54.13458654134586%" headers="mcps1.2.6.1.5 "><p id="p3243142510173"><a name="p3243142510173"></a><a name="p3243142510173"></a>通用的kv-cache量化配置。</p>
</td>
</tr>
<tr id="row888883816479"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p117112916181"><a name="p117112916181"></a><a name="p117112916181"></a>repeated</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p92427253175"><a name="p92427253175"></a><a name="p92427253175"></a>string</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p82423258171"><a name="p82423258171"></a><a name="p82423258171"></a>quant_layers</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p9241202541712"><a name="p9241202541712"></a><a name="p9241202541712"></a>支持量化的层名。</p>
</td>
</tr>
<tr id="row1422161319377"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p1292214716187"><a name="p1292214716187"></a><a name="p1292214716187"></a>optional</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p724072517172"><a name="p724072517172"></a><a name="p724072517172"></a>CalibrationConfig</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p132401225101718"><a name="p132401225101718"></a><a name="p132401225101718"></a>calibration_config</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p6238122591720"><a name="p6238122591720"></a><a name="p6238122591720"></a>量化配置。</p>
</td>
</tr>
<tr id="row198411100379"><td class="cellrowborder" rowspan="3" valign="top" width="11.148885111488852%" headers="mcps1.2.6.1.1 "><p id="p152911128188"><a name="p152911128188"></a><a name="p152911128188"></a>OverrideLayer</p>
</td>
<td class="cellrowborder" valign="top" width="6.52934706529347%" headers="mcps1.2.6.1.2 "><p id="p159221876187"><a name="p159221876187"></a><a name="p159221876187"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="13.628637136286374%" headers="mcps1.2.6.1.3 "><p id="p723814256176"><a name="p723814256176"></a><a name="p723814256176"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="14.558544145585442%" headers="mcps1.2.6.1.4 "><p id="p11237182514173"><a name="p11237182514173"></a><a name="p11237182514173"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="54.13458654134586%" headers="mcps1.2.6.1.5 "><p id="p723716255175"><a name="p723716255175"></a><a name="p723716255175"></a>重置某层量化配置。</p>
</td>
</tr>
<tr id="row73596813715"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p74351727121918"><a name="p74351727121918"></a><a name="p74351727121918"></a>repeated</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p10435112714196"><a name="p10435112714196"></a><a name="p10435112714196"></a>string</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p8235162551713"><a name="p8235162551713"></a><a name="p8235162551713"></a>layer_name</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p1423422551720"><a name="p1423422551720"></a><a name="p1423422551720"></a>被重置层的层名。</p>
</td>
</tr>
<tr id="row145071332133914"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p8435112711193"><a name="p8435112711193"></a><a name="p8435112711193"></a>optional</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p1043514272198"><a name="p1043514272198"></a><a name="p1043514272198"></a>CalibrationConfig</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p323302581719"><a name="p323302581719"></a><a name="p323302581719"></a>kv_data_quant_config</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p1923132512172"><a name="p1923132512172"></a><a name="p1923132512172"></a>重写的kv_cache量化配置参数。</p>
</td>
</tr>
<tr id="row1078145415203"><td class="cellrowborder" rowspan="3" valign="top" width="11.148885111488852%" headers="mcps1.2.6.1.1 "><p id="p77815542206"><a name="p77815542206"></a><a name="p77815542206"></a>CalibrationConfig</p>
<p id="p1778945632015"><a name="p1778945632015"></a><a name="p1778945632015"></a></p>
<p id="p327717252114"><a name="p327717252114"></a><a name="p327717252114"></a></p>
</td>
<td class="cellrowborder" valign="top" width="6.52934706529347%" headers="mcps1.2.6.1.2 "><p id="p134858254503"><a name="p134858254503"></a><a name="p134858254503"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="13.628637136286374%" headers="mcps1.2.6.1.3 "><p id="p1818261455119"><a name="p1818261455119"></a><a name="p1818261455119"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="14.558544145585442%" headers="mcps1.2.6.1.4 "><p id="zh-cn_topic_0240188735_p1363332811232"><a name="zh-cn_topic_0240188735_p1363332811232"></a><a name="zh-cn_topic_0240188735_p1363332811232"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="54.13458654134586%" headers="mcps1.2.6.1.5 "><p id="zh-cn_topic_0240188735_p363392862320"><a name="zh-cn_topic_0240188735_p363392862320"></a><a name="zh-cn_topic_0240188735_p363392862320"></a>Calibration量化的配置。</p>
</td>
</tr>
<tr id="row137881562201"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p1548512525015"><a name="p1548512525015"></a><a name="p1548512525015"></a>-</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p1183191465111"><a name="p1183191465111"></a><a name="p1183191465111"></a>FMRQuantize</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p425415411015"><a name="p425415411015"></a><a name="p425415411015"></a>ifmr_quantize</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="zh-cn_topic_0240188735_p1788134616316"><a name="zh-cn_topic_0240188735_p1788134616316"></a><a name="zh-cn_topic_0240188735_p1788134616316"></a>数据量化算法配置。</p>
<p id="zh-cn_topic_0240188735_p186119548317"><a name="zh-cn_topic_0240188735_p186119548317"></a><a name="zh-cn_topic_0240188735_p186119548317"></a>ifmr_quantize：IFMR量化算法配置。</p>
</td>
</tr>
<tr id="row127716222118"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p123369144483"><a name="p123369144483"></a><a name="p123369144483"></a>-</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p1233691404813"><a name="p1233691404813"></a><a name="p1233691404813"></a>HFMGQuantize</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p1833601494810"><a name="p1833601494810"></a><a name="p1833601494810"></a>hfmg_quantize</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p13337191454814"><a name="p13337191454814"></a><a name="p13337191454814"></a>数据量化算法配置。</p>
<p id="p8424174634818"><a name="p8424174634818"></a><a name="p8424174634818"></a>hfmg_quantize：HFMG量化算法配置。</p>
</td>
</tr>
<tr id="row81951911192118"><td class="cellrowborder" rowspan="8" valign="top" width="11.148885111488852%" headers="mcps1.2.6.1.1 "><p id="p219581111210"><a name="p219581111210"></a><a name="p219581111210"></a>FMRQuantize</p>
</td>
<td class="cellrowborder" valign="top" width="6.52934706529347%" headers="mcps1.2.6.1.2 "><p id="p144853253508"><a name="p144853253508"></a><a name="p144853253508"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="13.628637136286374%" headers="mcps1.2.6.1.3 "><p id="p918371455114"><a name="p918371455114"></a><a name="p918371455114"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="14.558544145585442%" headers="mcps1.2.6.1.4 "><p id="zh-cn_topic_0240188735_p1984611242417"><a name="zh-cn_topic_0240188735_p1984611242417"></a><a name="zh-cn_topic_0240188735_p1984611242417"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="54.13458654134586%" headers="mcps1.2.6.1.5 "><p id="zh-cn_topic_0240188735_p58460292417"><a name="zh-cn_topic_0240188735_p58460292417"></a><a name="zh-cn_topic_0240188735_p58460292417"></a>FMR数据量化算法配置。算法介绍请参见<a href="IFMR数据量化算法.md">IFMR数据量化算法</a>。</p>
<p id="p1877244707"><a name="p1877244707"></a><a name="p1877244707"></a>该参数与HFMGQuantize参数不能同时配置。</p>
</td>
</tr>
<tr id="row18756198132116"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p129891923173117"><a name="p129891923173117"></a><a name="p129891923173117"></a>optional</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p3183111418515"><a name="p3183111418515"></a><a name="p3183111418515"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="zh-cn_topic_0240188735_p20300132763310"><a name="zh-cn_topic_0240188735_p20300132763310"></a><a name="zh-cn_topic_0240188735_p20300132763310"></a>search_range_start</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="zh-cn_topic_0240188735_p11300627183317"><a name="zh-cn_topic_0240188735_p11300627183317"></a><a name="zh-cn_topic_0240188735_p11300627183317"></a>量化因子搜索范围左边界。</p>
</td>
</tr>
<tr id="row8394768212"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p182317254611"><a name="p182317254611"></a><a name="p182317254611"></a>optional</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p118237212462"><a name="p118237212462"></a><a name="p118237212462"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="zh-cn_topic_0240188735_p3189232123318"><a name="zh-cn_topic_0240188735_p3189232123318"></a><a name="zh-cn_topic_0240188735_p3189232123318"></a>search_range_end</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="zh-cn_topic_0240188735_p1954495473717"><a name="zh-cn_topic_0240188735_p1954495473717"></a><a name="zh-cn_topic_0240188735_p1954495473717"></a>量化因子搜索范围右边界。</p>
</td>
</tr>
<tr id="row162021072215"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p1861918118479"><a name="p1861918118479"></a><a name="p1861918118479"></a>optional</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p1461910118476"><a name="p1461910118476"></a><a name="p1461910118476"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="zh-cn_topic_0240188735_p13464113615338"><a name="zh-cn_topic_0240188735_p13464113615338"></a><a name="zh-cn_topic_0240188735_p13464113615338"></a>search_step</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="zh-cn_topic_0240188735_p104647369332"><a name="zh-cn_topic_0240188735_p104647369332"></a><a name="zh-cn_topic_0240188735_p104647369332"></a>量化因子搜索步长。</p>
</td>
</tr>
<tr id="row1921810122214"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p1861412254714"><a name="p1861412254714"></a><a name="p1861412254714"></a>optional</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p061418264717"><a name="p061418264717"></a><a name="p061418264717"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="zh-cn_topic_0240188735_p1810983233616"><a name="zh-cn_topic_0240188735_p1810983233616"></a><a name="zh-cn_topic_0240188735_p1810983233616"></a>max_percentile</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="zh-cn_topic_0240188735_p161093322365"><a name="zh-cn_topic_0240188735_p161093322365"></a><a name="zh-cn_topic_0240188735_p161093322365"></a>最大值搜索位置。</p>
</td>
</tr>
<tr id="row3211102225"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p1075110319475"><a name="p1075110319475"></a><a name="p1075110319475"></a>optional</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p117511331479"><a name="p117511331479"></a><a name="p117511331479"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="zh-cn_topic_0240188735_p1060420421361"><a name="zh-cn_topic_0240188735_p1060420421361"></a><a name="zh-cn_topic_0240188735_p1060420421361"></a>min_percentile</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="zh-cn_topic_0240188735_p186041242183616"><a name="zh-cn_topic_0240188735_p186041242183616"></a><a name="zh-cn_topic_0240188735_p186041242183616"></a>最小值搜索位置。</p>
</td>
</tr>
<tr id="row10253121413224"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p2876102645420"><a name="p2876102645420"></a><a name="p2876102645420"></a>optional</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p16876102655416"><a name="p16876102655416"></a><a name="p16876102655416"></a>bool</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p136291318105419"><a name="p136291318105419"></a><a name="p136291318105419"></a>asymmetric</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p18570132515213"><a name="p18570132515213"></a><a name="p18570132515213"></a>是否进行对称量化。用于控制逐层量化算法的选择。</p>
<a name="ul78601245115510"></a><a name="ul78601245115510"></a><ul id="ul78601245115510"><li>true：非对称量化</li><li>false：对称量化</li></ul>
<p id="p1784404618555"><a name="p1784404618555"></a><a name="p1784404618555"></a>如果override_layer_configs、common_config配置项都配置该参数，或者配置了activation_offset参数，则生效优先级为：</p>
<p id="p1630711108390"><a name="p1630711108390"></a><a name="p1630711108390"></a>override_layer_configs&gt;common_config&gt;activation_offset</p>
</td>
</tr>
<tr id="row9254201412218"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p15647105044715"><a name="p15647105044715"></a><a name="p15647105044715"></a>optional</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p12647350104719"><a name="p12647350104719"></a><a name="p12647350104719"></a>QuantGranularity</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p1764710503474"><a name="p1764710503474"></a><a name="p1764710503474"></a>quant_granularity</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p15112312194416"><a name="p15112312194416"></a><a name="p15112312194416"></a>量化粒度，支持如下两种方式：</p>
<a name="ul14112141214414"></a><a name="ul14112141214414"></a><ul id="ul14112141214414"><li>0：per_tensor，默认为0。</li><li>1：per_channel。</li></ul>
</td>
</tr>
<tr id="row1525501452218"><td class="cellrowborder" rowspan="4" valign="top" width="11.148885111488852%" headers="mcps1.2.6.1.1 "><p id="p162556143226"><a name="p162556143226"></a><a name="p162556143226"></a>HFMGQuantize</p>
</td>
<td class="cellrowborder" valign="top" width="6.52934706529347%" headers="mcps1.2.6.1.2 "><p id="p56910313515"><a name="p56910313515"></a><a name="p56910313515"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="13.628637136286374%" headers="mcps1.2.6.1.3 "><p id="p146916335113"><a name="p146916335113"></a><a name="p146916335113"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="14.558544145585442%" headers="mcps1.2.6.1.4 "><p id="p196910310511"><a name="p196910310511"></a><a name="p196910310511"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="54.13458654134586%" headers="mcps1.2.6.1.5 "><p id="p16989132195213"><a name="p16989132195213"></a><a name="p16989132195213"></a>HFMG数据量化算法配置。算法介绍请参见<a href="HFMG数据量化算法.md">HFMG数据量化算法</a>。</p>
<p id="p0610538149"><a name="p0610538149"></a><a name="p0610538149"></a>该参数与FMRQuantize参数不能同时配置。</p>
</td>
</tr>
<tr id="row5137512182316"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p3776259496"><a name="p3776259496"></a><a name="p3776259496"></a>optional</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p1777172594910"><a name="p1777172594910"></a><a name="p1777172594910"></a>uint32</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p1277122516495"><a name="p1277122516495"></a><a name="p1277122516495"></a>num_of_bins</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p14190312023"><a name="p14190312023"></a><a name="p14190312023"></a>直方图的bin（直方图中的一个最小单位直方图形）数目，支持的范围为{1024, 2048, 4096, 8192}。</p>
<p id="p1772259498"><a name="p1772259498"></a><a name="p1772259498"></a>默认值为4096。</p>
</td>
</tr>
<tr id="row5138101202316"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p1759374312114"><a name="p1759374312114"></a><a name="p1759374312114"></a>optional</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p10593144314119"><a name="p10593144314119"></a><a name="p10593144314119"></a>bool</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p359316431317"><a name="p359316431317"></a><a name="p359316431317"></a>asymmetric</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p1240373012612"><a name="p1240373012612"></a><a name="p1240373012612"></a>是否进行对称量化。用于控制逐层量化算法的选择。</p>
<a name="ul812310517497"></a><a name="ul812310517497"></a><ul id="ul812310517497"><li>true：非对称量化</li><li>false：对称量化</li></ul>
<p id="p64035306619"><a name="p64035306619"></a><a name="p64035306619"></a>如果override_layer_configs、common_config配置项都配置该参数，或者配置了</p>
<p id="p10403430769"><a name="p10403430769"></a><a name="p10403430769"></a>activation_offset参数，则生效优先级为：</p>
<p id="p8403163010613"><a name="p8403163010613"></a><a name="p8403163010613"></a>override_layer_configs&gt;common_config&gt;activation_offset</p>
</td>
</tr>
<tr id="row813881252319"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p121262037155118"><a name="p121262037155118"></a><a name="p121262037155118"></a>optional</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p485016249256"><a name="p485016249256"></a><a name="p485016249256"></a>QuantGranularity</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p13850162432518"><a name="p13850162432518"></a><a name="p13850162432518"></a>quant_granularity</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p47171749181715"><a name="p47171749181715"></a><a name="p47171749181715"></a>量化粒度，支持如下两种方式：</p>
<a name="ul740516543017"></a><a name="ul740516543017"></a><ul id="ul740516543017"><li>0：per_tensor，默认为0。</li><li>1：per_channel。</li></ul>
</td>
</tr>
</tbody>
</table>

基于该文件构造的**kv-cache量化简易配置文件**_quant_.cfg样例如下所示：

```
kv_cache_quant_config {
    quant_layers: 'matmul1'
    quant_layers: 'matmul2'
    calibration_config: {
        hfmg_quantize : {
        }
    }
}

override_layers_configs {
    layer_name: 'matmul3'
    kv_data_quant_config: {
        ifmr_quantize : {
        }
    }
}
```

