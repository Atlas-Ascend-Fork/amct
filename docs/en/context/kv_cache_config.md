# Simple Configuration File for KV Cache Quantization<a name="ZH-CN_TOPIC_0000002517028718"></a>

<!-- md-trans-meta sourceCommit=57c5f99db95a0bc1b70fa91a52c1a2c25d59ea20 translatedAt=2026-10-16T08:43:29.118Z pushedAt=2026-11-05T01:12:03.432Z nl=2 -->

The following table describes parameters in the **quant\_calibration\_config\_pytorch.proto** file. The file is located in the ***AMCT installation directory*/amct_pytorch/proto/**.

**Table 1**  Parameters in quant\_calibration\_config\_pytorch.proto

<a name="zh-cn_topic_0240188735_table1225503375617"></a>
<table><thead align="left"><tr id="zh-cn_topic_0240188735_row11255153315614"><th class="cellrowborder" valign="top" width="11.148885111488852%" id="mcps1.2.6.1.1"><p id="zh-cn_topic_0240188735_p6255193315616"><a name="zh-cn_topic_0240188735_p6255193315616"></a><a name="zh-cn_topic_0240188735_p6255193315616"></a>Message</p></th>
<th class="cellrowborder" valign="top" width="6.52934706529347%" id="mcps1.2.6.1.2"><p id="p194851125105014"><a name="p194851125105014"></a><a name="p194851125105014"></a>Required</p></th>
<th class="cellrowborder" valign="top" width="13.628637136286374%" id="mcps1.2.6.1.3"><p id="p466892519514"><a name="p466892519514"></a><a name="p466892519514"></a>Type</p></th>
<th class="cellrowborder" valign="top" width="14.558544145585442%" id="mcps1.2.6.1.4"><p id="zh-cn_topic_0240188735_p499481531910"><a name="zh-cn_topic_0240188735_p499481531910"></a><a name="zh-cn_topic_0240188735_p499481531910"></a>Field</p></th>
<th class="cellrowborder" valign="top" width="54.13458654134586%" id="mcps1.2.6.1.5"><p id="zh-cn_topic_0240188735_p122551335563"><a name="zh-cn_topic_0240188735_p122551335563"></a><a name="zh-cn_topic_0240188735_p122551335563"></a>Description</p></th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0240188735_row122551133185610"><td class="cellrowborder" rowspan="5" valign="top" width="11.148885111488852%" headers="mcps1.2.6.1.1 "><p id="zh-cn_topic_0240188735_p1925513338569"><a name="zh-cn_topic_0240188735_p1925513338569"></a><a name="zh-cn_topic_0240188735_p1925513338569"></a>AMCTQuantCaliConfig</p></td>
<td class="cellrowborder" valign="top" width="6.52934706529347%" headers="mcps1.2.6.1.2 "><p id="p1485162505012"><a name="p1485162505012"></a><a name="p1485162505012"></a>-</p></td>
<td class="cellrowborder" valign="top" width="13.628637136286374%" headers="mcps1.2.6.1.3 "><p id="p1617031410516"><a name="p1617031410516"></a><a name="p1617031410516"></a>-</p></td>
<td class="cellrowborder" valign="top" width="14.558544145585442%" headers="mcps1.2.6.1.4 "><p id="zh-cn_topic_0240188735_p2994161514196"><a name="zh-cn_topic_0240188735_p2994161514196"></a><a name="zh-cn_topic_0240188735_p2994161514196"></a>-</p></td>
<td class="cellrowborder" valign="top" width="54.13458654134586%" headers="mcps1.2.6.1.5 "><p id="zh-cn_topic_0240188735_p82551633105614"><a name="zh-cn_topic_0240188735_p82551633105614"></a><a name="zh-cn_topic_0240188735_p82551633105614"></a><span id="ph244511481409"><a name="ph244511481409"></a><a name="ph244511481409"></a>AMCT</span> simple configuration for kv-cache quantization.</p></td>
</tr>
<tr id="zh-cn_topic_0240188735_row9255123311566"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p11249539520"><a name="p11249539520"></a><a name="p11249539520"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p19715115505"><a name="p19715115505"></a><a name="p19715115505"></a>uint32</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="zh-cn_topic_0240188735_p11994171511911"><a name="zh-cn_topic_0240188735_p11994171511911"></a><a name="zh-cn_topic_0240188735_p11994171511911"></a>batch_num</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p960514816581"><a name="p960514816581"></a><a name="p960514816581"></a>Number of batches used for quantization, used by the ifmr/hfmg quantization algorithm to accumulate data for calculating the quantization factor.</p></td>
</tr>
<tr id="zh-cn_topic_0240188735_row11255533205611"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p142538635219"><a name="p142538635219"></a><a name="p142538635219"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p13170131414515"><a name="p13170131414515"></a><a name="p13170131414515"></a>bool</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p678714373177"><a name="p678714373177"></a><a name="p678714373177"></a>activation_offset</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p2179152617"><a name="p2179152617"></a><a name="p2179152617"></a>Whether data quantization carries an offset. Global configuration parameter.</p>
<a name="ul1970216412385"></a><a name="ul1970216412385"></a><ul id="ul1970216412385"><li>true: Carries an offset, and data quantization uses asymmetric quantization.</li><li>false: Does not carry an offset, and data quantization uses symmetric quantization.</li></ul></td>
</tr>
<tr id="row410750183617"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p998581133711"><a name="p998581133711"></a><a name="p998581133711"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p7248162531716"><a name="p7248162531716"></a><a name="p7248162531716"></a>CommonCalibrationConfig</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p1324719253175"><a name="p1324719253175"></a><a name="p1324719253175"></a>kv_cache_quant_config</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p2797259134011"><a name="p2797259134011"></a><a name="p2797259134011"></a>General kv-cache quantization configuration, a global quantization configuration parameter. If a layer is not overridden by override_layer_configs, this configuration is used.</p>
<p id="p31455545543"><a name="p31455545543"></a><a name="p31455545543"></a>Parameter priority: override_layer_configs &gt; kv_cache_quant_config</p></td>
</tr>
<tr id="row116751031143112"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p62711337163112"><a name="p62711337163112"></a><a name="p62711337163112"></a>Repeated</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p162461925131713"><a name="p162461925131713"></a><a name="p162461925131713"></a>OverrideLayer</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p22461254170"><a name="p22461254170"></a><a name="p22461254170"></a>override_layers_configs</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p15798105944016"><a name="p15798105944016"></a><a name="p15798105944016"></a>Overrides the quantization configuration of a specific layer, that is, applies differentiated quantization to certain layers.</p>
<p id="p4394114710214"><a name="p4394114710214"></a><a name="p4394114710214"></a>For example, if the global quantization configuration parameter sets the quantization factor search step to 0.01, this parameter can be used to apply differentiated quantization to some layers, such as by setting the search step to 0.02.</p>
<p id="p194543485613"><a name="p194543485613"></a><a name="p194543485613"></a>Parameter priority: override_layers_configs &gt; kv_cache_quant_config</p></td>
</tr>
<tr id="row344417284313"><td class="cellrowborder" rowspan="3" valign="top" width="11.148885111488852%" headers="mcps1.2.6.1.1 "><p id="p125293127186"><a name="p125293127186"></a><a name="p125293127186"></a>CommonCalibrationConfig</p></td>
<td class="cellrowborder" valign="top" width="6.52934706529347%" headers="mcps1.2.6.1.2 "><p id="p992417717189"><a name="p992417717189"></a><a name="p992417717189"></a>-</p></td>
<td class="cellrowborder" valign="top" width="13.628637136286374%" headers="mcps1.2.6.1.3 "><p id="p8244725171713"><a name="p8244725171713"></a><a name="p8244725171713"></a>-</p></td>
<td class="cellrowborder" valign="top" width="14.558544145585442%" headers="mcps1.2.6.1.4 "><p id="p62441325171717"><a name="p62441325171717"></a><a name="p62441325171717"></a>-</p></td>
<td class="cellrowborder" valign="top" width="54.13458654134586%" headers="mcps1.2.6.1.5 "><p id="p3243142510173"><a name="p3243142510173"></a><a name="p3243142510173"></a>General kv-cache quantization configuration.</p></td>
</tr>
<tr id="row888883816479"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p117112916181"><a name="p117112916181"></a><a name="p117112916181"></a>Repeated</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p92427253175"><a name="p92427253175"></a><a name="p92427253175"></a>string</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p82423258171"><a name="p82423258171"></a><a name="p82423258171"></a>quant_layers</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p9241202541712"><a name="p9241202541712"></a><a name="p9241202541712"></a>Names of layers that support quantization.</p></td>
</tr>
<tr id="row1422161319377"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p1292214716187"><a name="p1292214716187"></a><a name="p1292214716187"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p724072517172"><a name="p724072517172"></a><a name="p724072517172"></a>CalibrationConfig</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p132401225101718"><a name="p132401225101718"></a><a name="p132401225101718"></a>calibration_config</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p6238122591720"><a name="p6238122591720"></a><a name="p6238122591720"></a>Quantization configuration.</p></td>
</tr>
<tr id="row198411100379"><td class="cellrowborder" rowspan="3" valign="top" width="11.148885111488852%" headers="mcps1.2.6.1.1 "><p id="p152911128188"><a name="p152911128188"></a><a name="p152911128188"></a>OverrideLayer</p></td>
<td class="cellrowborder" valign="top" width="6.52934706529347%" headers="mcps1.2.6.1.2 "><p id="p159221876187"><a name="p159221876187"></a><a name="p159221876187"></a>-</p></td>
<td class="cellrowborder" valign="top" width="13.628637136286374%" headers="mcps1.2.6.1.3 "><p id="p723814256176"><a name="p723814256176"></a><a name="p723814256176"></a>-</p></td>
<td class="cellrowborder" valign="top" width="14.558544145585442%" headers="mcps1.2.6.1.4 "><p id="p11237182514173"><a name="p11237182514173"></a><a name="p11237182514173"></a>-</p></td>
<td class="cellrowborder" valign="top" width="54.13458654134586%" headers="mcps1.2.6.1.5 "><p id="p723716255175"><a name="p723716255175"></a><a name="p723716255175"></a>Resets the quantization configuration of a layer.</p></td>
</tr>
<tr id="row73596813715"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p74351727121918"><a name="p74351727121918"></a><a name="p74351727121918"></a>Repeated</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p10435112714196"><a name="p10435112714196"></a><a name="p10435112714196"></a>string</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p8235162551713"><a name="p8235162551713"></a><a name="p8235162551713"></a>layer_name</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p1423422551720"><a name="p1423422551720"></a><a name="p1423422551720"></a>Name of the layer to be reset.</p></td>
</tr>
<tr id="row145071332133914"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p8435112711193"><a name="p8435112711193"></a><a name="p8435112711193"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p1043514272198"><a name="p1043514272198"></a><a name="p1043514272198"></a>CalibrationConfig</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p323302581719"><a name="p323302581719"></a><a name="p323302581719"></a>kv_data_quant_config</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p1923132512172"><a name="p1923132512172"></a><a name="p1923132512172"></a>Rewritten kv_cache quantization configuration parameters.</p></td>
</tr>
<tr id="row1078145415203"><td class="cellrowborder" rowspan="3" valign="top" width="11.148885111488852%" headers="mcps1.2.6.1.1 "><p id="p77815542206"><a name="p77815542206"></a><a name="p77815542206"></a>CalibrationConfig</p>
<p id="p1778945632015"><a name="p1778945632015"></a><a name="p1778945632015"></a></p>
<p id="p327717252114"><a name="p327717252114"></a><a name="p327717252114"></a></p></td>
<td class="cellrowborder" valign="top" width="6.52934706529347%" headers="mcps1.2.6.1.2 "><p id="p134858254503"><a name="p134858254503"></a><a name="p134858254503"></a>-</p></td>
<td class="cellrowborder" valign="top" width="13.628637136286374%" headers="mcps1.2.6.1.3 "><p id="p1818261455119"><a name="p1818261455119"></a><a name="p1818261455119"></a>-</p></td>
<td class="cellrowborder" valign="top" width="14.558544145585442%" headers="mcps1.2.6.1.4 "><p id="zh-cn_topic_0240188735_p1363332811232"><a name="zh-cn_topic_0240188735_p1363332811232"></a><a name="zh-cn_topic_0240188735_p1363332811232"></a>-</p></td>
<td class="cellrowborder" valign="top" width="54.13458654134586%" headers="mcps1.2.6.1.5 "><p id="zh-cn_topic_0240188735_p363392862320"><a name="zh-cn_topic_0240188735_p363392862320"></a><a name="zh-cn_topic_0240188735_p363392862320"></a>Calibration quantization configuration.</p></td>
</tr>
<tr id="row137881562201"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p1548512525015"><a name="p1548512525015"></a><a name="p1548512525015"></a>-</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p1183191465111"><a name="p1183191465111"></a><a name="p1183191465111"></a>FMRQuantize</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p425415411015"><a name="p425415411015"></a><a name="p425415411015"></a>ifmr_quantize</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="zh-cn_topic_0240188735_p1788134616316"><a name="zh-cn_topic_0240188735_p1788134616316"></a><a name="zh-cn_topic_0240188735_p1788134616316"></a>Data quantization algorithm configuration.</p>
<p id="zh-cn_topic_0240188735_p186119548317"><a name="zh-cn_topic_0240188735_p186119548317"></a><a name="zh-cn_topic_0240188735_p186119548317"></a>ifmr_quantize: IFMR quantization algorithm configuration.</p></td>
</tr>
<tr id="row127716222118"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p123369144483"><a name="p123369144483"></a><a name="p123369144483"></a>-</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p1233691404813"><a name="p1233691404813"></a><a name="p1233691404813"></a>HFMGQuantize</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p1833601494810"><a name="p1833601494810"></a><a name="p1833601494810"></a>hfmg_quantize</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p13337191454814"><a name="p13337191454814"></a><a name="p13337191454814"></a>Data quantization algorithm configuration.</p>
<p id="p8424174634818"><a name="p8424174634818"></a><a name="p8424174634818"></a>hfmg_quantize: HFMG quantization algorithm configuration.</p></td>
</tr>
<tr id="row81951911192118"><td class="cellrowborder" rowspan="8" valign="top" width="11.148885111488852%" headers="mcps1.2.6.1.1 "><p id="p219581111210"><a name="p219581111210"></a><a name="p219581111210"></a>FMRQuantize</p></td>
<td class="cellrowborder" valign="top" width="6.52934706529347%" headers="mcps1.2.6.1.2 "><p id="p144853253508"><a name="p144853253508"></a><a name="p144853253508"></a>-</p></td>
<td class="cellrowborder" valign="top" width="13.628637136286374%" headers="mcps1.2.6.1.3 "><p id="p918371455114"><a name="p918371455114"></a><a name="p918371455114"></a>-</p></td>
<td class="cellrowborder" valign="top" width="14.558544145585442%" headers="mcps1.2.6.1.4 "><p id="zh-cn_topic_0240188735_p1984611242417"><a name="zh-cn_topic_0240188735_p1984611242417"></a><a name="zh-cn_topic_0240188735_p1984611242417"></a>-</p></td>
<td class="cellrowborder" valign="top" width="54.13458654134586%" headers="mcps1.2.6.1.5 "><p id="zh-cn_topic_0240188735_p58460292417"><a name="zh-cn_topic_0240188735_p58460292417"></a><a name="zh-cn_topic_0240188735_p58460292417"></a>FMR data quantization algorithm configuration. For details about the algorithm, see <a href="../algorithm_brief.md#ifmr-data-quantization-algorithm">IFMR data quantization algorithm</a>.</p>
<p id="p1877244707"><a name="p1877244707"></a><a name="p1877244707"></a>This parameter cannot be configured together with the HFMGQuantize parameter.</p></td>
</tr>
<tr id="row18756198132116"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p129891923173117"><a name="p129891923173117"></a><a name="p129891923173117"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p3183111418515"><a name="p3183111418515"></a><a name="p3183111418515"></a>float</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="zh-cn_topic_0240188735_p20300132763310"><a name="zh-cn_topic_0240188735_p20300132763310"></a><a name="zh-cn_topic_0240188735_p20300132763310"></a>search_range_start</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="zh-cn_topic_0240188735_p11300627183317"><a name="zh-cn_topic_0240188735_p11300627183317"></a><a name="zh-cn_topic_0240188735_p11300627183317"></a>Left boundary of the quantization factor search range.</p></td>
</tr>
<tr id="row8394768212"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p182317254611"><a name="p182317254611"></a><a name="p182317254611"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p118237212462"><a name="p118237212462"></a><a name="p118237212462"></a>float</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="zh-cn_topic_0240188735_p3189232123318"><a name="zh-cn_topic_0240188735_p3189232123318"></a><a name="zh-cn_topic_0240188735_p3189232123318"></a>search_range_end</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="zh-cn_topic_0240188735_p1954495473717"><a name="zh-cn_topic_0240188735_p1954495473717"></a><a name="zh-cn_topic_0240188735_p1954495473717"></a>Right boundary of the quantization factor search range.</p></td>
</tr>
<tr id="row162021072215"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p1861918118479"><a name="p1861918118479"></a><a name="p1861918118479"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p1461910118476"><a name="p1461910118476"></a><a name="p1461910118476"></a>float</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="zh-cn_topic_0240188735_p13464113615338"><a name="zh-cn_topic_0240188735_p13464113615338"></a><a name="zh-cn_topic_0240188735_p13464113615338"></a>search_step</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="zh-cn_topic_0240188735_p104647369332"><a name="zh-cn_topic_0240188735_p104647369332"></a><a name="zh-cn_topic_0240188735_p104647369332"></a>Quantization factor search step.</p></td>
</tr>
<tr id="row1921810122214"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p1861412254714"><a name="p1861412254714"></a><a name="p1861412254714"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p061418264717"><a name="p061418264717"></a><a name="p061418264717"></a>float</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="zh-cn_topic_0240188735_p1810983233616"><a name="zh-cn_topic_0240188735_p1810983233616"></a><a name="zh-cn_topic_0240188735_p1810983233616"></a>max_percentile</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="zh-cn_topic_0240188735_p161093322365"><a name="zh-cn_topic_0240188735_p161093322365"></a><a name="zh-cn_topic_0240188735_p161093322365"></a>Maximum value search position.</p></td>
</tr>
<tr id="row3211102225"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p1075110319475"><a name="p1075110319475"></a><a name="p1075110319475"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p117511331479"><a name="p117511331479"></a><a name="p117511331479"></a>float</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="zh-cn_topic_0240188735_p1060420421361"><a name="zh-cn_topic_0240188735_p1060420421361"></a><a name="zh-cn_topic_0240188735_p1060420421361"></a>min_percentile</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="zh-cn_topic_0240188735_p186041242183616"><a name="zh-cn_topic_0240188735_p186041242183616"></a><a name="zh-cn_topic_0240188735_p186041242183616"></a>Search position of the minimum value.</p></td>
</tr>
<tr id="row10253121413224"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p2876102645420"><a name="p2876102645420"></a><a name="p2876102645420"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p16876102655416"><a name="p16876102655416"></a><a name="p16876102655416"></a>bool</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p136291318105419"><a name="p136291318105419"></a><a name="p136291318105419"></a>asymmetric</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p18570132515213"><a name="p18570132515213"></a><a name="p18570132515213"></a>Whether to perform symmetric quantization. Used to control the selection of the per-layer quantization algorithm.</p>
<a name="ul78601245115510"></a><a name="ul78601245115510"></a><ul id="ul78601245115510"><li>true: asymmetric quantization</li><li>false: symmetric quantization</li></ul>
<p id="p1784404618555"><a name="p1784404618555"></a><a name="p1784404618555"></a>If this parameter is configured in both the override_layer_configs and common_config configuration items, or if the activation_offset parameter is configured, the effective priority is as follows:</p>
<p id="p1630711108390"><a name="p1630711108390"></a><a name="p1630711108390"></a>override_layer_configs&gt;common_config&gt;activation_offset</p></td>
</tr>
<tr id="row9254201412218"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p15647105044715"><a name="p15647105044715"></a><a name="p15647105044715"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p12647350104719"><a name="p12647350104719"></a><a name="p12647350104719"></a>QuantGranularity</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p1764710503474"><a name="p1764710503474"></a><a name="p1764710503474"></a>quant_granularity</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p15112312194416"><a name="p15112312194416"></a><a name="p15112312194416"></a>Quantization granularity, which supports the following two modes:</p>
<a name="ul14112141214414"></a><a name="ul14112141214414"></a><ul id="ul14112141214414"><li>0: per_tensor, defaulting to 0.</li><li>1: per_channel.</li></ul></td>
</tr>
<tr id="row1525501452218"><td class="cellrowborder" rowspan="4" valign="top" width="11.148885111488852%" headers="mcps1.2.6.1.1 "><p id="p162556143226"><a name="p162556143226"></a><a name="p162556143226"></a>HFMGQuantize</p></td>
<td class="cellrowborder" valign="top" width="6.52934706529347%" headers="mcps1.2.6.1.2 "><p id="p56910313515"><a name="p56910313515"></a><a name="p56910313515"></a>-</p></td>
<td class="cellrowborder" valign="top" width="13.628637136286374%" headers="mcps1.2.6.1.3 "><p id="p146916335113"><a name="p146916335113"></a><a name="p146916335113"></a>-</p></td>
<td class="cellrowborder" valign="top" width="14.558544145585442%" headers="mcps1.2.6.1.4 "><p id="p196910310511"><a name="p196910310511"></a><a name="p196910310511"></a>-</p></td>
<td class="cellrowborder" valign="top" width="54.13458654134586%" headers="mcps1.2.6.1.5 "><p id="p16989132195213"><a name="p16989132195213"></a><a name="p16989132195213"></a>HFMG data quantization algorithm configuration. For details about the algorithm, see <a href="../algorithm_brief.md#hfmg-data-quantization-algorithm">HFMG data quantization algorithm</a>.</p>
<p id="p0610538149"><a name="p0610538149"></a><a name="p0610538149"></a>This parameter cannot be configured together with the FMRQuantize parameter.</p></td>
</tr>
<tr id="row5137512182316"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p3776259496"><a name="p3776259496"></a><a name="p3776259496"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p1777172594910"><a name="p1777172594910"></a><a name="p1777172594910"></a>uint32</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p1277122516495"><a name="p1277122516495"></a><a name="p1277122516495"></a>num_of_bins</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p14190312023"><a name="p14190312023"></a><a name="p14190312023"></a>Number of bins in the histogram (the smallest unit in a histogram). The supported range is {1024, 2048, 4096, 8192}.</p>
<p id="p1772259498"><a name="p1772259498"></a><a name="p1772259498"></a>The default value is 4096.</p></td>
</tr>
<tr id="row5138101202316"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p1759374312114"><a name="p1759374312114"></a><a name="p1759374312114"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p10593144314119"><a name="p10593144314119"></a><a name="p10593144314119"></a>bool</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p359316431317"><a name="p359316431317"></a><a name="p359316431317"></a>asymmetric</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p1240373012612"><a name="p1240373012612"></a><a name="p1240373012612"></a>Whether to perform symmetric quantization. It controls the selection of the per-layer quantization algorithm.</p>
<a name="ul812310517497"></a><a name="ul812310517497"></a><ul id="ul812310517497"><li>true: asymmetric quantization</li><li>false: symmetric quantization</li></ul>
<p id="p64035306619"><a name="p64035306619"></a><a name="p64035306619"></a>If this parameter is configured in both override_layer_configs and common_config, or if the</p>
<p id="p10403430769"><a name="p10403430769"></a><a name="p10403430769"></a>activation_offset parameter is configured, the effective priority is:</p>
<p id="p8403163010613"><a name="p8403163010613"></a><a name="p8403163010613"></a>override_layer_configs &gt; common_config &gt; activation_offset</p></td>
</tr>
<tr id="row813881252319"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p121262037155118"><a name="p121262037155118"></a><a name="p121262037155118"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p485016249256"><a name="p485016249256"></a><a name="p485016249256"></a>QuantGranularity</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p13850162432518"><a name="p13850162432518"></a><a name="p13850162432518"></a>quant_granularity</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p47171749181715"><a name="p47171749181715"></a><a name="p47171749181715"></a>Quantization granularity. The following two modes are supported:</p>
<a name="ul740516543017"></a><a name="ul740516543017"></a><ul id="ul740516543017"><li>0: per_tensor, which is the default.</li><li>1: per_channel.</li></ul></td>
</tr>
</tbody>
</table>

The following is an example of the **kv-cache quantization simple configuration file** *quant*.cfg constructed based on this file:

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
