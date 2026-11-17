# Simple Configuration File for Post-Training Quantization<a name="ZH-CN_TOPIC_0000002548668511"></a>

<!-- md-trans-meta sourceCommit=57c5f99db95a0bc1b70fa91a52c1a2c25d59ea20 translatedAt=2026-10-16T08:45:56.307Z pushedAt=2026-11-05T01:12:03.434Z nl=2 -->

[Table 1](#zh-cn_topic_0240188735_table1225503375617) describes parameters in the **calibration\_config\_pytorch.proto** file. The file is located in ****AMCT installation directory**/amct\_pytorch/proto/calibration\_config\_pytorch.proto**.

**Table 1**  Parameters in calibration\_config\_pytorch.proto

<a name="zh-cn_topic_0240188735_table1225503375617"></a>
<table><thead align="left"><tr id="zh-cn_topic_0240188735_row11255153315614"><th class="cellrowborder" valign="top" width="11.148885111488852%" id="mcps1.2.6.1.1"><p id="zh-cn_topic_0240188735_p6255193315616"><a name="zh-cn_topic_0240188735_p6255193315616"></a><a name="zh-cn_topic_0240188735_p6255193315616"></a>Message</p></th>
<th class="cellrowborder" valign="top" width="6.52934706529347%" id="mcps1.2.6.1.2"><p id="p194851125105014"><a name="p194851125105014"></a><a name="p194851125105014"></a>Required</p></th>
<th class="cellrowborder" valign="top" width="11.188881111888811%" id="mcps1.2.6.1.3"><p id="p466892519514"><a name="p466892519514"></a><a name="p466892519514"></a>Type</p></th>
<th class="cellrowborder" valign="top" width="12.84871512848715%" id="mcps1.2.6.1.4"><p id="zh-cn_topic_0240188735_p499481531910"><a name="zh-cn_topic_0240188735_p499481531910"></a><a name="zh-cn_topic_0240188735_p499481531910"></a>Field</p></th>
<th class="cellrowborder" valign="top" width="58.284171582841715%" id="mcps1.2.6.1.5"><p id="zh-cn_topic_0240188735_p122551335563"><a name="zh-cn_topic_0240188735_p122551335563"></a><a name="zh-cn_topic_0240188735_p122551335563"></a>Description</p></th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0240188735_row122551133185610"><td class="cellrowborder" rowspan="11" valign="top" width="11.148885111488852%" headers="mcps1.2.6.1.1 "><p id="zh-cn_topic_0240188735_p1925513338569"><a name="zh-cn_topic_0240188735_p1925513338569"></a><a name="zh-cn_topic_0240188735_p1925513338569"></a>AMCTConfig</p></td>
<td class="cellrowborder" valign="top" width="6.52934706529347%" headers="mcps1.2.6.1.2 "><p id="p1485162505012"><a name="p1485162505012"></a><a name="p1485162505012"></a>-</p></td>
<td class="cellrowborder" valign="top" width="11.188881111888811%" headers="mcps1.2.6.1.3 "><p id="p1617031410516"><a name="p1617031410516"></a><a name="p1617031410516"></a>-</p></td>
<td class="cellrowborder" valign="top" width="12.84871512848715%" headers="mcps1.2.6.1.4 "><p id="zh-cn_topic_0240188735_p2994161514196"><a name="zh-cn_topic_0240188735_p2994161514196"></a><a name="zh-cn_topic_0240188735_p2994161514196"></a>-</p></td>
<td class="cellrowborder" valign="top" width="58.284171582841715%" headers="mcps1.2.6.1.5 "><p id="zh-cn_topic_0240188735_p82551633105614"><a name="zh-cn_topic_0240188735_p82551633105614"></a><a name="zh-cn_topic_0240188735_p82551633105614"></a><span id="ph244511481409"><a name="ph244511481409"></a><a name="ph244511481409"></a>AMCT</span> simple quantization configuration for post-training quantization.</p></td>
</tr>
<tr id="zh-cn_topic_0240188735_row9255123311566"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p11249539520"><a name="p11249539520"></a><a name="p11249539520"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p19715115505"><a name="p19715115505"></a><a name="p19715115505"></a>uint32</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="zh-cn_topic_0240188735_p11994171511911"><a name="zh-cn_topic_0240188735_p11994171511911"></a><a name="zh-cn_topic_0240188735_p11994171511911"></a>batch_num</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="zh-cn_topic_0240188735_p32551133175612"><a name="zh-cn_topic_0240188735_p32551133175612"></a><a name="zh-cn_topic_0240188735_p32551133175612"></a>Number of batches used for quantization.</p></td>
</tr>
<tr id="zh-cn_topic_0240188735_row11255533205611"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p142538635219"><a name="p142538635219"></a><a name="p142538635219"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p13170131414515"><a name="p13170131414515"></a><a name="p13170131414515"></a>bool</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="zh-cn_topic_0240188735_p1225516334564"><a name="zh-cn_topic_0240188735_p1225516334564"></a><a name="zh-cn_topic_0240188735_p1225516334564"></a>activation_offset</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p2179152617"><a name="p2179152617"></a><a name="p2179152617"></a>Whether data quantization uses an offset. Global configuration parameter.</p>
<a name="ul1970216412385"></a><a name="ul1970216412385"></a><ul id="ul1970216412385"><li>true: Uses an offset, and data quantization is asymmetric quantization.</li><li>false: Does not use an offset, and data quantization is symmetric quantization.</li></ul></td>
</tr>
<tr id="row116751031143112"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p62711337163112"><a name="p62711337163112"></a><a name="p62711337163112"></a>Repeated</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p16271153710317"><a name="p16271153710317"></a><a name="p16271153710317"></a>string</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p1527173716314"><a name="p1527173716314"></a><a name="p1527173716314"></a>skip_layers</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p6272113718317"><a name="p6272113718317"></a><a name="p6272113718317"></a>Names of layers that do not need to be quantized.</p></td>
</tr>
<tr id="row344417284313"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p1272173743111"><a name="p1272173743111"></a><a name="p1272173743111"></a>Repeated</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p17272193713114"><a name="p17272193713114"></a><a name="p17272193713114"></a>string</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p2272123719311"><a name="p2272123719311"></a><a name="p2272123719311"></a>skip_layer_types</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p627293783116"><a name="p627293783116"></a><a name="p627293783116"></a>Layer types that do not need to be quantized.</p></td>
</tr>
<tr id="row13308202915811"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p143081329145819"><a name="p143081329145819"></a><a name="p143081329145819"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p153081529135813"><a name="p153081529135813"></a><a name="p153081529135813"></a>FakequantPrecisionMode</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p930816292588"><a name="p930816292588"></a><a name="p930816292588"></a>fakequant_precision_mode</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p1254210571348"><a name="p1254210571348"></a><a name="p1254210571348"></a>Numeric precision mode of the scale_d parameter of the quant custom operator in the fakequant model.</p>
<a name="ul3401310203217"></a><a name="ul3401310203217"></a><ul id="ul3401310203217"><li>FORCE_FP16_QUANT: The value of the scale_d parameter is converted to float16 precision, and the value type is float32.</li><li>Empty, that is, not configured, which defaults to empty. The value of the scale_d parameter remains at float32 precision.</li></ul></td>
</tr>
<tr id="row1422161319377"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p6422191320375"><a name="p6422191320375"></a><a name="p6422191320375"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p134239134371"><a name="p134239134371"></a><a name="p134239134371"></a>CalibrationConfig</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p74231313103710"><a name="p74231313103710"></a><a name="p74231313103710"></a>common_config</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p2797259134011"><a name="p2797259134011"></a><a name="p2797259134011"></a>Common quantization configuration, that is, global quantization configuration parameters. If a layer is not overridden by override_layer_types or override_layer_configs, this configuration is used.</p>
<p id="p31455545543"><a name="p31455545543"></a><a name="p31455545543"></a>Parameter priority: override_layer_configs &gt; override_layer_types &gt; common_config</p></td>
</tr>
<tr id="row198411100379"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p10841410123712"><a name="p10841410123712"></a><a name="p10841410123712"></a>Repeated</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p1841151033713"><a name="p1841151033713"></a><a name="p1841151033713"></a>OverrideLayerType</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p6841191033712"><a name="p6841191033712"></a><a name="p6841191033712"></a>override_layer_types</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p979765994018"><a name="p979765994018"></a><a name="p979765994018"></a>Overrides the quantization configuration of a certain type of layer, that is, specifies which layers are subject to differentiated quantization.</p>
<p id="p2853153415116"><a name="p2853153415116"></a><a name="p2853153415116"></a>For example, if the global quantization configuration parameter sets the quantization factor search step to 0.01, you can use this parameter to apply differentiated quantization to some layers, for example, by setting the search step to 0.02.</p>
<p id="p61370245613"><a name="p61370245613"></a><a name="p61370245613"></a>Parameter priority: override_layer_configs&gt;override_layer_types&gt;common_config</p></td>
</tr>
<tr id="row73596813715"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p163597863717"><a name="p163597863717"></a><a name="p163597863717"></a>Repeated</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p1535910814376"><a name="p1535910814376"></a><a name="p1535910814376"></a>OverrideLayer</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p83598815375"><a name="p83598815375"></a><a name="p83598815375"></a>override_layer_configs</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p15798105944016"><a name="p15798105944016"></a><a name="p15798105944016"></a>Overrides the quantization configuration of a certain layer, that is, specifies which layers are subject to differentiated quantization.</p>
<p id="p4394114710214"><a name="p4394114710214"></a><a name="p4394114710214"></a>For example, if the global quantization configuration parameter sets the quantization factor search step to 0.01, you can use this parameter to apply differentiated quantization to some layers, for example, by setting the search step to 0.02.</p>
<p id="p194543485613"><a name="p194543485613"></a><a name="p194543485613"></a>Parameter priority: override_layer_configs&gt;override_layer_types&gt;common_config</p></td>
</tr>
<tr id="row145071332133914"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p350814326395"><a name="p350814326395"></a><a name="p350814326395"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p1650819327397"><a name="p1650819327397"></a><a name="p1650819327397"></a>bool</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p9508193293911"><a name="p9508193293911"></a><a name="p9508193293911"></a>do_fusion</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p07981859114018"><a name="p07981859114018"></a><a name="p07981859114018"></a>Whether to enable the BN fusion function. The default is true, indicating that this function is enabled.</p></td>
</tr>
<tr id="row12466115143717"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p84661857373"><a name="p84661857373"></a><a name="p84661857373"></a>Repeated</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p946614503717"><a name="p946614503717"></a><a name="p946614503717"></a>string</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p1746645143715"><a name="p1746645143715"></a><a name="p1746645143715"></a>skip_fusion_layers</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p1679819590407"><a name="p1679819590407"></a><a name="p1679819590407"></a>Layers that skip BN fusion. After configuration, these layers do not undergo BN fusion.</p></td>
</tr>
<tr id="row3932455380"><td class="cellrowborder" rowspan="3" valign="top" width="11.148885111488852%" headers="mcps1.2.6.1.1 "><p id="p47611242191418"><a name="p47611242191418"></a><a name="p47611242191418"></a>OverrideLayerType</p></td>
<td class="cellrowborder" valign="top" width="6.52934706529347%" headers="mcps1.2.6.1.2 "><p id="p1581853203818"><a name="p1581853203818"></a><a name="p1581853203818"></a>-</p></td>
<td class="cellrowborder" valign="top" width="11.188881111888811%" headers="mcps1.2.6.1.3 "><p id="p78053163810"><a name="p78053163810"></a><a name="p78053163810"></a>-</p></td>
<td class="cellrowborder" valign="top" width="12.84871512848715%" headers="mcps1.2.6.1.4 "><p id="p10885315383"><a name="p10885315383"></a><a name="p10885315383"></a>-</p></td>
<td class="cellrowborder" valign="top" width="58.284171582841715%" headers="mcps1.2.6.1.5 "><p id="p6909137153812"><a name="p6909137153812"></a><a name="p6909137153812"></a>Resets the quantization configuration of a layer type.</p></td>
</tr>
<tr id="row17134635121413"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p1476234213141"><a name="p1476234213141"></a><a name="p1476234213141"></a>required</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p37627428149"><a name="p37627428149"></a><a name="p37627428149"></a>string</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p6762342121414"><a name="p6762342121414"></a><a name="p6762342121414"></a>layer_type</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p07621142171414"><a name="p07621142171414"></a><a name="p07621142171414"></a>Name of the layer type that supports quantization.</p></td>
</tr>
<tr id="row1029523261414"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p776213421145"><a name="p776213421145"></a><a name="p776213421145"></a>required</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p0762174210140"><a name="p0762174210140"></a><a name="p0762174210140"></a>CalibrationConfig</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p876212426146"><a name="p876212426146"></a><a name="p876212426146"></a>calibration_config</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p19762942161410"><a name="p19762942161410"></a><a name="p19762942161410"></a>Quantization configuration to be overridden.</p></td>
</tr>
<tr id="zh-cn_topic_0240188735_row11783104092012"><td class="cellrowborder" rowspan="3" valign="top" width="11.148885111488852%" headers="mcps1.2.6.1.1 "><p id="zh-cn_topic_0240188735_p16795192442119"><a name="zh-cn_topic_0240188735_p16795192442119"></a><a name="zh-cn_topic_0240188735_p16795192442119"></a>OverrideLayer</p></td>
<td class="cellrowborder" valign="top" width="6.52934706529347%" headers="mcps1.2.6.1.2 "><p id="p9485142585010"><a name="p9485142585010"></a><a name="p9485142585010"></a>-</p></td>
<td class="cellrowborder" valign="top" width="11.188881111888811%" headers="mcps1.2.6.1.3 "><p id="p15182214175118"><a name="p15182214175118"></a><a name="p15182214175118"></a>-</p></td>
<td class="cellrowborder" valign="top" width="12.84871512848715%" headers="mcps1.2.6.1.4 "><p id="zh-cn_topic_0240188735_p37831240102014"><a name="zh-cn_topic_0240188735_p37831240102014"></a><a name="zh-cn_topic_0240188735_p37831240102014"></a>-</p></td>
<td class="cellrowborder" valign="top" width="58.284171582841715%" headers="mcps1.2.6.1.5 "><p id="zh-cn_topic_0240188735_p278314406207"><a name="zh-cn_topic_0240188735_p278314406207"></a><a name="zh-cn_topic_0240188735_p278314406207"></a>Resets the quantization configuration of a layer.</p></td>
</tr>
<tr id="zh-cn_topic_0240188735_row10956202614283"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p19444416194419"><a name="p19444416194419"></a><a name="p19444416194419"></a>required</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p1244491612447"><a name="p1244491612447"></a><a name="p1244491612447"></a>string</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="zh-cn_topic_0240188735_p6956152632810"><a name="zh-cn_topic_0240188735_p6956152632810"></a><a name="zh-cn_topic_0240188735_p6956152632810"></a>layer_name</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="zh-cn_topic_0240188735_p1195652613284"><a name="zh-cn_topic_0240188735_p1195652613284"></a><a name="zh-cn_topic_0240188735_p1195652613284"></a>Name of the layer to be reset.</p></td>
</tr>
<tr id="zh-cn_topic_0240188735_row1438216329281"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p576421834412"><a name="p576421834412"></a><a name="p576421834412"></a>required</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p1933192654411"><a name="p1933192654411"></a><a name="p1933192654411"></a>CalibrationConfig</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="zh-cn_topic_0240188735_p163821323286"><a name="zh-cn_topic_0240188735_p163821323286"></a><a name="zh-cn_topic_0240188735_p163821323286"></a>calibration_config</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="zh-cn_topic_0240188735_p382913237299"><a name="zh-cn_topic_0240188735_p382913237299"></a><a name="zh-cn_topic_0240188735_p382913237299"></a>Quantization configuration to be reset.</p></td>
</tr>
<tr id="zh-cn_topic_0240188735_row166322283234"><td class="cellrowborder" rowspan="6" valign="top" width="11.148885111488852%" headers="mcps1.2.6.1.1 "><p id="zh-cn_topic_0240188735_p9633152814236"><a name="zh-cn_topic_0240188735_p9633152814236"></a><a name="zh-cn_topic_0240188735_p9633152814236"></a>CalibrationConfig</p>
<p id="p3120433162119"><a name="p3120433162119"></a><a name="p3120433162119"></a></p></td>
<td class="cellrowborder" valign="top" width="6.52934706529347%" headers="mcps1.2.6.1.2 "><p id="p134858254503"><a name="p134858254503"></a><a name="p134858254503"></a>-</p></td>
<td class="cellrowborder" valign="top" width="11.188881111888811%" headers="mcps1.2.6.1.3 "><p id="p1818261455119"><a name="p1818261455119"></a><a name="p1818261455119"></a>-</p></td>
<td class="cellrowborder" valign="top" width="12.84871512848715%" headers="mcps1.2.6.1.4 "><p id="zh-cn_topic_0240188735_p1363332811232"><a name="zh-cn_topic_0240188735_p1363332811232"></a><a name="zh-cn_topic_0240188735_p1363332811232"></a>-</p></td>
<td class="cellrowborder" valign="top" width="58.284171582841715%" headers="mcps1.2.6.1.5 "><p id="zh-cn_topic_0240188735_p363392862320"><a name="zh-cn_topic_0240188735_p363392862320"></a><a name="zh-cn_topic_0240188735_p363392862320"></a>Calibration quantization configuration.</p></td>
</tr>
<tr id="zh-cn_topic_0240188735_row116226814307"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p1748519253502"><a name="p1748519253502"></a><a name="p1748519253502"></a>-</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p14182171417513"><a name="p14182171417513"></a><a name="p14182171417513"></a>ARQuantize</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p3275172510116"><a name="p3275172510116"></a><a name="p3275172510116"></a>arq_quantize</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="zh-cn_topic_0240188735_p1962317813010"><a name="zh-cn_topic_0240188735_p1962317813010"></a><a name="zh-cn_topic_0240188735_p1962317813010"></a>Weight quantization algorithm configuration.</p>
<p id="p245615017124"><a name="p245615017124"></a><a name="p245615017124"></a>arq_quantize: ARQ quantization algorithm configuration.</p></td>
</tr>
<tr id="row53671411153315"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p12367151111339"><a name="p12367151111339"></a><a name="p12367151111339"></a>-</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p9367811163310"><a name="p9367811163310"></a><a name="p9367811163310"></a>ADAquantize</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p136714114339"><a name="p136714114339"></a><a name="p136714114339"></a>ada_quantize</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p53671011193318"><a name="p53671011193318"></a><a name="p53671011193318"></a>Weight quantization algorithm configuration.</p>
<p id="p3778153020331"><a name="p3778153020331"></a><a name="p3778153020331"></a>ada_quantize: Adaptive rounding (AdaRound), adaptive rounding quantization algorithm configuration.</p></td>
</tr>
<tr id="zh-cn_topic_0240188735_row9881174683115"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p1548512525015"><a name="p1548512525015"></a><a name="p1548512525015"></a>-</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p1183191465111"><a name="p1183191465111"></a><a name="p1183191465111"></a>FMRQuantize</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p425415411015"><a name="p425415411015"></a><a name="p425415411015"></a>ifmr_quantize</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="zh-cn_topic_0240188735_p1788134616316"><a name="zh-cn_topic_0240188735_p1788134616316"></a><a name="zh-cn_topic_0240188735_p1788134616316"></a>Data quantization algorithm configuration.</p>
<p id="zh-cn_topic_0240188735_p186119548317"><a name="zh-cn_topic_0240188735_p186119548317"></a><a name="zh-cn_topic_0240188735_p186119548317"></a>ifmr_quantize: IFMR quantization algorithm configuration.</p></td>
</tr>
<tr id="row784910920348"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p123369144483"><a name="p123369144483"></a><a name="p123369144483"></a>-</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p1233691404813"><a name="p1233691404813"></a><a name="p1233691404813"></a>HFMGQuantize</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p1833601494810"><a name="p1833601494810"></a><a name="p1833601494810"></a>hfmg_quantize</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p13337191454814"><a name="p13337191454814"></a><a name="p13337191454814"></a>Data quantization algorithm configuration.</p>
<p id="p8424174634818"><a name="p8424174634818"></a><a name="p8424174634818"></a>hfmg_quantize: HFMG quantization algorithm configuration.</p></td>
</tr>
<tr id="row1312073315215"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p21209331216"><a name="p21209331216"></a><a name="p21209331216"></a>-</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p16120183319212"><a name="p16120183319212"></a><a name="p16120183319212"></a>DMQBalancer</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p21202332217"><a name="p21202332217"></a><a name="p21202332217"></a>dmq_balancer</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p1953551824417"><a name="p1953551824417"></a><a name="p1953551824417"></a>Equalization quantization algorithm configuration.</p>
<p id="p12535171816448"><a name="p12535171816448"></a><a name="p12535171816448"></a>dmq_balancer: DMQBalancer equalization algorithm configuration.</p></td>
</tr>
<tr id="zh-cn_topic_0240188735_row13931454162310"><td class="cellrowborder" rowspan="3" valign="top" width="11.148885111488852%" headers="mcps1.2.6.1.1 "><p id="zh-cn_topic_0240188735_p133932548230"><a name="zh-cn_topic_0240188735_p133932548230"></a><a name="zh-cn_topic_0240188735_p133932548230"></a>ARQuantize</p></td>
<td class="cellrowborder" valign="top" width="6.52934706529347%" headers="mcps1.2.6.1.2 "><p id="p2485725185011"><a name="p2485725185011"></a><a name="p2485725185011"></a>-</p></td>
<td class="cellrowborder" valign="top" width="11.188881111888811%" headers="mcps1.2.6.1.3 "><p id="p91831214185113"><a name="p91831214185113"></a><a name="p91831214185113"></a>-</p></td>
<td class="cellrowborder" valign="top" width="12.84871512848715%" headers="mcps1.2.6.1.4 "><p id="zh-cn_topic_0240188735_p16393454142316"><a name="zh-cn_topic_0240188735_p16393454142316"></a><a name="zh-cn_topic_0240188735_p16393454142316"></a>-</p></td>
<td class="cellrowborder" valign="top" width="58.284171582841715%" headers="mcps1.2.6.1.5 "><p id="zh-cn_topic_0240188735_p7393554192316"><a name="zh-cn_topic_0240188735_p7393554192316"></a><a name="zh-cn_topic_0240188735_p7393554192316"></a>ARQ quantization algorithm configuration. For algorithm introduction, see <a href="../algorithm_brief.md">ARQ weight quantization algorithm</a>.</p>
<p id="p817613334379"><a name="p817613334379"></a><a name="p817613334379"></a>This algorithm cannot be configured simultaneously with the ADAquantize and NUQuantize algorithms. If they are configured simultaneously, the quantization algorithm configured last in the configuration file prevails.</p></td>
</tr>
<tr id="zh-cn_topic_0240188735_row73046209318"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p169601842123113"><a name="p169601842123113"></a><a name="p169601842123113"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p0183111410516"><a name="p0183111410516"></a><a name="p0183111410516"></a>bool</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="zh-cn_topic_0240188735_p143040207319"><a name="zh-cn_topic_0240188735_p143040207319"></a><a name="zh-cn_topic_0240188735_p143040207319"></a>channel_wise</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="zh-cn_topic_0240188735_p3304220153114"><a name="zh-cn_topic_0240188735_p3304220153114"></a><a name="zh-cn_topic_0240188735_p3304220153114"></a>Whether to use a different quantization factor for each channel.</p>
<a name="zh-cn_topic_0240188002_ul1212114910429"></a><a name="zh-cn_topic_0240188002_ul1212114910429"></a><ul id="zh-cn_topic_0240188002_ul1212114910429"><li>true: Each channel is quantized independently with a different quantization factor.</li><li>false: All channels are quantized simultaneously and share the same quantization factor.</li></ul></td>
</tr>
<tr id="row0592142212598"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p1259332216599"><a name="p1259332216599"></a><a name="p1259332216599"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p3593162219599"><a name="p3593162219599"></a><a name="p3593162219599"></a>uint32</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p155931422185911"><a name="p155931422185911"></a><a name="p155931422185911"></a>quant_bits</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p138471244328"><a name="p138471244328"></a><a name="p138471244328"></a>Weight quantization bit width. Supports configuration as INT6, INT7, or INT8.</p>
<p id="p1593202215913"><a name="p1593202215913"></a><a name="p1593202215913"></a>Defaults to INT8 quantization.</p>
<p id="p26777718459"><a name="p26777718459"></a><a name="p26777718459"></a>When this field is configured as INT6 or INT7, it only supports the Conv2d operator type.</p>
<p id="p894111615316"><a name="p894111615316"></a><a name="p894111615316"></a>If quant_bits is configured as INT6 or INT7 in common_config, it takes effect only for the Conv2d operator, and other operators fall back to the default INT8.</p></td>
</tr>
<tr id="row27761940115319"><td class="cellrowborder" rowspan="7" valign="top" width="11.148885111488852%" headers="mcps1.2.6.1.1 "><p id="p87761040195315"><a name="p87761040195315"></a><a name="p87761040195315"></a>ADAquantize</p></td>
<td class="cellrowborder" valign="top" width="6.52934706529347%" headers="mcps1.2.6.1.2 "><p id="p14776134014536"><a name="p14776134014536"></a><a name="p14776134014536"></a>-</p></td>
<td class="cellrowborder" valign="top" width="11.188881111888811%" headers="mcps1.2.6.1.3 "><p id="p137771540125314"><a name="p137771540125314"></a><a name="p137771540125314"></a>-</p></td>
<td class="cellrowborder" valign="top" width="12.84871512848715%" headers="mcps1.2.6.1.4 "><p id="p9777144010535"><a name="p9777144010535"></a><a name="p9777144010535"></a>-</p></td>
<td class="cellrowborder" valign="top" width="58.284171582841715%" headers="mcps1.2.6.1.5 "><p id="p91642185612"><a name="p91642185612"></a><a name="p91642185612"></a>AdaRound adaptive rounding quantization algorithm. For algorithm introduction, see <a href="../algorithm_brief.md">ADA weight quantization algorithm</a>.</p>
<p id="p0329794386"><a name="p0329794386"></a><a name="p0329794386"></a>This algorithm cannot be configured simultaneously with the ARQuantize and NUQuantize algorithms. If they are configured simultaneously, the quantization algorithm configured last in the configuration file prevails.</p>
<p id="p9965162019211"><a name="p9965162019211"></a><a name="p9965162019211"></a>The activation functions supported by this quantization algorithm are: ReLU, RReLU, LeakyReLU, PReLU, GELU, ReLU6, Sigmoid, and Tanh.</p>
<p id="p154861059165910"><a name="p154861059165910"></a><a name="p154861059165910"></a>The ONNX operators corresponding to the preceding activation functions are: Relu, LeakyRelu, LeakyRelu, PRelu, Gelu, Clip, Sigmoid, and Tanh.</p>
<p id="p156871018154716"><a name="p156871018154716"></a><a name="p156871018154716"></a><span id="ph284172920174"><a name="ph284172920174"></a><a name="ph284172920174"></a>AMCT</span> obtains the quantizable layers in the quantization configuration (the quantization layers supported by this feature are torch.nn.Linear, torch.nn.Conv2d, and torch.nn.ConvTranspose2d), and obtains the quantizable modules in the topological order of the exported ONNX model. If a quantizable module is followed by one of the preceding activation functions, the quantizable module and the activation function are treated as a whole.</p>
<p id="p1785318562586"><a name="p1785318562586"></a><a name="p1785318562586"></a>Note:</p>
<p id="p4162741185615"><a name="p4162741185615"></a><a name="p4162741185615"></a>To export the preceding activation functions as ONNX operators, the torch.onnx.export interface of torch 2.1 is required. In addition, the Gelu operator requires Opset version v20, and the other operators require Opset version v17.</p></td>
</tr>
<tr id="row157031336145319"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p7704936155313"><a name="p7704936155313"></a><a name="p7704936155313"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p8704103614536"><a name="p8704103614536"></a><a name="p8704103614536"></a>uint32</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p15704163655316"><a name="p15704163655316"></a><a name="p15704163655316"></a>num_iteration</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p107041636145318"><a name="p107041636145318"></a><a name="p107041636145318"></a>Number of iterations. The value range is &gt;=0, and the default value is 10000.</p></td>
</tr>
<tr id="row189587384530"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p16959538165310"><a name="p16959538165310"></a><a name="p16959538165310"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p980283165517"><a name="p980283165517"></a><a name="p980283165517"></a>float</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p89591638175318"><a name="p89591638175318"></a><a name="p89591638175318"></a>reg_param</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p195903825318"><a name="p195903825318"></a><a name="p195903825318"></a><span>Regularization parameter, with a value range of (0,1) and a default value of 0.01.</span></p></td>
</tr>
<tr id="row1979535105311"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p47963505311"><a name="p47963505311"></a><a name="p47963505311"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p28101031145517"><a name="p28101031145517"></a><a name="p28101031145517"></a>float</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p17915350536"><a name="p17915350536"></a><a name="p17915350536"></a>beta_range_start</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p82744544620"><a name="p82744544620"></a><a name="p82744544620"></a>Beta start parameter, with a default value of 20, where beta_range_start &gt; beta_range_end &gt; 0.</p></td>
</tr>
<tr id="row094213125310"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p1594353112535"><a name="p1594353112535"></a><a name="p1594353112535"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p13816831125514"><a name="p13816831125514"></a><a name="p13816831125514"></a>float</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p1794319317539"><a name="p1794319317539"></a><a name="p1794319317539"></a>beta_range_end</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p1726314151973"><a name="p1726314151973"></a><a name="p1726314151973"></a>Beta end parameter, with a default value of 2.</p></td>
</tr>
<tr id="row18298122915546"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p1529892955410"><a name="p1529892955410"></a><a name="p1529892955410"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p1082210311552"><a name="p1082210311552"></a><a name="p1082210311552"></a>float</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p1829912975414"><a name="p1829912975414"></a><a name="p1829912975414"></a>warm_start</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p820812511358"><a name="p820812511358"></a><a name="p820812511358"></a>Warm-up factor, with a value range of (0,1) and a default value of 0.2.</p></td>
</tr>
<tr id="row129538314542"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p179538312543"><a name="p179538312543"></a><a name="p179538312543"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p11954131165420"><a name="p11954131165420"></a><a name="p11954131165420"></a>bool</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p1895483111547"><a name="p1895483111547"></a><a name="p1895483111547"></a>channel_wise</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p14805169105713"><a name="p14805169105713"></a><a name="p14805169105713"></a>Whether to use different quantization factors for each channel.</p>
<a name="ul680517915571"></a><a name="ul680517915571"></a><ul id="ul680517915571"><li>true: Each channel is quantized independently with different quantization factors.</li><li>false: All channels are quantized simultaneously and share the same quantization factor.</li></ul></td>
</tr>
<tr id="row14121356145210"><td class="cellrowborder" rowspan="2" valign="top" width="11.148885111488852%" headers="mcps1.2.6.1.1 "><p id="p983618745316"><a name="p983618745316"></a><a name="p983618745316"></a>DMQBalancer</p></td>
<td class="cellrowborder" valign="top" width="6.52934706529347%" headers="mcps1.2.6.1.2 "><p id="p1383615745310"><a name="p1383615745310"></a><a name="p1383615745310"></a>-</p></td>
<td class="cellrowborder" valign="top" width="11.188881111888811%" headers="mcps1.2.6.1.3 "><p id="p19836072536"><a name="p19836072536"></a><a name="p19836072536"></a>-</p></td>
<td class="cellrowborder" valign="top" width="12.84871512848715%" headers="mcps1.2.6.1.4 "><p id="p108368785317"><a name="p108368785317"></a><a name="p108368785317"></a>-</p></td>
<td class="cellrowborder" valign="top" width="58.284171582841715%" headers="mcps1.2.6.1.5 "><p id="p14836137155316"><a name="p14836137155316"></a><a name="p14836137155316"></a>DMQ equalization algorithm configuration. For algorithm introduction, see <a href="../algorithm_brief.md">DMQ equalization algorithm</a>.</p></td>
</tr>
<tr id="row579116575524"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p12836177175313"><a name="p12836177175313"></a><a name="p12836177175313"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p783619725310"><a name="p783619725310"></a><a name="p783619725310"></a>float</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p8837127185313"><a name="p8837127185313"></a><a name="p8837127185313"></a>migration_strength</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p128373710530"><a name="p128373710530"></a><a name="p128373710530"></a>Migration strength, representing the degree to which the quantization difficulty on activation data is migrated to the weight. The supported value range is [0.2, 0.8], and the default value is 0.5. The larger the outliers in the data distribution, the smaller the migration strength should be set.</p></td>
</tr>
<tr id="zh-cn_topic_0240188735_row188464215242"><td class="cellrowborder" rowspan="8" valign="top" width="11.148885111488852%" headers="mcps1.2.6.1.1 "><p id="zh-cn_topic_0240188735_p784622112414"><a name="zh-cn_topic_0240188735_p784622112414"></a><a name="zh-cn_topic_0240188735_p784622112414"></a>FMRQuantize</p></td>
<td class="cellrowborder" valign="top" width="6.52934706529347%" headers="mcps1.2.6.1.2 "><p id="p144853253508"><a name="p144853253508"></a><a name="p144853253508"></a>-</p></td>
<td class="cellrowborder" valign="top" width="11.188881111888811%" headers="mcps1.2.6.1.3 "><p id="p918371455114"><a name="p918371455114"></a><a name="p918371455114"></a>-</p></td>
<td class="cellrowborder" valign="top" width="12.84871512848715%" headers="mcps1.2.6.1.4 "><p id="zh-cn_topic_0240188735_p1984611242417"><a name="zh-cn_topic_0240188735_p1984611242417"></a><a name="zh-cn_topic_0240188735_p1984611242417"></a>-</p></td>
<td class="cellrowborder" valign="top" width="58.284171582841715%" headers="mcps1.2.6.1.5 "><p id="zh-cn_topic_0240188735_p58460292417"><a name="zh-cn_topic_0240188735_p58460292417"></a><a name="zh-cn_topic_0240188735_p58460292417"></a>FMR data quantization algorithm configuration. For the algorithm introduction, see <a href="../algorithm_brief.md">IFMR data quantization algorithm</a>.</p>
<p id="p1877244707"><a name="p1877244707"></a><a name="p1877244707"></a>This algorithm cannot be configured simultaneously with the HFMGQuantize algorithm. If both are configured, the quantization algorithm configured last in the configuration file prevails.</p></td>
</tr>
<tr id="zh-cn_topic_0240188735_row8300152716336"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p129891923173117"><a name="p129891923173117"></a><a name="p129891923173117"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p3183111418515"><a name="p3183111418515"></a><a name="p3183111418515"></a>float</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="zh-cn_topic_0240188735_p20300132763310"><a name="zh-cn_topic_0240188735_p20300132763310"></a><a name="zh-cn_topic_0240188735_p20300132763310"></a>search_range_start</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="zh-cn_topic_0240188735_p11300627183317"><a name="zh-cn_topic_0240188735_p11300627183317"></a><a name="zh-cn_topic_0240188735_p11300627183317"></a>Left boundary of the quantization factor search range.</p></td>
</tr>
<tr id="zh-cn_topic_0240188735_row1918914326335"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p182317254611"><a name="p182317254611"></a><a name="p182317254611"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p118237212462"><a name="p118237212462"></a><a name="p118237212462"></a>float</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="zh-cn_topic_0240188735_p3189232123318"><a name="zh-cn_topic_0240188735_p3189232123318"></a><a name="zh-cn_topic_0240188735_p3189232123318"></a>search_range_end</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="zh-cn_topic_0240188735_p1954495473717"><a name="zh-cn_topic_0240188735_p1954495473717"></a><a name="zh-cn_topic_0240188735_p1954495473717"></a>Right boundary of the quantization factor search range.</p></td>
</tr>
<tr id="zh-cn_topic_0240188735_row20464636113312"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p1861918118479"><a name="p1861918118479"></a><a name="p1861918118479"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p1461910118476"><a name="p1461910118476"></a><a name="p1461910118476"></a>float</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="zh-cn_topic_0240188735_p13464113615338"><a name="zh-cn_topic_0240188735_p13464113615338"></a><a name="zh-cn_topic_0240188735_p13464113615338"></a>search_step</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="zh-cn_topic_0240188735_p104647369332"><a name="zh-cn_topic_0240188735_p104647369332"></a><a name="zh-cn_topic_0240188735_p104647369332"></a>Quantization factor search step.</p></td>
</tr>
<tr id="zh-cn_topic_0240188735_row1710933263612"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p1861412254714"><a name="p1861412254714"></a><a name="p1861412254714"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p061418264717"><a name="p061418264717"></a><a name="p061418264717"></a>float</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="zh-cn_topic_0240188735_p1810983233616"><a name="zh-cn_topic_0240188735_p1810983233616"></a><a name="zh-cn_topic_0240188735_p1810983233616"></a>max_percentile</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="zh-cn_topic_0240188735_p161093322365"><a name="zh-cn_topic_0240188735_p161093322365"></a><a name="zh-cn_topic_0240188735_p161093322365"></a>Maximum value search position.</p></td>
</tr>
<tr id="zh-cn_topic_0240188735_row26036420362"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p1075110319475"><a name="p1075110319475"></a><a name="p1075110319475"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p117511331479"><a name="p117511331479"></a><a name="p117511331479"></a>float</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="zh-cn_topic_0240188735_p1060420421361"><a name="zh-cn_topic_0240188735_p1060420421361"></a><a name="zh-cn_topic_0240188735_p1060420421361"></a>min_percentile</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="zh-cn_topic_0240188735_p186041242183616"><a name="zh-cn_topic_0240188735_p186041242183616"></a><a name="zh-cn_topic_0240188735_p186041242183616"></a>Minimum value search position.</p></td>
</tr>
<tr id="row156362201313"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p2876102645420"><a name="p2876102645420"></a><a name="p2876102645420"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p16876102655416"><a name="p16876102655416"></a><a name="p16876102655416"></a>bool</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p136291318105419"><a name="p136291318105419"></a><a name="p136291318105419"></a>asymmetric</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p18570132515213"><a name="p18570132515213"></a><a name="p18570132515213"></a>Whether to perform symmetric quantization. Used to control the selection of the per-layer quantization algorithm.</p>
<a name="ul78601245115510"></a><a name="ul78601245115510"></a><ul id="ul78601245115510"><li>true: asymmetric quantization</li><li>false: symmetric quantization</li></ul>
<p id="p1784404618555"><a name="p1784404618555"></a><a name="p1784404618555"></a>If this parameter is configured in the override_layer_configs, override_layer_types, and common_config configuration items, or if the</p>
<p id="p58430467557"><a name="p58430467557"></a><a name="p58430467557"></a>activation_offset parameter is configured, the effective priority is:</p>
<p id="p1630711108390"><a name="p1630711108390"></a><a name="p1630711108390"></a>override_layer_configs&gt;override_layer_types&gt;common_config&gt;activation_offset</p></td>
</tr>
<tr id="row65481912144710"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p15647105044715"><a name="p15647105044715"></a><a name="p15647105044715"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p12647350104719"><a name="p12647350104719"></a><a name="p12647350104719"></a>CalibrationDataType</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p1764710503474"><a name="p1764710503474"></a><a name="p1764710503474"></a>dst_type</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p164710507471"><a name="p164710507471"></a><a name="p164710507471"></a>Quantization bit width, indicating whether data quantization uses INT8 or INT16 quantization, defaulting to INT8 quantization.<strong id="b7442433134911"><a name="b7442433134911"></a><a name="b7442433134911"></a>The current version only supports INT8 quantization.</strong></p></td>
</tr>
<tr id="row11961131163419"><td class="cellrowborder" rowspan="4" valign="top" width="11.148885111488852%" headers="mcps1.2.6.1.1 "><p id="p18771725134917"><a name="p18771725134917"></a><a name="p18771725134917"></a>HFMGQuantize</p></td>
<td class="cellrowborder" valign="top" width="6.52934706529347%" headers="mcps1.2.6.1.2 "><p id="p56910313515"><a name="p56910313515"></a><a name="p56910313515"></a>-</p></td>
<td class="cellrowborder" valign="top" width="11.188881111888811%" headers="mcps1.2.6.1.3 "><p id="p146916335113"><a name="p146916335113"></a><a name="p146916335113"></a>-</p></td>
<td class="cellrowborder" valign="top" width="12.84871512848715%" headers="mcps1.2.6.1.4 "><p id="p196910310511"><a name="p196910310511"></a><a name="p196910310511"></a>-</p></td>
<td class="cellrowborder" valign="top" width="58.284171582841715%" headers="mcps1.2.6.1.5 "><p id="p16989132195213"><a name="p16989132195213"></a><a name="p16989132195213"></a>HFMG data quantization algorithm configuration. For the algorithm introduction, see <a href="../algorithm_brief.md">HFMG data quantization algorithm</a>.</p>
<p id="p0610538149"><a name="p0610538149"></a><a name="p0610538149"></a>This algorithm cannot be configured simultaneously with the FMRQuantize algorithm. If both are configured, the quantization algorithm configured last in the configuration file prevails.</p></td>
</tr>
<tr id="row177009291348"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p3776259496"><a name="p3776259496"></a><a name="p3776259496"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p1777172594910"><a name="p1777172594910"></a><a name="p1777172594910"></a>uint32</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p1277122516495"><a name="p1277122516495"></a><a name="p1277122516495"></a>num_of_bins</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p14190312023"><a name="p14190312023"></a><a name="p14190312023"></a>Number of bins (the smallest unit in a histogram) in the histogram. The supported range is {1024, 2048, 4096, 8192}.</p>
<p id="p1772259498"><a name="p1772259498"></a><a name="p1772259498"></a>The default value is 4096.</p></td>
</tr>
<tr id="row1187284116114"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p1759374312114"><a name="p1759374312114"></a><a name="p1759374312114"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p10593144314119"><a name="p10593144314119"></a><a name="p10593144314119"></a>bool</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p359316431317"><a name="p359316431317"></a><a name="p359316431317"></a>asymmetric</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p1240373012612"><a name="p1240373012612"></a><a name="p1240373012612"></a>Whether to perform symmetric quantization. Used to control the selection of the per-layer quantization algorithm.</p>
<a name="ul812310517497"></a><a name="ul812310517497"></a><ul id="ul812310517497"><li>true: asymmetric quantization</li><li>false: symmetric quantization</li></ul>
<p id="p64035306619"><a name="p64035306619"></a><a name="p64035306619"></a>If this parameter is configured in all of the override_layer_configs, override_layer_types, and common_config configuration items, or if the</p>
<p id="p10403430769"><a name="p10403430769"></a><a name="p10403430769"></a>activation_offset parameter is configured, the effective priority is:</p>
<p id="p8403163010613"><a name="p8403163010613"></a><a name="p8403163010613"></a>override_layer_configs &gt; override_layer_types &gt; common_config &gt; activation_offset</p></td>
</tr>
<tr id="row131961176472"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p121262037155118"><a name="p121262037155118"></a><a name="p121262037155118"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p812683719512"><a name="p812683719512"></a><a name="p812683719512"></a>CalibrationDataType</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p16126537175117"><a name="p16126537175117"></a><a name="p16126537175117"></a>dst_type</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p47171749181715"><a name="p47171749181715"></a><a name="p47171749181715"></a>Data quantization bit width, indicating whether INT8 or INT16 quantization is used for data quantization. The default is INT8 quantization.<strong id="b1284153614327"><a name="b1284153614327"></a><a name="b1284153614327"></a>The current version only supports INT8 quantization.</strong></p></td>
</tr>
</tbody>
</table>

- The following is a **simple configuration file (*quant*.cfg) for post-training quantization**, which is constructed based on this file:

    ```
    # global quantize parameter
    batch_num : 2
    activation_offset : true
    skip_layers : "Opname"
    skip_layer_types:"Optype"
    do_fusion: true
    skip_fusion_layers : "Opname"
    common_config : {
        arq_quantize : {
            channel_wise : true
            quant_bits : 7
        }
        ifmr_quantize : {
            search_range_start : 0.7
            search_range_end : 1.3
            search_step : 0.01
            max_percentile : 0.999999
            min_percentile : 0.999999
            asymmetric : true
        }
         dmq_balancer : {
             migration_strength : 0.5
        }
    }

    override_layer_types : {
        layer_type : "Conv2d"
        calibration_config : {
            arq_quantize : {
                channel_wise : false
                quant_bits : 6
            }
            ifmr_quantize : {
                search_range_start : 0.8
                search_range_end : 1.2
                search_step : 0.02
                max_percentile : 0.999999
                min_percentile : 0.999999
                asymmetric : false
            }
             dmq_balancer : {
                  migration_strength : 0.5
             }
        }
    }

    override_layer_configs : {
        layer_name : "Opname"
        calibration_config : {
            arq_quantize : {
                channel_wise : true
            }
            ifmr_quantize : {
                search_range_start : 0.8
                search_range_end : 1.2
                search_step : 0.02
                max_percentile : 0.999999
                min_percentile : 0.999999
                 asymmetric : false
            }
             dmq_balancer : {
                  migration_strength : 0.5
             }
        }
    }
    ```

    If the HFMG data quantization algorithm is used, the bold parts in the preceding configuration file can be replaced with the following reference parameter information. An example is as follows (the following configuration information is only an example; modify it based on the actual situation):

    ```
    # global quantize parameter
    activation_offset : true
    batch_num : 2
    ...
    common_config : {
        hfmg_quantize : {
            num_of_bins : 4096
             asymmetric : false
        }
    ...
    }
    ```

- The following is a **simple configuration file (*ada\_round.cfg*) for adaptive rounding**, which is constructed based on this file:

    ```
    common_config : {
        ada_quantize : {
            num_iteration : 10000
            warm_start : 0.2
            reg_param : 0.01
            beta_range_start : 20
            beta_range_end : 2
            channel_wise : false
        }
    }
    ```

- The following is a **simple configuration file (*dmq\_balancer*.cfg) for quantization data equalization preprocessing**, which is constructed based on this file:

    ```
    # global quantize parameter
    batch_num : 2
    activation_offset : true
    skip_layers : "Opname"
    skip_layer_types:"Optype"
    do_fusion: true
    skip_fusion_layers : "Opname"
    common_config : {
        arq_quantize : {
            channel_wise : true
        }
        ifmr_quantize : {
            search_range_start : 0.7
            search_range_end : 1.3
            search_step : 0.01
            max_percentile : 0.999999
            min_percentile : 0.999999
            asymmetric : true
        }
         dmq_balancer : {
            migration_strength : 0.5
        }
    }

    override_layer_types : {
        layer_type : "Optype"
        calibration_config : {
            arq_quantize : {
                channel_wise : false
            }
            ifmr_quantize : {
                search_range_start : 0.8
                search_range_end : 1.2
                search_step : 0.02
                max_percentile : 0.999999
                min_percentile : 0.999999
                asymmetric : false
            }
           dmq_balancer : {
               migration_strength : 0.5
           }
        }
    }

    override_layer_configs : {
        layer_name : "Opname"
        calibration_config : {
            arq_quantize : {
                channel_wise : true
            }
            ifmr_quantize : {
                search_range_start : 0.8
                search_range_end : 1.2
                search_step : 0.02
                max_percentile : 0.999999
                min_percentile : 0.999999
                asymmetric : false
            }
           dmq_balancer : {
               migration_strength : 0.5
           }
        }
    }
    ```
