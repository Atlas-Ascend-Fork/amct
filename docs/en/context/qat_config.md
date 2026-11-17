# Simple Configuration File for Quantization-Aware Training<a name="ZH-CN_TOPIC_0000002548668605"></a>

<!-- md-trans-meta sourceCommit=4a47814eed18048e444f9a2acc42d9864ddc477f translatedAt=2026-10-16T08:50:39.718Z pushedAt=2026-11-05T01:12:03.436Z nl=2 -->

[Table 1](#zh-cn_topic_0240188735_table1225503375617) describes parameters in the **retrain\_config\_pytorch.proto** file. The directory where the file is located is: ****AMCT installation directory**/amct\_pytorch/proto/retrain\_config\_pytorch.proto**.

**This file can be used to configure a simple configuration file for quantization-aware training, a simple configuration file for sparsity, and a simple configuration file for combined compression. Users can configure it based on the scenario.**

**Table 1**  retrain\_config\_pytorch.proto parameter description

<a name="zh-cn_topic_0240188735_table1225503375617"></a>
<table><thead align="left"><tr id="zh-cn_topic_0240188735_row11255153315614"><th class="cellrowborder" valign="top" width="12.748725127487251%" id="mcps1.2.6.1.1"><p id="zh-cn_topic_0240188735_p6255193315616"><a name="zh-cn_topic_0240188735_p6255193315616"></a><a name="zh-cn_topic_0240188735_p6255193315616"></a>Message</p></th>
<th class="cellrowborder" valign="top" width="5.7694230576942305%" id="mcps1.2.6.1.2"><p id="p194851125105014"><a name="p194851125105014"></a><a name="p194851125105014"></a>Required</p></th>
<th class="cellrowborder" valign="top" width="13.218678132186781%" id="mcps1.2.6.1.3"><p id="p466892519514"><a name="p466892519514"></a><a name="p466892519514"></a>Type</p></th>
<th class="cellrowborder" valign="top" width="14.058594140585942%" id="mcps1.2.6.1.4"><p id="zh-cn_topic_0240188735_p499481531910"><a name="zh-cn_topic_0240188735_p499481531910"></a><a name="zh-cn_topic_0240188735_p499481531910"></a>Field</p></th>
<th class="cellrowborder" valign="top" width="54.20457954204579%" id="mcps1.2.6.1.5"><p id="zh-cn_topic_0240188735_p122551335563"><a name="zh-cn_topic_0240188735_p122551335563"></a><a name="zh-cn_topic_0240188735_p122551335563"></a>Description</p></th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0240188735_row122551133185610"><td class="cellrowborder" rowspan="14" valign="top" width="12.748725127487251%" headers="mcps1.2.6.1.1 "><p id="zh-cn_topic_0240188735_p1925513338569"><a name="zh-cn_topic_0240188735_p1925513338569"></a><a name="zh-cn_topic_0240188735_p1925513338569"></a>AMCTRetrainConfig</p></td>
<td class="cellrowborder" valign="top" width="5.7694230576942305%" headers="mcps1.2.6.1.2 "><p id="p1485162505012"><a name="p1485162505012"></a><a name="p1485162505012"></a>-</p></td>
<td class="cellrowborder" valign="top" width="13.218678132186781%" headers="mcps1.2.6.1.3 "><p id="p1617031410516"><a name="p1617031410516"></a><a name="p1617031410516"></a>-</p></td>
<td class="cellrowborder" valign="top" width="14.058594140585942%" headers="mcps1.2.6.1.4 "><p id="zh-cn_topic_0240188735_p2994161514196"><a name="zh-cn_topic_0240188735_p2994161514196"></a><a name="zh-cn_topic_0240188735_p2994161514196"></a>-</p></td>
<td class="cellrowborder" valign="top" width="54.20457954204579%" headers="mcps1.2.6.1.5 "><p id="zh-cn_topic_0240188735_p82551633105614"><a name="zh-cn_topic_0240188735_p82551633105614"></a><a name="zh-cn_topic_0240188735_p82551633105614"></a><span id="ph244511481409"><a name="ph244511481409"></a><a name="ph244511481409"></a>AMCT</span> simple configuration for quantization-aware training.</p></td>
</tr>
<tr id="row96311527496"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p112187245443"><a name="p112187245443"></a><a name="p112187245443"></a>Repeated</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p16218102415441"><a name="p16218102415441"></a><a name="p16218102415441"></a>string</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p18218524114415"><a name="p18218524114415"></a><a name="p18218524114415"></a>skip_layers</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p9218924114420"><a name="p9218924114420"></a><a name="p9218924114420"></a>Skips layers by layer name so that they are not compressed. This is a global parameter that facilitates skipping the same layers in other feature scenarios to be extended later. For example, for the extended quant_skip_layers and regular_prune_skip_layers parameters described below, the skip_layers parameter can be used to configure the layers to be skipped in both feature scenarios simultaneously, without configuring them separately in the quant_skip_layers and regular_prune_skip_layers parameters.</p>
<p id="p8244141942917"><a name="p8244141942917"></a><a name="p8244141942917"></a>If skip_layers and quant_skip_layers, or skip_layers and regular_prune_skip_layers, are configured simultaneously, the union of the two is taken.</p></td>
</tr>
<tr id="row943965013491"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p15481827144413"><a name="p15481827144413"></a><a name="p15481827144413"></a>Repeated</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p54811327134414"><a name="p54811327134414"></a><a name="p54811327134414"></a>string</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p11481162711442"><a name="p11481162711442"></a><a name="p11481162711442"></a>skip_layer_types</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p829720510115"><a name="p829720510115"></a><a name="p829720510115"></a>Skips layers by layer type that are not compressed. This is a global parameter that facilitates skipping layers that can be skipped simultaneously under other feature scenarios in future extensions. For example, for the extended quant_skip_types and regular_prune_skip_types parameters described below, the skip_layer_types parameter can be used to configure the layers to be skipped under both feature scenarios simultaneously, without configuring them separately in the quant_skip_types and regular_prune_skip_types parameters.</p>
<p id="p9401431133114"><a name="p9401431133114"></a><a name="p9401431133114"></a>If skip_layer_types and quant_skip_types, or skip_layer_types and regular_prune_skip_types, are configured simultaneously, the union of the two is taken.</p></td>
</tr>
<tr id="row2905315135013"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p183393054412"><a name="p183393054412"></a><a name="p183393054412"></a>Repeated</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p283373084415"><a name="p283373084415"></a><a name="p283373084415"></a>RetrainOverrideLayer</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p2833113013443"><a name="p2833113013443"></a><a name="p2833113013443"></a>override_layer_configs</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p7833193064415"><a name="p7833193064415"></a><a name="p7833193064415"></a>Rewrites layers by layer name, that is, performs differentiated compression on specific layers.</p>
<p id="p14204438183011"><a name="p14204438183011"></a><a name="p14204438183011"></a>For example, if the global quantization configuration parameter configures the quantization bit width as INT8, this parameter can be used to perform differentiated quantization on some layers, which can be configured as INT4 quantization. The current version only supports INT8 quantization.</p>
<p id="p3289174615465"><a name="p3289174615465"></a><a name="p3289174615465"></a>The parameter priority is:</p>
<a name="ul471255217464"></a><a name="ul471255217464"></a><ul id="ul471255217464"><li>Quantization scenario: override_layer_configs&gt;override_layer_types&gt;retrain_data_quant_config/retrain_weight_quant_config</li><li>Sparsity scenario: override_layer_configs&gt;override_layer_types&gt;prune_config</li></ul></td>
</tr>
<tr id="row3894313105014"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p11894113894416"><a name="p11894113894416"></a><a name="p11894113894416"></a>Repeated</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p17894123817449"><a name="p17894123817449"></a><a name="p17894123817449"></a>RetrainOverrideLayerType</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p5894153819449"><a name="p5894153819449"></a><a name="p5894153819449"></a>override_layer_types</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p6894938144418"><a name="p6894938144418"></a><a name="p6894938144418"></a>Rewrites layers by layer type, that is, applies differentiated compression to specific layers.</p>
<p id="p24851712113217"><a name="p24851712113217"></a><a name="p24851712113217"></a>For example, if the global quantization configuration parameter sets the quantization bit width to INT8, this parameter can be used to apply differentiated quantization to some layers, such as configuring them to INT4 quantization. The current version only supports INT8 quantization.</p>
<p id="p79981423114814"><a name="p79981423114814"></a><a name="p79981423114814"></a>The parameter priority is as follows:</p>
<a name="ul1765922611489"></a><a name="ul1765922611489"></a><ul id="ul1765922611489"><li>Quantization scenario: override_layer_configs &gt; override_layer_types &gt; retrain_data_quant_config/retrain_weight_quant_config</li><li>Sparsity scenario: override_layer_configs &gt; override_layer_types &gt; prune_config</li></ul></td>
</tr>
<tr id="row0494202212456"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p11540926131416"><a name="p11540926131416"></a><a name="p11540926131416"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p0540926171413"><a name="p0540926171413"></a><a name="p0540926171413"></a>FakequantPrecisionMode</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p1054042691418"><a name="p1054042691418"></a><a name="p1054042691418"></a>fakequant_precision_mode</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p1254210571348"><a name="p1254210571348"></a><a name="p1254210571348"></a>Numeric precision mode of the scale_d parameter of the quant custom operator in the fakequant model.</p>
<a name="ul3401310203217"></a><a name="ul3401310203217"></a><ul id="ul3401310203217"><li>FORCE_FP16_QUANT: The value of the scale_d parameter is converted to float16 precision, and the numeric type is float32.</li><li>Empty, that is, not configured, and the default is empty. The value of the scale_d parameter remains at float32 precision.</li></ul></td>
</tr>
<tr id="row1910222725412"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p168381635155416"><a name="p168381635155416"></a><a name="p168381635155416"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p113881528155412"><a name="p113881528155412"></a><a name="p113881528155412"></a>uint32</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p183881328125417"><a name="p183881328125417"></a><a name="p183881328125417"></a>batch_num</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p938816282543"><a name="p938816282543"></a><a name="p938816282543"></a>Number of batches used for quantization.</p></td>
</tr>
<tr id="row23612610447"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p193621662444"><a name="p193621662444"></a><a name="p193621662444"></a>required</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p23621968447"><a name="p23621968447"></a><a name="p23621968447"></a>RetrainDataQuantConfig</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p183621644414"><a name="p183621644414"></a><a name="p183621644414"></a>retrain_data_quant_config</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p193621260447"><a name="p193621260447"></a><a name="p193621260447"></a>Data quantization configuration parameter for quantization-aware training. Global quantization configuration parameter.</p>
<p id="p1468523716344"><a name="p1468523716344"></a><a name="p1468523716344"></a>Parameter priority: override_layer_configs&gt;override_layer_types&gt;retrain_data_quant_config/retrain_weight_quant_config</p></td>
</tr>
<tr id="row468062044418"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p96801020114415"><a name="p96801020114415"></a><a name="p96801020114415"></a>required</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p168032016441"><a name="p168032016441"></a><a name="p168032016441"></a>RetrainWeightQuantConfig</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p1368019204448"><a name="p1368019204448"></a><a name="p1368019204448"></a>retrain_weight_quant_config</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p20680132018441"><a name="p20680132018441"></a><a name="p20680132018441"></a>Weight quantization configuration parameter for quantization-aware training. Global quantization configuration parameter.</p>
<p id="p1191764012342"><a name="p1191764012342"></a><a name="p1191764012342"></a>Parameter priority: override_layer_configs&gt;override_layer_types&gt;retrain_data_quant_config/retrain_weight_quant_config</p></td>
</tr>
<tr id="row172182024194419"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p2011511274810"><a name="p2011511274810"></a><a name="p2011511274810"></a>Repeated</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p15115524488"><a name="p15115524488"></a><a name="p15115524488"></a>string</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p132355834817"><a name="p132355834817"></a><a name="p132355834817"></a>quant_skip_layers</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p411515215487"><a name="p411515215487"></a><a name="p411515215487"></a>Skip layers that do not need to be quantized by layer name. Parameter used under the quantization scenario.</p>
<p id="p524872363211"><a name="p524872363211"></a><a name="p524872363211"></a>If skip_layers and quant_skip_layers are configured simultaneously, take the union of the two.</p></td>
</tr>
<tr id="row2828930174410"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p1011517294819"><a name="p1011517294819"></a><a name="p1011517294819"></a>Repeated</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p131151274811"><a name="p131151274811"></a><a name="p131151274811"></a>string</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p1611517224810"><a name="p1611517224810"></a><a name="p1611517224810"></a>quant_skip_types</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p14115152194819"><a name="p14115152194819"></a><a name="p14115152194819"></a>Skip layers that do not need to be quantized by layer type. Parameter used under the quantization scenario.</p>
<p id="p9166455153210"><a name="p9166455153210"></a><a name="p9166455153210"></a>If skip_layer_types and quant_skip_types are configured simultaneously, take the union of the two.</p></td>
</tr>
<tr id="row926199172716"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p72611393273"><a name="p72611393273"></a><a name="p72611393273"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p19261109122719"><a name="p19261109122719"></a><a name="p19261109122719"></a>PruneConfig</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p1926117917273"><a name="p1926117917273"></a><a name="p1926117917273"></a>prune_config</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p10326161418515"><a name="p10326161418515"></a><a name="p10326161418515"></a>Sparse configuration. Global sparse configuration parameter.</p>
<p id="p14840278498"><a name="p14840278498"></a><a name="p14840278498"></a>The parameter priority level is: override_layer_configs &gt; override_layer_types &gt; prune_config</p></td>
</tr>
<tr id="row17743447152714"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p18743247122718"><a name="p18743247122718"></a><a name="p18743247122718"></a>Repeated</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p5743194762713"><a name="p5743194762713"></a><a name="p5743194762713"></a>string</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p12743104719273"><a name="p12743104719273"></a><a name="p12743104719273"></a>regular_prune_skip_layers</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p15800152718615"><a name="p15800152718615"></a><a name="p15800152718615"></a>Skips layers that do not require structured sparsity by layer name. Parameter used under the sparsity scenario.</p>
<p id="p1123117299327"><a name="p1123117299327"></a><a name="p1123117299327"></a>If the skip_layers and regular_prune_skip_layers parameters are configured simultaneously, the union of the two is taken.</p></td>
</tr>
<tr id="row1689645514274"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p5896125592716"><a name="p5896125592716"></a><a name="p5896125592716"></a>Repeated</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p68971055172711"><a name="p68971055172711"></a><a name="p68971055172711"></a>string</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p1589715562720"><a name="p1589715562720"></a><a name="p1589715562720"></a>regular_prune_skip_types</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p68963399615"><a name="p68963399615"></a><a name="p68963399615"></a>Skips layers that do not require structured sparsity by layer type. Parameter used under the sparsity scenario.</p>
<p id="p092614683313"><a name="p092614683313"></a><a name="p092614683313"></a>If the skip_layer_types and regular_prune_skip_types parameters are configured simultaneously, the union of the two is taken.</p></td>
</tr>
<tr id="row835113511449"><td class="cellrowborder" rowspan="2" valign="top" width="12.748725127487251%" headers="mcps1.2.6.1.1 "><p id="p2351035174414"><a name="p2351035174414"></a><a name="p2351035174414"></a>RetrainDataQuantConfig</p></td>
<td class="cellrowborder" valign="top" width="5.7694230576942305%" headers="mcps1.2.6.1.2 "><p id="p435113554413"><a name="p435113554413"></a><a name="p435113554413"></a>-</p></td>
<td class="cellrowborder" valign="top" width="13.218678132186781%" headers="mcps1.2.6.1.3 "><p id="p143543514449"><a name="p143543514449"></a><a name="p143543514449"></a>-</p></td>
<td class="cellrowborder" valign="top" width="14.058594140585942%" headers="mcps1.2.6.1.4 "><p id="p33516355446"><a name="p33516355446"></a><a name="p33516355446"></a>-</p></td>
<td class="cellrowborder" valign="top" width="54.20457954204579%" headers="mcps1.2.6.1.5 "><p id="p1358112517220"><a name="p1358112517220"></a><a name="p1358112517220"></a>Quantization-aware training data quantization configuration.</p></td>
</tr>
<tr id="row66701120490"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p106717184920"><a name="p106717184920"></a><a name="p106717184920"></a>-</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p767228172514"><a name="p767228172514"></a><a name="p767228172514"></a>ActULQquantize</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p767119119493"><a name="p767119119493"></a><a name="p767119119493"></a>ulq_quantize</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p767110164914"><a name="p767110164914"></a><a name="p767110164914"></a>Algorithm for data quantization. Currently only supports ULQ.</p></td>
</tr>
<tr id="row27126389467"><td class="cellrowborder" rowspan="4" valign="top" width="12.748725127487251%" headers="mcps1.2.6.1.1 "><p id="p1291913387254"><a name="p1291913387254"></a><a name="p1291913387254"></a>ActULQquantize</p></td>
<td class="cellrowborder" valign="top" width="5.7694230576942305%" headers="mcps1.2.6.1.2 "><p id="p371283864612"><a name="p371283864612"></a><a name="p371283864612"></a>-</p></td>
<td class="cellrowborder" valign="top" width="13.218678132186781%" headers="mcps1.2.6.1.3 "><p id="p11712103813467"><a name="p11712103813467"></a><a name="p11712103813467"></a>-</p></td>
<td class="cellrowborder" valign="top" width="14.058594140585942%" headers="mcps1.2.6.1.4 "><p id="p13712123824620"><a name="p13712123824620"></a><a name="p13712123824620"></a>-</p></td>
<td class="cellrowborder" valign="top" width="54.20457954204579%" headers="mcps1.2.6.1.5 "><p id="p6712538104620"><a name="p6712538104620"></a><a name="p6712538104620"></a>ULQ data quantization algorithm configuration. For the algorithm introduction, see also <a href="../algorithm_brief.md">ULQ Data Quantization Algorithm</a>.</p></td>
</tr>
<tr id="row10327195261215"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p1015622491618"><a name="p1015622491618"></a><a name="p1015622491618"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p17812182818426"><a name="p17812182818426"></a><a name="p17812182818426"></a>DataType</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p14156172421615"><a name="p14156172421615"></a><a name="p14156172421615"></a>dst_type</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p9542423532"><a name="p9542423532"></a><a name="p9542423532"></a>Data quantization bit width. The following configurations are supported (the current version only supports INT8 quantization):</p>
<a name="ul883322015534"></a><a name="ul883322015534"></a><ul id="ul883322015534"><li>INT4</li><li>INT8, default is INT8.</li><li>INT16</li></ul></td>
</tr>
<tr id="row17632182254916"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p1563272213491"><a name="p1563272213491"></a><a name="p1563272213491"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p763282217493"><a name="p763282217493"></a><a name="p763282217493"></a>ClipMaxMin</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p16337222490"><a name="p16337222490"></a><a name="p16337222490"></a>clip_max_min</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p463372224914"><a name="p463372224914"></a><a name="p463372224914"></a>Initial upper and lower bound values. If not configured, ifmr is used for initialization by default.</p></td>
</tr>
<tr id="row2086612605014"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p8867963502"><a name="p8867963502"></a><a name="p8867963502"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p08671969506"><a name="p08671969506"></a><a name="p08671969506"></a>bool</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p188679619509"><a name="p188679619509"></a><a name="p188679619509"></a>fixed_min</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p138678625013"><a name="p138678625013"></a><a name="p138678625013"></a>Whether the lower bound is not learned and is fixed to 0. The default is true after ReLu and false otherwise.</p></td>
</tr>
<tr id="row19982124274611"><td class="cellrowborder" rowspan="3" valign="top" width="12.748725127487251%" headers="mcps1.2.6.1.1 "><p id="p6983242154619"><a name="p6983242154619"></a><a name="p6983242154619"></a>ClipMaxMin</p></td>
<td class="cellrowborder" valign="top" width="5.7694230576942305%" headers="mcps1.2.6.1.2 "><p id="p298314294618"><a name="p298314294618"></a><a name="p298314294618"></a>-</p></td>
<td class="cellrowborder" valign="top" width="13.218678132186781%" headers="mcps1.2.6.1.3 "><p id="p998317423461"><a name="p998317423461"></a><a name="p998317423461"></a>-</p></td>
<td class="cellrowborder" valign="top" width="14.058594140585942%" headers="mcps1.2.6.1.4 "><p id="p1098314284620"><a name="p1098314284620"></a><a name="p1098314284620"></a>-</p></td>
<td class="cellrowborder" valign="top" width="54.20457954204579%" headers="mcps1.2.6.1.5 "><p id="p8983124213468"><a name="p8983124213468"></a><a name="p8983124213468"></a>Initial upper and lower bounds.</p>
<p id="p1029819505536"><a name="p1029819505536"></a><a name="p1029819505536"></a>This parameter is used in computing the scaling factor. Subsequently, training-based tuning is performed on top of this initial value. It is recommended that the initial value be consistent with the upper and lower bounds of the actual inference data; otherwise, the quantized accuracy may be poor.</p></td>
</tr>
<tr id="row485785615017"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p0794145910158"><a name="p0794145910158"></a><a name="p0794145910158"></a>required</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p08571556195010"><a name="p08571556195010"></a><a name="p08571556195010"></a>float</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p1485765613504"><a name="p1485765613504"></a><a name="p1485765613504"></a>clip_max</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p985785605011"><a name="p985785605011"></a><a name="p985785605011"></a>Initial upper limit value.</p></td>
</tr>
<tr id="row4421551105019"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p1380975912154"><a name="p1380975912154"></a><a name="p1380975912154"></a>required</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p84221251195017"><a name="p84221251195017"></a><a name="p84221251195017"></a>float</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p342205185013"><a name="p342205185013"></a><a name="p342205185013"></a>clip_min</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p1242245111501"><a name="p1242245111501"></a><a name="p1242245111501"></a>Initial lower limit value.</p></td>
</tr>
<tr id="row1903547104616"><td class="cellrowborder" rowspan="3" valign="top" width="12.748725127487251%" headers="mcps1.2.6.1.1 "><p id="p17903747154615"><a name="p17903747154615"></a><a name="p17903747154615"></a>RetrainWeightQuantConfig</p></td>
<td class="cellrowborder" valign="top" width="5.7694230576942305%" headers="mcps1.2.6.1.2 "><p id="p1290316473462"><a name="p1290316473462"></a><a name="p1290316473462"></a>-</p></td>
<td class="cellrowborder" valign="top" width="13.218678132186781%" headers="mcps1.2.6.1.3 "><p id="p1990354714612"><a name="p1990354714612"></a><a name="p1990354714612"></a>-</p></td>
<td class="cellrowborder" valign="top" width="14.058594140585942%" headers="mcps1.2.6.1.4 "><p id="p16903547134615"><a name="p16903547134615"></a><a name="p16903547134615"></a>-</p></td>
<td class="cellrowborder" valign="top" width="54.20457954204579%" headers="mcps1.2.6.1.5 "><p id="p89031347134617"><a name="p89031347134617"></a><a name="p89031347134617"></a>Quantization-aware training weight quantization configuration.</p></td>
</tr>
<tr id="row34911413523"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p1749291115215"><a name="p1749291115215"></a><a name="p1749291115215"></a>-</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p1049219185211"><a name="p1049219185211"></a><a name="p1049219185211"></a>ARQRetrain</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p949211145217"><a name="p949211145217"></a><a name="p949211145217"></a>arq_retrain</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p54929165210"><a name="p54929165210"></a><a name="p54929165210"></a>ARQ weight quantization algorithm.</p></td>
</tr>
<tr id="row1286911121282"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p487071210283"><a name="p487071210283"></a><a name="p487071210283"></a>-</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p138703125287"><a name="p138703125287"></a><a name="p138703125287"></a>WtsULQRetrain</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p2870201210285"><a name="p2870201210285"></a><a name="p2870201210285"></a>ulq_retrain</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p952365884315"><a name="p952365884315"></a><a name="p952365884315"></a>ULQ weight quantization algorithm.</p></td>
</tr>
<tr id="row118213572512"><td class="cellrowborder" rowspan="3" valign="top" width="12.748725127487251%" headers="mcps1.2.6.1.1 "><p id="p7182557125116"><a name="p7182557125116"></a><a name="p7182557125116"></a>ARQRetrain</p></td>
<td class="cellrowborder" valign="top" width="5.7694230576942305%" headers="mcps1.2.6.1.2 "><p id="p1118213572511"><a name="p1118213572511"></a><a name="p1118213572511"></a>-</p></td>
<td class="cellrowborder" valign="top" width="13.218678132186781%" headers="mcps1.2.6.1.3 "><p id="p41827576517"><a name="p41827576517"></a><a name="p41827576517"></a>-</p></td>
<td class="cellrowborder" valign="top" width="14.058594140585942%" headers="mcps1.2.6.1.4 "><p id="p8183175795111"><a name="p8183175795111"></a><a name="p8183175795111"></a>-</p></td>
<td class="cellrowborder" valign="top" width="54.20457954204579%" headers="mcps1.2.6.1.5 "><p id="p41831257135111"><a name="p41831257135111"></a><a name="p41831257135111"></a>ARQ weight quantization algorithm configuration. For the algorithm introduction, see also <a href="../algorithm_brief.md">ARQ weight quantization algorithm</a>.</p></td>
</tr>
<tr id="row033113172131"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p9716349171711"><a name="p9716349171711"></a><a name="p9716349171711"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p1719533654412"><a name="p1719533654412"></a><a name="p1719533654412"></a>DataType</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p201954368449"><a name="p201954368449"></a><a name="p201954368449"></a>dst_type</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p47171749181715"><a name="p47171749181715"></a><a name="p47171749181715"></a>Used to select the INT8 or INT4 quantization bit width. The default is INT8. The current version only supports INT8 quantization.</p></td>
</tr>
<tr id="row123036212479"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p1995313815164"><a name="p1995313815164"></a><a name="p1995313815164"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p53031221144718"><a name="p53031221144718"></a><a name="p53031221144718"></a>bool</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p173031621174711"><a name="p173031621174711"></a><a name="p173031621174711"></a>channel_wise</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p73031121184714"><a name="p73031121184714"></a><a name="p73031121184714"></a>Whether to perform channel-wise ARQ.</p></td>
</tr>
<tr id="row56455471127"><td class="cellrowborder" rowspan="3" valign="top" width="12.748725127487251%" headers="mcps1.2.6.1.1 "><p id="p8885712143016"><a name="p8885712143016"></a><a name="p8885712143016"></a>WtsULQRetrain</p></td>
<td class="cellrowborder" valign="top" width="5.7694230576942305%" headers="mcps1.2.6.1.2 "><p id="p664519471123"><a name="p664519471123"></a><a name="p664519471123"></a>-</p></td>
<td class="cellrowborder" valign="top" width="13.218678132186781%" headers="mcps1.2.6.1.3 "><p id="p16462473124"><a name="p16462473124"></a><a name="p16462473124"></a>-</p></td>
<td class="cellrowborder" valign="top" width="14.058594140585942%" headers="mcps1.2.6.1.4 "><p id="p264616477128"><a name="p264616477128"></a><a name="p264616477128"></a>-</p></td>
<td class="cellrowborder" valign="top" width="54.20457954204579%" headers="mcps1.2.6.1.5 "><p id="p8281113418465"><a name="p8281113418465"></a><a name="p8281113418465"></a>ULQ weight quantization algorithm configuration. For the algorithm introduction, see also <a href="../algorithm_brief.md">ULQ Data Quantization Algorithm</a>.</p></td>
</tr>
<tr id="row088521215307"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p204013810308"><a name="p204013810308"></a><a name="p204013810308"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p24013382308"><a name="p24013382308"></a><a name="p24013382308"></a>DataType</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p4406383306"><a name="p4406383306"></a><a name="p4406383306"></a>dst_type</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p187026176455"><a name="p187026176455"></a><a name="p187026176455"></a>Used to select the INT8 or INT4 quantization bit width, with INT8 as the default. The current version only supports INT8 quantization.</p></td>
</tr>
<tr id="row4910023103012"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p1540193803019"><a name="p1540193803019"></a><a name="p1540193803019"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p7401838173013"><a name="p7401838173013"></a><a name="p7401838173013"></a>bool</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p04023819309"><a name="p04023819309"></a><a name="p04023819309"></a>channel_wise</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p1977013514458"><a name="p1977013514458"></a><a name="p1977013514458"></a>Whether to perform channel-wise ULQ.</p></td>
</tr>
<tr id="row4706041144711"><td class="cellrowborder" rowspan="5" valign="top" width="12.748725127487251%" headers="mcps1.2.6.1.1 "><p id="p470619419472"><a name="p470619419472"></a><a name="p470619419472"></a>RetrainOverrideLayer</p></td>
<td class="cellrowborder" valign="top" width="5.7694230576942305%" headers="mcps1.2.6.1.2 "><p id="p1170610419472"><a name="p1170610419472"></a><a name="p1170610419472"></a>-</p></td>
<td class="cellrowborder" valign="top" width="13.218678132186781%" headers="mcps1.2.6.1.3 "><p id="p1370644154716"><a name="p1370644154716"></a><a name="p1370644154716"></a>-</p></td>
<td class="cellrowborder" valign="top" width="14.058594140585942%" headers="mcps1.2.6.1.4 "><p id="p1970618411472"><a name="p1970618411472"></a><a name="p1970618411472"></a>-</p></td>
<td class="cellrowborder" valign="top" width="54.20457954204579%" headers="mcps1.2.6.1.5 "><p id="p070616416472"><a name="p070616416472"></a><a name="p070616416472"></a>Overridden layer configuration.</p></td>
</tr>
<tr id="row36791757135318"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p26792571537"><a name="p26792571537"></a><a name="p26792571537"></a>required</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p1667925716533"><a name="p1667925716533"></a><a name="p1667925716533"></a>string</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p167965718534"><a name="p167965718534"></a><a name="p167965718534"></a>layer_name</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p467965710535"><a name="p467965710535"></a><a name="p467965710535"></a>Layer name.</p></td>
</tr>
<tr id="row47391115411"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p8517230115410"><a name="p8517230115410"></a><a name="p8517230115410"></a>required</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p2073711115412"><a name="p2073711115412"></a><a name="p2073711115412"></a>RetrainDataQuantConfig</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p177361118549"><a name="p177361118549"></a><a name="p177361118549"></a>retrain_data_quant_config</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p147371155412"><a name="p147371155412"></a><a name="p147371155412"></a>Rewritten data layer quantization parameters.</p></td>
</tr>
<tr id="row1334152175418"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p156753125418"><a name="p156753125418"></a><a name="p156753125418"></a>required</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p23418211544"><a name="p23418211544"></a><a name="p23418211544"></a>RetrainWeightQuantConfig</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p13413265416"><a name="p13413265416"></a><a name="p13413265416"></a>retrain_weight_quant_config</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p1934162195416"><a name="p1934162195416"></a><a name="p1934162195416"></a>Rewritten weight layer quantization parameters.</p></td>
</tr>
<tr id="row72381731202915"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p92391931102915"><a name="p92391931102915"></a><a name="p92391931102915"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p112391431102914"><a name="p112391431102914"></a><a name="p112391431102914"></a>PruneConfig</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p72391319292"><a name="p72391319292"></a><a name="p72391319292"></a>prune_config</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p182391231192913"><a name="p182391231192913"></a><a name="p182391231192913"></a>Rewritten sparse configuration parameters.</p></td>
</tr>
<tr id="row101343874717"><td class="cellrowborder" rowspan="5" valign="top" width="12.748725127487251%" headers="mcps1.2.6.1.1 "><p id="p171363810472"><a name="p171363810472"></a><a name="p171363810472"></a>RetrainOverrideLayerType</p></td>
<td class="cellrowborder" valign="top" width="5.7694230576942305%" headers="mcps1.2.6.1.2 "><p id="p3133387471"><a name="p3133387471"></a><a name="p3133387471"></a>-</p></td>
<td class="cellrowborder" valign="top" width="13.218678132186781%" headers="mcps1.2.6.1.3 "><p id="p414938134712"><a name="p414938134712"></a><a name="p414938134712"></a>-</p></td>
<td class="cellrowborder" valign="top" width="14.058594140585942%" headers="mcps1.2.6.1.4 "><p id="p11414386476"><a name="p11414386476"></a><a name="p11414386476"></a>-</p></td>
<td class="cellrowborder" valign="top" width="54.20457954204579%" headers="mcps1.2.6.1.5 "><p id="p1114123811478"><a name="p1114123811478"></a><a name="p1114123811478"></a>Rewritten layer type configuration.</p></td>
</tr>
<tr id="row372413817554"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p7736142519558"><a name="p7736142519558"></a><a name="p7736142519558"></a>required</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p572613805511"><a name="p572613805511"></a><a name="p572613805511"></a>string</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p167268818559"><a name="p167268818559"></a><a name="p167268818559"></a>layer_type</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p137260825513"><a name="p137260825513"></a><a name="p137260825513"></a>Layer type.</p></td>
</tr>
<tr id="row884420120551"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p27371325115514"><a name="p27371325115514"></a><a name="p27371325115514"></a>required</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p28451012155515"><a name="p28451012155515"></a><a name="p28451012155515"></a>RetrainDataQuantConfig</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p28455122550"><a name="p28455122550"></a><a name="p28455122550"></a>retrain_data_quant_config</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p1845112145511"><a name="p1845112145511"></a><a name="p1845112145511"></a>Rewritten data layer quantization parameters.</p></td>
</tr>
<tr id="row969710176559"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p873718256558"><a name="p873718256558"></a><a name="p873718256558"></a>required</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p1770061710558"><a name="p1770061710558"></a><a name="p1770061710558"></a>RetrainWeightQuantConfig</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p67004178554"><a name="p67004178554"></a><a name="p67004178554"></a>retrain_weight_quant_config</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p87004178556"><a name="p87004178556"></a><a name="p87004178556"></a>Overridden weight layer quantization parameter.</p></td>
</tr>
<tr id="row19715133812917"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p105481683304"><a name="p105481683304"></a><a name="p105481683304"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p17548587304"><a name="p17548587304"></a><a name="p17548587304"></a>PruneConfig</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p154816893012"><a name="p154816893012"></a><a name="p154816893012"></a>prune_config</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p10548584301"><a name="p10548584301"></a><a name="p10548584301"></a>Overridden sparse configuration parameter.</p></td>
</tr>
<tr id="row516841563015"><td class="cellrowborder" rowspan="3" valign="top" width="12.748725127487251%" headers="mcps1.2.6.1.1 "><p id="p1416871533018"><a name="p1416871533018"></a><a name="p1416871533018"></a>PruneConfig</p></td>
<td class="cellrowborder" valign="top" width="5.7694230576942305%" headers="mcps1.2.6.1.2 "><p id="p7168161515300"><a name="p7168161515300"></a><a name="p7168161515300"></a>-</p></td>
<td class="cellrowborder" valign="top" width="13.218678132186781%" headers="mcps1.2.6.1.3 "><p id="p8168115103019"><a name="p8168115103019"></a><a name="p8168115103019"></a>-</p></td>
<td class="cellrowborder" valign="top" width="14.058594140585942%" headers="mcps1.2.6.1.4 "><p id="p171689151305"><a name="p171689151305"></a><a name="p171689151305"></a>-</p></td>
<td class="cellrowborder" valign="top" width="54.20457954204579%" headers="mcps1.2.6.1.5 "><p id="p1816818156308"><a name="p1816818156308"></a><a name="p1816818156308"></a>Sparse configuration.</p></td>
</tr>
<tr id="row826710183301"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p1426812181300"><a name="p1426812181300"></a><a name="p1426812181300"></a>-</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p162684182303"><a name="p162684182303"></a><a name="p162684182303"></a>FilterPruner</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p162681218113018"><a name="p162681218113018"></a><a name="p162681218113018"></a>filter_pruner</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p5268191810300"><a name="p5268191810300"></a><a name="p5268191810300"></a>Sparse configuration of filter (output dimension channel).</p></td>
</tr>
<tr id="row37125236714"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p729418994812"><a name="p729418994812"></a><a name="p729418994812"></a>-</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p19294129174816"><a name="p19294129174816"></a><a name="p19294129174816"></a>NOutOfMPruner</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p3294693487"><a name="p3294693487"></a><a name="p3294693487"></a>n_out_of_m_pruner</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p1529439134817"><a name="p1529439134817"></a><a name="p1529439134817"></a>4:2 sparse configuration.</p>
<p id="p268413597138"><a name="p268413597138"></a><a name="p268413597138"></a><strong id="b1168415951310"><a name="b1168415951310"></a><a name="b1168415951310"></a>Due to hardware constraints, the <span id="ph2357057154619"><a name="ph2357057154619"></a><a name="ph2357057154619"></a>Ascend 950PR&950DT products</span> do not support the 4:2 structured sparsity feature.</strong></p></td>
</tr>
<tr id="row13753152020303"><td class="cellrowborder" rowspan="2" valign="top" width="12.748725127487251%" headers="mcps1.2.6.1.1 "><p id="p167537204302"><a name="p167537204302"></a><a name="p167537204302"></a>FilterPruner</p></td>
<td class="cellrowborder" valign="top" width="5.7694230576942305%" headers="mcps1.2.6.1.2 "><p id="p4753220183010"><a name="p4753220183010"></a><a name="p4753220183010"></a>-</p></td>
<td class="cellrowborder" valign="top" width="13.218678132186781%" headers="mcps1.2.6.1.3 "><p id="p107532020143018"><a name="p107532020143018"></a><a name="p107532020143018"></a>-</p></td>
<td class="cellrowborder" valign="top" width="14.058594140585942%" headers="mcps1.2.6.1.4 "><p id="p1175315209307"><a name="p1175315209307"></a><a name="p1175315209307"></a>-</p></td>
<td class="cellrowborder" valign="top" width="54.20457954204579%" headers="mcps1.2.6.1.5 "><p id="p075317206301"><a name="p075317206301"></a><a name="p075317206301"></a>Configuration for filter sparsity.</p></td>
</tr>
<tr id="row326317244308"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p152631924173014"><a name="p152631924173014"></a><a name="p152631924173014"></a>-</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p526319241303"><a name="p526319241303"></a><a name="p526319241303"></a>BalancedL2NormFilterPruner</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p18263162463011"><a name="p18263162463011"></a><a name="p18263162463011"></a>balanced_l2_norm_filter_prune</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p02631724113012"><a name="p02631724113012"></a><a name="p02631724113012"></a>BalancedL2NormFilterPruner (BCP) algorithm. For details about the algorithm, see <a href="../algorithm_brief.md">Manual Channel Sparsity Algorithm</a>.</p></td>
</tr>
<tr id="row10390424113913"><td class="cellrowborder" rowspan="3" valign="top" width="12.748725127487251%" headers="mcps1.2.6.1.1 "><p id="p7391162453918"><a name="p7391162453918"></a><a name="p7391162453918"></a>BalancedL2NormFilterPruner</p></td>
<td class="cellrowborder" valign="top" width="5.7694230576942305%" headers="mcps1.2.6.1.2 "><p id="p139142483915"><a name="p139142483915"></a><a name="p139142483915"></a>-</p></td>
<td class="cellrowborder" valign="top" width="13.218678132186781%" headers="mcps1.2.6.1.3 "><p id="p12391152453910"><a name="p12391152453910"></a><a name="p12391152453910"></a>-</p></td>
<td class="cellrowborder" valign="top" width="14.058594140585942%" headers="mcps1.2.6.1.4 "><p id="p17391224143915"><a name="p17391224143915"></a><a name="p17391224143915"></a>-</p></td>
<td class="cellrowborder" valign="top" width="54.20457954204579%" headers="mcps1.2.6.1.5 "><p id="p16391182443916"><a name="p16391182443916"></a><a name="p16391182443916"></a>Configuration of the BalancedL2NormFilterPruner algorithm.</p></td>
</tr>
<tr id="row11575938173912"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p10575143818399"><a name="p10575143818399"></a><a name="p10575143818399"></a>required</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p18575038183913"><a name="p18575038183913"></a><a name="p18575038183913"></a>float</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p1357593818390"><a name="p1357593818390"></a><a name="p1357593818390"></a>prune_ratio</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p5575193820398"><a name="p5575193820398"></a><a name="p5575193820398"></a>Sparsity ratio, which is the ratio of the number of pruned filters to the total number of filters. The recommended configuration is 0.2, meaning that 20% of the output channels will be pruned.</p></td>
</tr>
<tr id="row1695819481431"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p1495834814312"><a name="p1495834814312"></a><a name="p1495834814312"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p1595817485435"><a name="p1595817485435"></a><a name="p1595817485435"></a>bool</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p12958114811434"><a name="p12958114811434"></a><a name="p12958114811434"></a>ascend_optimized</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p1095864884311"><a name="p1095864884311"></a><a name="p1095864884311"></a>Whether to perform Ascend affinity optimization. If the pruned model is to be deployed on the <span id="ph8545943164612"><a name="ph8545943164612"></a><a name="ph8545943164612"></a>AI processor</span>, it is recommended to set this parameter to true.</p></td>
</tr>
<tr id="row1516615551772"><td class="cellrowborder" rowspan="2" valign="top" width="12.748725127487251%" headers="mcps1.2.6.1.1 "><p id="p17327158174917"><a name="p17327158174917"></a><a name="p17327158174917"></a>NOutOfMPruner</p></td>
<td class="cellrowborder" valign="top" width="5.7694230576942305%" headers="mcps1.2.6.1.2 "><p id="p11328881495"><a name="p11328881495"></a><a name="p11328881495"></a>-</p></td>
<td class="cellrowborder" valign="top" width="13.218678132186781%" headers="mcps1.2.6.1.3 "><p id="p19328188124913"><a name="p19328188124913"></a><a name="p19328188124913"></a>-</p></td>
<td class="cellrowborder" valign="top" width="14.058594140585942%" headers="mcps1.2.6.1.4 "><p id="p9328108154914"><a name="p9328108154914"></a><a name="p9328108154914"></a>-</p></td>
<td class="cellrowborder" valign="top" width="54.20457954204579%" headers="mcps1.2.6.1.5 "><p id="p183284834919"><a name="p183284834919"></a><a name="p183284834919"></a>Configuration for 4:2 sparsity.</p></td>
</tr>
<tr id="row112115591673"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p260617227491"><a name="p260617227491"></a><a name="p260617227491"></a>-</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p6606112210498"><a name="p6606112210498"></a><a name="p6606112210498"></a>L1SelectivePruner</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p1606132210497"><a name="p1606132210497"></a><a name="p1606132210497"></a>l1_selective_prune</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p460752224919"><a name="p460752224919"></a><a name="p460752224919"></a>L1SelectivePrune algorithm. For the algorithm introduction, see <a href="../algorithm_brief.md">4:2 structured sparsity algorithm</a>.</p></td>
</tr>
<tr id="row2433194089"><td class="cellrowborder" rowspan="3" valign="top" width="12.748725127487251%" headers="mcps1.2.6.1.1 "><p id="p173074260497"><a name="p173074260497"></a><a name="p173074260497"></a>L1SelectivePruner</p></td>
<td class="cellrowborder" valign="top" width="5.7694230576942305%" headers="mcps1.2.6.1.2 "><p id="p14308526144910"><a name="p14308526144910"></a><a name="p14308526144910"></a>-</p></td>
<td class="cellrowborder" valign="top" width="13.218678132186781%" headers="mcps1.2.6.1.3 "><p id="p163081426124918"><a name="p163081426124918"></a><a name="p163081426124918"></a>-</p></td>
<td class="cellrowborder" valign="top" width="14.058594140585942%" headers="mcps1.2.6.1.4 "><p id="p0308102618499"><a name="p0308102618499"></a><a name="p0308102618499"></a>-</p></td>
<td class="cellrowborder" valign="top" width="54.20457954204579%" headers="mcps1.2.6.1.5 "><p id="p14308102615491"><a name="p14308102615491"></a><a name="p14308102615491"></a>Configuration of the L1SelectivePrune algorithm.</p></td>
</tr>
<tr id="row13434441582"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p12710191811495"><a name="p12710191811495"></a><a name="p12710191811495"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p2710818164918"><a name="p2710818164918"></a><a name="p2710818164918"></a>NOutOfMType</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p471081813494"><a name="p471081813494"></a><a name="p471081813494"></a>n_out_of_m_type</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="li1249420013282p0"><a name="li1249420013282p0"></a><a name="li1249420013282p0"></a>Currently only supports M4N2, that is, retaining 2 weights out of every 4 consecutive weights.</p></td>
</tr>
<tr id="row172841886819"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p16464926125015"><a name="p16464926125015"></a><a name="p16464926125015"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p9464122605019"><a name="p9464122605019"></a><a name="p9464122605019"></a>uint32</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p646432675017"><a name="p646432675017"></a><a name="p646432675017"></a>update_freq</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p246432685013"><a name="p246432685013"></a><a name="p246432685013"></a>Interval for updating the 4:2 sparsity selection. When update_freq=0, the 4:2 sparsity selection is updated only in the first batch; when update_freq=2, it is updated every 2 batches, and so on. Defaults to 0.</p></td>
</tr>
</tbody>
</table>

- The following is a **simple configuration file (*quant*.cfg) for quantization-aware training**, which is constructed based on this file:

    ```protobuf
    # global quantize parameter
    retrain_data_quant_config: {
        ulq_quantize: {
            clip_max_min: {
                clip_max: 6.0
                clip_min: -6.0
            }
            fixed_min: true
            dst_type: INT8
        }
    }

    retrain_weight_quant_config: {
        arq_retrain: {
           channel_wise: true
           dst_type: INT8
           }
      }

    skip_layers: "Opname"
    skip_layer_types: "Optype"

    override_layer_types : {
        layer_type: "Optype"
        retrain_weight_quant_config: {
            arq_retrain: {
               channel_wise: false
               dst_type: INT8
            }
        }
    }

    override_layer_configs : {
       layer_name: "Opname"
       retrain_data_quant_config: {
           ulq_quantize: {
               clip_max_min: {
                   clip_max: 3.0
                   clip_min: -3.0
               }
               dst_type: INT8
           }
       }
    }
    ```

- The following is a **simple configuration file (*prune*.cfg) for channel sparsity**, which is constructed based on this file:

    ```protobuf
    # global prune parameter
    prune_config{
        filter_pruner {
            balanced_l2_norm_filter_prune {
                prune_ratio: 0.3
                ascend_optimized: True
            }
        }
    }

    # skip layers
    regular_prune_skip_layers: "Opname"
    regular_prune_skip_layers: "Opname"

    # override specific layers
    override_layer_configs: {
        layer_name: "Opname"
        prune_config : {
            filter_pruner: {
                balanced_l2_norm_filter_prune: {
                    prune_ratio: 0.5
                    ascend_optimized: True
                }
            }
        }
    }
    ```

- The following is a **simple configuration file (*selective\_prune*.cfg) for 4:2 structured sparsity**, which is constructed based on this file:

    ```protobuf
    # global prune parameter
    prune_config{
        n_out_of_m_pruner {
            l1_selective_prune {
                n_out_of_m_type: M4N2
                update_freq: 0
            }
        }
    }

    # skip layers
    regular_prune_skip_layers: "Opname"
    regular_prune_skip_layers: "Opname"

    # override specific layers
    override_layer_configs: {
        layer_name: "Opname"
        prune_config : {
            n_out_of_m_pruner: {
                l1_selective_prune: {
                    n_out_of_m_type: M4N2
                    update_freq: 1
                }
            }
        }
    }
    ```

- The following is a **simple configuration file (*compressed1*.cfg) for combined compression (channel sparsity + INT8 quantization)**, which is constructed based on this file:

    ```protobuf
    prune_config : {
        filter_pruner : {
            balanced_l2_norm_filter_prune : {
                prune_ratio : 0.3
                ascend_optimized: True
            }
        }
    }

    # skip_layers: "skip_layers_name_0"
    skip_layer_types: "Optype"

    quant_skip_layers: "Opname"
    quant_skip_types: "Optype"

    retrain_weight_quant_config: {
        arq_retrain: {
        channel_wise: true
        dst_type: INT8
        }
    }

    override_layer_types : {
        layer_type: "Optype"
        retrain_weight_quant_config: {
            arq_retrain: {
            channel_wise: false
            dst_type: INT8
            }
        }
        retrain_data_quant_config : {
            ulq_quantize : {
                clip_max_min : {
                    clip_max : 6.0
                    clip_min : -6.0
                }
            }
        }
        prune_config : {
            filter_pruner : {
                balanced_l2_norm_filter_prune : {
                    prune_ratio : 0.5
                    ascend_optimized: True
                }
            }
        }
    }
    ```

- The following is a **simple configuration file (*compressed2*.cfg) for combined compression (4:2 structured sparsity + INT8 quantization)**, which is constructed based on this file:

    ```protobuf
    prune_config{
        n_out_of_m_pruner {
            l1_selective_prune {
                n_out_of_m_type: M4N2
                update_freq: 0
            }
        }
    }
    # skip_layers: "skip_layers_name_0"
    skip_layer_types: "Optype"

    quant_skip_layers: "quant_skip_layers_name_0"
    quant_skip_types: "Optype"

    retrain_weight_quant_config: {
        arq_retrain: {
        channel_wise: true
        dst_type: INT8
        }
    }

    override_layer_types : {
        layer_type: "Optype"
        retrain_weight_quant_config: {
            arq_retrain: {
            channel_wise: false
            dst_type: INT8
            }
        }
        retrain_data_quant_config : {
            ulq_quantize : {
                clip_max_min : {
                    clip_max : 6.0
                    clip_min : -6.0
                }
            }
        }
        prune_config{
            n_out_of_m_pruner {
                l1_selective_prune {
                    n_out_of_m_type: M4N2
                    update_freq: 1
                }
            }
        }
    }
    ```
