# Simple Configuration File for Automatic Channel Sparsity Search<a name="ZH-CN_TOPIC_0000002517028846"></a>

<!-- md-trans-meta sourceCommit=4a47814eed18048e444f9a2acc42d9864ddc477f translatedAt=2026-10-16T08:42:10.826Z pushedAt=2026-11-05T01:12:03.431Z nl=2 -->

The configuration description related to automatic channel sparsity search is in the **basic\_info.proto** file, which is located in the ***_AMCT_ installation directory*/amct_pytorch/proto/basic\_info.proto**.

<a name="zh-cn_topic_0240188735_table1225503375617"></a>
<table><tbody><thead align="left"><tr id="zh-cn_topic_0000002517028726_zh-cn_topic_0240188735_row11255153315614"><th class="cellrowborder" valign="top" width="15.818418158184183%" id="mcps1.1.6.1.1"><p id="zh-cn_topic_0000002517028726_zh-cn_topic_0240188735_p6255193315616"><a name="zh-cn_topic_0000002517028726_zh-cn_topic_0240188735_p6255193315616"></a><a name="zh-cn_topic_0000002517028726_zh-cn_topic_0240188735_p6255193315616"></a>Message</p></th>
<th class="cellrowborder" valign="top" width="8.729127087291271%" id="mcps1.1.6.1.2"><p id="zh-cn_topic_0000002517028726_p194851125105014"><a name="zh-cn_topic_0000002517028726_p194851125105014"></a><a name="zh-cn_topic_0000002517028726_p194851125105014"></a>Required</p></th>
<th class="cellrowborder" valign="top" width="8.7991200879912%" id="mcps1.1.6.1.3"><p id="zh-cn_topic_0000002517028726_p466892519514"><a name="zh-cn_topic_0000002517028726_p466892519514"></a><a name="zh-cn_topic_0000002517028726_p466892519514"></a>Type</p></th>
<th class="cellrowborder" valign="top" width="13.338666133386662%" id="mcps1.1.6.1.4"><p id="zh-cn_topic_0000002517028726_zh-cn_topic_0240188735_p499481531910"><a name="zh-cn_topic_0000002517028726_zh-cn_topic_0240188735_p499481531910"></a><a name="zh-cn_topic_0000002517028726_zh-cn_topic_0240188735_p499481531910"></a>Field</p></th>
<th class="cellrowborder" valign="top" width="53.314668533146694%" id="mcps1.1.6.1.5"><p id="zh-cn_topic_0000002517028726_zh-cn_topic_0240188735_p122551335563"><a name="zh-cn_topic_0000002517028726_zh-cn_topic_0240188735_p122551335563"></a><a name="zh-cn_topic_0000002517028726_zh-cn_topic_0240188735_p122551335563"></a>Description</p></th>
</tr>
</thead>

<tr id="zh-cn_topic_0000002517028726_row10838214165614"><td class="cellrowborder" rowspan="6" valign="top" width="15.818418158184183%" headers="mcps1.1.6.1.1 "><p id="zh-cn_topic_0000002517028726_p1772863191517"><a name="zh-cn_topic_0000002517028726_p1772863191517"></a><a name="zh-cn_topic_0000002517028726_p1772863191517"></a>AutoChannelPruneConfig</p></td>
<td class="cellrowborder" valign="top" width="8.729127087291271%" headers="mcps1.1.6.1.2 "><p id="zh-cn_topic_0000002517028726_p13136183020565"><a name="zh-cn_topic_0000002517028726_p13136183020565"></a><a name="zh-cn_topic_0000002517028726_p13136183020565"></a>-</p></td>
<td class="cellrowborder" valign="top" width="8.7991200879912%" headers="mcps1.1.6.1.3 "><p id="zh-cn_topic_0000002517028726_p7136530195617"><a name="zh-cn_topic_0000002517028726_p7136530195617"></a><a name="zh-cn_topic_0000002517028726_p7136530195617"></a>-</p></td>
<td class="cellrowborder" valign="top" width="13.338666133386662%" headers="mcps1.1.6.1.4 "><p id="zh-cn_topic_0000002517028726_p113623095612"><a name="zh-cn_topic_0000002517028726_p113623095612"></a><a name="zh-cn_topic_0000002517028726_p113623095612"></a>-</p></td>
<td class="cellrowborder" valign="top" width="53.314668533146694%" headers="mcps1.1.6.1.5 "><p id="zh-cn_topic_0000002517028726_p1713633035616"><a name="zh-cn_topic_0000002517028726_p1713633035616"></a><a name="zh-cn_topic_0000002517028726_p1713633035616"></a><span id="zh-cn_topic_0000002517028726_ph18136630175615"><a name="zh-cn_topic_0000002517028726_ph18136630175615"></a><a name="zh-cn_topic_0000002517028726_ph18136630175615"></a>Simple configuration for AMCT</span> automatic channel sparsity search.</p></td>
</tr>
<tr id="zh-cn_topic_0000002517028726_row7728163120153"><td class="cellrowborder" valign="top" headers="mcps1.1.6.1.1 "><p id="zh-cn_topic_0000002517028726_p16728153191519"><a name="zh-cn_topic_0000002517028726_p16728153191519"></a><a name="zh-cn_topic_0000002517028726_p16728153191519"></a>Yes</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.2 "><p id="zh-cn_topic_0000002517028726_p107281131111519"><a name="zh-cn_topic_0000002517028726_p107281131111519"></a><a name="zh-cn_topic_0000002517028726_p107281131111519"></a>float</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.3 "><p id="zh-cn_topic_0000002517028726_p12728131141515"><a name="zh-cn_topic_0000002517028726_p12728131141515"></a><a name="zh-cn_topic_0000002517028726_p12728131141515"></a>compress_ratio</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.4 "><p id="zh-cn_topic_0000002517028726_p14729113118158"><a name="zh-cn_topic_0000002517028726_p14729113118158"></a><a name="zh-cn_topic_0000002517028726_p14729113118158"></a>Compression ratio, which is the compression multiple based on the computation amount of all quantizable layers.</p></td>
</tr>
<tr id="zh-cn_topic_0000002517028726_row144521540201512"><td class="cellrowborder" valign="top" headers="mcps1.1.6.1.1 "><p id="zh-cn_topic_0000002517028726_p2730115111616"><a name="zh-cn_topic_0000002517028726_p2730115111616"></a><a name="zh-cn_topic_0000002517028726_p2730115111616"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.2 "><p id="zh-cn_topic_0000002517028726_p0452114061511"><a name="zh-cn_topic_0000002517028726_p0452114061511"></a><a name="zh-cn_topic_0000002517028726_p0452114061511"></a>bool</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.3 "><p id="zh-cn_topic_0000002517028726_p184531240151513"><a name="zh-cn_topic_0000002517028726_p184531240151513"></a><a name="zh-cn_topic_0000002517028726_p184531240151513"></a>ascend_optimized</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.4 "><p id="zh-cn_topic_0000002517028726_p1095864884311"><a name="zh-cn_topic_0000002517028726_p1095864884311"></a><a name="zh-cn_topic_0000002517028726_p1095864884311"></a>Whether to perform Ascend affinity optimization. If the pruned model is to be deployed on an <span id="zh-cn_topic_0000002517028726_ph8545943164612"><a name="zh-cn_topic_0000002517028726_ph8545943164612"></a><a name="zh-cn_topic_0000002517028726_ph8545943164612"></a>AI processor</span>, it is recommended to set this item to true.</p></td>
</tr>
<tr id="zh-cn_topic_0000002517028726_row5453940171518"><td class="cellrowborder" valign="top" headers="mcps1.1.6.1.1 "><p id="zh-cn_topic_0000002517028726_p167611251121611"><a name="zh-cn_topic_0000002517028726_p167611251121611"></a><a name="zh-cn_topic_0000002517028726_p167611251121611"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.2 "><p id="zh-cn_topic_0000002517028726_p6453134018156"><a name="zh-cn_topic_0000002517028726_p6453134018156"></a><a name="zh-cn_topic_0000002517028726_p6453134018156"></a>float</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.3 "><p id="zh-cn_topic_0000002517028726_p1445310408151"><a name="zh-cn_topic_0000002517028726_p1445310408151"></a><a name="zh-cn_topic_0000002517028726_p1445310408151"></a>max_prune_ratio</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.4 "><p id="zh-cn_topic_0000002517028726_p144531540191516"><a name="zh-cn_topic_0000002517028726_p144531540191516"></a><a name="zh-cn_topic_0000002517028726_p144531540191516"></a>Maximum sparse ratio of a single layer, which limits the maximum sparse ratio in the sparse configuration output by the API. The default value is 1.</p></td>
</tr>
<tr id="zh-cn_topic_0000002517028726_row16764229181512"><td class="cellrowborder" valign="top" headers="mcps1.1.6.1.1 "><p id="zh-cn_topic_0000002517028726_p279512513164"><a name="zh-cn_topic_0000002517028726_p279512513164"></a><a name="zh-cn_topic_0000002517028726_p279512513164"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.2 "><p id="zh-cn_topic_0000002517028726_p5764129191511"><a name="zh-cn_topic_0000002517028726_p5764129191511"></a><a name="zh-cn_topic_0000002517028726_p5764129191511"></a>int64</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.3 "><p id="zh-cn_topic_0000002517028726_p12765229171515"><a name="zh-cn_topic_0000002517028726_p12765229171515"></a><a name="zh-cn_topic_0000002517028726_p12765229171515"></a>test_iteration</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.4 "><p id="zh-cn_topic_0000002517028726_p14765122921518"><a name="zh-cn_topic_0000002517028726_p14765122921518"></a><a name="zh-cn_topic_0000002517028726_p14765122921518"></a>Number of batches of the input test data.</p></td>
</tr>
<tr id="zh-cn_topic_0000002517028726_row589591715203"><td class="cellrowborder" valign="top" headers="mcps1.1.6.1.1 "><p id="zh-cn_topic_0000002517028726_p68961117192013"><a name="zh-cn_topic_0000002517028726_p68961117192013"></a><a name="zh-cn_topic_0000002517028726_p68961117192013"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.2 "><p id="zh-cn_topic_0000002517028726_p1989614174203"><a name="zh-cn_topic_0000002517028726_p1989614174203"></a><a name="zh-cn_topic_0000002517028726_p1989614174203"></a>string</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.3 "><p id="zh-cn_topic_0000002517028726_p1589617173209"><a name="zh-cn_topic_0000002517028726_p1589617173209"></a><a name="zh-cn_topic_0000002517028726_p1589617173209"></a>override_prune_cfg</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.4 "><p id="zh-cn_topic_0000002517028726_p88961317102012"><a name="zh-cn_topic_0000002517028726_p88961317102012"></a><a name="zh-cn_topic_0000002517028726_p88961317102012"></a>User-configured simple configuration file for specified channel sparsity search. It may contain only skip and override configurations. The configured layers will follow the specified configuration and will not be rewritten by the automatic channel sparsity search API.</p></td>
</tr>
<tr id="zh-cn_topic_0000002517028726_row1280415861711"><td class="cellrowborder" rowspan="17" valign="top" width="15.818418158184183%" headers="mcps1.1.6.1.1 "><p id="zh-cn_topic_0000002517028726_p11804458141710"><a name="zh-cn_topic_0000002517028726_p11804458141710"></a><a name="zh-cn_topic_0000002517028726_p11804458141710"></a>AttrProto</p></td>
<td class="cellrowborder" valign="top" width="8.729127087291271%" headers="mcps1.1.6.1.2 "><p id="zh-cn_topic_0000002517028726_p12805125881715"><a name="zh-cn_topic_0000002517028726_p12805125881715"></a><a name="zh-cn_topic_0000002517028726_p12805125881715"></a>-</p></td>
<td class="cellrowborder" valign="top" width="8.7991200879912%" headers="mcps1.1.6.1.3 "><p id="zh-cn_topic_0000002517028726_p1980511589175"><a name="zh-cn_topic_0000002517028726_p1980511589175"></a><a name="zh-cn_topic_0000002517028726_p1980511589175"></a>-</p></td>
<td class="cellrowborder" valign="top" width="13.338666133386662%" headers="mcps1.1.6.1.4 "><p id="zh-cn_topic_0000002517028726_p1280575861710"><a name="zh-cn_topic_0000002517028726_p1280575861710"></a><a name="zh-cn_topic_0000002517028726_p1280575861710"></a>-</p></td>
<td class="cellrowborder" valign="top" width="53.314668533146694%" headers="mcps1.1.6.1.5 "><p id="zh-cn_topic_0000002517028726_p98051358141716"><a name="zh-cn_topic_0000002517028726_p98051358141716"></a><a name="zh-cn_topic_0000002517028726_p98051358141716"></a>General-purpose attribute definition message used to store attribute values of different types.</p></td>
</tr>
<tr id="zh-cn_topic_0000002517028726_row39403711911"><td class="cellrowborder" valign="top" headers="mcps1.1.6.1.1 "><p id="zh-cn_topic_0000002517028726_p4941337101910"><a name="zh-cn_topic_0000002517028726_p4941337101910"></a><a name="zh-cn_topic_0000002517028726_p4941337101910"></a>AttrType</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.2 "><p id="zh-cn_topic_0000002517028726_p1594203761910"><a name="zh-cn_topic_0000002517028726_p1594203761910"></a><a name="zh-cn_topic_0000002517028726_p1594203761910"></a>-</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.3 "><p id="zh-cn_topic_0000002517028726_p139413379190"><a name="zh-cn_topic_0000002517028726_p139413379190"></a><a name="zh-cn_topic_0000002517028726_p139413379190"></a>-</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.4 "><p id="zh-cn_topic_0000002517028726_p8945378196"><a name="zh-cn_topic_0000002517028726_p8945378196"></a><a name="zh-cn_topic_0000002517028726_p8945378196"></a>Attribute type enumeration, which determines the field used to store the value.</p></td>
</tr>
<tr id="zh-cn_topic_0000002517028726_row119454031916"><td class="cellrowborder" valign="top" headers="mcps1.1.6.1.1 "><p id="zh-cn_topic_0000002517028726_p169494017191"><a name="zh-cn_topic_0000002517028726_p169494017191"></a><a name="zh-cn_topic_0000002517028726_p169494017191"></a>-</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.2 "><p id="zh-cn_topic_0000002517028726_p394140101910"><a name="zh-cn_topic_0000002517028726_p394140101910"></a><a name="zh-cn_topic_0000002517028726_p394140101910"></a>-</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.3 "><p id="zh-cn_topic_0000002517028726_p109484091918"><a name="zh-cn_topic_0000002517028726_p109484091918"></a><a name="zh-cn_topic_0000002517028726_p109484091918"></a>UNDEFINED</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.4 "><p id="zh-cn_topic_0000002517028726_p1894640121914"><a name="zh-cn_topic_0000002517028726_p1894640121914"></a><a name="zh-cn_topic_0000002517028726_p1894640121914"></a>Undefined type.</p></td>
</tr>
<tr id="zh-cn_topic_0000002517028726_row922215429192"><td class="cellrowborder" valign="top" headers="mcps1.1.6.1.1 "><p id="zh-cn_topic_0000002517028726_p112224421193"><a name="zh-cn_topic_0000002517028726_p112224421193"></a><a name="zh-cn_topic_0000002517028726_p112224421193"></a>-</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.2 "><p id="zh-cn_topic_0000002517028726_p2022244211917"><a name="zh-cn_topic_0000002517028726_p2022244211917"></a><a name="zh-cn_topic_0000002517028726_p2022244211917"></a>-</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.3 "><p id="zh-cn_topic_0000002517028726_p822224213193"><a name="zh-cn_topic_0000002517028726_p822224213193"></a><a name="zh-cn_topic_0000002517028726_p822224213193"></a>FLOAT</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.4 "><p id="zh-cn_topic_0000002517028726_p15222942131911"><a name="zh-cn_topic_0000002517028726_p15222942131911"></a><a name="zh-cn_topic_0000002517028726_p15222942131911"></a>A single floating-point number.</p></td>
</tr>
<tr id="zh-cn_topic_0000002517028726_row_new1"><td class="cellrowborder" valign="top" headers="mcps1.1.6.1.1 "><p id="zh-cn_topic_0000002517028726_p_new1"><a name="zh-cn_topic_0000002517028726_p_new1"></a><a name="zh-cn_topic_0000002517028726_p_new1"></a>-</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.2 "><p id="zh-cn_topic_0000002517028726_p_new2"><a name="zh-cn_topic_0000002517028726_p_new2"></a><a name="zh-cn_topic_0000002517028726_p_new2"></a>-</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.3 "><p id="zh-cn_topic_0000002517028726_p_new3"><a name="zh-cn_topic_0000002517028726_p_new3"></a><a name="zh-cn_topic_0000002517028726_p_new3"></a>INT</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.4 "><p id="zh-cn_topic_0000002517028726_p_new4"><a name="zh-cn_topic_0000002517028726_p_new4"></a><a name="zh-cn_topic_0000002517028726_p_new4"></a>A single integer.</p></td>
</tr>
<tr id="zh-cn_topic_0000002517028726_row_new5"><td class="cellrowborder" valign="top" headers="mcps1.1.6.1.1 "><p id="zh-cn_topic_0000002517028726_p_new5"><a name="zh-cn_topic_0000002517028726_p_new5"></a><a name="zh-cn_topic_0000002517028726_p_new5"></a>-</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.2 "><p id="zh-cn_topic_0000002517028726_p_new6"><a name="zh-cn_topic_0000002517028726_p_new6"></a><a name="zh-cn_topic_0000002517028726_p_new6"></a>-</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.3 "><p id="zh-cn_topic_0000002517028726_p_new7"><a name="zh-cn_topic_0000002517028726_p_new7"></a><a name="zh-cn_topic_0000002517028726_p_new7"></a>STRING</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.4 "><p id="zh-cn_topic_0000002517028726_p_new8"><a name="zh-cn_topic_0000002517028726_p_new8"></a><a name="zh-cn_topic_0000002517028726_p_new8"></a>A single string.</p></td>
</tr>
<tr id="zh-cn_topic_0000002517028726_row_new9"><td class="cellrowborder" valign="top" headers="mcps1.1.6.1.1 "><p id="zh-cn_topic_0000002517028726_p_new9"><a name="zh-cn_topic_0000002517028726_p_new9"></a><a name="zh-cn_topic_0000002517028726_p_new9"></a>-</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.2 "><p id="zh-cn_topic_0000002517028726_p_new10"><a name="zh-cn_topic_0000002517028726_p_new10"></a><a name="zh-cn_topic_0000002517028726_p_new10"></a>-</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.3 "><p id="zh-cn_topic_0000002517028726_p_new11"><a name="zh-cn_topic_0000002517028726_p_new11"></a><a name="zh-cn_topic_0000002517028726_p_new11"></a>FLOATS</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.4 "><p id="zh-cn_topic_0000002517028726_p_new12"><a name="zh-cn_topic_0000002517028726_p_new12"></a><a name="zh-cn_topic_0000002517028726_p_new12"></a>Floating-point number array.</p></td>
</tr>
<tr id="zh-cn_topic_0000002517028726_row_new13"><td class="cellrowborder" valign="top" headers="mcps1.1.6.1.1 "><p id="zh-cn_topic_0000002517028726_p_new13"><a name="zh-cn_topic_0000002517028726_p_new13"></a><a name="zh-cn_topic_0000002517028726_p_new13"></a>-</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.2 "><p id="zh-cn_topic_0000002517028726_p_new14"><a name="zh-cn_topic_0000002517028726_p_new14"></a><a name="zh-cn_topic_0000002517028726_p_new14"></a>-</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.3 "><p id="zh-cn_topic_0000002517028726_p_new15"><a name="zh-cn_topic_0000002517028726_p_new15"></a><a name="zh-cn_topic_0000002517028726_p_new15"></a>INTS</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.4 "><p id="zh-cn_topic_0000002517028726_p_new16"><a name="zh-cn_topic_0000002517028726_p_new16"></a><a name="zh-cn_topic_0000002517028726_p_new16"></a>Integer array.</p></td>
</tr>
<tr id="zh-cn_topic_0000002517028726_row_new17"><td class="cellrowborder" valign="top" headers="mcps1.1.6.1.1 "><p id="zh-cn_topic_0000002517028726_p_new17"><a name="zh-cn_topic_0000002517028726_p_new17"></a><a name="zh-cn_topic_0000002517028726_p_new17"></a>-</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.2 "><p id="zh-cn_topic_0000002517028726_p_new18"><a name="zh-cn_topic_0000002517028726_p_new18"></a><a name="zh-cn_topic_0000002517028726_p_new18"></a>-</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.3 "><p id="zh-cn_topic_0000002517028726_p_new19"><a name="zh-cn_topic_0000002517028726_p_new19"></a><a name="zh-cn_topic_0000002517028726_p_new19"></a>STRINGS</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.4 "><p id="zh-cn_topic_0000002517028726_p_new20"><a name="zh-cn_topic_0000002517028726_p_new20"></a><a name="zh-cn_topic_0000002517028726_p_new20"></a>String array.</p></td>
</tr>
<tr id="zh-cn_topic_0000002517028726_row_new21"><td class="cellrowborder" valign="top" headers="mcps1.1.6.1.1 "><p id="zh-cn_topic_0000002517028726_p_new21"><a name="zh-cn_topic_0000002517028726_p_new21"></a><a name="zh-cn_topic_0000002517028726_p_new21"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.2 "><p id="zh-cn_topic_0000002517028726_p_new22"><a name="zh-cn_topic_0000002517028726_p_new22"></a><a name="zh-cn_topic_0000002517028726_p_new22"></a>string</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.3 "><p id="zh-cn_topic_0000002517028726_p_new23"><a name="zh-cn_topic_0000002517028726_p_new23"></a><a name="zh-cn_topic_0000002517028726_p_new23"></a>name</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.4 "><p id="zh-cn_topic_0000002517028726_p_new24"><a name="zh-cn_topic_0000002517028726_p_new24"></a><a name="zh-cn_topic_0000002517028726_p_new24"></a>Attribute name.</p></td>
</tr>
<tr id="zh-cn_topic_0000002517028726_row_new25"><td class="cellrowborder" valign="top" headers="mcps1.1.6.1.1 "><p id="zh-cn_topic_0000002517028726_p_new25"><a name="zh-cn_topic_0000002517028726_p_new25"></a><a name="zh-cn_topic_0000002517028726_p_new25"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.2 "><p id="zh-cn_topic_0000002517028726_p_new26"><a name="zh-cn_topic_0000002517028726_p_new26"></a><a name="zh-cn_topic_0000002517028726_p_new26"></a>AttrType</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.3 "><p id="zh-cn_topic_0000002517028726_p_new27"><a name="zh-cn_topic_0000002517028726_p_new27"></a><a name="zh-cn_topic_0000002517028726_p_new27"></a>type</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.4 "><p id="zh-cn_topic_0000002517028726_p_new28"><a name="zh-cn_topic_0000002517028726_p_new28"></a><a name="zh-cn_topic_0000002517028726_p_new28"></a>Attribute type (using the AttrType enum).</p></td>
</tr>
<tr id="zh-cn_topic_0000002517028726_row_new29"><td class="cellrowborder" valign="top" headers="mcps1.1.6.1.1 "><p id="zh-cn_topic_0000002517028726_p_new29"><a name="zh-cn_topic_0000002517028726_p_new29"></a><a name="zh-cn_topic_0000002517028726_p_new29"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.2 "><p id="zh-cn_topic_0000002517028726_p_new30"><a name="zh-cn_topic_0000002517028726_p_new30"></a><a name="zh-cn_topic_0000002517028726_p_new30"></a>float</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.3 "><p id="zh-cn_topic_0000002517028726_p_new31"><a name="zh-cn_topic_0000002517028726_p_new31"></a><a name="zh-cn_topic_0000002517028726_p_new31"></a>f</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.4 "><p id="zh-cn_topic_0000002517028726_p_new32"><a name="zh-cn_topic_0000002517028726_p_new32"></a><a name="zh-cn_topic_0000002517028726_p_new32"></a>Floating-point value (used when type is FLOAT).</p></td>
</tr>
<tr id="zh-cn_topic_0000002517028726_row_new33"><td class="cellrowborder" valign="top" headers="mcps1.1.6.1.1 "><p id="zh-cn_topic_0000002517028726_p_new33"><a name="zh-cn_topic_0000002517028726_p_new33"></a><a name="zh-cn_topic_0000002517028726_p_new33"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.2 "><p id="zh-cn_topic_0000002517028726_p_new34"><a name="zh-cn_topic_0000002517028726_p_new34"></a><a name="zh-cn_topic_0000002517028726_p_new34"></a>int64</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.3 "><p id="zh-cn_topic_0000002517028726_p_new35"><a name="zh-cn_topic_0000002517028726_p_new35"></a><a name="zh-cn_topic_0000002517028726_p_new35"></a>i</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.4 "><p id="zh-cn_topic_0000002517028726_p_new36"><a name="zh-cn_topic_0000002517028726_p_new36"></a><a name="zh-cn_topic_0000002517028726_p_new36"></a>Integer value (used when type is INT).</p></td>
</tr>
<tr id="zh-cn_topic_0000002517028726_row_new37"><td class="cellrowborder" valign="top" headers="mcps1.1.6.1.1 "><p id="zh-cn_topic_0000002517028726_p_new37"><a name="zh-cn_topic_0000002517028726_p_new37"></a><a name="zh-cn_topic_0000002517028726_p_new37"></a>No</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.2 "><p id="zh-cn_topic_0000002517028726_p_new38"><a name="zh-cn_topic_0000002517028726_p_new38"></a><a name="zh-cn_topic_0000002517028726_p_new38"></a>bytes</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.3 "><p id="zh-cn_topic_0000002517028726_p_new39"><a name="zh-cn_topic_0000002517028726_p_new39"></a><a name="zh-cn_topic_0000002517028726_p_new39"></a>s</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.4 "><p id="zh-cn_topic_0000002517028726_p_new40"><a name="zh-cn_topic_0000002517028726_p_new40"></a><a name="zh-cn_topic_0000002517028726_p_new40"></a>String value (used when type is STRING).</p></td>
</tr>
<tr id="zh-cn_topic_0000002517028726_row_new41"><td class="cellrowborder" valign="top" headers="mcps1.1.6.1.1 "><p id="zh-cn_topic_0000002517028726_p_new41"><a name="zh-cn_topic_0000002517028726_p_new41"></a><a name="zh-cn_topic_0000002517028726_p_new41"></a>Repeated</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.2 "><p id="zh-cn_topic_0000002517028726_p_new42"><a name="zh-cn_topic_0000002517028726_p_new42"></a><a name="zh-cn_topic_0000002517028726_p_new42"></a>float</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.3 "><p id="zh-cn_topic_0000002517028726_p_new43"><a name="zh-cn_topic_0000002517028726_p_new43"></a><a name="zh-cn_topic_0000002517028726_p_new43"></a>floats</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.4 "><p id="zh-cn_topic_0000002517028726_p_new44"><a name="zh-cn_topic_0000002517028726_p_new44"></a><a name="zh-cn_topic_0000002517028726_p_new44"></a>Floating-point array (used when type is FLOATS).</p></td>
</tr>
<tr id="zh-cn_topic_0000002517028726_row_new45"><td class="cellrowborder" valign="top" headers="mcps1.1.6.1.1 "><p id="zh-cn_topic_0000002517028726_p_new45"><a name="zh-cn_topic_0000002517028726_p_new45"></a><a name="zh-cn_topic_0000002517028726_p_new45"></a>Repeated</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.2 "><p id="zh-cn_topic_0000002517028726_p_new46"><a name="zh-cn_topic_0000002517028726_p_new46"></a><a name="zh-cn_topic_0000002517028726_p_new46"></a>int64</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.3 "><p id="zh-cn_topic_0000002517028726_p_new47"><a name="zh-cn_topic_0000002517028726_p_new47"></a><a name="zh-cn_topic_0000002517028726_p_new47"></a>ints</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.4 "><p id="zh-cn_topic_0000002517028726_p_new48"><a name="zh-cn_topic_0000002517028726_p_new48"></a><a name="zh-cn_topic_0000002517028726_p_new48"></a>Integer array (used when type is INTS).</p></td>
</tr>
<tr id="zh-cn_topic_0000002517028726_row_new49"><td class="cellrowborder" valign="top" headers="mcps1.1.6.1.1 "><p id="zh-cn_topic_0000002517028726_p_new49"><a name="zh-cn_topic_0000002517028726_p_new49"></a><a name="zh-cn_topic_0000002517028726_p_new49"></a>Repeated</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.2 "><p id="zh-cn_topic_0000002517028726_p_new50"><a name="zh-cn_topic_0000002517028726_p_new50"></a><a name="zh-cn_topic_0000002517028726_p_new50"></a>bytes</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.3 "><p id="zh-cn_topic_0000002517028726_p_new51"><a name="zh-cn_topic_0000002517028726_p_new51"></a><a name="zh-cn_topic_0000002517028726_p_new51"></a>strings</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.1.6.1.4 "><p id="zh-cn_topic_0000002517028726_p_new52"><a name="zh-cn_topic_0000002517028726_p_new52"></a><a name="zh-cn_topic_0000002517028726_p_new52"></a>Integer array (used when type is STRINGS).</p></td>
</tr>
</tbody>
</table>

The following is a sample **simple configuration file (_amc_.cfg) for automatic channel sparsity search**, which is constructed based on this file:

```text
compress_ratio: 1.5
ascend_optimized: true
max_prune_ratio: 0.8
test_iteration: 1
override_prune_cfg: 'your/path/to/override_channel_prune.cfg'
```
