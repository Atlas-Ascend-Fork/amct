# save\_model<a name="ZH-CN_TOPIC_0000002548668677"></a>

## 产品支持情况<a name="section185612964420"></a>

<a name="zh-cn_topic_0000002517188794_table38301303189"></a>

| 产品                                        | 是否支持 |
| ------------------------------------------- | -------- |
| Ascend 950PR/Ascend 950DT                   | √        |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √        |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √        |


## 功能说明<a name="zh-cn_topic_0240188006_section15406195619561"></a>

训练后量化接口，根据量化因子记录文件record\_file以及修改后的模型，插入AscendQuant、AscendDequant等算子，然后保存为可以在ONNX Runtime环境进行精度仿真的fake\_quant模型，和可以在AI处理器做推理的部署模型。

## 函数原型<a name="zh-cn_topic_0240188006_zh-cn_topic_0122830089_section428121323411"></a>

```python
save_model(modfied_onnx_file, record_file, save_path)
```

## 参数说明<a name="zh-cn_topic_0240188006_zh-cn_topic_0122830089_section795991810344"></a>

<a name="zh-cn_topic_0240188006_zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0240188006_zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="10.69%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0240188006_zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0240188006_zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0240188006_zh-cn_topic_0122830089_p1438834363520"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="7.290000000000001%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0240188006_p1769255516412"><a name="zh-cn_topic_0240188006_p1769255516412"></a><a name="zh-cn_topic_0240188006_p1769255516412"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="82.02000000000001%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0240188006_p15231205416325"><a name="zh-cn_topic_0240188006_p15231205416325"></a><a name="zh-cn_topic_0240188006_p15231205416325"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0240188006_zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="10.69%" headers="mcps1.1.4.1.1 "><p id="p61329565575"><a name="p61329565575"></a><a name="p61329565575"></a>modfied_onnx_file</p>
</td>
<td class="cellrowborder" valign="top" width="7.290000000000001%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0240188006_p1255438123917"><a name="zh-cn_topic_0240188006_p1255438123917"></a><a name="zh-cn_topic_0240188006_p1255438123917"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="82.02000000000001%" headers="mcps1.1.4.1.3 "><p id="p18172549175815"><a name="p18172549175815"></a><a name="p18172549175815"></a>含义：文件名，存储融合后模型的onnx格式。</p>
<p id="zh-cn_topic_0240188006_p182551638143919"><a name="zh-cn_topic_0240188006_p182551638143919"></a><a name="zh-cn_topic_0240188006_p182551638143919"></a>数据类型：string</p>
</td>
</tr>
<tr id="zh-cn_topic_0240188006_row12264154113552"><td class="cellrowborder" valign="top" width="10.69%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0240188006_p172650416558"><a name="zh-cn_topic_0240188006_p172650416558"></a><a name="zh-cn_topic_0240188006_p172650416558"></a>record_file</p>
</td>
<td class="cellrowborder" valign="top" width="7.290000000000001%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0240188006_p82657418551"><a name="zh-cn_topic_0240188006_p82657418551"></a><a name="zh-cn_topic_0240188006_p82657418551"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="82.02000000000001%" headers="mcps1.1.4.1.3 "><p id="p1071345865812"><a name="p1071345865812"></a><a name="p1071345865812"></a>含义：量化因子记录文件路径及名称。</p>
<p id="zh-cn_topic_0240188006_p1760663182215"><a name="zh-cn_topic_0240188006_p1760663182215"></a><a name="zh-cn_topic_0240188006_p1760663182215"></a>数据类型：string</p>
</td>
</tr>
<tr id="zh-cn_topic_0240188006_row73395458557"><td class="cellrowborder" valign="top" width="10.69%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0240188006_p9489111813337"><a name="zh-cn_topic_0240188006_p9489111813337"></a><a name="zh-cn_topic_0240188006_p9489111813337"></a>save_path</p>
</td>
<td class="cellrowborder" valign="top" width="7.290000000000001%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0240188006_p934014518550"><a name="zh-cn_topic_0240188006_p934014518550"></a><a name="zh-cn_topic_0240188006_p934014518550"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="82.02000000000001%" headers="mcps1.1.4.1.3 "><p id="p1749919013596"><a name="p1749919013596"></a><a name="p1749919013596"></a>含义：模型存放路径。该路径需要包含模型名前缀，例如./quantized_model/*<em id="zh-cn_topic_0240188713_i162671794111"><a name="zh-cn_topic_0240188713_i162671794111"></a><a name="zh-cn_topic_0240188713_i162671794111"></a>model</em>。</p>
<p id="zh-cn_topic_0240188006_p021914293368"><a name="zh-cn_topic_0240188006_p021914293368"></a><a name="zh-cn_topic_0240188006_p021914293368"></a>数据类型：string</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0240188006_zh-cn_topic_0122830089_section293415513458"></a>

无

## 约束说明<a name="zh-cn_topic_0240188006_section1443392021419"></a>

-   在网络推理的batch数目达到batch\_num后，再调用该接口，否则量化因子不正确，量化结果不正确。
-   该接口只接收[quantize\_model](./quantize_model.md)接口产生的ONNX类型模型文件。
-   该接口需要输入量化因子记录文件，量化因子记录文件在[quantize\_model](./quantize_model.md)阶段生成，在模型推理阶段填充有效值。

## 调用示例<a name="zh-cn_topic_0240188006_section149503402234"></a>

```python
import amct_pytorch as amct
# 进行网络推理，期间完成量化
for i in batch_num:
    output = calibration_model(input_batch)

# 插入API，将量化的模型存为ONNX文件
amct.save_model(modfied_onnx_file="./tmp/modfied_model.onnx",
                record_file="./tmp/scale_offset_record.txt",
                save_path="./results/model")
```

落盘文件说明：

-   精度仿真模型文件：ONNX格式的模型文件，模型名中包含fake\_quant，可以在ONNX Runtime环境进行精度仿真。
-   部署模型文件：ONNX格式的模型文件，模型名中包含deploy，经过ATC转换工具转换后可部署到AI处理器。
-   （可选）\*.external文件，包括\*deploy.external和\*fakequant.external：

    只有保存的精度仿真模型以及部署模型文件大小\>=2GB才会生成该类文件，且与压缩后的\*.onnx模型文件生成在同级目录，用于保存Tensor中的数据，每个Tensor数据单独保存一份\*.external文件，文件名与Tensor相同，例如_conv1.weight_\_deploy.external和_conv1.weight_\_fakequant.external。

    后续通过ATC工具加载压缩后的\*.onnx部署模型文件进行模型转换时，会自动读取同级目录下\*.external文件中的Tensor数据。

重新执行量化时，该接口输出的上述文件将会被覆盖。

