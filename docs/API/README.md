# 整体约束和接口列表

## 整体约束

-   若接口中存在需要用户输入文件路径的参数，请确保输入路径正确，AMCT不会对路径做安全校验。
-   若接口中存在需要用户输入文件路径的参数，重新执行量化时，该参数相关取值将会被覆盖；量化打屏日志中也会有相关文件被覆盖的warning风险提示信息。

## 接口列表

| 分类                       | 接口名称                                                     | 功能描述                                                     |
| -------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 公共接口                   | [ModelEvaluator](./ModelEvaluator.md)                        | 针对某一个模型，根据模型的bin类型输入数据，提供一个Python实例，可对该模型执行校准和推理的评估器。 |
| 训练后量化接口             | [quantize](quantize.md)                                      | 高精度模型转换为校准模型，得到量化校准模型，推理后计算得到量化参数。 |
|                            | [convert](convert.md)                                        | 将量化校准模型转换为量化部署模型。                           |
|                            | [algorithm\_register](algorithm_register.md)                 | 将用户提供的自定义算法注册到AMCT工具。                       |
|                            | [create\_quant\_config](create_quant_config.md)              | 训练后量化接口，根据图的结构找到所有可量化的层，自动生成量化配置文件，并将可量化层的量化配置信息写入文件。 |
|                            | [quantize\_model](quantize_model.md)                         | 训练后量化接口，将输入的待量化的图结构按照给定的量化配置文件进行量化处理，在传入的图结构中插入权重量化、数据量化相关的算子，生成量化因子记录文件record_file，返回修改后的torch.nn.Module校准模型。 |
|                            | [save\_model](save_model.md)                                 | 训练后量化接口，根据量化因子记录文件record_file以及修改后的模型，插入AscendQuant、AscendDequant等算子，然后保存为可以在ONNX Runtime环境进行精度仿真的fake_quant模型，和可以在AI处理器做推理的部署模型。 |
|                            | [accuracy\_based\_auto\_calibration](accuracy_based_auto_calibration.md) | 根据用户输入的模型、配置文件进行自动的校准过程，搜索得到一个满足目标精度的量化配置，输出可以在ONNX Runtime环境下做精度仿真的fake_quant模型，和可在AI处理器上做推理的deploy模型。 |
|                            | [quantize\_preprocess](quantize_preprocess.md)               | 量化数据均衡预处理接口，将输入的待量化的图结构按照给定的量化配置文件进行量化处理，在传入的图结构中插入均衡量化相关的算子，生成均衡量化因子记录文件record_file，返回修改后的torch.nn.Module校准模型。 |
| 量化感知训练接口           | [create\_quant\_retrain\_config](create_quant_retrain_config.md) | 量化感知训练接口，根据图的结构找到所有可量化的层，自动生成量化配置文件，并将可量化层的量化配置信息写入配置文件。 |
|                            | [create\_quant\_retrain\_model](create_quant_retrain_model.md) | 量化感知训练接口，将输入的待量化的图结构按照给定的量化配置文件进行量化处理，在传入的图结构中插入量化相关的算子（数据和权重的量化感知训练层以及找N的层），生成量化因子记录文件record_file，返回修改后可用于量化感知训练的torch.nn.Module模型。 |
|                            | [restore\_quant\_retrain\_model](restore_quant_retrain_model.md) | 量化感知训练接口，将输入的待量化的图结构按照给定的量化感知训练配置文件进行量化处理，在传入的图结构中插入量化感知训练相关的算子（数据和权重的量化感知训练层以及找N的层），生成量化因子记录文件record_file，加载训练过程中保存的checkpoint权重参数，返回修改后的torch.nn.Module量化感知训练模型。 |
|                            | [save\_quant\_retrain\_model](save_quant_retrain_model.md)   | 量化感知训练接口，根据用户最终的重训练好的模型，插入AscendQuant、AscendDequant等算子，生成最终量化精度仿真模型以及量化部署模型。 |
| 单算子模式量化感知训练接口 | [Conv2dQAT](Conv2dQAT.md)                                    | 构造Conv2d的QAT算子。                                        |
|                            | [ConvTranspose2dQAT](ConvTranspose2dQAT.md)                  | 构造ConvTranspose2d的QAT算子。                               |
|                            | [Conv3dQAT](Conv3dQAT.md)                                    | 构造Conv3d的QAT算子。                                        |
|                            | [LinearQAT](LinearQAT.md)                                    | 构造Linear的QAT算子。                                        |
| 稀疏接口                   | [create\_prune\_retrain\_model](create_prune_retrain_model.md) | 通道稀疏或4选2结构化稀疏接口，两种稀疏特性每次只能使能一个：将输入的待稀疏的图结构按照给定的稀疏配置文件进行稀疏处理，在传入的图结构中插入或者替换相关的算子，生成记录稀疏信息的record_file，返回修改后可用于稀疏后训练的torch.nn.Module模型。 |
|                            | [restore\_prune\_retrain\_model](restore_prune_retrain_model.md) | 通道稀疏或4选2结构化稀疏接口，两种稀疏特性每次只能使能一个：将输入的待稀疏的图结构按照给定的record_file中稀疏记录进行稀疏，返回修改后可用于稀疏后训练的torch.nn.Module模型。 |
|                            | [save\_prune\_retrain\_model](save_prune_retrain_model.md)   | 稀疏接口，根据用户最终的重训练好的稀疏模型，生成最终ONNX仿真模型以及部署模型。 |
| 自动通道稀疏搜索接口       | [auto\_channel\_prune\_search](auto_channel_prune_search.md) | 自动通道稀疏接口，根据用户模型来计算各通道的稀疏敏感度（影响精度）以及稀疏收益（影响性能），然后搜索策略依据该输入来搜索最优的逐层通道稀疏率，以平衡精度和性能。最终输出一个配置文件。 |
| 组合压缩接口               | [create\_compressed\_retrain\_model](create_compressed_retrain_model.md) | 静态组合压缩接口，将输入的待静态组合压缩的模型按照给定的组合压缩配置文件进行压缩处理，即将传入的模型先进行稀疏（通道稀疏或者4选2结构化稀疏，二选一），后对模型插入量化相关的算子（数据和权重的量化感知训练层以及searchN的层），生成稀疏和量化因子记录文件record_file（如果配置存在），返回修改后的torch.nn.Module模型。 |
|                            | [restore\_compressed\_retrain\_model](restore_compressed_retrain_model.md) | 静态组合压缩训练接口，将输入的待静态组合压缩的模型按照给定的组合压缩配置文件和record记录文件进行压缩处理（先稀疏后量化），加载保存的权重。将传入的模型按照给定record_file中稀疏记录进行稀疏，后对模型插入量化相关的算子（数据和权重的量化感知训练层以及searchN的层）。加载训练过程中保存的checkpoint权重参数，返回修改后的torch.nn.Module模型。 |
|                            | [save\_compressed\_retrain\_model](save_compressed_retrain_model.md) | 静态组合压缩接口，根据用户最终的重训练好的模型，生成最终静态组合压缩精度仿真模型以及部署模型。 |
| 张量分解接口               | [auto\_decomposition](auto_decomposition.md)                 | 对用户输入的PyTorch模型对象进行张量分解，得到分解后的模型对象和分解前后层的对应名称，并保存分解信息文件（可选）。 |
|                            | [decompose\_network](decompose_network.md)                   | 用户输入PyTorch模型对象和通过auto_decomposition保存的分解信息文件，根据分解信息文件将模型对象改变为张量分解后的结构，得到分解后的模型对象和分解前后层的对应名称。 |
| 蒸馏接口                   | [create\_distill\_config](create_distill_config.md)          | 蒸馏接口，根据图的结构找到所有可蒸馏量化的层和可蒸馏量化的结构，自动生成蒸馏量化配置文件，并将可蒸馏量化层的量化配置和蒸馏结构写入配置文件。 |
|                            | [create\_distill\_model](create_distill_model.md)            | 蒸馏接口，将输入的待量化压缩的图结构按照给定的蒸馏量化配置文件进行量化处理，在传入的图结构中插入量化相关的算子（数据和权重的蒸馏量化层以及找N的层），返回修改后可用于蒸馏的torch.nn.Module模型。 |
|                            | [distill](distill.md)                                        | 蒸馏接口，将输入的待蒸馏的图结构按照给定的蒸馏量化配置文件进行蒸馏处理，返回修改后的torch.nn.Module蒸馏模型。 |
|                            | [save\_distill\_model](save_distill_model.md)                | 蒸馏接口，根据用户最终的蒸馏好的模型，生成最终量化精度仿真模型以及量化部署模型。 |
| KV Cache量化接口           | [create\_quant\_cali\_config](create_quant_cali_config.md)   | KV-cache量化接口，根据用户传入模型、量化层信息与量化配置信息，生成每个层的详细量化配置。 |
|                            | [create\_quant\_cali\_model](create_quant_cali_model.md)     | KV-cache量化接口，根据模型和量化详细配置，对用户模型进行改图，将待量化Linear算子替换为输出后进行IFMR/HFMG量化的量化算子，后续用户拿到模型后进行在线校准，校准后生成量化因子保存在record_file中。 |
|                            | [QuantCalibrationOp](QuantCalibrationOp.md)                  | KV Cache量化接口，用于用户构图，在前向传播时，根据用户的量化算法配置调用IFMR/HFMG量化算法对输出做校准，校准后，将量化因子依据对应格式输出到record_file文件指定层名中。 |

