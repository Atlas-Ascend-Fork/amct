# AMCT大模型torchair离线模型导出

> 本文档说明如何使用`src/export_torchair_sample.py`将AMCT量化后的deploy模型导出为torchair AIR离线计算图，供下游离线推理使用。

## 1 前置条件

- 模型已完成AMCT量化，且为`amct.convert`后的量化deploy模型（量化流程参见对应算法的量化文档，如[smoothquant](../smoothquant/README.md)）；
- 已安装CANN、torch_npu及torchair，且版本匹配；
- 执行脚本前需先source CANN环境变量（按实际安装路径调整）：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

- Python依赖（`torch`/`torch_npu`/`torchair`/`transformers`/`amct_pytorch`/`Pillow`）已安装且版本匹配，也可参考[requirements.txt](requirements.txt)；若tokenizer需在线下载而网络不通，设置镜像`export HF_ENDPOINT=https://hf-mirror.com`。
- 安全提示：脚本默认关闭`--trust_remote_code`；加载的模型目录须为可信的本地目录，开启`--trust_remote_code`或`torch.load`反序列化可能执行模型目录中的自定义代码，请勿加载来源不明的模型。

**量化deploy模型的保存方式**：`amct.convert`后的模型包含AMCT自定义量化模块（如`NpuQuantizationLinear`），无法通过`save_pretrained`/`AutoModel.from_pretrained`还原，需在`amct.convert`后立即使用`torch.save`保存整模型，并保留原模型的tokenizer等文件：

```python
amct.convert(quant_model)
torch.save(quant_model.cpu(), '/path/to/quantized_llm/model.pth')
enc.save_pretrained('/path/to/quantized_llm/')  # tokenizer文件，供导出脚本构造输入
```

`--model_path`需指向该目录（目录内含`model.pth`），或直接指向`model.pth`文件。目录结构要求如下：

```
/path/to/quantized_llm/
├── model.pth              # torch.save保存的量化deploy模型
├── tokenizer.json         # enc.save_pretrained保存的tokenizer文件
├── tokenizer_config.json
└── ...                    # 其他tokenizer文件（vocab.json/merges.txt等，视模型而定）
```

## 2 使用方式

LLM网络（如Qwen/Llama类模型），导出输入为真实文本token化后的`[1, seqlen]` int64张量，文本默认使用脚本内置样例，也可通过`--input_file`指定本地文本文件：

```python
python3 src/export_torchair_sample.py --model_path=/path/to/quantized_llm/ --seqlen=2048
```

VLM网络（CLIP-ViT类，纯图输入，`--image_path`可为单张图片文件或含多张图片的目录，脚本将图片预处理为`[batch_size, 3, H, W]`张量）：

```python
python3 src/export_torchair_sample.py --model_type=vlm --model_path=/path/to/quantized_clip/ --image_path=/path/to/image.jpg --batch_size=1 --dtype=float16

python3 src/export_torchair_sample.py --model_type=vlm --model_path=/path/to/quantized_clip/ --image_path=/path/to/image_dir/ --batch_size=1 --dtype=float16
```

> 说明：当前脚本仅支持纯图输入的VLM模型（如CLIP），不支持图文混合输入的多模态模型（Qwen2-VL/Qwen2.5-VL/LLaVA类）。

常用参数说明：

| 参数 | 必填 | 默认值 | 说明 |
|:--|:--:|:--:|:--|
|--model_path|是|/|量化后deploy模型路径或所在目录|
|--model_type|否|llm|`llm`/`vlm`，vlm仅支持纯图输入的CLIP类模型|
|--output_dir|否|./torchair_output|导出产物目录|
|--export_name|否|--model_path 为目录时取模型目录名，为 .pth 文件时取不含后缀的文件名|AIR文件名（不含.air后缀）|
|--seqlen|否|2048|LLM导出输入序列长度（仅llm场景生效）|
|--input_file|否|内置样例文本|指定LLM导出输入所用的文本文件（其内容将被token化作为导出输入）|
|--image_path|否|/|VLM导出用图片文件或图片目录（vlm场景可选，未提供时用processor默认尺寸的占位输入；支持jpg/jpeg/png/bmp/tiff/webp）|
|--batch_size|否|1|VLM导出输入batch大小（须不大于图片数量）|
|--dtype|否|float16|VLM图像输入精度（float16/bfloat16/float32，须与模型浮点权重精度一致）；仅vlm场景适用，LLM导出输入为int64 token id，无浮点精度可调|
|--trust_remote_code|否|False|允许执行模型目录内自定义Python代码；默认关闭，仅对信任的本地模型开启|

## 3 导出产物

导出成功后，`--output_dir`目录下的产物布局如下：

```
output_dir/
├── <export_name>.air      # AIR离线计算图（权重内嵌）
├── input.bin              # 导出输入二进制
├── dynamo.pbtxt           # TorchDynamo中间表示（权重内嵌的小模型时生成于顶层）
└── weights/               # 大模型（权重总量超过约1.8G）时自动生成
    ├── dynamo.pbtxt       # TorchDynamo中间表示（大模型时生成于weights/下）
    └── xxx_weight ...     # 外置权重文件（.air通过FileConstant引用）
```

| 文件 | 说明 |
|:--|:--|
|`<export_name>.air`|torchair导出的AIR离线计算图，供离线推理使用；权重总量较小时内嵌全部权重，较大时通过FileConstant引用`weights/`下的外置文件|
|`dynamo.pbtxt`|TorchDynamo中间表示（文本格式）；权重内嵌的小模型生成于顶层，权重外置的大模型生成于`weights/`下|
|`weights/xxx_weight`|外置权重二进制文件，与`.air`需一起交付，不可单独移动或改名|
|`input.bin`|导出输入的二进制文件，供下游离线推理对帧使用|

**注意**：若生成`weights/`目录，离线推理部署时须保持`.air`与`weights/`的相对目录结构一并拷贝。

## 3.1 使用 ATC 转 om

导出 `.air` 后，可继续使用 ATC 工具转成板端离线推理模型 `.om`：

```bash
cd ./<output_dir>
atc --model=./<export_name>.air --framework=1 --output=./<export_name> --soc_version=<目标芯片soc_version>
```

其中 `--framework=1` 表示输入为 AIR 格式；`--soc_version` 为目标芯片对应的版本，可通过 `npu-smi info` 查看。

## 4 注意事项

- 导出输入的shape和dtype需与下游离线推理保持一致，请通过`--seqlen`/`--batch_size`/`--dtype`控制；VLM导出的`--dtype`须与模型浮点权重精度一致；
- LLM导出默认强制eager注意力实现（脚本内自动设置`config._attn_implementation='eager'`）：sdpa注意力在torch_npu上会被重派发为`npu_fusion_attention_v3`算子，当前torchair版本暂无该算子的AscendIR转换器；
- 加载后脚本会自动将量化模块中未随`.npu()`迁移的张量（如`bias`/`offset_bias`）迁移至NPU；
