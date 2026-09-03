# AMCT大模型SMOOTHQUANT量化

## 1 量化前提

### 1.1 安装依赖

本sample依赖包可参考[requirements.txt](requirements.txt)

需要注意的是torch_npu包版本需要与Python、torch包版本相匹配，需要安装CANN包

### 1.2 模型和数据集准备

本sample以Llama2-7b，qwen2-7b，qwen3-8b模型，pileval数据，wikitext2数据集为示例, 数据为在线加载，模型需要用户自己下载并在执行脚本时指定模型路径。

注意：量化数据类型组合float8_e4m3fn * float4_e2m1只支持量化原始数据类型为torch.bfloat16，请注意修改src/utils.py文件中获取模型的数据类型。

### 1.3 简易量化配置
本sample中使用的量化配置已经内置在工具中，可以通过下述方式获取并使用：

量化数据类型组合int8 * int8的配置：
`from amct_pytorch import INT8_SMOOTHQUANT_CFG`
量化数据类型组合float8_e4m3fn * float4_e2m1的配置：
```python
cfg = {
    'batch_num': 1,
    'quant_cfg': {
        'weights': {
            'type': 'float4_e2m1',
            'symmetric': True,
            'strategy': 'group',
            'group_size': 32
        },
        'inputs': {
            'type': 'float8_e4m3fn',
            'symmetric': True,
            'strategy': 'tensor',
        },
    },
    'algorithm': {'smoothquant': {'smooth_strength': 0.77}},
    'skip_layers': {'lm_head', 'down_proj'}
}
```
量化数据类型组合hifloat8 * hifloat8的配置：
`from amct_pytorch import HIFP8_SMOOTHQUANT_CFG`

基础配置如下，可分别测试smooth_strength=0.5和0.8：
```python
{
    'batch_num': 1,
    'quant_cfg': {
        'weights': {
            'type': 'hifloat8',
            'symmetric': True,
            'strategy': 'channel',
        },
        'inputs': {
            'type': 'hifloat8',
            'symmetric': True,
            'strategy': 'tensor',
        },
    },
    'algorithm': {'smoothquant': {'smooth_strength': 0.5}},
    'skip_layers': {'lm_head'}
}
```

如果需要修改详细配置，请参考资料构造需要的量化配置dict。

smoothquant算法仅支持全量化，支持的量化类型以及量化配置：

| 字段 |类型| 说明 | 取值范围 | 注意事项 |
|:--| :-: | :-- | :-: | :-- |
|batch_num|uint32|量化使用的batch数量 |1|/|
|skip_layers|str|跳过量化的层 |/|跳过量化层支持模糊匹配，当配置字符串为层名字串，或与层名一致时，跳过该层量化，不生成量化配置。字符串必须包含数字或字母|
|weights.type|str|量化后权重类型|'int8'/'float4_e2m1'/'hifloat8'|/|
|weights.symmetric|bool|对称量化|TRUE/FALSE|量化数据类型为float4_e2m1时，只支持对称量化|
|weights.strategy|str|量化粒度|'tensor'/'channel'/'group'|量化策略为group时，只支持量化数据类型为float4_e2m1，且float4_e2m1只支持配group|
|inputs.type|str|量化后激活类型|'int8'/'float8_e4m3fn'/'hifloat8'|/|
|inputs.symmetric|bool|对称量化|TRUE/FALSE|量化策略为token时，不支持非对称量化，不支持pergroup量化；量化数据类型为float8_e4m3fn时，只支持对称量化|
|inputs.strategy|str|量化粒度|'tensor'/'token'|量化数据类型为float8_e4m3fn时，只支持量化策略为tensor|
|algorithm|dict|量化使用的算法配置|{'smoothquant'}|/|
|algorithm.smoothquant.smooth_strength|float|smoothquant算法参数：迁移强度|0-1|不包含0/1|


## 2 量化示例

### 2.1 使用接口方式调用

**step 1.**  请在当前目录执行如下命令运行示例程序，用户需根据实际情况修改示例程序中的模型和数据集路径：

```python
python3 src/run_llama2_samples.py --model_path=/data/Llama2_7b_hf/
```

```python
python3 src/run_qwen_samples.py --model_path=/data/Qwen2-7b/
```

```python
python3 src/run_qwen_samples.py --model_path=/data/Qwen3-8b/
```


若出现如下信息，则说明量化成功：

```none
Test time taken:  1.0 min  59.24865388870239 s
Score:  5.477707
```
其中Score为量化模型PPL，具体数值参考下表：

| 模型 | 校准集 | 数据集 | 量化前PPL | int8*int8量化后PPL | float8_e4m3fn*float4_e2m1量化后PPL | 
| :-: | :-: | :-: | :-: | :-: | :-: |
|LLAMA2-7B|pileval|wikitext2|5.472|5.673|5.589|
|QWEN2-7B|pileval|wikitext2|7.137|7.155|7.252|
|QWEN3-8B|pileval|wikitext2|9.715|9.861|9.931|


### 2.2 hifloat8 * hifloat8 精度参考（Qwen3-8B + wikitext2）

使用`HIFP8_SMOOTHQUANT_CFG`（weights: hifloat8/channel，inputs: hifloat8/tensor，skip_layers=`{'lm_head'}`），
在Qwen3-8B、wikitext2-raw-v1/test上做全量PPL评估，量化前后精度对比如下：

2048-token全量评估（评估段数146，量化前PPL 9.7252）：

| smooth_strength | 量化后PPL | PPL delta | 相对劣化 |
| ---: | ---: | ---: | ---: |
| 0.5 | 9.7719 | +0.0467 | +0.4804% |
| 0.8 | 9.7357 | +0.0105 | +0.1080% |

4096-token全量评估（评估段数73，量化前PPL 8.9975）：

| smooth_strength | 量化后PPL | PPL delta | 相对劣化 |
| ---: | ---: | ---: | ---: |
| 0.5 | 9.0288 | +0.0313 | +0.3481% |
| 0.8 | 9.0359 | +0.0385 | +0.4276% |

其中2048-token评估下`smooth_strength=0.8`精度优于默认`0.5`；4096-token评估下`smooth_strength=0.5`略优于`0.8`。实际使用时可结合序列长度调整`smooth_strength`。


推理成功后，在当前目录会生成量化日志文件./amct_log/amct_pytorch.log

## 3 torchair离线模型导出

### 3.1 前置条件

- 模型已完成AMCT量化，且为`amct.convert`后的量化deploy模型（量化流程参见[2.1 使用接口方式调用](#21-使用接口方式调用)）；
- 已安装CANN、torch_npu及torchair，且版本匹配；
- 执行脚本前需先source CANN环境变量（按实际安装路径调整）：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

- Python依赖（`torch`/`torch_npu`/`torchair`/`transformers`/`amct_pytorch`）已安装且版本匹配；若tokenizer需在线下载而网络不通，设置镜像`export HF_ENDPOINT=https://hf-mirror.com`。

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

### 3.2 使用方式

LLM网络（如Qwen/Llama类模型），导出输入为真实文本token化后的`[1, seqlen]` int64张量，文本默认使用脚本内置样例，也可通过`--input_file`指定本地文本文件：

```python
python3 src/export_torchair_sample.py --model_path=/path/to/quantized_llm/ --seqlen=2048
```

VLM网络（如CLIP-ViT类模型），导出输入为真实图片预处理后的`[batch_size, 3, H, W]`张量：

```python
python3 src/export_torchair_sample.py --model_type=vlm --model_path=/path/to/quantized_clip/ --image_path=/path/to/image.jpg --batch_size=1 --dtype=float16
```

常用参数说明：

| 参数 | 必填 | 默认值 | 说明 |
|:--|:--:|:--:|:--|
|--model_path|是|/|已量化deploy模型目录（内含`model.pth`）或pth文件路径|
|--model_type|否|llm|`llm`/`vlm`，vlm走CLIP-ViT路径|
|--output_dir|否|./torchair_output|导出产物目录|
|--export_name|否|模型目录名|AIR文件名（不含.air后缀）|
|--seqlen|否|2048|LLM导出输入序列长度|
|--input_file|否|内置样例文本|LLM导出输入文本文件|
|--image_path|vlm必填|/|VLM导出用真实图片路径|
|--batch_size|否|1|VLM导出输入batch大小|
|--dtype|否|float16|VLM图像输入精度（float16/bfloat16）|

### 3.3 导出产物

导出成功后，`--output_dir`目录下生成三个文件：

| 文件 | 说明 |
|:--|:--|
|`<export_name>.air`|torchair导出的AIR离线计算图，供离线推理使用|
|`dynamo.pbtxt`|TorchDynamo中间表示（文本格式）|
|`input.bin`|导出输入的二进制文件，供下游离线推理对帧使用|

### 3.4 注意事项

- 导出输入的shape和dtype需与下游离线推理保持一致，请通过`--seqlen`/`--batch_size`/`--dtype`控制；
- LLM导出默认强制eager注意力实现（脚本内自动设置`config._attn_implementation='eager'`）：sdpa注意力在torch_npu上会被重派发为`npu_fusion_attention_v3`算子，当前torchair版本暂无该算子的AscendIR转换器；
- 加载后脚本会自动将量化模块中未随`.npu()`迁移的张量（如`bias`/`offset_bias`）迁移至NPU；
