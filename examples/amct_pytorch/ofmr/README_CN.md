# AMCT大模型量化

## 1 量化前提

### 1.1 安装依赖

本sample依赖包可参考[requirements.txt](requirements.txt)

需要注意的是torch_npu包版本需要与Python、torch包版本相匹配，需要安装CANN包

### 1.2 模型和数据集准备

本sample以Llama2-7b，qwen2-7b，qwen3-8b模型，pileval数据，wikitext2数据集为示例。
模型请用户自行下载，并传模型路径到脚本，数据集为在线加载。

### 1.3 简易量化配置
本sample中使用的量化配置已经内置在工具中，可以通过下述方式获取并使用：

`from amct_pytorch import HIFP8_OFMR_CFG`

如果需要修改详细配置，请参考资料构造需要的量化配置dict。

ofmr算法支持仅权重量化和全量化，支持的量化类型以及量化配置：

| 字段 |类型| 说明 | 取值范围 | 注意事项 |
|:--| :-: | :-- | :-: | :-- |
|batch_num|uint32|量化使用的batch数量 |1|/|
|skip_layers|str|跳过量化的层 |/|跳过量化层支持模糊匹配，当配置字符串为层名字串，或与层名一致时，跳过该层量化，不生成量化配置。字符串必须包含数字或字母|
|weights.type|str|量化后权重类型|'float8_e4m3fn'/'hifloat8'|/|
|weights.symmtric|bool|对称量化|TRUE|/|
|weights.strategy|str|量化粒度|'tensor'/'channel'|/|
|inputs.type|str|量化后激活值类型|'float8_e4m3fn'/'hifloat8'|/|
|inputs.symmtric|bool|对称量化|TRUE|/|
|inputs.strategy|str|量化粒度|'tensor'|/|
|algorithm|dict|量化使用的算法配置|{'ofmr'}|/|

## 2 量化示例

### 2.1 使用接口方式调用

**step 1.**  请在当前目录执行如下命令运行示例程序，用户需根据实际情况修改示例程序中的模型路径：

```python
python3 src/run_llama2_samples.py --model_path=/data/Llama2_7b_hf/
```

```python
python3 src/run_qwen_samples.py --model_path=/data/Qwen2-7b/
```

```python
python3 src/run_qwen_samples.py --model_path=/data/Qwen3-8B/
```


若出现如下信息，则说明量化成功：

```none
Test time taken:  1.0 min  59.24865388870239 s
Score:  5.477707
```

其中Score为量化模型PPL，具体数值参考下表：

| 模型 | 校准集 | 数据集 | 量化前PPL | 量化后PPL | 
| :-: | :-: | :-: | :-: | :-: |
|LLAMA2-7B|pileval|wikitext2|5.472|5.505|
|QWEN2-7B|pileval|wikitext2|7.137|7.196|
|QWEN3-8B|pileval|wikitext2|9.715|9.808|

推理成功后，在当前目录会生成量化日志文件./amct_log/amct_pytorch.log
