# AMCT大模型AWQ量化

## 1 量化前提

### 1.1 安装依赖

本sample依赖包可参考[requirements.txt](requirements.txt)

需要注意的是torch_npu包版本需要与Python、torch包版本相匹配，需要安装CANN包。

### 1.2 模型和数据集准备

本sample以Llama2-7b，qwen2-7b，pileval数据，wikitext2数据集为示例，请用户自行下载，并根据实际保存目录修改utils.py文件中get_llama2，get_loaders， get_calib_dataset等获取模型或数据函数中的读取路径。

### 1.3 简易量化配置
本sample中使用的量化配置已经内置在工具中，可以通过下述方式获取并使用：

`from amct_pytorch import INT4_AWQ_WEIGHT_QUANT_CFG`

如果需要修改详细配置，请参考资料构造需要的量化配置dict。

awq算法仅支持权重量化，支持的量化类型以及量化配置：

| 字段 |类型| 说明 | 取值范围 | 注意事项 |
|:--| :-: | :-- | :-: | :-- |
|batch_num|uint32|量化使用的batch数量 |1|/|
|skip_layers|str|跳过量化的层 |/|跳过量化层支持模糊匹配，当配置字符串为层名字串，或与层名一致时，跳过该层量化，不生成量化配置。字符串必须包含数字或字母|
|weights.type|str|量化后权重类型|'int4'/'int8'|/|
|weights.symmetric|bool|对称量化|TRUE/FALSE|/|
|weights.strategy|str|量化粒度|'tensor'/'channel'/'group'|/|
|algorithm|dict|量化使用的算法配置|{'awq'}|/|
|algorithm.awq.grids_num|int|awq算法参数：搜索格点数量|/|/|

## 2 量化示例

### 2.1 使用接口方式调用

**step 1.**  请在当前目录执行如下命令运行示例程序，用户需根据实际情况修改示例程序中的模型和数据集路径：

```python
python3 src/run_llama2_samples.py
```

```python
python3 src/run_qwen2_samples.py
```

若出现如下信息，则说明量化成功：

```none
Test time taken:  1.0 min  59.24865388870239 s
Score:  5.477707
```
其中Score为量化模型PPL，具体数值参考下表：

| 模型 | 校准集 | 数据集 | 量化前PPL | 量化后PPL | 
| :-: | :-: | :-: | :-: | :-: |
|LLAMA2-7B|pileval|wikitext2|5.472|5.550|
|QWEN2-7B|pileval|wikitext2|7.137|7.268|


推理成功后，在当前目录会生成量化日志文件./amct_log/amct_pytorch.log