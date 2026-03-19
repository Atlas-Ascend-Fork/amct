# AMCT大模型SINQ量化

## 1 量化前提

### 1.1 方法简介

SINQ是一种免校准的仅权重量化算法，适用于大模型 `Linear` 算子。与依赖校准数据的量化方法不同，SINQ直接基于权重本身完成量化参数搜索，不需要准备校准集，也不需要执行额外的校准前向。该方法的一个突出特点是量化准备速度快，同时通常能够保持较高精度；在大多数场景下，直接使用免校准SINQ即可获得较好的量化效果。

当前AMCT中的SINQ实现面向首版原生Ascend部署场景，使用 `quantize() + convert()` 接口完成量化与部署转换，复用AMCT已有的权重量化Linear部署算子。

如果用户希望在保持SINQ接口不变的前提下进一步利用少量校准数据，也可以在 `quantize()` 与 `convert()` 之间执行一次或多次准备前向。此时SINQ会利用输入激活重估量化参数，行为上接近A-SINQ；如果不执行该步骤，则默认使用免校准SINQ。可选校准属于额外增强手段，适合在需要进一步逼近最佳精度表现时使用。

### 1.2 安装依赖

本sample依赖包可参考[requirements.txt](requirements.txt)。

需要注意的是 `torch_npu` 包版本需要与Python、torch包版本相匹配，并且需要正确安装CANN运行环境。

### 1.3 模型和数据集准备

本sample以Qwen3-1.7B模型、wikitext2数据集为示例。数据集可在线加载，模型需要用户自行下载并在执行脚本时指定本地路径。

### 1.4 使用约束

当前SINQ实现支持范围如下：

- 仅支持仅权重量化
- 仅支持 `Linear` 算子
- 支持 `int4`、`int8` 量化
- 仅支持 `group` 粒度
- 当前版本保留 `batch_num` 配置字段，但SINQ为免校准算法，该字段不参与量化搜索

若模型中存在不适合量化的层，例如输出头层，可通过 `skip_layers` 配置跳过。

## 2 量化配置

### 2.1 简易量化配置

推荐配置如下：

```python
cfg = {
    'batch_num': 1,
    'quant_cfg': {
        'weights': {
            'type': 'int4',
            'symmetric': False,
            'strategy': 'group',
            'group_size': 64
        },
    },
    'algorithm': {'sinq': {'order': 16}},
    'skip_layers': {'lm_head'}
}
```

其中：

- `weights.symmetric=False` 为大模型常用推荐配置
- `weights.group_size=64` 为默认推荐配置
- `algorithm.sinq.order=16` 表示Sinkhorn迭代次数，通常可作为默认值使用

如果希望直接使用AMCT内置预置配置，可使用：

```python
from amct_pytorch import INT4_SINQ_WEIGHT_QUANT_CFG, INT8_SINQ_WEIGHT_QUANT_CFG
```

其中：

- `INT4_SINQ_WEIGHT_QUANT_CFG`：内置 `int4 + group_size=64 + asymmetric`
- `INT8_SINQ_WEIGHT_QUANT_CFG`：内置 `int8 + group_size=64 + asymmetric`

### 2.2 支持的量化配置

| 字段 |类型| 说明 | 取值范围 | 注意事项 |
|:--| :-: | :-- | :-: | :-- |
|batch_num|uint32|量化使用的batch数量|1|SINQ为免校准算法，当前实现中该参数保留但不参与量化搜索|
|skip_layers|str/set|跳过量化的层|/|支持模糊匹配；当配置字符串为层名字串，或与层名一致时，跳过该层量化，不生成量化配置|
|weights.type|str|量化后权重类型|'int4'/'int8'|推荐优先使用 `int4` 或 `int8`|
|weights.symmetric|bool|对称量化|TRUE/FALSE|建议大模型场景使用FALSE|
|weights.strategy|str|量化粒度|'group'|当前SINQ仅支持group策略|
|weights.group_size|int|group粒度大小|32的整数倍|默认推荐配置64|
|algorithm|dict|量化使用的算法配置|{'sinq': {'order': 16}}|当前需显式指定 `sinq` 算法|
|algorithm.sinq.order|int|SINQ算法参数：Sinkhorn迭代次数|正整数|推荐配置16|

## 3 量化示例

### 3.1 使用接口方式调用

用户可以参考如下流程调用接口完成量化与部署转换：

```python
from amct_pytorch import quantize, convert
from amct_pytorch import INT4_SINQ_WEIGHT_QUANT_CFG

quantize(model, INT4_SINQ_WEIGHT_QUANT_CFG)
convert(model)
```

完成转换后，模型中的目标 `Linear` 层会被替换为AMCT部署态权重量化算子，可直接进行推理或精度评估。

如果需要启用激活辅助的SINQ，可在 `quantize()` 与 `convert()` 之间增加一次准备前向，例如：

```python
amct.quantize(model, amct.INT4_SINQ_WEIGHT_QUANT_CFG)

# 可选：使用少量校准输入执行准备前向
amct.prepare_sinq(model, calib_inputs)

amct.convert(model)
```

若省略该前向，则仍为默认的免校准SINQ流程；若执行该前向，则可以在保持相同部署接口的前提下进一步提升量化质量，用于追求更优精度表现的场景。

### 3.2 示例脚本流程

典型流程如下：

1. 加载大模型和tokenizer
2. 构造SINQ量化配置
3. 调用 `quantize(model, cfg)` 执行量化
4. 调用 `convert(model)` 将量化模块转换为部署模块
5. 使用wikitext2等数据集评估量化后模型PPL

示例代码如下：

```python
import amct_pytorch as amct

# 方式一：直接使用内置配置
cfg = amct.INT4_SINQ_WEIGHT_QUANT_CFG
amct.quantize(model, cfg)
amct.convert(model)

# 方式二：自定义为int8
cfg = amct.INT8_SINQ_WEIGHT_QUANT_CFG
amct.quantize(model, cfg)
amct.convert(model)
```

### 3.3 精度评估说明

SINQ本身不依赖校准数据，通常能够以较快的量化准备时间获得较高质量结果；量化完成后仍建议通过下游评测数据集验证精度。大模型场景中，可使用wikitext2等常见数据集评估PPL，以观察量化前后的精度变化。如果需要进一步逼近原始模型精度，也可以增加一次准备前向，利用少量校准输入作为额外增强手段提升量化质量。

## 4 注意事项

- 当前版本不支持激活量化
- 当前版本不支持除 `Linear` 之外的其他算子
- 当前版本仅支持 `int4`、`int8` 两种权重量化类型
- 当前版本不支持 `tensor`、`channel` 等其他量化粒度
- 如果模型中存在对精度敏感的层，建议通过 `skip_layers` 先跳过，再根据结果逐步收敛配置

推理成功后，在当前目录会生成量化日志文件 `./amct_log/amct_pytorch.log`。
