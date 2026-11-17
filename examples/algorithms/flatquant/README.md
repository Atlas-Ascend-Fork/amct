# AMCT大模型FlatQuant量化

> Qwen3走正式量化链路`quantize -> sequential_calibrate -> convert`，
> 默认构建包即可运行（无需experimental构建选项）。
> Llama2为experimental链路遗留样例（`run_llama2_samples.py`），
> 需`bash build.sh --torch --experimental`构建后运行，后续将迁移或下线。

## 1 量化前提

### 1.1 安装依赖

本sample依赖包可参考[requirements.txt](requirements.txt)

需要注意的是TorchNPU包版本需要与Python、torch包版本相匹配，需要安装CANN包

### 1.2 模型和数据集准备

本sample以Llama2-7b/Qwen3-0.6B，wikitext2数据集为示例，请用户自行下载，并在脚本中传入实际目录。

### 1.3 简易量化配置
本sample中使用的量化配置已经内置在工具中，可以通过下述方式获取并使用：

正式链路（Qwen3）的内置量化配置可通过下述方式获取：

`from amct_pytorch.common.config.config import INT8_INT4_FLAT_QUANT_CFG`（W4A8，仅用于伪量化精度验收）
`from amct_pytorch.common.config.config import INT4_INT4_FLAT_QUANT_CFG`（W4A4，支持全链路含部署）

experimental链路（Llama2）使用：

`from amct_pytorch.experimental.flatquant.config import INT4_FLAT_QUANT_CFG`

我们在量化配置中增加了'use_down_quant'配置，用来控制down_proj是否进行量化，对于down_proj量化敏感的模型，可以跳过down_proj的量化。
如果需要修改详细配置，请参考资料构造需要的量化配置dict。

flatquant算法支持如下部分的量化：
- 真量化：self_attn中q_proj，k_proj，v_proj以及mlp中up_proj，gate_proj，down_proj部分的权重及输入共同量化（使用Kronecker product），其中输入为per token，权重为per channel，两者均为对称量化
- 伪量化：kv_cache及o_proj（现阶段建议关闭，参见`INT4_FLAT_QUANT_CFG`）

支持的量化类型以及量化配置：

| 字段 | 类型 | 说明 | 取值范围 | 注意事项 |
|:--| :-: | :-- | :-: | :-- |
|skip_layers|list[str] / set[str]|跳过量化的层 |/|对配置列表或集合中的每个字符串做模糊匹配；当层名包含任一配置字符串时，跳过该层量化，不生成量化配置。例如：{"lm_head"}。|
|algorithm|dict|量化使用的算法配置|{'flatquant'}|参考`INT8_INT4_FLAT_QUANT_CFG`/`INT4_INT4_FLAT_QUANT_CFG`示例。|

> 说明：激活/权重的位宽（如int8/int4、int4/int4）、对称性与粒度（激活per-token、权重per-channel）在`quant_cfg.inputs`/`quant_cfg.weights`中配置，与算法参数分开。组合支持：int8/int4与int4/int4。

## 2 量化示例

### 2.1 llama2量化

**step 1.**  请在当前目录执行如下命令运行示例程序，并根据实际情况修改示例程序中的模型路径：
```python
python3 src/run_llama2_samples.py --model_path <llama2 model path>
```

若出现如下信息，则说明量化成功：
```none
INFO All done!
```

其中日志里如下信息为评测任务结果（百分比准确率）：
```
INFO ACC: {'arc_challenge': 42.83, 'arc_easy': 70.88, 'hellaswag': 73.63, 'lambada_openai': 72.0, 'piqa': 77.48, 'winogrande': 67.88, 'acc_avg': 67.45}
```
如下信息为perplexity（wikitext, max length 512）：
```
INFO PPL score: 5.870388984680176
```
如下信息为原始模型及真量化模型的推理速度（ms）:
```
INFO Time diff orig: 929.0580000000001 ms
INFO Time diff after real quant: 139.707 ms
```

脚本运行结束后，在当前目录会生成并保存校准后参数`./outputs/llama2_7b/flat_matrices.pth`及量化日志文件`./amct_log/amct_pytorch.log`。如果想直接加载校准参数则使用如下设定：
```python
python3 src/run_llama2_samples.py --model_path <llama2 model path> --load_matrix --flat_matrix_path <matrix path, e.g. ./outputs/llama2_7b/flat_matrices.pth>
```

### 2.2 qwen3量化（正式链路）

正式链路示例：`quantize`生成量化校准模型后，由`sequential_calibrate`逐层做梯度校准
（框架按block捕获输入，FlatQuant的gradient block calibrator对每层做AdamW训练），
校准产物`flat_matrices.pth`可保存/复用，最后`convert`固化为NPU部署模块（W4A4真量化）。

**step 1.**  请在当前目录执行如下命令运行示例程序，并根据实际情况修改示例程序中的模型路径：
```python
python3 src/run_qwen_samples.py --model_path <qwen3-0.6b model path>
```
复用已校准矩阵跳过校准：
```python
python3 src/run_qwen_samples.py --model_path <qwen3-0.6b model path> --load_matrix --flat_matrix_path ./outputs/qwen/flat_matrices.pth
```

若出现如下信息，则说明量化成功：
```none
INFO All done!
```

示例展示的是模型量化前后根据prompt生成的不同结果：
prompt为:
```
prompt = "Give me a short introduction to the Ascend Model Compression Toolkit(AMCT). /no_think"
```

量化前的生成结果为：
```
content: <think>
<>
The Ascend Model Compression Toolkit (AMCT) is a powerful tool designed to ...
```

量化后的生成结果为：
```
content: <think>
<>
The Ascend Model Compression Toolkit (AMCT) is a powerful tool designed to ...
```

> 量化完成后，如需将量化模型导出为 torchair 离线模型（.air）用于 ATC 转 om 板端推理，请参考 [AMCT大模型torchair离线模型导出](../export/README.md)。
