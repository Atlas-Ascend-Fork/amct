# Kimi K3 MXFP 真量化评测

`--eval_mode npu` 在 Kimi K3 的 block 粒度评测中启用 MXFP A4W4/A8W8
NPU 矩阵乘。各投影的位宽由 bit YAML 决定，A16W16 投影保留 BF16。
默认 `bf16`、`quant` 路径及其他模型的加载逻辑不变。

## 运行

```bash
python -m amct_pytorch.cli.llm.eval \
  --model /path/to/kimi-k3 \
  --model_name kimi_k3 \
  --eval_mode npu \
  --granularity block \
  --device npu:0 \
  --quant_dtype mxfp \
  --quant_target moe \
  --bit_config amct_pytorch/configs/kimi_k3_mxfp4_npu.yaml \
  --mxfp_checkpoint_format e2m1-e8m0 \
  --output_dir ./outputs/kimi_k3_npu
```

示例 YAML 仅将路由专家的 `w1/w3/w2`（分别映射到策略名称
`gate_proj/up_proj/down_proj`）设为 A4W4。共享专家、额外的
`routed_expert_up_proj/down_proj` 和其他层保持 A16W16。

## 权重与 scale

Kimi 适配器在 `kimi_k3.py` 中按每个投影的完整键名成对读取，可以跨分片。
原始模型在 meta 上构建，逐投影加载；不会先展开整个路由专家层的 BF16 权重。

| 来源 | 权重 | scale | 执行前处理 |
|---|---|---|---|
| BF16/FP16/FP32 | `.weight`，逻辑 `[N,K]` | 无 | 转 BF16，在首次使用时按 W4/W8 动态量化并缓存 |
| MXFP4 | `.weight_packed`，uint8 `[N,K/2]` | `.weight_scale`，uint8 `[N,K/32]` | 保留四位编码，转置后重新打包，不反量化、不重新舍入 |
| MXFP8 | `.weight`，float8_e4m3fn `[N,K]` | `.weight_scale`，uint8 `[N,K/32]` | 转置低比特权重 |

约定 FP4 为 E2M1，同一字节的低四位在前；scale 为 E8M0 的原始字节，
不是普通浮点倍率。二维 scale 转为 NPU 所需的
`[ceil(K/64), N, 2]` 布局，奇数分组补零。

索引文件不能证明编码类型，因此加载低比特模型时需显式声明
`--mxfp_checkpoint_format`：FP4 使用 `e2m1-e8m0`，FP8 使用
`e4m3-e8m0`，同时包含两种格式时使用 `mixed-e8m0`。
这是一项输入格式声明，必须与实际 checkpoint 相符；不会自动猜测 INT4、
int32 打包、浮点 scale 或其他布局。纯浮点模型不需要该参数。

适配器检查键名配对、有效位宽、dtype、shape、K 的 32 对齐、FP4 的 N 偶数、
scale 的 E8M0 NaN 编码 255，以及 FP8 权重非有限值。
文件中已经量化的权重若被配置为 A16W16，或者没有选中对应 quant_target，
会报冲突，不会偷偷反量化或忽略权重。

## 前向计算

真量化模块 `NpuMxLinear` 位于 `quantization/dtypes/mxfp.py`。
它直接替换 Kimi 原始投影，不经过已有的激活伪量化 wrapper。
每次 forward 将当前激活转为 BF16，调用 `npu_dynamic_mx_quant` 生成
低比特激活和动态 scale，再调用 `npu_quant_matmul`；输出为 BF16。
权重和 scale 注册为 buffer，预量化权重不保留 BF16 副本。
各专家沿用 Kimi 的多 NPU 放置和输入输出搬运逻辑。

当前支持 A4W4、A8W8 和 A16W16；A8W4、A16W4 等组合明确报错。
首版不接收 PTQ 算法/参数目录，也不支持量化 attention cache 或 model 粒度真量化评测。
PTQ 权重应先导出为符合上述约定的最终 checkpoint；本路径不会再次应用 PTQ 参数。

## 验证与环境

需要目标硬件、CANN 和 torch_npu 支持 MXFP 动态量化及矩阵乘；仅安装
仓库的默认依赖不能保证硬件支持。接口依据：
[npu_dynamic_mx_quant](https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_dynamic_mx_quant.md)、
[npu_quant_matmul](https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_quant_matmul.md)。

CPU 单测使用显式的算子替身，覆盖布局保真、scale 配对、位宽选择、混合加载、
权重只量化一次和激活每次量化；不能替代目标 NPU 上的算子数值测试与整模型 PPL。
运行真实模型前，应先在目标 NPU 上比较 A4W4/A8W8 单层输出与反量化参考，
再用上述命令验证 PPL 和峰值内存。
