# quant export 产物校验工具

`scripts/verify_export.py` —— 校验 amct `deploy` 导出目录的 **int 量化产物是否反量化保真**，与具体模型解耦（按 `*.weight` / `*.weight_scale` 命名通用，适用任意 LLM adapter 的导出结果）。

## 校验内容

对导出目录里每个量化 Linear（`*.weight` 为 I8、配对 `*.weight_scale` F32 per-output-channel）：

1. **结构自洽**：`weight`(I8) 与 `weight_scale` 配对、dtype、形状一致。
2. **反量化保真**（给 `--ref_model` 时）：导出反量化 `w_deq = weight.int8 * weight_scale` 应等于「对原始 BF16 权重按同 scale 做对称伪量化-反量化」的结果（容差按量化步长）；并报告 `w_deq` vs 原始 BF16 的平均余弦相似度与平均相对误差（量化引入的权重误差量级）。

只读校验，**不跑模型前向、不需要完整模型**，按需抽样层 —— 适合大模型只导出/抽取部分层做产物核验。

## 用法

```bash
python scripts/verify_export.py \
  --deploy_dir <amct deploy 输出目录> \
  --ref_model  <原始 BF16 模型目录>   \   # 可选，做量化误差对比
  --layers 0,1,2 \                        # 可选，默认全部导出层
  --max_tensors_per_layer 8               # 每层抽查上限
```

## 输出示例

```
[verify] 量化张量数: 23; 形状自洽: 23/23
[verify] vs 原始BF16: 抽查 7 个; 平均余弦 1.000000; 最大相对误差 0.0190
[verify] 通过：所有抽查量化张量 weight/scale 配对、I8、形状自洽，且反量化与原始 BF16 同 scale 伪量化一致
```

## 适用范围与边界

- 当前覆盖 **int 对称 per-output-channel**（`weight_scale` 形状 `[out,1]`）的导出，对应 amct W8A8/W4A8 的 int 路径。
- mxfp（`weight_packed` + block scale）路径暂未覆盖（后续可扩 dtype 分支）。
- 校验产物“是否忠实落盘量化方案”，不替代端到端 PPL（精度结论仍以 eval 的 `ppl_quant`/`delta` 为准）。
