# Validation Checklist

## 最低必做

- 对所有修改文件做语法检查。
- BF16 baseline 路径至少验证一次，或明确记录阻塞原因。
- 至少构建一层新模型的 quant block。
- 至少枚举出一个 `PtqUnit`。
- 在适配路径上验证：关闭量化后，wrapper 前向与原始浮点模块一致或足够接近。
- 至少验证一个 unit 的 PTQ 参数可以导出。

## 推荐再做

- 跑一个 unit 级 PTQ smoke test。
- 回载保存的 PTQ 参数并跑一次 forward。
- 检查模型特有逻辑是否泄漏到了 `amct_pytorch/workflows/...`。
- 检查是否无必要地改动了 solver。

## 收尾自检

结束前确认：

1. 改动是否主要停留在 `amct_pytorch/common/models/llm/...`？
2. 是否先复用了现有 wrapper，再新增抽象？
3. 关闭量化时，wrapper 是否保住了原始浮点行为？
4. 是否避免了没有必要的框架级重构？
