# sequential_calibrate

## 产品支持情况

| 产品                    | 是否支持 |
| ----------------------- | -------- |
| Ascend 950PR&950DT系列产品 | √        |
| Atlas A3系列产品        | √        |
| Atlas A2系列产品        | √        |

## 功能说明

逐层校准接口。将 `quantize` 接口生成的量化校准模型按 block（如大语言模型的每个 decoder layer、视觉 Transformer 的每个 encoder layer）切分，逐 block 运行校准数据收集统计信息，并将该 block 的算法级校准逻辑分派给对应算法注册的 block calibrator，把搜索/优化得到的量化参数写回模型。

当前该接口主要配合 `auto_smooth` 算法使用：对每个 block 在预设的平滑强度范围内做网格搜索，选择使 block 量化输出与原始输出均方误差（MSE）最小的平滑强度与量化参数。接口本身与算法解耦，后续算法（如 FlatQuant 等）只需注册对应的 block calibrator 即可复用。

> 说明：本接口为试验接口，后续版本可能变更，请谨慎使用。

## 函数原型

```python
sequential_calibrate(model, calib_set, blocks=None, quant_result_dir=None, **kwargs)
```

## 参数说明

| 参数名     | 输入/输出 | 说明 |
| ---------- | --------- | ---- |
| model      | 输入/输出 | 含义：`quantize` 接口生成的量化校准模型，校准完成后量化参数会写回到该模型。<br/>数据类型：torch.nn.Module。 |
| calib_set  | 输入      | 含义：校准数据。可迭代对象，每一项为一个输入 tensor、一个 dict（按关键字参数方式传给模型前向），或一个 tuple/list（按位置参数顺序展开传给模型前向，如 `(input_ids, attention_mask)`）。接口入口会先将该可迭代对象物化为 list，因此一次性迭代器（如 generator）也被完整支持（每个 block 重放一次）。<br/>数据类型：iterable。 |
| blocks     | 输入      | 含义：可选，block 模块名列表，或 block 名到其量化层名列表的 dict。不传时自动推断：取量化层名中最后一个数字段所在的最近祖先模块作为 block（如 `layers.0`）。<br/>数据类型：list/dict，默认 None。 |
| quant_result_dir | 输入 | 含义：可选，调用方已创建的目录路径，用于持久化校准结果（`quant_manifest.json` + 逐层 safetensors）；不传则仅内存校准不落盘。<br/>数据类型：str，默认 None。 |
| kwargs     | 输入      | 含义：透传给 block calibrator 的附加参数。当前支持 `forward_kwargs`（模型前向的额外参数）、`log_blocks`（list，仅对指定 block 名列表打印完整网格搜索日志）、`log_block_num`（int，仅对前 N 个 block 打印完整日志，其余仅一行摘要），以及算法特定参数（如 `args`、`data_loader`）。 |

## 返回值说明

无

## 调用示例

```python
# 建立待进行量化的网络图结构
ori_model = build_model()
model = copy.deepcopy(ori_model)
# 量化配置：auto_smooth 自动平滑量化算法
cfg = {
    'batch_num': 1,
    'quant_cfg': {
        'weights': {'type': 'int8', 'symmetric': True, 'strategy': 'channel'},
        'inputs': {'type': 'int8', 'symmetric': True, 'strategy': 'tensor'},
    },
    'algorithm': {'auto_smooth': {'grid_num': 9, 'smooth_range': [0.1, 0.99]}},
}
# 1. 生成量化校准模型
quantize(model, cfg)
# 2. 逐 block 校准，网格搜索最优平滑强度
calib_set = [torch.randn(4, 768)]
sequential_calibrate(model, calib_set)
# 3. 转换为量化部署模型
convert(model)
```

## 约束

- 调用前必须先通过 `quantize` 接口对模型插入量化算子；否则接口会报错提示无可用校准模块。
- 配置了 `auto_smooth` 算法的层必须通过本接口校准：`auto_smooth` 的平滑强度由 block 级网格搜索确定，legacy 的 `quantize` → 前向校准 → `convert` 流程不会使其生效，此时直接调用 `convert` 会报错提示 `auto_smooth layers are not calibrated`。
- `auto_smooth` 算法的 `grid_num`、`smooth_range` 等参数在 `quantize` 的 config 中配置（详见 [quantize](./quantize.md) 的 `algorithm` 配置）。
- 默认 block 自动推断依赖层名含数字段（如 `layers.0`、`encoder.layers.5`），若模型结构特殊，请通过 `blocks` 参数显式指定 block 边界。

## 扩展说明（注册 block calibrator）

`sequential_calibrate` 按校准器类别（kind）分派，算法名不参与：`forward`（无梯度，如网格搜索 / MSE 重建）与 `gradient`（需梯度，如可学习参数 + 反向传播）各注册一个通用校准器。新增算法无需重复调用 `register_block_calibrator`，而是让自身量化模块满足对应 matcher 并实现块级搜索契约，即可被自动识别与派发。

**calibrator 契约**：一个可调用对象，签名为

```python
calibrator(model, block_name, block, quant_modules, block_inputs, **kwargs)
```

| 参数 | 说明 |
| ---- | ---- |
| model | 完整模型。 |
| block_name | block 模块名，如 `layers.0`。 |
| block | block 子模块。 |
| quant_modules | 该 block 内的量化模块列表。 |
| block_inputs | 校准期间捕获到的该 block 输入 tensor 列表（每个校准样本一个元素）。仅需单个代表输入的算法（如网格搜索）可取 `block_inputs[-1]`；需要整批输入的算法（如梯度下降）可对整个列表 `stack`。 |
| **kwargs | 透传参数（如 `forward_kwargs`）。 |

**注册接口**：`@register_block_calibrator(kind, matcher=None)`。`kind` 只接受 `'forward'` 或 `'gradient'`；`matcher(module) -> bool` 声明该类校准器可处理的量化模块。每类只注册一个通用校准器，新增算法无需重复注册，而是让自身量化模块满足对应 matcher 并实现块级搜索契约（`calib_param_grid` / `apply_calib_param` / `reset_calib`）。

示例（无梯度算法接入：模块实现搜索契约，经 forward matcher 自动识别，无需注册新校准器）：

```python
class NewSearchQuant(BaseQuantizeModule):
    # 无 trainable_params → 满足 forward matcher
    def calib_param_grid(self):
        return [0.1, 0.3, 0.5, 0.7, 0.99]

    def apply_calib_param(self, param):
        self.finalize(float(param))

    def reset_calib(self):
        self.finalized = False
```

说明：该模块经 forward matcher 自动识别，由 `calibrate_forward` 驱动块级网格搜索；算法名不参与分派。
