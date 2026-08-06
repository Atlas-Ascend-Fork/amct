# 预定义剪枝配置（Predefined Pruning Configs）<a name="ZH-CN_TOPIC_0000002600000601"></a>

## 产品支持情况<a name="zh-cn_topic_0000002600000601_section106000010001"></a>

<a name="zh-cn_topic_0000002600000601_table106000010002"></a>

| 产品                                        | 是否支持 |
| ------------------------------------------- | -------- |
| Ascend 950PR/Ascend 950DT                   | √        |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √        |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √        |



## 功能说明<a name="zh-cn_topic_0000002600000601_section106000010003"></a>

开箱即用的预设剪枝配置（dict），作为 `amct.prune(model, CFG, data=calib)` 的第二个参数传入。各预设针对不同网络结构（CNN、Dense、MoE）封装了对应的剪枝方法与超参。

## 配置列表<a name="zh-cn_topic_0000002600000601_section106000010004"></a>

下表中的常量均在 `amct_pytorch` 顶层导出，可直接 `amct.<常量名>` 引用。

| 配置常量 | 适用域 | 关键 kwargs（默认值） | 说明 |
| -------- | ------ | -------------------- | ---- |
| CNN_RECONSTRUCT_PRUNE_CFG | cnn | prune_ratio=0.30 | CNN 卷积通道重建剪枝，剪枝后对下游权重做最小二乘重建。 |
| CNN_VARIANCE_PRUNE_CFG | cnn | prune_ratio=0.30 | CNN 卷积通道方差剪枝，按激活方差移除信息量最低的通道。 |
| DENSE_LOWVAR_PRUNE_CFG | dense | prune_ratio=0.50 | Dense（全连接）低方差剪枝，移除方差最低的神经元。 |
| MOE_MASSVAR_PRUNE_CFG | moe | prune_ratio=0.50, boundary=10 | MoE 质量-方差（mass_variance）剪枝，结合权重质量与方差，boundary 控制边界。 |
| MOE_OUTPUT_MERGE_PRUNE_CFG | moe | keep_ratio=0.50, selector="calib_nll" | MoE 输出空间专家合并，以 calib-NLL 选择器在合并与 drop 之间择优。 |
| SENSITIVITY_ALLOC_PRUNE_CFG | dense | prune_ratio=0.50；allocation: strategy="sensitivity", ref_ratio=0.50, min_ratio=0.05, max_ratio=0.90, guard="calib_nll" | 按逐层敏感度分配剪枝率（稳健层多剪、敏感层少剪），以 calib_nll 守卫回退到均匀分配；需传入 data=。 |
| FULL_STRUCTURED_PRUNE_CFG | cnn + dense + moe | cnn: prune_ratio=0.30；dense: prune_ratio=0.50；moe: prune_ratio=0.50, boundary=10 | 一次性对 CNN、Dense、MoE 三种结构同时做结构化剪枝的组合预设。 |

> 说明：以上常量均开启 `"missing_data_policy": "warn_skip"`（缺校准数据时告警并跳过对应方法）。`MOE_VARIANCE_MENU_CFG`、`DENSE_RECOVERY_MENU_CFG`、`CNN_RECOVERY_MENU_CFG` 为守卫式菜单模式（menu）专用配置：`prune` 检测到配置带 `menu` 即改走菜单择优，在 `eval_data` 上实测各候选后择优应用，无需额外开关。三者均从 `amct_pytorch.pruning` 导入，不在顶层 `amct_pytorch` 导出。

## 调用示例<a name="zh-cn_topic_0000002600000601_section106000010012"></a>

```python
import amct_pytorch as amct

# 直接传入预定义剪枝配置作为第二个参数
amct.prune(model, amct.DENSE_LOWVAR_PRUNE_CFG, data=calib)
```
