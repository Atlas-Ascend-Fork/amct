# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""AMCT 风格 façade：amct.prune(model, cfg, ...) 和 ModelOptimizer 兼容的 PrunePass。

设计目标：
 1. 函数签名与 amct.quantize(model, config) 1:1 对齐，dict 配置即可使用。
 2. 同一份配置可被 ModelOptimizer 当作 Pass 加入 prune→quantize→convert 流水线。
 3. 不在库内部触发任何模型下载，调用方负责传入已实例化的 torch.nn.Module。
"""

from typing import Any, Callable, Mapping, Optional, Union

from torch import nn

from amct_pytorch.common.utils.check_params import check_params

from .config import PruneConfig
from .context import BatchAdapter
from .pruner import AutoPruner
from .report import PruneReport

ConfigLike = Union[dict, PruneConfig, type(None)]

# 与 amct.quantize/convert 保持一致：façade 返回 None（in-place 修改模型）。
# 剪枝报告通过 get_last_report() 获取。
_LAST_REPORT: Optional[PruneReport] = None
# tolerance 模式（容差驱动自动剪枝）的结果，通过 get_last_autotune() 获取。
_LAST_AUTOTUNE = None


def _coerce_config(config: ConfigLike) -> PruneConfig:
    if config is None:
        return PruneConfig()
    if isinstance(config, PruneConfig):
        return config
    if isinstance(config, Mapping):
        return PruneConfig(**dict(config))
    raise TypeError(
        f"prune config must be dict / PruneConfig / None, got {type(config)!r}"
    )


@check_params(model=nn.Module, config=(dict, PruneConfig, type(None)))
def prune(
    model: nn.Module,
    config: ConfigLike = None,
    *,
    data: Any = None,
    batch_adapter: Optional[BatchAdapter] = None,
    tolerance: Optional[float] = None,
    quality_fn: Optional[Callable[[nn.Module], float]] = None,
    eval_data: Any = None,
    ratio_grid: Optional[Any] = None,
) -> None:
    """对已实例化的 torch 模型执行结构化剪枝（in-place 语义，与 amct.quantize 一致）。

    两种模式：

    1. **固定剪枝率**（``tolerance=None``，默认）：按 ``config`` 中的 prune_ratio 直接剪枝。
    2. **容差驱动自动剪枝**（给定 ``tolerance``）：面向不熟悉剪枝的用户 ——
       只需说“最多可接受多少精度损失”，自动搜索并应用满足容差的最大剪枝率，
       并自动跳过注意力子模块以免破坏 LLM。结果经 :func:`get_last_autotune` 获取::

           amct.prune(model, data=calib, tolerance=0.02)   # 最多掉 2%
           print(amct.get_last_autotune().summary())

    Parameters
    ----------
    model : torch.nn.Module
        已加载的待剪枝模型，调用方负责通过 ``from_pretrained(local_path)`` 等方式准备。
    config : dict | PruneConfig | None
        剪枝配置；为 None 时使用默认 PruneConfig。
    data : Any
        校准数据（DataLoader / 张量 / 字典 等），方差类剪枝方法需要。
    batch_adapter : Callable | None
        批数据适配函数，可将自定义 batch 结构展开为 ``args, kwargs``。
    tolerance : float | None
        若给定，则启用容差驱动自动剪枝（见 :func:`prune_to_tolerance`）。
    quality_fn, eval_data, ratio_grid :
        仅在 tolerance 模式下透传给 :func:`prune_to_tolerance`。

    Notes
    -----
    本函数与 ``amct.quantize`` 一样不返回值；统计与警告经 :func:`get_last_report`，
    容差模式的搜索结果经 :func:`get_last_autotune` 获取。
    """
    global _LAST_REPORT, _LAST_AUTOTUNE
    if tolerance is not None:
        # 延迟导入避免循环依赖。
        from .autotune import DEFAULT_RATIO_GRID, prune_to_tolerance
        result = prune_to_tolerance(
            model, config, data=data, tolerance=tolerance,
            quality_fn=quality_fn, eval_data=eval_data,
            ratio_grid=ratio_grid if ratio_grid is not None else DEFAULT_RATIO_GRID,
            batch_adapter=batch_adapter, apply=True,
        )
        _LAST_AUTOTUNE = result
        _LAST_REPORT = result.report
        return

    cfg = _coerce_config(config)
    # AutoPruner 默认 copy_model=True，会克隆模型；
    # 为对齐 amct.quantize 的 in-place 语义，这里强制关闭克隆。
    cfg.copy_model = False
    pruner = AutoPruner(cfg)
    pruned = pruner(model, data=data, batch_adapter=batch_adapter)
    # 即便 copy_model=False，AutoPruner 仍可能返回新对象（兼容路径），
    # 兜底将状态拷回原模型，保证调用方持有的引用始终有效。
    if pruned is not model:
        model.__dict__.update(pruned.__dict__)
    _LAST_REPORT = pruner.last_report
    _LAST_AUTOTUNE = None


def get_last_report() -> Optional[PruneReport]:
    """返回上一次 :func:`prune` 调用产生的 :class:`PruneReport`，未调用过时返回 None。"""
    return _LAST_REPORT


def get_last_autotune():
    """返回上一次 tolerance 模式剪枝的 ``AutoTuneResult``；非容差模式或未调用时为 None。"""
    return _LAST_AUTOTUNE


class PrunePass:
    """ModelOptimizer 风格 Pass：可与 InsertQuantizeModulePass 等串接。

    用法::

        from amct_pytorch.classic.ptq.optimizer.model_optimizer import ModelOptimizer
        from amct_pytorch.experimental.pruning import PrunePass

        opt = ModelOptimizer()
        opt.add_pass(PrunePass(prune_cfg, data=calib))
        opt.add_pass(InsertQuantizeModulePass(layer_cfg))
        opt.do_optimizer(model)
    """

    def __init__(
        self,
        config: ConfigLike = None,
        *,
        data: Any = None,
        batch_adapter: Optional[BatchAdapter] = None,
    ) -> None:
        cfg = _coerce_config(config)
        cfg.copy_model = False
        self._pruner = AutoPruner(cfg)
        self._data = data
        self._batch_adapter = batch_adapter
        self.last_report: Optional[PruneReport] = None

    def run(self, model: nn.Module) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError(f"PrunePass.run expects torch.nn.Module, got {type(model)!r}")
        pruned = self._pruner(model, data=self._data, batch_adapter=self._batch_adapter)
        if pruned is not model:
            model.__dict__.update(pruned.__dict__)
        self.last_report = self._pruner.last_report
