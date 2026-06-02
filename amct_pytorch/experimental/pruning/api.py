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

from typing import Any, Mapping, Optional, Union

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
) -> None:
    """对已实例化的 torch 模型执行结构化剪枝（in-place 语义，与 amct.quantize 一致）。

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

    Notes
    -----
    本函数与 ``amct.quantize`` 一样不返回值；如需读取本次剪枝的统计与警告，
    请调用 :func:`get_last_report`。
    """
    global _LAST_REPORT
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


def get_last_report() -> Optional[PruneReport]:
    """返回上一次 :func:`prune` 调用产生的 :class:`PruneReport`，未调用过时返回 None。"""
    return _LAST_REPORT


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
