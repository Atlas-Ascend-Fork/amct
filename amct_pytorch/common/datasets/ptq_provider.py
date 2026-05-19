from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader, TensorDataset

from amct_pytorch.common.models.llm.common.base import PtqUnit


@dataclass
class BlockPtqBatch:
    layer_idx: int
    unit_name: str
    data_loader: DataLoader
    num_samples: int
    kwargs: Optional[dict[str, Any]]
    has_gts: bool = False
    metadata: Optional[dict] = None


class LlmPtqDataProvider:
    """Build iterable PTQ batches for block/attn/expert units."""

    def __init__(self, args, pipeline):
        self.args = args
        self.device = args.device
        self.pipeline = pipeline

    def load_unit_inputs(self, unit: PtqUnit):
        return self.pipeline.load_unit_inputs(self.args.data_dir, unit)

    def build_unit_batch(self, unit: PtqUnit, inps, kwargs, gts=None) -> BlockPtqBatch:
        tensors = [inps]
        if gts is not None:
            tensors.append(gts)
        dataset = TensorDataset(*tensors)
        data_loader = DataLoader(
            dataset,
            batch_size=self.args.cali_bsz,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        return BlockPtqBatch(
            layer_idx=unit.layer_idx,
            unit_name=unit.name,
            data_loader=data_loader,
            kwargs=kwargs,
            num_samples=len(dataset),
            has_gts=gts is not None,
            metadata=unit.metadata or None,
        )

    def materialize_gt(self, inps, ori_module, kwargs=None):
        ori_module.float().eval().to(self.device)
        forward_kwargs = kwargs or {}
        ori_loader = DataLoader(TensorDataset(inps), batch_size=self.args.cali_bsz, shuffle=False)
        gts = []
        with torch.no_grad():
            for (x,) in ori_loader:
                if x.is_floating_point():
                    x = x.to(self.device, dtype=torch.float32)
                else:
                    x = x.to(self.device)
                gt = ori_module(x, **forward_kwargs)
                if isinstance(gt, (tuple, list)):
                    gt = gt[0]
                gts.append(gt.detach())
        torch.npu.empty_cache()
        return torch.cat(gts, dim=0)


    def get_model_data(self) -> dict[str, int | str]:
        return {
            "data_dir": self.args.data_dir,
            "start_block_idx": self.args.start_block_idx,
            "end_block_idx": self.args.end_block_idx,
        }
