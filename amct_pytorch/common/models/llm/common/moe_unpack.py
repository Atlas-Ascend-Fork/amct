import torch.nn as nn


class ExpertLinearView(nn.Module):
    def __init__(
        self,
        experts_module,
        expert_idx: int,
        weight_name: str,
        start: int | None = None,
        end: int | None = None,
        materialize: bool = False,
    ):
        super().__init__()
        self.expert_idx = expert_idx
        self.weight_name = weight_name
        self.start = start
        self.end = end
        self.bias = None
        self._weight = None
        if materialize:
            weight_tensor = self._slice_weight(experts_module)
            self._weight = nn.Parameter(weight_tensor.detach().clone(), requires_grad=False)
        else:
            object.__setattr__(self, "_experts_module", experts_module)

    @property
    def weight(self):
        if self._weight is not None:
            return self._weight
        return self._slice_weight(self._experts_module)

    def _slice_weight(self, experts_module):
        weight_tensor = getattr(experts_module, self.weight_name)[self.expert_idx]
        if self.start is None and self.end is None:
            return weight_tensor
        return weight_tensor[self.start:self.end]


class GatedExpertView(nn.Module):
    def __init__(
        self,
        experts_module,
        expert_idx: int,
        hidden_attr: str = "hidden_dim",
        intermediate_attr: str = "intermediate_dim",
        act_attr: str = "act_fn",
        gate_up_name: str = "gate_up_proj",
        down_name: str = "down_proj",
        materialize: bool = False,
    ):
        super().__init__()
        self.hidden_size = getattr(experts_module, hidden_attr)
        self.intermediate_size = getattr(experts_module, intermediate_attr)
        self.act_fn = getattr(experts_module, act_attr)
        self.gate_proj = ExpertLinearView(
            experts_module,
            expert_idx,
            gate_up_name,
            0,
            self.intermediate_size,
            materialize=materialize,
        )
        self.up_proj = ExpertLinearView(
            experts_module,
            expert_idx,
            gate_up_name,
            self.intermediate_size,
            None,
            materialize=materialize,
        )
        self.down_proj = ExpertLinearView(experts_module, expert_idx, down_name, materialize=materialize)



def find_moe_module(block):
    mlp = getattr(block, "mlp", None)
    if mlp is not None and hasattr(mlp, "experts"):
        return mlp

    for module in block.modules():
        if hasattr(module, "experts"):
            return module
    return None