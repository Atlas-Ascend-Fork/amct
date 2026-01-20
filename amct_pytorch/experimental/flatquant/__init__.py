from .flat_quant_module.default_model_utils import FlatQuantAttention, FlatQuantMLP
from .npu_flat_quant_module.flat_quant_module import NpuFlatQuantAttention, NpuFlatQuantMLP
from amct_pytorch.quantize import algorithm_register

algorithm_register('flatquant_attn', 'LlamaAttention', FlatQuantAttention, NpuFlatQuantAttention)
algorithm_register('flatquant_attn_spda', 'LlamaSdpaAttention', FlatQuantAttention, NpuFlatQuantAttention)
algorithm_register('flatquant_mlp', 'LlamaMLP', FlatQuantMLP, NpuFlatQuantMLP)