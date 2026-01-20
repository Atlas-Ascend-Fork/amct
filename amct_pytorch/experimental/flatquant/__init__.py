from .flat_quant_module.default_model_utils import FlatQuantAttention, FlatQuantMLP
from .npu_flat_quant_module.flat_quant_module import NpuFlatQuantAttention, NpuFlatQuantMLP
from amct_pytorch.algorithm import AlgorithmRegistry

AlgorithmRegistry.register('flatquant_attn', 'LlamaAttention', FlatQuantAttention, NpuFlatQuantAttention)
AlgorithmRegistry.register('flatquant_attn_spda', 'LlamaSdpaAttention', FlatQuantAttention, NpuFlatQuantAttention)
AlgorithmRegistry.register('flatquant_mlp', 'LlamaMLP', FlatQuantMLP, NpuFlatQuantMLP)