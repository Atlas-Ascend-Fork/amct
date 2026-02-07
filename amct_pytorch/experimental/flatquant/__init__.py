from .flat_quant_module.default_model_utils import FlatQuantAttention, FlatQuantMLP
from .npu_flat_quant_module.flat_quant_module import NpuFlatQuantAttention, NpuFlatQuantMLP

# TODO: hope to change to amct_pytorch.quantize.algorithm_register, but currently this would result in circular import
from amct_pytorch.algorithm import AlgorithmRegistry

AlgorithmRegistry.register('flatquant', 'LlamaAttention', FlatQuantAttention, NpuFlatQuantAttention)
AlgorithmRegistry.register('flatquant', 'LlamaMLP', FlatQuantMLP, NpuFlatQuantMLP)

AlgorithmRegistry.register('flatquant', 'Qwen3Attention', FlatQuantAttention, NpuFlatQuantAttention)
AlgorithmRegistry.register('flatquant', 'Qwen3MLP', FlatQuantMLP, NpuFlatQuantMLP)