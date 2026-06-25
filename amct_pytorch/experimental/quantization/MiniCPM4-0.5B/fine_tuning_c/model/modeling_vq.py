import torch
from .minicpm4.modeling_minicpm import MiniCPMForCausalLM, MiniCPMConfig
from .lcvq import LCVQHadLinear

class LCVQConfig(MiniCPMConfig):
    model_type = "lcvq"
    def __init__(self, *args, vec_size=4, vec_bit=2, group_size=128, proj_type="bypass", use_hadamard=False, codebook_trainable=False, **kwargs):
        self.vec_size = vec_size
        self.vec_bit = vec_bit
        self.group_size = group_size
        self.proj_type = proj_type
        self.use_hadamard = use_hadamard
        self.codebook_trainable = codebook_trainable
        super().__init__(*args, **kwargs)
    
class LCVQForCausalLM(MiniCPMForCausalLM):
    config_class = LCVQConfig
    def __init__(self, config: LCVQConfig):
        super().__init__(config)
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = getattr(config, "head_dim", hidden_size//num_heads)
        num_key_value_heads = config.num_key_value_heads
        intermediate_size = config.intermediate_size
        vec_size = config.vec_size
        vec_bit = config.vec_bit
        proj_type = config.proj_type
        code_book_args = {"vec_size": vec_size, "vec_bit": vec_bit, "proj_type": proj_type}
        attention_bias = getattr(config, "attention_bias", True)
        for layer in self.model.layers:
            layer.self_attn.q_proj = LCVQHadLinear(hidden_size, num_heads * head_dim, bias=attention_bias, **code_book_args)
            layer.self_attn.k_proj = LCVQHadLinear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias, **code_book_args)
            layer.self_attn.v_proj = LCVQHadLinear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias, **code_book_args)
            layer.self_attn.o_proj = LCVQHadLinear(num_heads * head_dim, hidden_size, bias=False, **code_book_args)
            layer.mlp.gate_proj = LCVQHadLinear(hidden_size, intermediate_size, bias=False, **code_book_args)
            layer.mlp.up_proj = LCVQHadLinear(hidden_size, intermediate_size, bias=False, **code_book_args)
            layer.mlp.down_proj = LCVQHadLinear(intermediate_size, hidden_size, bias=False, **code_book_args)                
        self.config = config