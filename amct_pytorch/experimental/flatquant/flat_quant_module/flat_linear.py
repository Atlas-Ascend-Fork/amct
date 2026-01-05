import torch
import torch.nn as nn
import torch.nn.functional as F

from .quant_utils import WeightQuantizer, ActivationQuantizer
from .flat_utils import kronecker_matmul
from .config_utils import InnerConfig

class FlatQuantizedLinear(nn.Module):
    def __init__(self, linear: nn.Linear, quantizer_size: int=4096):
        super(FlatQuantizedLinear, self).__init__()
        self.linear = linear

        self.weight_quantizer = WeightQuantizer(shape=(quantizer_size, 1))
        self.weight_quantizer.configure(InnerConfig.w_bits.value, perchannel=True, sym=not(InnerConfig.w_asym.value), mse=False)
        self.act_quantizer = ActivationQuantizer(bits=InnerConfig.a_bits.value, sym=not(InnerConfig.a_asym.value), lac=InnerConfig.lac.value)

        self.lwc = InnerConfig.lwc.value
        if self.lwc:
            lwc_dim = self.linear.weight.shape[0] if self.lwc else -1
            init_value = 4.
            linear_device = self.linear.weight.device

            self.clip_factor_w_max = nn.Parameter(torch.ones((lwc_dim, 1)).to(linear_device)*init_value, requires_grad=True)
            self.clip_factor_w_min = nn.Parameter(torch.ones((lwc_dim, 1)).to(linear_device)*init_value, requires_grad=True)

            self.sigmoid = nn.Sigmoid()

        self._eval_mode = False

    def apply_wclip(self, weight):
        wmin, wmax = weight.min(1, keepdim=True)[0], weight.max(1, keepdim=True)[0]
        wmax *= self.sigmoid(self.clip_factor_w_max)
        wmin *= self.sigmoid(self.clip_factor_w_min)
        weight = torch.clamp(weight, min=wmin, max=wmax)
        return weight

    def apply_trans(self, weight, qa_trans):
        if isinstance(qa_trans, list):
            weight = kronecker_matmul(weight, qa_trans[0].to(weight), qa_trans[1].to(weight))
        else:
            weight = qa_trans(weight, inv_t=True)
        return weight
    
    def get_quantized_weight(self, qa_trans=None, out_trans=None, quantonly=False):
        weight = self.linear.weight.data
        # quantization-adaptive transform
        if qa_trans is not None:
            weight = self.apply_trans(weight, qa_trans)
        # learnable weight clipping 
        if self.lwc:
            weight = self.apply_wclip(weight)
        if out_trans is not None:
            weight = out_trans(weight.T).T
        
        # quantize weight
        self.weight_quantizer.find_params(weight)
        if quantonly:
            weight, scale = self.weight_quantizer.quantize(weight, quantonly=True)
            return weight, scale
        else:
            weight = self.weight_quantizer.quantize(weight)
            return weight

    def _ori_forward(self, hidden_states):
        return self.linear(hidden_states)

    def _train_forward(self, hidden_states, qa_trans=None, out_trans=None):
        weight = self.get_quantized_weight(qa_trans=qa_trans, out_trans=out_trans)

        # quantize activation
        hidden_states = self.act_quantizer(hidden_states)

        if out_trans is not None and self.linear.bias is not None:
            bias = out_trans(self.linear.bias.data)
        else:
            bias = self.linear.bias
        output = F.linear(hidden_states, weight, bias)
        return output

    def forward(self, hidden_states, qa_trans=None, out_trans=None):
        if not self._eval_mode:
            return self._train_forward(hidden_states, qa_trans=qa_trans, out_trans=out_trans)
        else:
            return self._eval_forward(hidden_states)

    def _eval_forward(self, hidden_states):
        x_dtype = hidden_states.dtype
        hidden_states = self.act_quantizer(hidden_states).to(x_dtype)

        output = self.linear(hidden_states)
        return output

    def reparameterize(self, qa_trans=None, out_trans=None):
        weight = self.linear.weight.data
        ori_dtype = weight.dtype
        weight = weight.to(torch.float64)
        # quantization-adaptive transform
        if qa_trans is not None:
            weight = self.apply_trans(weight, qa_trans)
        if self.lwc:
            weight = self.apply_wclip(weight)
        if out_trans is not None:
            weight = out_trans(weight.T).T
        if out_trans is not None and self.linear.bias is not None:
            self.linear.bias.data = out_trans(self.linear.bias.data)
        
        self.linear.weight.data = weight.to(ori_dtype)
        self._eval_mode = True