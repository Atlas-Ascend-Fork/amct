import copy
import shutil
from pathlib import Path

import torch
from torch import nn

from amct_pytorch.classic.quantize import quantize
from amct_pytorch.classic.quant_result import reload_quant_params
from amct_pytorch.classic.quantize_op.gptq_module import GPTQuant
from amct_pytorch.classic.quantize_op.linear_awq_module import LinearAWQuant
from amct_pytorch.classic.quantize_op.minmax_module import MinMaxQuant


class RealTwoLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 1024, bias=False)
        self.fc2 = nn.Linear(1024, 1024, bias=False)

    def forward(self, x):
        return self.fc2(torch.nn.functional.silu(self.fc1(x)))


def cfg(alg):
    base = {
        'batch_num': 2,
        'quant_cfg': {
            'weights': {'type': 'int8', 'symmetric': True, 'strategy': 'channel'}
        },
    }
    if alg == 'minmax':
        base['algorithm'] = {'minmax': {}}
    elif alg == 'awq':
        base['algorithm'] = {'awq': {'grids_num': 20}}
    elif alg == 'gptq':
        base['algorithm'] = {'gptq': {}}
    return base


def run(name, config, cls):
    torch.manual_seed(123)
    source = RealTwoLayer().to(torch.bfloat16)
    dumped = copy.deepcopy(source)
    root = Path('/tmp/real-full-algorithm-matrix') / name
    shutil.rmtree(root, ignore_errors=True)
    (root / 'result').mkdir(parents=True)
    out = root / 'result'
    quantize(dumped, config, quant_result_dir=str(out))
    batches = [torch.randn(1, 16, 1024, dtype=torch.bfloat16) for _ in range(2)]
    for batch in batches:
        dumped(batch)
    torch.manual_seed(321)
    probe = torch.randn(1, 16, 1024, dtype=torch.bfloat16)
    with torch.no_grad():
        direct = dumped(probe)
    reload_quant_params(source, str(out))
    assert type(source.fc1).__name__ == cls.__name__
    assert type(source.fc2).__name__ == cls.__name__
    with torch.no_grad():
        reloaded = source(probe)
    if name == 'awq':
        diff = (direct.float() - reloaded.float()).abs()
        assert torch.isfinite(reloaded.float()).all()
        print(name, 'PASS', 'module', cls.__name__, 'max_abs', diff.max().item())
    else:
        assert torch.equal(direct, reloaded)
        print(name, 'PASS', 'module', cls.__name__, 'bit_exact', True)


if __name__ == '__main__':
    run('minmax', cfg('minmax'), MinMaxQuant)
    run('awq', cfg('awq'), LinearAWQuant)
    run('gptq', cfg('gptq'), GPTQuant)
