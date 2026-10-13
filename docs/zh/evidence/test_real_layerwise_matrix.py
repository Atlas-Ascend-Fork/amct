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

import json
import os
import shutil
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM

from amct_pytorch.classic.layerwise import (
    _needs_extract,
    load_calibration_data,
    prepare_layerwise_quantization,
    run_layerwise_quantization,
)
from amct_pytorch.classic.quant_result import (
    compute_model_fingerprint,
    load_quant_manifest,
    reload_quant_params,
)

MODEL_PATH = os.environ.get('AMCT_QWEN3_MODEL_PATH')
if not MODEL_PATH:
    raise RuntimeError('Set AMCT_QWEN3_MODEL_PATH to the local Qwen3-0.6B checkpoint.')
ROOT = Path('/tmp/real-layerwise-algorithm-matrix')


def reset(name):
    path = ROOT / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    return path


def run_weight_only():
    root = reset('weight-only')
    out = root / 'result'
    out.mkdir()
    config = {
        'pipeline': 'layerwise',
        'batch_num': 1,
        'batch_size': 1,
        'num_calibration_samples': 1,
        'max_seq_length': 16,
        'device': 'cpu',
        'work_dir': str(root / 'work'),
        'quant_cfg': {
            'weights': {'type': 'int8', 'symmetric': True, 'strategy': 'channel'},
        },
        'algorithm': {'minmax': {}},
    }
    plan = prepare_layerwise_quantization(MODEL_PATH, config, str(out))
    assert _needs_extract(plan.args) is False
    result = run_layerwise_quantization(plan)
    manifest = load_quant_manifest(result)
    assert len(manifest['units']) == 84
    assert all(
        [item['key'] for item in unit['tensors']] == ['scale_w']
        for unit in manifest['units']
    )

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16)
    assert compute_model_fingerprint(model) == manifest['source_model']
    reload_quant_params(model, result)
    assert type(model.model.layers[0].mlp.gate_proj).__name__ == 'MinMaxQuant'
    assert type(model.model.layers[27].mlp.down_proj).__name__ == 'MinMaxQuant'
    input_ids = torch.tensor([[151643] + list(range(1, 16))])
    with torch.no_grad():
        logits = model(input_ids).logits
    assert logits.shape == (1, 16, 151936)
    assert torch.isfinite(logits.float()).all()
    print('weight-only: PASS, units=84, reload=MinMaxQuant, forward=finite')


def run_static_minmax():
    root = reset('static-minmax')
    out = root / 'result'
    out.mkdir()
    data = root / 'dataset'
    text = 'The quick brown fox jumps over the lazy dog. '
    Dataset.from_dict({'text': [text, text]}).save_to_disk(str(data))
    config = {
        'pipeline': 'layerwise',
        'batch_num': 1,
        'batch_size': 1,
        'num_calibration_samples': 1,
        'max_seq_length': 16,
        'device': 'cpu',
        'work_dir': str(root / 'work'),
        'dataset': str(data),
        'quant_cfg': {
            'weights': {'type': 'int8', 'symmetric': True, 'strategy': 'channel'},
            'inputs': {
                'type': 'int8',
                'symmetric': True,
                'strategy': 'tensor',
            },
        },
        'algorithm': {'minmax': {}},
    }
    plan = prepare_layerwise_quantization(MODEL_PATH, config, str(out))
    assert _needs_extract(plan.args) is True
    samples = load_calibration_data(plan.adapter)
    assert len(samples) == 1 and samples[0].shape == (1, 16)
    result = run_layerwise_quantization(plan)
    manifest = load_quant_manifest(result)
    assert len(manifest['units']) == 84
    assert all(
        [item['key'] for item in unit['tensors']] == ['scale_d', 'scale_w']
        for unit in manifest['units']
    )
    cache = json.loads(
        (root / 'work' / 'ptq_data' / 'mlp' / 'cache_manifest.json').read_text()
    )
    assert len(cache['files']) == 28

    # Reuse must skip extract and produce a second complete artifact.
    out2 = root / 'result-reuse'
    out2.mkdir()
    plan.args.quant_result_dir = str(out2)
    before = list((root / 'work' / 'ptq_data' / 'mlp').glob('block_*_in.pkl'))
    result2 = run_layerwise_quantization(plan)
    after = list((root / 'work' / 'ptq_data' / 'mlp').glob('block_*_in.pkl'))
    assert len(before) == len(after) == 28
    assert load_quant_manifest(result2)['source_model'] == manifest['source_model']

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16)
    assert compute_model_fingerprint(model) == manifest['source_model']
    reload_quant_params(model, result)
    input_ids = torch.tensor([[151643] + list(range(1, 16))])
    with torch.no_grad():
        logits = model(input_ids).logits
    assert logits.shape == (1, 16, 151936)
    assert torch.isfinite(logits.float()).all()
    print(
        'static-minmax: PASS, units=84, cache_files=28, reuse=PASS, reload+forward=PASS'
    )


def run_rejected_algorithms():
    root = reset('rejected')
    cases = {
        'awq': {'awq': {'grids_num': 4}},
        'gptq': {'gptq': {}},
        'smoothquant': {'smoothquant': {'smooth_strength': 0.5}},
    }
    for name, algorithm in cases.items():
        config = {
            'pipeline': 'layerwise',
            'quant_cfg': {
                'weights': {'type': 'int8', 'symmetric': True, 'strategy': 'channel'},
            },
            'algorithm': algorithm,
        }
        try:
            prepare_layerwise_quantization(MODEL_PATH, config, str(root / name))
        except ValueError as error:
            assert "only no algorithm or algorithm 'minmax'" in str(error)
            print(name + ': correctly rejected (layerwise P0)')
        else:
            raise AssertionError(name + ' was not rejected')


if __name__ == '__main__':
    run_weight_only()
    run_static_minmax()
    run_rejected_algorithms()
