import argparse
import torch
from utils import get_qwen, get_test_dataset, test_ppl
from hifloat8_fakequant_linear import Hifloat8FakequantLinear
import amct_pytorch as amct

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='example')
    parser.add_argument("--model_path", type=str, required=True, help="model path")
    args = parser.parse_args()

    model, tokenizer = get_qwen(args.model_path)
    quant_model = model.eval().npu()

    amct.algorithm_register('hifloat8_fakequant', 'Linear', Hifloat8FakequantLinear, None)

    # Phase1: quantize model
    hifloat8_fakequant_cfg = {
        'quant_cfg': {
            'weights': {
                'type': 'hifloat8',
                'symmetric': True,
                'strategy': 'channel',
            },
            'inputs': {
                'type': 'hifloat8',
                'symmetric': True,
                'strategy': 'tensor',
            },
        },
        'algorithm': {'hifloat8_fakequant'},
        'skip_layers': {'lm_head'}
    }
    amct.quantize(quant_model, hifloat8_fakequant_cfg)

    # Phase2: Test ppl result
    testenc = get_test_dataset(tokenizer)
    testenc = testenc.input_ids.npu()
    test_ppl(quant_model, testenc)
