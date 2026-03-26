# Compression Feature Introduction

Model compression is a collection of techniques that achieve model miniaturization and inference acceleration by reducing the number of parameters, computational load, memory usage, and bandwidth requirements while ensuring that the model performance remains basically unchanged.
The core methods of model compression include quantization, pruning, knowledge distillation, and low-rank decomposition, etc.

Quantization is the process of converting high-precision data such as FP32/FP16/BF16 to low-precision data such as INT8/FP8/INT4/FP4. By reducing the number of bits per parameter, it reduces storage usage and memory bandwidth, while using hardware low-precision computing units to achieve inference acceleration.
According to different optimization objects, quantization can be divided into weight quantization, activation quantization, and KV-cache quantization. For the explanation of the above concepts about compression features, please refer to [Compression Concepts](../压缩概念.md).

## AMCT Quantization Process

The quantization process is shown in the following figure:
![](../figures/quant.png "Call Process")

The blue part is implemented by the user, and the gray part is implemented by the user calling the API provided by AMCT. The brief process is as follows:

1. The user first constructs the original PyTorch model (please ensure that the original model can perform normal inference) and quantization configuration, then calls the [quantize](../api/quantize.md) interface to modify the original model and generate a quantized calibration model. During the modification process, different data categories are used and different quantization algorithms are applied.

2. The user calls the [convert](../api/convert.md) interface to convert the quantized calibration operators to the corresponding NPU quantized deployment operators, and outputs the quantized deployment model (type is torch.nn.Module).

## Call Example

The following is a code example of key steps, which cannot be directly copied, compiled and run, and is for reference only.

```python
# 1. Import AMCT package
import amct_pytorch as amct

# 2. Call AMCT to quantize the model
# 2.1 Generate quantized calibration model
# Build the network graph structure to be quantized
ori_model = build_model()
model = copy.deepcopy(ori_model)
# Quantization configuration
cfg = {
        'batch_num': 1,
        'quant_cfg': {
            'weights': {
                'type': 'int8',
                'symmetric': True,
                'strategy': 'tensor',
            },
        },
        'algorithm': {'minmax'},
      }
# Call the quantization interface to generate the quantized calibration model
amct.quantize(model, cfg)

# 2.2 Generate quantized deployment model
# Call the interface to convert the quantized calibration model to the quantized deployment model
amct.convert(model)
```
