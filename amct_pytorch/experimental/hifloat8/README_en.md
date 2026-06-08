# HiFloat8 Fake Quantization Module

This module provides PyTorch model fake quantization functionality based on HiFloat8 format, including conversion between HiFloat8 and FP32/FP16/BF16 formats, fake quantization linear modules, and test scripts.

*Note*
Current HiFloat8 format conversion is fully implemented based on CPU, using OpenMP for parallel acceleration. Compared to NPU hardware acceleration, there is a certain performance gap. Recommendations:
- Use for small batch data processing
- Use for model accuracy verification and debugging
- For production environments, recommend using NPU hardware accelerated version

## File Description

```
├── hifloat8_cast.cpp               # C++ implementation of HiFloat8 and FP32/FP16/BF16 format conversion
├── setup.py                        # Python compilation script
├── build.sh                        # Shell compilation script
├── hifloat8_fakequant_linear.py    # HiFloat8 fake quantization linear module
├── test.py                         # Test script
├── utils.py                        # Utility functions
└── README.md
```

## Function Description
### 1. HiFloat8 Format Conversion

#### 1.1 Compile C++ Extension

Compile C++ extension module through shell:

```bash
# Set parallel compilation task count (optional, default is 8)
export AMCT_NUM_BUILD_JOBS=8

# Compile extension
./build.sh
```

Or compile directly using Python:

```bash
python setup.py build_ext --inplace
```

#### 1.2 Conversion Function Description

After successful compilation, you can use the following conversion functions:

```python
import torch
import hifloat8_cast

# FP32/FP16/BF16 to HiFloat8
float32_tensor = torch.randn(10, 10, dtype=torch.float32)
hifloat8_tensor = hifloat8_cast.float_to_hifloat8(float32_tensor)

# HiFloat8 to FP32
float32_tensor = hifloat8_cast.hifloat8_to_float32(hifloat8_tensor)

# HiFloat8 to FP16
float16_tensor = hifloat8_cast.hifloat8_to_float16(hifloat8_tensor)

# HiFloat8 to BFloat16
bfloat16_tensor = hifloat8_cast.hifloat8_to_bfloat16(hifloat8_tensor)
```

### 2. Fake Quantization Module

`Hifloat8FakequantLinear` is a fake quantization linear layer module based on HiFloat8 format, inheriting from `BaseQuantizeModule`.

- **Weight Quantization**: Uses per-tensor quantization strategy
- **Quantization Range**: HiFloat8 high-precision range is 16
- **Activation Quantization**: Uses data direct conversion strategy

### 3. Algorithm Registration

amct_pytorch provides a flexible algorithm registration mechanism, allowing users to define custom quantization algorithms and integrate them into the quantization process.
Through the [`algorithm_register`](../../../docs/zh/api/algorithm_register.md) function, you can register custom quantization modules into the system.

```python
import torch
from hifloat8_fakequant_linear import Hifloat8FakequantLinear
import amct_pytorch as amct

# Register HiFloat8 fake quantization algorithm
amct.algorithm_register('hifloat8_fakequant', 'Linear', Hifloat8FakequantLinear, None)
```

### 4. Test Script

The test script performs the following steps:
1. **Load Model**: Load pre-trained model and tokenizer from specified path
2. **Register Algorithm**: Register HiFloat8 fake quantization algorithm
3. **Model Quantization**: Perform fake quantization on the model using HiFloat8 format
4. **Dataset Loading**: Load Wikitext-2 test dataset
5. **Performance Evaluation**: Calculate perplexity of the quantized model

### 5. Complete Usage Process

#### 5.1 Prepare Environment
1. Install amct tool, refer to [Tool Build](../../../docs/zh/build.md)
2. Install other dependencies
```bash
pip install transformers datasets
```
3. Compile HiFloat8 extension
```bash
cd amct_pytorch/experimental/hifloat8
./build.sh
```

#### 5.2 Run Test
```bash
# Run test script
python test.py --model_path /path/to/qwen/model
```