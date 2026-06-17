# SVDQuant — SVD Quantization With Mixed Mxfp4/Bf16 Operator

SVDQuant is a 4-bit quantization paradigm. SVDQuant method uses a low-rank branch to absorb  outliers. Firstly, outliers have merged by transferring them from activations to weights, and then use a high-precision low-rank branch to process weight outliers using singular value decomposition (SVD). This process simplifies quantization on both sides.
An effective for usage in Large Language Models and AI agentic.

SVDQuant operator has 2 branches for calculation:

> 1. Low-rank branch performs high-presision but low-rank MatMul operations. Low-rank branch has 2 BF16 MatMul which performs Down Projection and Up Projection multiplications by low-rank weights with BF16 precision  
> 2. Low-precision branch performs high-rank but low-precision MxFp4 MatMul operation with preliminary BF16->fp4x2_e2m1 quantization of activation  

Finally, both branches represent the combined BF16 results obtained by tensors summation.

```mermaid
stateDiagram-v2
	Active: SVDQuant Operator
	Quantize: Quantize<br>BF16->FP4
	MatMul1: DownProj<sub>BF16</sub><br>X @ L<sub>1</sub>
	MatMul2: UpProj<sub>BF16</sub><br>DP @ L<sub>2</sub>
	MatMul3: MatMul<sub>MxFp4</sub><br>Q(X) @ Q(R)
	Add: (+)

	classDef ActiveState font-size:16pt,font-weight:bold,font-style:italic,color:black,fill:white
	classDef CUBState font-size:14pt,font-weight:bold,color:black,fill:#dd80aa,stroke:#dd80aa
	classDef VECState font-size:14pt,font-weight:bold,color:black,fill:#94dae2,stroke:#94dae2

    state Active {
		direction LR
		[*] --> MatMul1: X<sub>BF16</sub> {B,S,K}
		MatMul1 --> MatMul2: DP<sub>BF16</sub> {B,S,R}
		[*] --> Quantize: X<sub>BF16</sub> {B,S,K}
		Quantize --> MatMul3: Scale<sub>U8</sub>, X<sub>MxFp4</sub>
		MatMul2 --> Add
		MatMul3 --> Add
		Add --> [*]: Out<sub>BF16</sub> {B,S,N}
	}

	class Active ActiveState
	class MatMul1, MatMul2, MatMul3, Add CUBState
	class Quantize VECState
```

## Supported Features

- Batches
- Prefill and Decode LLM modes
- Wide range of shapes, but it shows good performance on target shapes
- BF16 activation/low-rank weights and fp4x2_e2m1 weights for low-precision calculations
- An SVDQuant operator designed for Ascend David A5
- <1% accuracy drop
- Target Seq-len = 1, 32K; Rank = 32, 64, 128; weight shapes:

|   N   |   K   |
| :---: | :---: |
| 10944 | 2048  |
| 2048  | 10944 |
| 2816  | 2048  |
| 2048  | 2816  |
| 1408  | 2048  |
| 2048  | 1408  |
| 2048  | 2048  |
| 2048  | 2048  |
| 2048  | 3072  |
|  512  | 4096  |

## Interface Description

### svd_quant (x: Tensor, w: Tensor, s: Tensor, d: Tensor, u: Tensor) -> Tensor

Matrix multiplication of mixed MxFp4/Bf16 precision.

**Parameters description:**
`B`: batch size is zero or more batch dimensions  
`S`: sequence length or activation row size dimension  
`N`: weights column size  
`K`: activation column size  
`ScaleK`: derived from K-dimension  
`R`: SVD rank size  

**Input Parameters**
NPU device

| Tensor | Shape        | torch.dType      | Description                                                          |
| :----: | :----------- | :--------------- | :------------------------------------------------------------------- |
| x      | (B,S,K)      | bfloat16         | Activation tensor, where `'B'` is empty or one and more batch dimensions |
| w      | (N,K)        | float4_e2m1fn_x2 | Weight low-precision tensor                                          |
| s      | (N,ScaleK,2) | float8_e8m0fnu   | Weight scales tensor                                                 |
| d      | (K,R)        | bfloat16         | Down projection low-rank tensor                                      |
| u      | (R,N)        | bfloat16         | Up projection low-rank tensor                                        |

**Returns**:
NPU device

`torch.bfloat16` tensor, (B, S, N) shape

**Exceptions**:
`RuntimeError`: Input is not on NPU  
`RuntimeError`: Data Type is wrong  
`RuntimeError`: Invalid input shapes  
`RuntimeError`: Input shapes are incompatible  

---

## Directory Structure

```text
svd_quant/             # SVDQuant components
|
├── ascendc            # AscendC operator for installation to CANN
|  ├── cmake/          # CMake utilities
|  ├── svd_quant/      # SVDQuant AscendC implementation
|  |  ├── op_host/     # Host part of SVDQuant
|  |  └── op_kernel/   # SVDQuant kernel files for A5
|  ├── CMakeLists.txt  # CMake build configuration
|  └── build.sh        # Script for compilation of CANN operator
|
├── torch_amct         # PyTorch operator implementation
│  ├── amct/
|  |  ├── csrc/        # PyTorch host implementation and TORCH_LIBRARY registration
|  |  └── __init__.py  # Python package entry
│  ├─ tests/
|  |  ├── quantize_ref.py   # Quantization reference function
|  |  └── test_svd_quant.py # SVDQuant accuracy tests
│  ├─ setup.py
│  └─ build_and_install.sh  # Script for PyTorch operator compilation and installation
└── README.md               # Operator documentation
```

## Environment Dependencies

| SOC         | Platform                | Nominal UB / core |
| ----------- | ----------------------- | ----------------- |
| `ascend950` | Ascend950PR/Ascend950DT | 512 KB+           |

- CANN 9.0.0
- Python ≥ 3.9
- PyTorch + torch_npu (matching corresponding CANN version)

## Compilation And Usage

### 1. CANN Custom Operator Compilation And Installation @ A5

```bash
source $ASCEND_HOME_PATH/set_env.sh
path="amct"
cd $path$/amct_ops/svd_quant/ascendc
bash build.sh -c ascend950
./output/CANN-custom_ops--linux.aarch64.run --install-path=$ASCEND_HOME_PATH/opp
source $ASCEND_HOME_PATH/opp/vendors/customize/bin/set_env.bash
```

### 2. PyTorch Operator Compilation And Installation

```bash
cd $path$/amct_ops/svd_quant/torch_amct
bash build_and_install.sh
```

### 3. Usage Example

```python
import torch
import torch_npu
import numpy as np
import amct

# Select NPU Device
device = torch.device('npu:0')

# Setup Shapes
bs, seq_len, n, k, rank = (4, 32 * 1024, 512, 4096, 64)
a_shape = (bs, seq_len, k)
w_shape = (n, k)
dp_shape = (k, rank)
up_shape = (rank, n)

# Generate Tensors
x = torch.tensor(np.random.uniform(-10, 10, a_shape), dtype=torch.bfloat16).npu()
w = torch.tensor(np.random.uniform(-10, 10, w_shape), dtype=torch.bfloat16).npu()
dp = torch.tensor(np.random.uniform(-10, 10, dp_shape), dtype=torch.bfloat16).npu()
up = torch.tensor(np.random.uniform(-10, 10, up_shape), dtype=torch.bfloat16).npu()

# Quantize Weights
w_quant, scale = torch_npu.npu_dynamic_mx_quant(w, block_size=32, round_mode="round")

# SVDQuant Execution
svd_quant_out = torch.ops.amct.svd_quant(x, w_quant, scale,  dp, up)
```

## Performance Verification

SVDQuant gives performance advantages for long contexts in compare with MatMul BF16 with low-accuracy drop.

**Test Platform**: Ascend950PR (A5, ascend950, HBM 114688 MB), CANN 9.1.0  
**Scenario**: Batch size = 1, Seq-len = 32K, Rank = 32  

### MatMulV3 BF16 ↔ SVDQuant

| W Shape (N,K)   | BF16 (us) | SVDQuant (us) | Ratio |
|:------------:|:----:|:----:|:---:|
| (10944,2048) | 3446 | 2188 | 1.6 |
| (2816,2048)  | 888  | 691  | 1.3 |
| (1408,2048)  | 488  | 406  | 1.2 |
| (2048,2816)  | 885  | 719  | 1.2 |
| (2048,3072)  | 965  | 808  | 1.2 |

> Small weights matrix <(2K, 2K) SVDQuant effectiveness is relatively low, because a target is long contexts with big weight tensor sizes.  

## Accuracy Verification

Accuracy verification is performed through the following conditions:
**Golden data**: calculated on random BF16 data (activation and weight tensors) on low-rank and low-precision branches with de-quantized activation and weights  
**MxFp4 weights**: preliminary calculated with `npu_dynamic_mx_quant` operator from `torch_npu` with `block_size=32` and `round_mode="round"`  
**Absolute tolerance**: threshold is set to `1e-07`  

## Test Method

```bash
cd $path$/amct_ops/svd_quant/torch_amct/tests
pytest test_svd_quant.py
```

Can also install wheel first then run tests:

```bash
cd $path$/amct_ops/svd_quant/torch_amct
pip install dist/amct-1.0-*.whl
pytest test_svd_quant.py
```

