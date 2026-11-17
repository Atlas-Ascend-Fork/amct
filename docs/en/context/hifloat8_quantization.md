# HiFloat8 Data Format and Quantization Algorithm Introduction

<!-- md-trans-meta sourceCommit=aabeafe332131a5f1e13adfd04a61bdc1ec9fdab translatedAt=2026-10-16T08:42:45.942Z pushedAt=2026-11-05T01:12:03.433Z nl=2 -->

## 1. Format Overview

**HiFloat8** is an 8-bit floating-point data format developed by Huawei. Compared with the traditional **FP8** format, **HiFloat8** optimizes the allocation of the exponent and mantissa through a variable-length prefix-encoded dot field (**Dot**), significantly improving the dynamic range and representation precision.

### Field Structure

HiFloat8 consists of four fields: **Sign**, **Dot**, **Exponent**, and **Mantissa**:

| Field | Width (Bit) | Description |
| --- | --- | --- |
| Sign (S) | 1 | Value sign. **1** indicates negative, **0** indicates positive. |
| Dot (D) | 2-4 | Indicates the exponent bit width and encoding mode. |
| Exponent (E) | 0-4 | Uses Sign-Magnitude encoding. |
| Mantissa (M) | 1-3 | Mantissa bits with the leading 1 omitted. |

**Field description:**

- **Sign Field**: 1 bit, determining the sign of the value. **1** indicates negative, **0** indicates positive.
- **Dot Field**: 2 to 4 bits, using prefix encoding to explicitly indicate the exponent bit width and the Normal/Denormal flag.
- **Exponent Field**: 0 to 4 bits, signed integer encoding, with the most significant bit as the sign bit.
- **Mantissa Field**: 1 to 3 bits, unsigned integer encoding. For normalized numbers, the mantissa represents the fractional part of the significant digits (to the right of the binary point).

**Dot encoding table:**

| Bit Width | Encoding | Encoded Value |
| --- | --- | --- |
| 2 | `11` | 4 |
| 2 | `10` | 3 |
| 2 | `01` | 2 |
| 3 | `001` | 1 |
| 4 | `0001` | 0 |
| 4 | `0000` | Denormal (DML) |

### Normal Encoding Mode

![Normal encoding mode](../figures/hifloat8_normal_encoding.png)

The normal encoding mode ensures that the exponent ranges corresponding to different exponent bit widths do not overlap, achieving **redundancy-free encoding**. HiFloat8 adopts the Sign-Magnitude encoding scheme, setting the most significant bit of the magnitude as an implicit bit that does not occupy storage space.

**Encoding rules:**

- **Dot = 0**: The exponent occupies no bit width, and E = 0.
- **Dot = 1-4**: The most significant bit of the exponent magnitude is fixed to 1 (the red digits in the figure).

The specific correspondence is as follows:

- When Dot = 1, E = ±1, excluding the already encoded 0.
- When Dot = 2, E = ±[2, 3], excluding the already encoded [-1, 1].
- By analogy, the five exponent bit widths from 0 to 4 indicated by the **Dot** field encode exponents in [-15, 15] without duplication.

**Tapered precision characteristics:**

HiFloat8 adopts a design in which a large-bit-width **Dot** field matches a small-bit-width exponent field, achieving smooth tapering of mantissa precision:

- E ∈ [-3, 3]: 3-bit mantissa.
- As the exponent magnitude increases, the mantissa bit width gradually shrinks to 1 bit.

This tapered precision characteristic is the key reason HiFloat8 outperforms the Posit format.

### Denormal Encoding Mode

![Denormal encoding mode](../figures/hifloat8_denormal_encoding.png)

To support a larger dynamic range, HiFloat8 adopts a Denormal scheme that differs from the IEEE 754 style.

**Encoding rules:**

- When the **Dot** field indicates Denormal mode, the exponent field is not retained, and the 3-bit mantissa space is used to map the exponent representation of extremely small values.
- Normal mode: supports 31 exponents in the range [-15, 15].
- Denormal mode: supports 7 exponents in the range [-22, -16].
- The overall coverage is 38 exponents in the range [-22, 15], close to the [-24, 15] range of FP16.

**Continuity:**

The [-15, -8] range has a 1-bit mantissa, and the [-22, -16] range corresponds to a 0-bit mantissa. The transition from Normal to Denormal remains continuous in precision with no abrupt change.

### Special Values

| Condition | Output | HiFloat8 (Binary) | HiFloat8 (Hex) | Meaning |
| --- | --- | --- | --- | --- |
| S=0, D=DML, M=3'b0 | Zero | `00000000` | `0x00` | Zero |
| S=1, D=DML, M=3'b0 | NaN | `10000000` | `0x80` | Not a number |
| D=4, Es=4'b0111, M=1 | ±Inf | `X1101111` | `0x6F`/`0xEF` | Positive/negative infinity |

HiFloat8 fully supports the special values Zero, NaN, and ±Inf. It does not distinguish between positive 0 and negative 0, using a single encoding to save encoding space, making it a numerically complete and efficient 8-bit single format.

## 2. Core Features

HiFloat8 is designed specifically for large language model (LLM) training and inference, offering three core advantages:

### Tapered Precision

Traditional FP8 (E4M3/E5M2) struggles to balance dynamic range and precision simultaneously because of its fixed exponent bit width. Considering that neural network parameters and activation values follow a Gaussian-like distribution, HiFloat8 adopts a tapered precision distribution, concentrating higher precision in the high-frequency value range.

![Precision mode](../figures/hifloat8_precision.png)

**Precision segments:**

| Range | Exponent Range | Significant Bits |
| --- | --- | --- |
| High-precision range | E ∈ [-3, 3] | 4-bit |
| Medium-precision range | E ∈ ±[4, 7] | 3-bit |
| Low-precision range | E ∈ ±[8, 15] | 2-bit |
| Extremely small value range | E ∈ [-22, -16] | 1-bit |

#### No Redundant Encoding

The exponent uses sign-magnitude encoding, with a fixed 1-bit high bit set as the implicit bit, ensuring that the exponent ranges of different exponent bit widths do not overlap.

#### Large Dynamic Range

![Typical value comparison](../figures/hifloat8_values.png)

Under the 8-bit constraint, HiFloat8 matches the data distribution through its tapered precision feature, significantly expanding the dynamic range while ensuring training and inference precision, and providing a more comprehensive 8-bit single format.

## 3. Core Algorithms

### Cast (Direct Conversion)

**Algorithm principle:**

**Cast** is the simplest quantization algorithm. It requires no calibration data and directly converts high-precision floating-point numbers (**Float16**/**BFloat16**) to the **HiFloat8** format.

**Conversion process:**

1. For activation data, directly convert from **Float16**/**BFloat16** to **HiFloat8**.
2. For weight data, compute the maximum absolute value of the weights offline and scale it into the high-precision representation range of **HiFloat8**.

**Applicable scenarios:**

- Scenarios with high requirements on deployment speed.
- Models with relatively concentrated data distribution.
- Scenarios with high tolerance for precision loss.

**Algorithm advantages:**

- No calibration data or additional computation is required, making deployment the simplest.
- Inference is the fastest, with no additional overhead.
- Thanks to the large dynamic range and high precision of HiFloat8, direct conversion from Float16/BFloat16 incurs minimal precision loss.

### Quantile

**Algorithm principle:**

The Quantile algorithm considers the numerical distribution characteristics of the model and dynamically adjusts the quantization range through quantile statistics, avoiding the impact of extreme values on quantization precision.

**Algorithm flow:**

Taking calibration data of 512 batches as an example:

1. **Initialization**: Record the maximum value of the first batch as the initial quantization threshold.
2. **Moving average**: Perform exponential moving average on the maximum values of subsequent batches to smooth the impact of extreme values.
3. **Calculate scaling factors**: Calculate the quantization scaling factors based on the final statistical values.

**Mathematical representation:**

$$
\begin{align*}
&q_{max0} = batch_{max0} \\ 
&q_{max1} = 0.99 \times q_{max0} + 0.01 \times batch_{max1} \\ 
&q_{max2} = 0.99 \times q_{max1} + 0.01 \times batch_{max2} \\ 
&... \\ 
&q_{max511} = 0.99 \times q_{max510} + 0.01 \times batch_{max511} \\ 
&scale_d = q_{max511} / 16 \\ 
&scale_w = max(|weight|) / 16
\end{align*}
$$

Here, $batch_{max_i}$ represents the maximum value of the $i$-th batch, and $scale_d$ and $scale_w$ are the quantization scaling factors for activations and weights, respectively.

**Applicable scenarios:**

- Scenarios that require high quantization precision.
- Data distributions that contain outliers or extreme points.
- Calibration data is available.

**Algorithm advantages:**

- Higher precision than the Cast algorithm.
- Low resource consumption for quantization calibration.
- Effectively suppresses the impact of extreme points.

### OFMR (Output FeatureMap Reconstruct)

**Algorithm principle:**

OFMR (Output FeatureMap Reconstruct) is a quantization factor search algorithm based on output error minimization. It determines the optimal quantization parameters by comparing the differences in model outputs before and after quantization.

**Algorithm flow:**

1. **Baseline inference**: Use the original high-precision model for inference and record the output feature maps of each layer.
2. **Candidate search**: Traverse the candidate quantization factor set and perform quantized inference for each candidate factor.
3. **Error evaluation**: Calculate the error between the quantized output and the original output (such as MSE, cosine similarity, etc.).
4. **Optimal selection**: Select the quantization factor with the smallest error as the final parameter.

    ![OFMR algorithm principle](../figures/ofmrquant.png)

**Applicable scenarios:**

- Scenarios that require extremely high quantization precision.
- Post-training quantization (PTQ).
- Sufficient calibration data and computing resources are available.

**Algorithm advantages:**

- The final output is directly optimized, delivering high quantization precision.
- No model retraining is required.
- The optimization objective can be customized for specific tasks.

**Note:**

- A calibration dataset is required, resulting in high computational cost.
- When the search space is large, the process takes a long time.
