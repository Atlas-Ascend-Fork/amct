# Compression Concepts

<!-- md-trans-meta sourceCommit=4a47814eed18048e444f9a2acc42d9864ddc477f translatedAt=2026-10-16T07:53:38.323Z pushedAt=2026-11-05T01:12:03.400Z nl=2 -->

This section describes the concepts used in model compression and introduces the principles of different compression methods.

## Quantization

Quantization performs low-bit processing on the weights and activations of a model to make the final network model more lightweight, thereby saving model storage space, reducing transmission latency, improving computational efficiency, and achieving the goal of performance improvement and optimization.

**AMCT** separates quantization from model conversion, independently quantizes the quantizable operators in the model, and outputs the quantized model. The quantized model can run on Ascend AI Processors to improve inference performance. Taking quantization to the **INT8** data type as an example, its operating principle is shown in the following figure.

 <img src="./figures/quant_running_principles.png" alt="Quantization operating principle"  width="700px" height="320px">

Based on whether retraining is required, quantization is divided into Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT). The concepts are explained as follows:

### Data Type Concept

#### HIFloat8 (HiF8)

HIFloat8 (HiF8) is a new 8-bit floating-point data format customized for AI processors. It consists of a Sign bit (indicating the sign of the value), a Dot bit, an Exponent bit (used to represent the range of the value), and a Mantissa bit (used to represent the significant digits of the value). HiF8 extends IEEE 754 by adding a dot field (Dot) to dynamically adjust the bit width of the exponent field and the mantissa field.

$$

X=
\begin{cases}
(-1)^S \times 2^E \times 1.M, & \text{NormalValue} \\
(-1)^S \times 2^{M-23}, & \text{DenormalValue}
\end{cases}

$$

Here, S is the value of the HiF8 sign field, E is the value of the HiF8 exponent field, and M is the value of the HiF8 mantissa field. When the dot bit is 0b'0000, the HiF8 data is a Denormal value.

The conversion principles are as follows. Users can configure them in the configuration file. For details, see [config details](./api/quantize.md):

- The NAN of float32/bfloat16/float16 is converted to the NAN of HiF8.
- The INF of float32/bfloat16/float16 is converted to the INF of HiF8, with the sign bit unchanged.
- If the absolute value of float32/bfloat16/float16 is greater than the upper bound of the representable range of HiF8, it is converted to the MAX of HiF8, with the sign bit unchanged.
- If the absolute value of a float32/bfloat16/float16 number is smaller than the lower bound of the representable range of HiF8, it is converted to 0 in HiF8.

Other values are converted according to the rounding mode.

**Note:** Due to the inherent data precision limitations of HiF8 or float8, when the original input data contains values whose absolute value is smaller than `1.5*2^-10`, the precision of conversion to float8 cannot be guaranteed; when the original data contains values whose absolute value is smaller than `1.5*10^-24`, the precision of conversion to HiF8 cannot be guaranteed.

#### FLOAT8 (float8)

float8 is a derivative of float16 and has two encoding formats: E4M3 and E5M2. Currently, only the E4M3 encoding format is supported.

- E4M3 contains a 4-bit exponent, a 3-bit mantissa, and a 1-bit sign bit. It is used for forward propagation during training.
- E5M2 contains a 5-bit exponent, a 2-bit mantissa, and a 1-bit sign bit. It is used for backward propagation during training. Compared with E5M2, E4M3 has a smaller data range but higher precision, and is mostly used in inference scenarios.

The conversion principles are as follows. You can configure them in the configuration file. For details, see [config details](./api/quantize.md):

- The NAN of float32/bfloat16/float16 is converted to the NAN of float8.
- If the absolute value of float32/bfloat16/float16 is greater than the upper bound of the representable range of float8, it is converted to the MAX of float8, with the sign bit unchanged.
- If the absolute value of float32/bfloat16/float16 is less than the lower bound of the representable range of float8, it is converted to 0 of float8.

Other values are converted according to the "Round to nearest, tie to Even" rounding mode.

#### MXFP8

MXFP8 is a low-precision 8-bit floating-point number representation method based on the microscaling format. It consists of a scale factor and elements, where multiple elements share a single scale factor. Compared with traditional 16-bit floating-point numbers (fp16) or 32-bit floating-point numbers (fp32), it can significantly reduce data storage and transmission requirements and reduce the memory usage of large models, thereby improving product inference competitiveness.

FP8_E4M3FN is a type of FP8 data format, an 8-bit floating-point quantization format that contains a 4-bit exponent, a 3-bit mantissa, and a 1-bit sign bit (FP8_E4M3FN). Compared with FP8, MXFP8 adds a shared exponent (a data representation method in which multiple elements share a single exponent). Through microscaling technology, it can provide higher precision and finer quantization granularity than FP8.

The conversion principle is as follows. Users can configure this type in the configuration file. For details, see [config details](./api/quantize.md):

- Conversion from float32/bfloat16/float16 to float8:

  - The NAN of float32/bfloat16/float16 is converted to the NAN of float8.
  - If the absolute value of float32/bfloat16/float16 is greater than the upper bound of the representable range of float8, it is converted to the MAX of float8, with the sign bit unchanged.
  - If the absolute value of float32/bfloat16/float16 is less than the lower bound of the representable range of float8, it is converted to 0 of float8.

  Other values are converted to FP8_E4M3FN according to the "Round to nearest, tie to Even" rounding mode (RINT).

- MXFP8 data format conversion: For the weight tensor data of the **torch.nn.Linear** operator, one shared exponent is calculated for every 32 data elements along the -1 axis direction. The calculation formula is as follows:

  $$

  exponent = \lfloor \log_2(\max(|x|)) \rfloor
  $$

#### MXFP4

MXFP4 is also a low-precision 4-bit floating-point representation method based on the microscaling format. It consists of a scale factor and elements, with multiple elements sharing a single scale factor. Compared with traditional 16-bit floating-point numbers (fp16) or 32-bit floating-point numbers (fp32), it significantly reduces data storage and transmission requirements and lowers the memory usage of large models, thereby improving product inference competitiveness.

FP4_E2M1 is a type of FP4 data format, a 4-bit floating-point quantization format that contains a 2-bit exponent, a 1-bit mantissa, and a 1-bit sign bit (FP4_E2M1). MXFP4 adds a shared exponent (a data representation method in which multiple elements share a single exponent) on top of FP4. Through microscaling technology, it provides higher precision and finer quantization granularity than FP4.

The conversion principles are as follows. Users can configure this type in the configuration file. For details, see [config details](./api/quantize.md):

- Conversion from float16/bfloat16 to FP4_E2M1: Values smaller than the FP4_E2M1 minimum are converted to the FP4_E2M1 minimum, values larger than the FP4_E2M1 maximum are converted to the FP4_E2M1 maximum, NAN and INF are not supported for conversion, and the remaining values are converted to FP4_E2M1 using the "Round to nearest, tie to Even" rounding mode (RINT).

- MXFP4 data format conversion: For the weight tensor data of the torch.nn.Linear operator, one shared exponent is calculated for every 32 data elements along the -1 axis. The calculation formula is as follows:

  $$

  exponent = \lfloor \log_2(\max(|x|)) \rfloor

  $$

  $$

  mantissa = \frac{\max(|x|)}{2^{exponent}}

  $$

  $$

  exponent = \begin{cases}
  exponent + 1, & \text{if } mantissa > 1.75 \\
  exponent, & \text{otherwise}
  \end{cases}

  $$

#### FP4_E2M1

FP4_E2M1 is a type of FP4 data format, a 4-bit floating-point quantization format that contains a 2-bit exponent, a 1-bit mantissa, and a 1-bit sign bit (FP4_E2M1).

The conversion principle from float16/bfloat16 high precision to FP4_E2M1 low precision is as follows. You can configure this type in the configuration file. For details, see [config details](./api/quantize.md):

Values smaller than the FP4_E2M1 minimum are converted to the FP4_E2M1 minimum, values greater than the FP4_E2M1 maximum are converted to the FP4_E2M1 maximum, NAN and INF are not supported for conversion, and the remaining values are converted to FP4_E2M1 using the "Round to nearest, tie to Even" rounding mode (RINT).

FP4_E2M1 can represent 16 values, namely `{±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}`.

#### INT8

8-bit int8 data is used to represent, for example, 32-bit float32 data, converting float matrix multiplication into int8 matrix multiplication to accelerate computation and achieve model compression.

#### INT4

4-bit int4 data is used to represent, for example, 32-bit float32 data. Compared with int8 quantization, int4 quantization achieves better quantization results but may introduce greater accuracy loss.

#### Overview

- Compared with float8, HiF8 better balances accuracy and dynamic range, while MXFP8 and MXFP4 can reduce memory usage of large models.
- MXFP8 and MXFP4: Dedicated hardware acceleration units are available, which process quantization factors faster.
- FP4_E2M1 and FP4_E1M2: For uniformly distributed data, FP4_E1M2 is recommended; for Gaussian distribution, FP4_E2M1 is recommended.

### Post-Training Quantization

Post-Training Quantization (PTQ) refers to quantization performed after model training is complete. It quantizes the weights of the trained model from floating-point numbers to low-bit integers, and calibrates and quantizes the data (activation) based on the inference process using a small amount of calibration data, thereby minimizing the accuracy loss during quantization. PTQ is simple and easy to use, requiring only a small amount of calibration data, and is suitable for scenarios that prioritize high usability and lack training resources.

Typically, the weights of a trained model are already determined, so the quantization parameters of the weights can be computed offline based on their values. However, data is usually input online, so the value range of the data cannot be accurately obtained. A small representative dataset is generally required to simulate the distribution of online data. Forward inference is performed using this dataset to obtain the corresponding intermediate floating-point results, from which the quantization parameters of the data are computed offline. The principle is illustrated in the following figure.

 ![](./figures/ptq_principles.png "Post-Training Quantization (PTQ) Principle")

### Quantization-Aware Training

Quantization-aware training (QAT) is a quantization method that introduces quantization during retraining. By retraining, it improves the model's ability to adapt to quantization effects, thereby achieving higher quantized model accuracy. QAT leverages the user's complete training dataset and introduces fake quantization operations (converting from floating-point to fixed-point and then back to floating-point) during training to simulate the errors introduced by quantization in forward inference. Through training, the model weights can better adapt to this quantization-induced information loss, thereby improving quantization accuracy.

Generally, compared with PTQ, QAT incurs less accuracy loss, but its main drawback is that the overall quantization takes longer. In addition, the quantization process requires more data, typically the complete training dataset.
Its operating principle is shown in the following figure.

​    ![](./figures/qat_principles.png "Quantization-aware training principle")

Other concepts in the quantization process are explained as follows:

### Weight Quantization

Weight quantization processes weights to low bit based on their numerical distribution.

### Activation Quantization

Activation quantization, also known as data quantization, refers to processing the input data (activation) to low bit based on the numerical distribution of the data. The data distribution of each layer is unknown and enormous, and can only be determined during the forward process (inference or training). Therefore, data quantization is based on the inference or training process.

### Calibration Dataset

In quantization scenarios, the calibration dataset is the dataset used for forward inference. Its distribution represents the distribution of all datasets. The calibration dataset should be representative when obtained, and it is recommended to use a subset of the test set as the calibration dataset. If the dataset does not match the model or is not representative enough, the quantization factor calculated from the calibration dataset performs poorly on the full dataset, resulting in large quantization loss and low post-quantization accuracy.

### Quantization Factor

Parameters used to convert floating-point numbers into integers, including the scale factor and the offset. The formula for converting a floating-point number into an integer (using INT8 as an example) is as follows:

$$
int\_val=clip(round(float\_val/scale+offset),-128,127)
$$

- **scale**: Quantization factor, the scale factor of floating-point numbers. This parameter is further divided into:
  - **scale_d**: Data quantization scale factor, which supports only uniform quantization of data.
  - **scale_w**: Weight quantization scale factor, which supports two modes: scalar (uniform quantization of the weights of the current layer) and vector.
- **offset**: Quantization factor, the offset. This parameter is further divided into:
  - **offset_d**: Data quantization offset factor, which supports only uniform quantization of data.
  - **offset_w**: Weight quantization offset factor. Like **scale_w**, it supports both scalar and vector modes, and its dimension must be consistent with that of **scale_w**.

### Quantization Granularity

It refers to applying different quantization computation levels to different input tensors of matrix operators such as Matmul in a neural network. Common quantization computation granularities include:

  > [!NOTE] Note
  >
  >- The variables m, n, and k represent the sizes of different axes of the tensor computation.
  >- The left matrix and right matrix refer to the two input tensors used for matrix multiplication in a cube operator. Generally, the left matrix represents the activation and the right matrix represents the weight. Users should understand and use them according to the actual situation.

#### Per-Channel Quantization

Abbreviated as C quantization, the quantization object is the right matrix, and each channel uses an independent quantization parameter. Assuming the right matrix has a shape of (k, n), where k is the reduce axis, the shape of the generated quantization parameter is (n, ).

![](figures/perchannel.png)

#### Per-Group Quantization

Abbreviated as G quantization, the quantization object can be either the left matrix or the right matrix (AMCT supports only the right matrix). Data is grouped on the reduce axis, and each group uses independent quantization parameters.

Assume the right matrix has a shape of (k, n), where k is the reduce axis. Grouping is performed on the k axis with a group size of gs, and the generated quantization parameters have a shape of (k/gs, n).

![](figures/pergroup.png)

#### Per-Tensor Quantization

Abbreviated as T quantization, the quantization object can be either the left matrix or the right matrix, and each tensor shares the same quantization parameter.

Assume the left matrix has a shape of (m, k), the right matrix has a shape of (k, n), and k is the reduce axis. The shape of the generated quantization parameter is (1, ).

![](figures/pertensor.png)

#### Per-Token Quantization

Abbreviated as K quantization, the quantization object is the left matrix, and each token uses an independent quantization parameter.

Assume the left matrix has a shape of (m, k), where k is the reduce axis, and the shape of the generated quantization parameter is (m, ).

![](figures/pertoken.png)

## Sparsity

Sparsity applies structural pruning to some operators in a model to sparsify their weights, thereby producing a network model with fewer parameters and less computation. AMCT currently supports two sparsity methods: channel sparsity and 4:2 structured sparsity. Only one sparsity method can be enabled at a time. That is, for the same compressible operator in a layer, channel sparsity and 4:2 structured sparsity cannot be configured simultaneously.

Compared with **4:2 structured sparsity**, channel sparsity has a coarser sparsity granularity and a greater impact on model accuracy, but it also delivers a greater performance gain. You can select a sparsity method based on the actual situation.

### Channel Sparsity

Channel sparsity is based on retraining. By pruning the number of network channels, it reduces the number of model parameters while preserving network functionality, thereby lowering the overall computational load of the network. Since channel sparsity prunes channels based on their importance, it removes channels of relatively low importance. However, directly pruning channels has a significant impact on network accuracy, so the pruned model must be retrained to ensure service accuracy. The implementation of channel sparsity typically involves two steps: first, channel selection, which requires selecting an appropriate combination of channels to retain rich information; second, reconstruction, which requires using the selected channels to reconstruct the output of the next layer. The principle of channel sparsity is shown in the following figure.

 <img src="./figures/channel_sparse.png" alt="Channel sparsity diagram"  width="750px" height="320px">

### 4:2 Structured Sparsity

**Due to hardware constraints, Atlas inference products, Atlas training products, and Ascend 950PR&950DT products do not support the 4:2 structured sparsity feature.**

4:2 structured sparsity is based on retraining. Among every four consecutive weights, the two weights with relatively higher importance are retained, and the remaining weights are set to 0. Because the sparsity granularity is small, 4:2 structured sparsity can retain more important information and offers the accuracy advantage of fine-grained sparsity. Meanwhile, 4:2 structured sparsity can reduce the computational workload on specially designed hardware, offering the performance advantage of structured sparsity. Unlike channel sparsity, 4:2 sparsity does not change the shape of the weights, and therefore does not affect the operators in the upper or lower layers.

The principle is shown in the following figure. Four adjacent elements along the cin dimension form a group. In each group of four elements, the two elements with the largest absolute values are retained. If cin is not a multiple of 4, zeros are padded to make it a multiple of 4.

 <img src="./figures/m4n2.png" alt="4:2 sparsity diagram"  width="650px" height="150px">

## Combined Compression

Combined compression combines the characteristics of **sparsity** and **quantization**. Based on the configuration file, sparsity is performed first, followed by quantization. During sparsity, sparsity operators are inserted according to the corresponding algorithm. Then, during quantization, quantization layers for data and weights and SearchN layers are inserted into the sparsified model to generate a combined compression model, with the expectation of achieving higher performance gains. After the combined compression model is generated, the model is retrained and saved as a quantized model that supports both accuracy simulation and deployment.

## Layer-wise Distillation

Distillation quantization is a model compression method that uses the supervision information of the original model to train the quantized model, so as to achieve better quantization accuracy.

The pre-trained original model is used as the teacher network, and the output of the teacher network layers is used as the target to supervise the training of the student network. By computing the loss between the predicted outputs of the teacher network and the student network, gradient updates are performed, ultimately yielding a quantized model with higher accuracy.

- Compared with post-training quantization, knowledge distillation quantization can achieve better accuracy results.
- Compared with quantization-aware training, knowledge distillation quantization does not require a labeled dataset, and can obtain satisfactory quantization results within a shorter quantization time.

![](./figures/distill.png "Layer-wise distillation diagram")

## Tensor Decomposition

Deep learning operations, especially CV (computer vision) tasks, involve a large number of convolution operations. Tensor decomposition decomposes the tensor of a convolution kernel, splitting a large convolution kernel into the product of two small convolution kernels, that is, decomposing the convolution kernel into low-rank tensors. This reduces storage space and computation, thereby lowering inference overhead.

Taking the decomposition of a 64\*64\*3\*3 convolution into cascaded 32\*64\*3\*1 and 64\*32\*1\*3 convolutions as an example, the computation can be reduced by 1 - \(32\*64\*3\*1 + 64\*32\*1\*3\) / 64\*64\*3\*3 = 66.7%, delivering more cost-effective performance gains while producing approximately equivalent results. The operating principle of tensor decomposition is shown in the following figure (using the PyTorch framework as an example).

![](./figures/decomposition.png "Operating principle of tensor decomposition")

## Model Deployment Optimization

It mainly refers to operator fusion, which mathematically equivalently fuses multiple operator operations in a model into a single operator operation to reduce the computation workload in the actual forward process, for example, fusing a convolution layer and a BN layer into a single convolution layer.

Its operating principle is shown in the following figure (using the PyTorch framework as an example).

![](./figures/model_deploy_optimize.png "Model deployment optimization principle")
