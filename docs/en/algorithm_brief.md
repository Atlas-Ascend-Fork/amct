# Algorithm Introduction

<!-- md-trans-meta sourceCommit=4a47814eed18048e444f9a2acc42d9864ddc477f translatedAt=2026-10-16T07:51:37.440Z pushedAt=2026-11-05T01:12:03.399Z nl=2 -->

## Algorithm Support Matrix

**Weight Quantization (W)**

| Algorithm | INT8 | INT4 | MXFP8 | MXFP4 | HiFloat8 | HiFloat4 |
|--------|:----------:|:----------:|:-----------:|:-----------:|:--------:|:--------:|
| Min-Max    | ✅ | ✅ | — | — | — | — |
| AWQ        | ✅ | ✅ | — | — | — | — |
| GPTQ       | ✅ | ✅ | — | ✅ | ✅ | ✅ |
| Cast       | — | — | — | — | ✅ | — |
| Quantile   | — | — | — | — | ✅ | — |
| OFMR       | — | — | — | — | ✅ | — |
| MXQuant    | — | — | ✅ | ✅ | — | — |

**Full Quantization (W+A)**

| Algorithm | INT8 | INT4 | MXFP8 | HiFloat8 |
|--------|:----------:|:----------:|:-----------:|:--------:|
| Min-Max     | ✅ | — | — | — |
| SmoothQuant | ✅ | — | — | ✅ |
| Cast        | — | — | — | ✅ |
| Quantile    | — | — | — | ✅ |
| OFMR        | — | — | — | ✅ |
| MXQuant     | — | — | ✅ | — |
| FlatQuant   | — | ✅ | — | — |

## Weight Quantization Algorithms

### ARQ

ARQ (Adaptive Range Quantization) is an algorithm that directly quantizes weights. This algorithm provides two methods: **channel_wise** (whether to use a different quantization factor for each channel; for parameter explanation, see [Post-Training Quantization Configuration Parameters](./context/ptq_config_param.md)) quantization and non-**channel_wise** quantization. The quantization principle is shown in the following figures:

**Figure 1** Non-channel\_wise quantization<a name="fig7600113961719"></a>  
![](./figures/nonchannel_wise_quant.png "Non-channel_wise quantization")

**Figure 2** channel_wise quantization<a name="fig6351104815177"></a>  
![](./figures/channel_wise_quant.png "channel_wise quantization")

In the quantization configuration, the **channel\_wise** parameter provided by **ARQuantize** (using the PyTorch framework as an example; for parameter explanation, see [Table 1](./context/ptq_config.md#zh-cn_topic_0240188735_table1225503375617)) is used to select the quantization method:

- When **channel\_wise** is **False**, the quantization method in [Figure 1](#fig7600113961719) is selected. All filters are analyzed together for data distribution and quantized, and all different channels in a layer share one quantization factor.
- When **channel\_wise** is **True**, the quantization method in [Figure 2](#fig6351104815177) is selected. Each filter is analyzed independently for data and quantized, and each channel in the same layer has an independent quantization factor.

Generally, quantizing each filter independently, that is, using the quantization method in [Figure 2](#fig6351104815177), yields higher quantization precision. If the amount of data in each filter is relatively small, the quantization effect is poor; in this case, the quantization method in [Figure 1](#fig7600113961719) is recommended.

Note: Fully-connected layers and average downsampling layers (Pooling, where the downsampling method is AVE and it is not global pooling) have no channel. Setting **channel\_wise** to **True** for these layers triggers an error message.

### ADA

When quantizing a neural network, the main approach is to assign each floating-point weight to its nearest fixed-point value. However, this is not the optimal quantization strategy. This document introduces **AdaRound\(Adaptive Rounding\)**, a better weight rounding mechanism for post-training quantization that can adapt to the data and task loss, and requires only a small amount of unlabeled data.

The ADA weight quantization algorithm is used in the PTQ process. Because the rounding mode of quantized weights affects model precision, ADA mainly fine-tunes the rounding mode, adaptively rounding up or down for each quantized data point. It requires unlabeled data for training and tuning to determine different rounding modes. The overall process is shown in the following figure. First, the KL divergence algorithm is used to compute the initial weight quantization factor scale and offset, and then unlabeled data is used for training to adjust the rounding mode of the weights.

**Figure 1**  ADA weight quantization algorithm<a name="fig432315576612"></a>  
![](./figures/adaquant.png "ADA weight quantization algorithm")

The formula for the quantized weight is as follows:

$$

W_q = \mathrm{clip}\left( \left| \frac{W}{\mathrm{scale}} \right| + h(alpha), n, p \right)

$$

Where **n** and **p** are the upper and lower bounds after quantization, respectively.

The training process mainly optimizes the alpha parameter, with the goal of making h\(alpha\) equal to 0 or 1:

$$

alpha = -\ln\left( \frac{\varsigma-\gamma}{\dfrac{W}{\mathrm{scale}} - \left\lfloor \dfrac{W}{\mathrm{scale}} \right\rfloor} - 1 \right)

$$

Where **$\varsigma$** and **$\gamma$** are scaling parameters with values of **1.1** and **-0.1**, respectively.

Quantize the original weight, then add h\(alpha\) to dequantize it after quantization, and introduce the dequantized weight into the final operator computation, so that the final output carries the quantization error:

$$

h(alpha) = \mathrm{clip}\big(\mathrm{sigmoid}(alpha) \times (\varsigma-\gamma),\, 0,\, 1\big)

$$

The loss function is calculated as follows:

$$

\mathrm{Loss} = \left\| WX - \widetilde{W}X \right\|^2 + \lambda \sum 1 - \left| 2h(alpha) - 1 \right|^\beta

$$

The regularization term (the latter half of the loss function) constrains h\(alpha\) to converge toward the two values **0** and **1**. In the early iterations, a larger **$\beta$** increases the impact of the regularization term on the overall loss, prompting the optimization process to rapidly adjust h\(alpha\) and approach the region of the optimal solution. In the later iterations, when h\(alpha\) is already in the optimal solution region, a smaller **$\beta$** is used to avoid missing the optimal solution due to over-adjustment, allowing h\(alpha\) to approach the optimal solution.

### AutoRound

AutoRound learns the Rounding Offset and is a lightweight post-training quantization algorithm for LLMs. Its core idea is to make the "rounding compilation V" a learnable parameter and jointly adjust the min/max scale, so that weights can select a better rounding direction during low-bit quantization.
For detailed algorithm description, see [Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs](https://arxiv.org/abs/2309.05516).

Note: `autoround` carries a custom `quantize()` hook. Currently, PTQ fake quant is available, but `export_deploy()` in deployment export does not support this type of hook yet.

### AWQ

AWQ (Activation-aware Weight Quantization) is a post-training quantization (PTQ) method for large models. Through an activation-aware weight quantization strategy, it significantly reduces memory usage and improves inference speed while maintaining model precision. This algorithm assumes that not all weights are equally sensitive to quantization error, and protecting only about 1% of the salient weights can greatly reduce the error. Using a small amount of calibration data, it employs grid search to find the optimal scaling factor within a preset range, amplifies the salient weights before quantization, and reduces the error by narrowing the quantization interval.

For detailed algorithm information, click [here](https://arxiv.org/abs/2306.00978).

### GPTQ

GPTQ (Generative Pre-trained Transformer Quantization) is a weight-only post-training quantization (PTQ) method for large models. Its core idea is layer-wise greedy quantization, which minimizes the quantization error through mathematical optimization. For each quantization operator, it computes the Hessian matrix based on the activation values, fine-tunes the weights block by block to minimize the quantization error, and finally outputs the adjusted weights and the quantization factors.

For detailed algorithm information, click [here](https://arxiv.org/abs/2210.17323).

Note: For the HiFloat4 format (corresponding to the HiGPTQ algorithm), GPTQ automatically switches to group quantization of 64 elements to match the three-level hierarchical block scale of HiFloat4. The columns within a group must be solved jointly and cannot be quantized independently per column as with int/mxfp. The rest of the Hessian calibration process is the same as standard GPTQ, and only weight quantization is supported.

### LWC

LWC (Learnable Weight Clipping) is a differentiable weight range optimization technique for neural network quantization, especially for post-training quantization of LLMs. It adds channel-scaling-like structural transformations on top of learnable clipping to enhance the quantization precision of low-bit W/A quantization. For detailed algorithm information, see [OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models](https://arxiv.org/abs/2308.13137).

## Activation Quantization Algorithm

### LAC

LAC (Learnable Activation Clipping) is a differentiable activation range optimization technique for neural network quantization (especially LLMs). It adds channel-scaling-like structural transformations on top of learnable clipping to enhance low-bit W/A quantization precision. For detailed algorithm information, see [OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models](https://arxiv.org/abs/2308.13137).

## Full Quantization Algorithm

### DMQ Balancing Algorithm

This algorithm is used in post-training quantization scenarios.

DMQ (Diagonal Matrix Quantization Balancer) balances the data before computing the quantization factor, transferring part of the quantization difficulty of the data to the weights. As shown in the following figure, X is the data and W is the weight.

![](./figures/dmq.png)

The balancing principle is as follows:

1. For each channel of the input data of the operator to be quantized, compute the balancing factor.

2. Divide the data by the balancing factor and multiply the weight by the balancing factor, so that the computation result of the operator to be quantized is mathematically equivalent before and after balancing.

### HFMG Data Quantization Algorithm

This algorithm is used in post-training quantization scenarios.

The HFMG (Histogram Feature Map Glutton) algorithm records the data distribution of activation data by means of histograms and determines the optimal quantization truncation position through searching. The quantization principle is as follows:

1. First, create a histogram based on the input activation data.
2. If there is data from more batches, create a histogram for the data of each batch, and then merge the histograms, as shown in [Figure 1](#fig1126323519407).
3. Based on the histogram of the activation data, determine the truncation point of the data through searching, as shown in [Figure 2](#fig1419882412).

The data of this quantization algorithm is stored in memory. Compared with the [IFMR Data Quantization Algorithm](#ifmr-data-quantization-algorithm), it saves more memory. This algorithm and the IFMR data quantization algorithm cannot be configured simultaneously on the same layer.

The **create\_quant\_config** API uses the IFMR data quantization algorithm by default for quantization. To use the HFMG data quantization algorithm, you can only configure it through the **config\_defination** parameter of the **create\_quant\_config** API in a simple configuration file.

**Figure 1**  Histogram merging process<a name="fig1126323519407"></a>  
![](./figures/histogram_merg_process.png "Histogram merging process")

**Figure 2**  Searching for the optimal left and right truncation positions<a name="fig1419882412"></a>  
![](./figures/clip_min_max_search.png "Searching for the optimal left and right truncation positions")

If the optimal quantization result is not obtained, you can change the number of bins in the histogram (the smallest unit histogram bar in a histogram) and select a group of parameters with a smaller quantization error as the final quantization result. In the HFMG algorithm, the **num\_of\_bins** configuration parameter is used to adjust the number of bins in the histogram. For details about the parameter, see the configuration parameters of the **simple configuration file for post-training quantization \> HFMGQuantize**. Generally, a larger **num\_of\_bins** value means a stronger ability of the histogram to fit the original data distribution, which may yield a better quantization result, but the post-training quantization process also takes more time.

### IFMR Data Quantization Algorithm

The IFMR (Input Feature Map Reconstruction) algorithm determines the optimal quantization method through searching under a given data distribution. This algorithm is used in post-training quantization scenarios. Its quantization principle is shown in the following figure:

**Figure 1**  IFMR data quantization algorithm<a name="fig11850413154412"></a>  
![](./figures/ifmrquant.png "IFMR data quantization algorithm")

In the figure above, number \[1,3\] represents \[clip\_min\_start, clip\_min\_end\], 2 represents clip\_min, \[4,6\] represents \[clip\_max\_start, clip\_max\_end\], and 5 represents clip\_max.

The quantization process of this algorithm consists of two steps, as shown in [Figure 1](#fig11850413154412). In the first step, the floating-point numbers are truncated to the \[clip\_min, clip\_max\] range, that is, the \[2,5\] point in [Figure 1](#fig11850413154412). In the second step, the floating-point data is quantized to the int range. Generally, values near the boundary of the data distribution are relatively sparse and can all be truncated to improve quantization precision.

To achieve the optimal quantization result, the truncation range \[clip\_min, clip\_max\] can be continuously changed, and the set of ranges that yields the best quantization result is selected as the final quantization result. In the IFMR algorithm, **clip\_min** and **clip\_max** are used to set the search range and search step, and the optimal quantization result is obtained through traversal. The parameters provided by **FMRQuantize** in the quantization configuration (taking the PyTorch framework as an example, see [Table 1](./context/ptq_config.md#zh-cn_topic_0240188735_table1225503375617) for parameter explanation) are used to adjust the truncation range.

- The traversal value range of **clip\_min** (that is, number 2 in [Figure 1](#fig11850413154412)) is \[clip\_min\_start, clip\_min\_end\] (that is, \[1,3\] in [Figure 1](#fig11850413154412)), with a step of **search\_step**.
- The traversal value range of **clip\_max** (that is, number 5 in [Figure 1](#fig11850413154412)) is \[clip\_max\_start, clip\_max\_end\] (that is, \[4,6\] in [Figure 1](#fig11850413154412)), with a step of **search\_step**.

In the descending data sequence, **clip\_min\_init** is obtained based on the **min\_percentile** parameter, thereby obtaining the **clip\_min\_start** and **clip\_min\_end** parameters, where:

- clip\_min\_start = clip\_min\_init\*search\_range\_start
- clip\_min\_end = clip\_min\_init\* search\_range\_end

In the ascending data sequence, the **clip\_max\_init** is obtained based on the **max\_percentile** parameter, thereby deriving the **clip\_max\_start** and **clip\_max\_end** parameters, where:

- clip\_max\_start =clip\_max\_init\*search\_range\_start
- clip\_max\_end =clip\_max\_init\* search\_range\_end

Generally, the larger the search range \[clip\_min\_start, clip\_min\_end\] and the smaller the search step **search\_step**, the higher the quantization precision that may be achieved, but the more time quantization takes.

### MIN-MAX

This algorithm collects the maximum and minimum values of the data in activations and weights. Based on these maximum and minimum values, it scales the data to the boundary value range of the quantization bit width, and then calculates the corresponding scaling factor scale and offset value offset according to the quantization formula.

During initialization, the maximum and minimum values of weight are computed offline according to the weight quantization granularity (for example, **PER_TENSOR**), and **scale_w**/**offset_w** are computed offline and added to the record object. During forward propagation, the maximum and minimum values of the current batch are obtained, and the global maximum and minimum values are refreshed. After the min/max values of weight and activation are obtained, scale and offset are calculated according to the following formulas (for weight, only the symmetric scenario is supported, and offset is 0).

- Symmetric quantization scenario

  ```bash
  if abs(max) > abs(min):
      scale = abs(max)/(2^(quant_bit-1)-1)
  else:
      scale = abs(max)/(2^(quant_bit-1))
      
  offset = 0
  ```

- Asymmetric quantization scenario

  ```bash
  scale=(max-min)/(2^quant_bit-1)
    
  offset=-2^(quant_bit-1)-min/scale
  ```

### OFMR

The OFMR (Output FeatureMap Reconstruct) algorithm determines the optimal quantization factor through a search process under a given data distribution. This algorithm is used for post-training quantization. Its quantization principle is illustrated as follows:

1. Infer the original model to obtain the original output of the model to be quantized.
2. Traverse all candidate quantization factors, infer the model, and obtain the quantized output.
3. Compare the error between the quantized output and the original output, and select the quantization factor with the smallest error.

**Figure 1**  OFMR algorithm principle<a name="fig1393075517419"></a>  
![](./figures/ofmrquant.png "OFMR algorithm principle")

### OmniQuant

OmniQuant is a lightweight, differentiable post-training quantization algorithm for LLMs. Its core is structural transformation that learns channel scaling: it adds channel-scaling structural transformations on top of learnable clipping to enhance low-bit W/A quantization precision.

For detailed algorithm information, see [OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models](https://arxiv.org/abs/2308.13137).

### SmoothQuant

SmoothQuant is a post-training quantization (PTQ) method that ensures both accuracy and inference efficiency. Since weights are easy to quantize while activations are difficult to quantize, SmoothQuant smooths activation outliers by introducing a smoothing factor, and shifts the quantization difficulty from activations to weights through a mathematically equivalent transformation. Generally, the more activation outliers there are, the greater the migration intensity.

For detailed algorithm information, see [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438).

### FlatQuant

FlatQuant is a post-training quantization (PTQ) method for large models. Its core idea is to apply mathematically equivalent transformations to activations and weights through learnable affine transformations, making the activation distribution flatter and thereby reducing the difficulty of low-bit quantization. Specifically, FlatQuant introduces structured parameters such as learnable affine transformations, diagonal scaling, and Kronecker decomposition matrices. While preserving the computational equivalence of linear layers, it redistributes activations and weights along the hidden dimension so that outliers are flattened across more dimensions. In general, the sharper the activation distribution and the more concentrated the outliers, the more FlatQuant's structural transformation helps improve the quantized precision.

For detailed algorithm information, see [FlatQuant: Flatness Matters for LLM Quantization](https://arxiv.org/abs/2410.09426).

### ULQ Data Quantization Algorithm

This algorithm is used in quantization-aware training scenarios.

The ULQ (Universal Linear Quantization) algorithm continuously trains the quantization factor during training to reduce quantization loss. During initialization, this algorithm quantizes the values (including truncation) and is sensitive to initialization. In the quantization configuration, ULQuantize (taking the PyTorch framework as an example; for parameter explanation, see the [simple configuration file for quantization-aware training](./context/qat_config.md)) controls the ULQ algorithm.

- **clip\_max\_min** is the truncation range for initialization, **clip\_min** is the lower truncation bound, and **clip\_max** is the upper truncation bound. They can be set based on the data range; if not set, AMCT uses its built-in default initialization method.
- **fixed\_min** indicates whether to set the lower truncation bound to 0 during training. For example, when the layer preceding the quantization layer is a ReLU layer, the data range is \[0, +inf\]. In this scenario, data quantization can set **fixed\_min** to **true**, indicating that the lower truncation bound is always 0.

## Sparsity Algorithm

### Automatic Channel Sparsity Search Algorithm

The automatic channel sparsity search uses the greedy search algorithm by default. Under a given model computation constraint, it automatically selects the appropriate channels that can be removed from each layer (channels with lower value ranking), reducing the impact of channel sparsity on the model's precision loss after fine-tuning.

This problem is a combinatorial optimization problem, modeled as a knapsack problem. Given the value v<sub>i</sub> and weight F<sub>i</sub> of each item, find the items with the maximum total value that can fit within the knapsack capacity C. It can be expressed as:

Maximize:

$$

\max_{b}\sum_{i} b_i v_i

$$

Subject to:

$$
\text{s.t.}\quad \sum_{i} b_i F_i \le c,\; b_i \in \{0,1\}
$$

Here, v<sub>i</sub> corresponds to the sparsity sensitivity; F<sub>i</sub> is the bit complexity; b is the channel sparsity scheme, that is, whether the channel is retained; and C is the compression ratio set by the user.

For the channel sparsity task, the weight F<sub>i</sub> is the computational cost of the i-th channel, and v<sub>i</sub> is the change in the loss function of network w after pruning the i-th channel:

$$

v_i = \text{loss}(w) - \text{loss}(w - W_i)

$$

Here, loss\(w - w<sub>i</sub>\) can be approximated using a first-order or second-order Taylor expansion.

Based on the target sparsity ratio specified by the user, the value density (sparsity sensitivity/bit complexity) of each channel in each layer is calculated. After global sorting, the batch of channels with the lowest value density is selected for pruning, and the solution is obtained through the combinatorial optimization method described above, thereby completing the search task.

### Manual Channel Sparsity Algorithm

The channel sparsity algorithm used by AMCT is BalancedL2Norm. This algorithm computes the L2 norm (the square root of the sum of squares of all elements) of each filter (output-dimension channel) of the weights, ranks the output channels by importance, and prunes the channels with lower importance. That is, channels with lower importance are sparsified first. The principle is shown in the following figure:

**Figure 1** Channel sparsity algorithm principle<a name="fig856211635614"></a>  
![](./figures/channel_sparsity_algorithm.png "Channel sparsity algorithm principle")

In the sparsity configuration file, the BalancedL2Norm algorithm is controlled by the **BalancedL2NormFilterPruner** field (using the PyTorch framework as an example; for parameter explanation, see the [simple configuration file for quantization-aware training](./context/qat_config.md)).

- **prune\_ratio**: Sparsity rate, which is the ratio of the number of sparsified filters to the total number of filters. The degree of sparsification is controlled by the sparsity rate. For example, a sparsity rate of 0.3 means that 30% of the output channels will be pruned.
- **ascend\_optimized**: Ascend affinity optimization. The currently used optimization scheme is 16-alignment. When 16-alignment is enabled, the number of channels retained after sparsification is aligned to a multiple of 16. For example, if the original number of channels is 20 and a sparsity rate of 0.25 is configured without enabling 16-alignment, the number of channels retained after sparsification is 15. If the same sparsity rate is used but 16-alignment is enabled, the number of channels retained after sparsification is 16. This configuration can improve the inference performance of the sparsified model on AI processors.

### 4:2 Structured Sparsity Algorithm

The 4:2 structured sparsity algorithm used by AMCT is L1SelectivePrune. It determines which weights to retain by comparing the l1 values (absolute values) of the weights, retaining the 2 weights with the largest l1 values among every 4 consecutive weights.

In the sparsity configuration file, the L1SelectivePrune algorithm is controlled through the **L1SelectivePruner** field (for parameter explanation, see [Quantization-Aware Training Simple Configuration File](./context/qat_config.md)).

- **update\_freq**: The interval for updating 4:2 sparsity, that is, the interval at which the elements to be retained are computed. During retraining, the weights change with each training batch, and the corresponding l1 value ranking may change. For example, the first two of the four elements may originally be retained, but after an update, the first and third elements may be retained instead.

  **update\_freq=0** means that during training, the elements to be retained are computed only in the first batch; **update\_freq=2** means that during training, the elements to be retained are computed every 2 batches, and so on. The default value is **0**.

- **n\_out\_of\_m\_type**: Currently, only M4N2 is supported, that is, 2 weights are retained among every 4 consecutive weights.

## Distillation Algorithm

The idea of distillation quantization is to use the quantized model as the student model and the original model as the teacher model, guiding the quantized model to "imitate" the floating-point model to achieve better precision results. Distillation quantization requires only a small amount of unlabeled data to achieve good precision results within a short quantization time.

Steps of quantization distillation:

1. Quantize the original model to obtain a quantized model with the same structure as the floating-point model.

2. Divide several cascaded quantized layers into a distillation unit.

3. Use the output of the floating-point distillation module as the soft label to fine-tune the quantized distillation module.

4. Complete the distillation of all modules to obtain a quantized model with better precision.

   **Figure 1**  Distillation diagram<a name="fig275521214542"></a>  
   ![](./figures/distill_algorithm.png "Distillation diagram")
