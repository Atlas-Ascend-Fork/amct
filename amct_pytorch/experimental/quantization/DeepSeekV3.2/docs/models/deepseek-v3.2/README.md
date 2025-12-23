# NPU DeepSeek-V3.2 量化训练及推理

DeepSeek团队发布了最新的模型DeepSeek-V3.2，可利用稀疏架构 **DeepSeek Sparse Attention(DSA)** 来提高长序列的计算效率，降低推理成本。长上下文场景和其新颖的DSA结构，共同对推理优化系统提出了新诉求。

## 量化策略

相对于BF16推理，Int8量化可以有效降低端到端时延，提升系统吞吐。目前本sample已经支持W8A8C8/W4A8C8量化，量化架构如下：
![w8a8c8_quantization.png](figures/w8a8c8_quantization.png)
其中MLA量化位置如下：
![w8a8c8_quantization_details.png](figures/w8a8c8_quantization_details.png)
- MLAProlog：除Q_b_proj使用W8A8，其他Linear均不量化；KVCache量化到C8；
- Sparse Flash Attention：KVCache Int8存储，BF16计算；
- IndexerProlog：除Q_b_proj使用W8A8，其他Linear均不量化；Indexer Q使用A8量化；Indexer Cache使用C8量化；
- Lightning Indexer: BatchMatmtul使用Int8计算；
- MoE：路由专家使用W8A8/W4A8量化，共享专家使用W8A8量化；
- MLAEpilog：O_proj使用W8A8量化；
- LM_Head：暂不量化。

**注：
W8A8：W8指权重使用静态Per-Channel Int8量化，A8指数据使用动态Per-Token Int8量化；
A8C8：A8表示Lightning Indexer中的Q使用动态Per-Token-Head Int8量化，Indexer Cache使用动态Per-Token-Head Int8量化；
MLAEpilog：O_proj使用W8A8量化；
KVCache C8：表示KVCache 使用动态Per-Token-Head-Tile-128 Int8量化；**
### 量化目的
本sample量化位置与Ascend硬件性能强耦合，对性能瓶颈处做了竞争力的量化，部署友好 

在当前W8A8C8量化策略下，线性层的量化覆盖率较低，MLA线性层中只对`q_b_proj`和`w_o_proj`进行了量化，Indexer模块只量化了`wq_b_proj`。主要原因是IndexerProlog融合算子设计成`weights_proj`模块的输出格式为fp16，且不做量化，因此MLA输入关联的Linear统一不做量化，好处是可将同一份BF16数据输入IndexerProlog和MLAProlog。

其次，MLAProlog KVCache的量化策略使用了动态存8算16。在超长序列情况下，W8A8C8量化精度接近无损，同时权重内存占用优化2倍。MLA C8算16获取内存收益，可以打高吞吐量。另一方面，LightningIndexer的A8C8获取计算收益，降低LI计算时延，TTFT和TPOT也同步优化。

W4A8C8量化版本针对`DeepSeek-V3.2`使用基于学习的量化算法优化Clamp参数，缓解W4A8离群值量化困难的问题，实现了较优的量化模型精度。同时，W4A8C8版本比W8A8C8节约MoE权重显存2x，因此在大EP场景下，利用W4A8 MoEGMM算子，同一张卡可以装下更多的专家，节约资源，优化计算访存比，实现单机部署。


