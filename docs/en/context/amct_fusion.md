# Fusion Function Implemented by the Tool<a name="ZH-CN_TOPIC_0000002548668643"></a>

<!-- md-trans-meta sourceCommit=57c5f99db95a0bc1b70fa91a52c1a2c25d59ea20 translatedAt=2026-10-16T08:39:57.021Z pushedAt=2026-11-05T01:12:03.429Z nl=2 -->

## Fusion Function<a name="section0331458296"></a>

Currently, the tool mainly implements Conv+BN fusion. Before quantization, AMCT performs Conv+BN fusion on the "**torch.nn.Conv2d**+**torch.nn.BatchNorm2d**" structure in the model, and the fused **torch.nn.BatchNorm2d** layer is deleted.
