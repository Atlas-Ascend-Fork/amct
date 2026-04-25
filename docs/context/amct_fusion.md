# 工具实现的融合功能<a name="ZH-CN_TOPIC_0000002548668643"></a>

## 融合功能<a name="section0331458296"></a>

当前该工具主要实现的为Conv+BN融合：AMCT在量化前会对模型中的"torch.nn.Conv2d+torch.nn.BatchNorm2d"结构做Conv+BN融合，融合后的torch.nn.BatchNorm2d层会被删除。

