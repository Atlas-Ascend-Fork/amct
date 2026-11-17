# algorithm\_register

<!-- md-trans-meta sourceCommit=4a47814eed18048e444f9a2acc42d9864ddc477f translatedAt=2026-10-16T08:04:07.040Z pushedAt=2026-11-05T01:12:03.402Z nl=2 -->

## Supported Products

| Product        | Supported |
| -------------- | --------- |
| Atlas A3 products | √        |
| Atlas A2 products | √        |

## Description

Registers a user-provided custom algorithm with the AMCT tool.

## Prototype

```python
algorithm_register(name, src_op, quant_op, deploy_op)
```

## Parameters

| Parameter | Input/Output | Description                                             |
| --------- | ------------ | ------------------------------------------------------- |
| name      | Input        | Meaning: Algorithm name.<br/>Data type: string.          |
| src_op    | Input        | Meaning: Operator to be replaced.<br/>Data type: string.        |
| quant_op  | Input        | Meaning: Quantization operator.<br/>Data type: torch.nn.Module. |
| deploy_op | Input        | Meaning: Deployment operator.<br/>Data type: torch.nn.Module. |

## Return Value

None

## Example

```python
# Custom algorithm name.
name = 'customize_algo'
# Operator type to be quantized.
src_op = 'Linear'
# User-implemented quantization operator.
class CustomizedQuantOp(BaseQuantizeModule):
    def __init__(self,
                 ori_module,
                 layer_name,
                 quant_config):
        super().__init__(ori_module, layer_name, quant_config)
        
    @torch.no_grad()
    def forward(self, inputs):
        return
quant_op = CustomizedQuantOp
# User-implemented deployment operator.
class CustomizedDeployOp(torch.nn.Module):
    def __init__(self, quant_module):
        super().__init__()
    
    def forward(self, x):
        return
deploy_op = CustomizedDeployOp
# Register the custom algorithm.
algorithm_register(name, src_op, quant_op, deploy_op)
```
