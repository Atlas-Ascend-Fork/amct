# convert

<!-- md-trans-meta sourceCommit=4a47814eed18048e444f9a2acc42d9864ddc477f translatedAt=2026-10-16T08:12:00.840Z pushedAt=2026-11-05T01:12:03.406Z nl=2 -->

## Supported Products

| Product | Supported |
| ----------- | -------- |
| Ascend 950PR&950DT products | √ |
| Atlas A3 products | √ |
| Atlas A2 products | √ |

## Description

Converts a quantized calibration model into a quantized deployment model.

## Prototype

```python
convert(model)
```

## Parameters

| Parameter | Input/Output | Description |
| ------ | --------- | ------------------------------------------------------------ |
| model | Input/Output | Input meaning: quantized calibration model.<br>Output meaning: quantized deployment model.<br>Data type: torch.nn.Module. |

## Return Value

Quantized deployment model.

## Example

```python
# Call the preceding API to generate a quantized calibration model.
quantize(model, cfg)
# Call the API to convert the quantized calibration model into a quantized deployment model.
convert(model)
```
