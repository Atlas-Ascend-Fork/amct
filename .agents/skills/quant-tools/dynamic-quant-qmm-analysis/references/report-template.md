# Dynamic Quant + QMM 估算结果

## 结论摘要

- Benefit:
- Estimated speedup:
- Dominant bottleneck:
- Confidence:
- Platform assumption confidence:
- Key assumptions:

## 输入口径

- Scenario:
- Shape:
- Quant scheme:
- Platform:
- Baseline path:
- Target path:
- Weight preparation:
- A5 bandwidth variant:

## 时间分解

| Term | Estimate | Note |
| --- | ---: | --- |
| `T_baseline` |  | float/bf16 matmul baseline |
| `T_base_cube` |  | baseline cube compute |
| `T_base_mte2` |  | baseline input movement with cold GM and hot L2 bytes |
| `T_base_mte3` |  | baseline result write movement |
| `T_dynamic_quant` |  | activation quantization, including `X_q + scale/meta` write |
| `T_qmm_gm_to_l1` |  | GM/L2->L1, with cold/hot data assumptions |
| `T_qmm_l1_to_l0` |  | L1->L0A/L0B/L0C movement |
| `T_qmm_cube` |  | selected qmm cube candidate |
| `T_qmm_fixpipe` |  | FixPipe result handling cost |
| `T_qmm_antiquant_rescale` |  | int path antiquant/rescale after cube |
| `T_qmm_mte3` |  | target result write movement |
| `T_qmm_candidate` |  | selected qmm candidate |
| `T_target` |  | dynamic quant + qmm |

## 风险与待确认项

- Missing inputs:
- Sensitivity:
- What could change the conclusion:
- What to confirm next:
