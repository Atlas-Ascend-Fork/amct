# Kimi K3 MXFP NPU Implementation Plan

**Goal:** Add opt-in `--real_quant` to Kimi K3 blockwise quant evaluation, retaining BF16 layers.

**Architecture:** Kimi owns checkpoint naming, validation and streaming loading. An inference-only Linear in `mxfp.py` owns low-bit buffers, lossless layout conversion and A8W8/A4W4 execution. Existing fake quantizers and other model loaders remain unchanged.

**Spec:** User-approved design in this task: paired checkpoint weight/scale loading, per-projection bit policy, dynamic activation quantization, isolated Kimi validation.

**Constraints:** Preserve existing workspace changes. No NPU hardware available locally; CPU tests use an explicit operator double and are not hardware validation. First version supports direct quantization and prequantized checkpoints, not PTQ parameter application or cache quantization. Reject unsupported modes explicitly.

- [x] Add failing tests for layout preservation, dispatch, checkpoint validation and mixed precision loading.
- [x] Implement NPU MXFP Linear and Kimi-specific streaming loader; preserve floating and packed tensor types.
- [x] Enable and validate eval CLI, add example YAML and usage documentation.
- [x] Run targeted CPU regressions, syntax/lint checks; document hardware validation remaining.

## Verification

- Expanded CPU suite: 230 passed; two existing Kimi cache/PTQ tests fail identically against HEAD sources.
- New tests include an independent decoded FP4 matmul reference, cross-shard loading, scale validation, policy conflicts and BF16 bypass.
- Independent review found no actionable new correctness issue.
- Ruff on touched production files reports pre-existing diagnostics outside the new implementation; new tests pass Ruff.
- Hardware NPU kernel/PPL validation remains outstanding; local execution used CPU PyTorch 2.14.0, not the production NPU runtime.
