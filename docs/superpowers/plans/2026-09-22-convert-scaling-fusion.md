# Convert Scaling Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement shared-input quantization groups, group-level calibration parameters, and convert-time deployment fusion from `docs/2026-09-08-convert-scaling-fusion/{srs,sd,stc}.md`.

**Architecture:** Add a fusion registry and hook-based structure discovery after quantization insertion. Coordinate MinMax/SmoothQuant modules through `FusionGroup` during calibration, then run a new fusion pass before the existing NPU replacement pass. Static activation granularities use `FusedPreOp`; dynamic per-token uses model-provided layer replicas.

**Tech Stack:** Python, PyTorch, AMCT classic quantization modules, pytest.

## Global Constraints

- Public `quantize(model, config)` and `convert(model)` signatures remain unchanged.
- Registry is stored as `_amct_fusion_registry`, not as a buffer/parameter.
- One quantization module belongs to at most one group.
- Paths containing reshape/transpose/cat/split/chunk do not create groups.
- Group failure keeps all members on the existing independent path.
- `convert` must not recalculate activation scales or SmoothQuant factors.

---

### Task 1: Fusion registry and first-forward structure discovery

**Files:**
- Create: `amct_pytorch/classic/optimizer/fusion_group.py`
- Create: `amct_pytorch/classic/optimizer/fusion_hooks.py`
- Modify: `amct_pytorch/classic/quantize.py`
- Test: `tests/unit_test/classic/optimizer/test_fusion_group.py`

**Interfaces:**
- `install_fusion_hooks(model) -> FusionHookHandle`
- `build_fusion_groups(model, records) -> FusionRegistry`
- `attach_fusion_registry(model, registry)`
- `FusionGroup` fields required by SD 3.1.3.

- [ ] Write failing tests for Q/K/V discovery, up/gate discovery, layout transform rejection, outside consumer rejection, configuration mismatch rejection, and unique ownership.
- [ ] Implement hook collection and group construction.
- [ ] Call hook installation from `quantize` after `InsertQuantizeModulePass`.
- [ ] Verify all structure-discovery tests pass.

### Task 2: Group-level MinMax and SmoothQuant calibration

**Files:**
- Modify: `amct_pytorch/classic/optimizer/fusion_group.py`
- Modify: `amct_pytorch/classic/quantize_op/minmax_module.py`
- Modify: `amct_pytorch/classic/quantize_op/smooth_quant_module.py`
- Test: `tests/unit_test/classic/optimizer/test_fusion_calibration.py`

**Interfaces:**
- `FusionGroup.update_activation_stat(inputs)`
- `FusionGroup.finalize_quant_params()`
- Quant modules expose `fusion_group` and clear stale fake-quant caches after parameter write-back.

- [ ] Write failing tests proving shared activation statistics are counted once per batch.
- [ ] Write failing tests proving group members receive identical activation scales and SmoothQuant factors.
- [ ] Write failing tests proving member weight scales remain independent.
- [ ] Modify module forward paths to delegate final parameter calculation to an attached healthy group.
- [ ] Verify `batch_num=1` group results override member-only results and clear caches.
- [ ] Verify failed groups fall back to independent module calculation.

### Task 3: Convert-time scaling fusion and static shared quantization

**Files:**
- Modify: `amct_pytorch/classic/optimizer/fusion_group.py`
- Create: `amct_pytorch/classic/optimizer/apply_scaling_fusion_pass.py`
- Create: `amct_pytorch/classic/deploy_op/shared_quantize.py`
- Modify: `amct_pytorch/classic/deploy_op/npu_quantization_linear.py`
- Modify: `amct_pytorch/classic/optimizer/replace_npu_quant_pass.py`
- Modify: `amct_pytorch/classic/quantize.py`
- Test: `tests/unit_test/classic/optimizer/test_apply_scaling_fusion.py`

**Interfaces:**
- `ApplyScalingFusionPass.run(model)`
- `FusedPreOp(pre_op, shared_quantize)`
- `SharedQuantize` for static granularities.
- `NpuQuantizationLinear(input_quant_disabled=...)` variant behavior.

- [ ] Write failing tests for Norm and Linear pre-op parameter formulas.
- [ ] Write failing tests proving SmoothQuant weight and activation scaling are applied exactly once.
- [ ] Write failing tests for MinMax shared quantization without parameter fusion.
- [ ] Add fusion pass before `ReplaceNpuQuantModulePass`.
- [ ] Verify repeat convert is idempotent.

### Task 4: Dynamic per-token layer replacement

**Files:**
- Create: `amct_pytorch/common/models/llm/qwen/qwen3/fused_layer.py`
- Create: model-specific registry hook for dynamic per-token replicas.
- Modify: `amct_pytorch/classic/optimizer/apply_scaling_fusion_pass.py`
- Test: `tests/unit_test/common/models/llm/qwen/test_qwen3_fused_layer.py`

**Interfaces:**
- Qwen3 dynamic per-token layer replica with the original public signature.
- Shared dynamic quantization returns `(quantized_input, pertoken_scale)`.

- [ ] Write failing tests proving the layer replica consumes one dynamic quantization result for Q/K/V or up/gate.
- [ ] Implement Qwen3 layer replica by moving the original layer into the replacement wrapper.
- [ ] Skip dynamic groups without a registered replica and emit WARNING.
- [ ] Verify the replacement path preserves residual connections and layer outputs.

### Task 5: Regression and smoke verification

**Files:**
- Modify: `tests/smoke/run_smoke.py` only if needed for explicit group assertions.

- [ ] Run focused unit tests.
- [ ] Run existing classic quantization tests.
- [ ] Run AMCT smoke MinMax and SmoothQuant cases on available NPU.
- [ ] Check status, logs, and repeat-convert behavior.
- [ ] Run static code checks used by this repository.
