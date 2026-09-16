# Qwen3.6 Ascend Tensorwise Deploy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Qwen3.6 packed-MoE aware W8A8_DYNAMIC Ascend tensorwise deployment.

**Architecture:** Keep model-specific naming and packed expert expansion in the existing Qwen3.6 adapter. Extend the shared workflow only with optional adapter hooks and allow-list validation. Use a JSON selection file over the expanded vLLM module names.

**Tech Stack:** Python, PyTorch, safetensors, pytest, vLLM-Ascend 0.23.0 metadata contract.

**Spec:** `docs/superpowers/specs/2026-09-16-qwen3-6-ascend-deploy-design.md`

## Global Constraints

- Preserve `legacy` deploy behavior.
- Ascend path accepts only tensorwise `int` with W8A8 defaults.
- Do not implement or modify Qwen3.6 forward/PTQ behavior for this deploy-only task.
- Use repository-relative paths and existing shared exporter/validator APIs.

### Task 1: Tests for Qwen3.6 deploy contract

**Files:**
- Create: `tests/unit_test/common/models/llm/qwen/test_qwen3_6_deploy.py`
- Modify: `tests/unit_test/common/models/llm/common/test_deploy_selection.py`

- [x] Add tests that fail until Qwen3.6 is allowed, packed virtual inventory expands to 256 experts, JSON patterns select routed experts/full-attention QKV, and packed `gate_up_proj` splits into gate/up tensors.
- [x] Run the focused tests and confirm expected failures.

### Task 2: Implement Qwen3.6 adapter hooks

**Files:**
- Modify: `amct_pytorch/common/models/llm/qwen/qwen3_6/qwen3_6_moe.py`

- [x] Add config-driven role detection, virtual inventory expansion, and packed state-dict expansion while preserving inherited PTQ methods.
- [x] Run Task 1 tests and make them pass.

### Task 3: Add shared workflow hooks and selection JSON

**Files:**
- Modify: `amct_pytorch/common/models/llm/common/deploy_selection.py`
- Modify: `amct_pytorch/workflows/llm_deploy.py`
- Create: `amct_pytorch/configs/deploy/qwen3_6_ascend_w8a8.json`

- [x] Allow `qwen3_6_moe`, resolve names with non-suffixed packed tensors, invoke adapter virtual inventory/state expansion, and keep legacy unchanged.
- [x] Add the confirmed W8A8 JSON rules and run focused workflow tests.

### Task 4: Full verification and commit

- [x] Run Qwen/deploy unit tests, `compileall`, and `git diff --check`.
- [x] Use the supplied real indexes for selection statistics and report unresolved NPU runtime risks.
- [x] Commit the branch with a focused message.
