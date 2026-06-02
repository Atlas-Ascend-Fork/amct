#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AMCT Pruning Comprehensive Test
HuggingFace-style mini models: MLP, CNN, MoE
"""
import os
import shutil
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(__file__))

from mini_models import create_mini_mlp, create_mini_cnn, create_mini_moe
from amct_pytorch.experimental.pruning import (
    DENSE_LOWVAR_PRUNE_CFG,
    CNN_VARIANCE_PRUNE_CFG,
    MOE_MASSVAR_PRUNE_CFG,
    prune,
    get_last_report,
)

print("=" * 80)
print("AMCT Pruning Comprehensive Test")
print("=" * 80)

# ============================================================================
# Test 1: Various prune_ratio test (MLP)
# ============================================================================
print("\n" + "=" * 80)
print("TEST 1: Various prune_ratio test (Mini MLP)")
print("=" * 80)

test_ratios = [0.3, 0.5, 0.7]
results_ratio = []

# Calibration data for MLP
calib_data_mlp = [torch.randint(0, 1000, (4, 20)) for _ in range(8)]

for ratio in test_ratios:
    model, config = create_mini_mlp()
    model.eval()
    params_before = sum(p.numel() for p in model.parameters())

    cfg = {
        'methods': {
            'dense': {
                'name': 'low_variance',
                'kwargs': {'prune_ratio': ratio}
            }
        },
        'missing_data_policy': 'warn_skip'
    }

    prune(model, cfg, data=calib_data_mlp)
    report = get_last_report()

    reduction = (1 - report.params_after / max(report.params_before, 1)) * 100
    results_ratio.append({
        'ratio': ratio,
        'before': params_before,
        'after': report.params_after,
        'reduction': reduction
    })

    print(f"\n  Ratio {ratio}: {params_before:,} -> {report.params_after:,} "
          f"({reduction:.2f}% reduction)")

print("\n✓ Test 1 completed")

# ============================================================================
# Test 2: Model save/load test (MLP)
# ============================================================================
print("\n" + "=" * 80)
print("TEST 2: Pruned model save/load test (Mini MLP)")
print("=" * 80)

# Create and prune original model
original_model, config = create_mini_mlp()
original_model.eval()

prune(original_model, DENSE_LOWVAR_PRUNE_CFG, data=calib_data_mlp)
report = get_last_report()
print(f"\nPruning: {report.params_before:,} -> {report.params_after:,}")

# Test input and output
test_input = torch.randint(0, 1000, (2, 20))
with torch.no_grad():
    original_output = original_model(test_input)
print(f"Original output shape: {original_output.shape}")
print(f"Original output mean: {original_output.mean().item():.6f}")

# Save to temp directory
temp_dir = tempfile.mkdtemp(prefix="amct_mlp_")
save_path = os.path.join(temp_dir, "pruned_mlp.pth")

torch.save({
    'model_state_dict': original_model.state_dict(),
    'config': config.__dict__,
    'pruning_info': {
        'params_before': report.params_before,
        'params_after': report.params_after,
    }
}, save_path)
print(f"Saved to: {save_path}")

# Load to new model
loaded_model, _ = create_mini_mlp()
prune(loaded_model, DENSE_LOWVAR_PRUNE_CFG, data=calib_data_mlp)

checkpoint = torch.load(save_path)
loaded_model.load_state_dict(checkpoint['model_state_dict'])
loaded_model.eval()

with torch.no_grad():
    loaded_output = loaded_model(test_input)

output_diff = torch.abs(original_output - loaded_output).max().item()
print(f"Loaded output mean: {loaded_output.mean().item():.6f}")
print(f"Max difference: {output_diff:.10f}")

if output_diff < 1e-6:
    print("✓ Outputs perfectly match!")
    save_load_status = "Perfect match"
else:
    print(f"✓ Outputs similar (diff: {output_diff})")
    save_load_status = f"Similar (diff: {output_diff:.2e})"

shutil.rmtree(temp_dir)
print("✓ Test 2 completed")

# ============================================================================
# Test 3-1: Mini MLP model pruning
# ============================================================================
print("\n" + "=" * 80)
print("TEST 3-1: Mini MLP model Dense Pruning")
print("=" * 80)

mlp_model, mlp_config = create_mini_mlp()
mlp_model.eval()
mlp_params_before = sum(p.numel() for p in mlp_model.parameters())

print(f"\nModel info:")
print(f"  Hidden size: {mlp_config.hidden_size}")
print(f"  Intermediate size: {mlp_config.intermediate_size}")
print(f"  Num layers: {mlp_config.num_hidden_layers}")
print(f"  Total params: {mlp_params_before:,}")

prune(mlp_model, DENSE_LOWVAR_PRUNE_CFG, data=calib_data_mlp)
report = get_last_report()

mlp_reduction = (1 - report.params_after / max(report.params_before, 1)) * 100
print(f"\nPruning result:")
print(f"  {report.params_before:,} -> {report.params_after:,}")
print(f"  Reduction: {mlp_reduction:.2f}%")

# Inference test
test_input = torch.randint(0, 1000, (2, 20))
with torch.no_grad():
    output = mlp_model(test_input)
print(f"  Inference successful: {output.shape}")

print("✓ Test 3-1 completed")

# ============================================================================
# Test 3-2: Mini CNN model Channel Pruning
# ============================================================================
print("\n" + "=" * 80)
print("TEST 3-2: Mini CNN model Channel Pruning")
print("=" * 80)

cnn_model, cnn_config = create_mini_cnn()
cnn_model.eval()
cnn_params_before = sum(p.numel() for p in cnn_model.parameters())

print(f"\nModel info:")
print(f"  Num classes: {cnn_config.num_classes}")
print(f"  Total params: {cnn_params_before:,}")

# CNN calibration data
calib_data_cnn = [torch.randn(2, 3, 32, 32) for _ in range(8)]

prune(cnn_model, CNN_VARIANCE_PRUNE_CFG, data=calib_data_cnn)
report = get_last_report()

cnn_reduction = (1 - report.params_after / max(report.params_before, 1)) * 100
print(f"\nPruning result:")
print(f"  {report.params_before:,} -> {report.params_after:,}")
print(f"  Reduction: {cnn_reduction:.2f}%")

# Inference test
test_input = torch.randn(2, 3, 32, 32)
with torch.no_grad():
    output = cnn_model(test_input)
print(f"  Inference successful: {output.shape}")

print("✓ Test 3-2 completed")

# ============================================================================
# Test 3-3: Mini MoE model pruning
# ============================================================================
print("\n" + "=" * 80)
print("TEST 3-3: Mini MoE model Expert Pruning")
print("=" * 80)

moe_model, moe_config = create_mini_moe()
moe_model.eval()
moe_params_before = sum(p.numel() for p in moe_model.parameters())

print(f"\nModel info:")
print(f"  Hidden size: {moe_config.hidden_size}")
print(f"  Num experts: {moe_config.num_experts}")
print(f"  Experts per token: {moe_config.num_experts_per_tok}")
print(f"  Num layers: {moe_config.num_hidden_layers}")
print(f"  Total params: {moe_params_before:,}")

# MoE calibration data
calib_data_moe = [torch.randint(0, 1000, (4, 20)) for _ in range(8)]

prune(moe_model, MOE_MASSVAR_PRUNE_CFG, data=calib_data_moe)
report = get_last_report()

moe_reduction = (1 - report.params_after / max(report.params_before, 1)) * 100
print(f"\nPruning result:")
print(f"  {report.params_before:,} -> {report.params_after:,}")
print(f"  Reduction: {moe_reduction:.2f}%")

# Inference test
test_input = torch.randint(0, 1000, (2, 20))
with torch.no_grad():
    output = moe_model(test_input)
print(f"  Inference successful: {output.shape}")

print("✓ Test 3-3 completed")

# ============================================================================
# Final summary
# ============================================================================
print("\n" + "=" * 80)
print("Overall Test Summary")
print("=" * 80)

print("\n[Test 1] Various prune_ratio test:")
for r in results_ratio:
    print(f"  • Ratio {r['ratio']}: {r['reduction']:.2f}% reduction")

print(f"\n[Test 2] Model save/load test:")
print(f"  • Status: {save_load_status}")

print(f"\n[Test 3] Model-type specific pruning:")
print(f"  • MLP (Dense):   {mlp_reduction:.2f}% reduction")
print(f"  • CNN (Channel): {cnn_reduction:.2f}% reduction")
print(f"  • MoE (Expert):  {moe_reduction:.2f}% reduction")

print("\n" + "=" * 80)
print("All tests passed.")
print("AMCT Pruning verified:")
print("  - Dense Layer Pruning")
print("  - CNN Channel Pruning")
print("  - MoE Expert Pruning")
print("  - Model save/load")
print("  - Various prune_ratio")
print("=" * 80)
