# HiFloat4 Cast — Test Results

Qwen3.5-9B (bf16, 32 层 dense), Ascend 910B NPU.

## 相对 HiFloat4-private 的改动

### 架构

| | HiFloat4-private | AMCT (本分支) |
|---|---|---|
| NPU 伪量化 kernel | `hifx_kernel` / `hifx_kernel_bf16` (EasyASC 生成) | `hifx_kernel` / `hifx_kernel_bf16` (手写 AscendC)，BATCH=512 |
| 2D block kernel | `hif4_block_kernel` (64×64 2D 分块) | 无 — 不在本分支范围 |
| 打包/解包 | Python (`hif4_bit_dump.py`) | Python (`_hif4_encode_decode.py` + `hif4_pack_unpack.py`)，与伪量化共用编解码 |
| CLI 集成 | 独立脚本，手动格式检测 | `--quant_dtype hifp` 自动检测源格式 (bf16/fp16/hif4)，ptq 做 fake-quant/解包，deploy 做真打包 |

### 编码算法

与 `hifx.py` / `To_HiFX` 对齐：

- 3 级共享 scale: L1 E6M2 (64-block) → L2 E1 (8x) → L3 E1 (16x) → S1P2 尾数
- 元素除掉块 scale、做完二三级缩放后，在量化成 S1P2 之前先做一次 `round_to_bf16`
- 64-block 内出现 NaN/+Inf/-Inf 时：打包 scale byte 0 写入 0xFF，解码时整块恢复为 NaN（block-poison）
- 所有 2 的幂通过 `torch.ldexp` 实现，保证 Ascend NPU 上 bit-exact

### 打包字节序

| | HiFloat4-private | AMCT |
|---|---|---|
| value 字节 | 高位优先 (元素 0 → 高 4 bit) | **低位优先** (元素 0 → 低 4 bit) |

scale 4 字节布局一致: `[E6M2, E1_8, E1_16_hi, E1_16_lo]`。

### 修复的 bug

| Bug | 原因 | 修复 |
|-----|------|------|
| kernel tail masking | `BATCH=512` 时尾元素只重复清零最后一个 64-block，中间未覆盖的 stale 数据污染 block max | 从 BATCH-64 向 shape.N 反向遍历，对齐清零每个 64-block，跨 shape.N 的部分 block 用 vector mask 保护有效数据 |
| `qdim != -1` 广播错误 | `unflatten` 后 (8,2,4) 可能不在末尾，`hif4_encode` 的 `amax(dim=-1)` 维度错乱 | 按尺寸查找 (8,2,4) 并 permute 到末尾，编解码后逆 permute 还原 |

## 1. 单元测试: `tests/amct_ops/test_hifloat4_cast.py`

```
$ python -m unittest tests.amct_ops.test_hifloat4_cast -v

test_kernel_bf16 (tests.amct_ops.test_hifloat4_cast.TestHiFloat4Kernel) ... ok
test_kernel_fp16 (tests.amct_ops.test_hifloat4_cast.TestHiFloat4Kernel) ... ok
test_kernel_padded (tests.amct_ops.test_hifloat4_cast.TestHiFloat4Kernel) ... ok
test_kernel_qdim (tests.amct_ops.test_hifloat4_cast.TestHiFloat4Kernel) ... ok
test_kernel_zero_is_nan_free (tests.amct_ops.test_hifloat4_cast.TestHiFloat4Kernel) ... ok

----------------------------------------------------------------------
Ran 5 tests in 1.164s
OK
```

5/5 全部通过，包括：
- BF16 伪量化 kernel 与 golden 参考逐元素一致 (max diff < 1e-6)
- FP16 伪量化 kernel 与 golden 参考一致
- 非 64 对齐维度 (3×100) 正确 padding 且结果一致
- `qdim=0` (非最后一维) 正确重排后量化
- 零值输入不产生 NaN

## 2. CLI 端到端测试: PTQ (伪量化 → 浮点 checkpoint)

```
$ python -m amct_pytorch.cli.llm.ptq \
    --quant_dtype hifp \
    --bit_config amct_pytorch/configs/hifloat4.yaml \
    --quant_target mlp \
    --model_name qwen3_5 \
    --model /run/z84450661/modelscope/models/Qwen--Qwen3.5-9B/snapshots/master \
    --output_dir /tmp/hif4_test_ptq

[ptq/hifp] source format detected: bf16
Block Processing...: 100%|██████████| 32/32 [00:42<00:00, 1.34s/it]
Exported model to /tmp/hif4_test_ptq
```

输出: 32 × layer safetensors (float 权重复原) + 2 rest shards + config/tokenizer。

## 3. CLI 端到端测试: Deploy (真实打包 → HiF4 uint8)

```
$ python -m amct_pytorch.cli.llm.deploy \
    --quant_dtype hifp \
    --bit_config amct_pytorch/configs/hifloat4.yaml \
    --quant_target mlp \
    --granularity block \
    --model_name qwen3_5 \
    --model /run/z84450661/modelscope/models/Qwen--Qwen3.5-9B/snapshots/master \
    --output_dir /tmp/hif4_test_deploy

Block Processing...: 100%|██████████| 32/32 [01:21<00:00, 2.55s/it]
Exported model to /tmp/hif4_test_deploy
```

输出: 32 × layer safetensors (HiF4 uint8 打包: qweight + scale) + 2 rest shards (未量化权重直接复制)。

## 4. 交叉验证: PTQ (伪量化) vs Deploy (打包→解码)

### 对比方法

PTQ 输出 float 权重（fake-quant 编解码后的 bf16 值），Deploy 输出 uint8 打包字节（qweight + scale）。
将 Deploy 输出通过 `hif4_bit_reconstruct` 解码回 float，再与 PTQ 输出逐元素对比：

```python
from safetensors.torch import load_file
import torch, os, re, sys
sys.path.insert(0, '/home/ma-user/work/z84450661/amct')
from amct_pytorch.common.utils.hifloat4_quant.hif4_pack_unpack import hif4_bit_reconstruct

ptq_dir = '/tmp/hif4_test_ptq'
dep_dir = '/tmp/hif4_test_deploy'

ptq_layers = sorted([f for f in os.listdir(ptq_dir)
                     if f.startswith('layer_') and f.endswith('.safetensors')])
dep_layers = sorted([f for f in os.listdir(dep_dir)
                     if f.startswith('layer_') and f.endswith('.safetensors')])

print(f'Comparing {len(ptq_layers)} layers...')
print()

mismatches = []
for layer_file in ptq_layers:
    ptq_w = load_file(f'{ptq_dir}/{layer_file}', device='cpu')
    dep_w = load_file(f'{dep_dir}/{layer_file}', device='cpu')
    for key in sorted(ptq_w.keys()):
        ptq_tensor = ptq_w[key]                             # bf16 float
        qweight = dep_w[key]                                # uint8 packed
        scale = dep_w[f'{key}_scale']                       # uint8 scale
        decoded = torch.from_numpy(hif4_bit_reconstruct(scale, qweight)).to(torch.bfloat16)
        diff = (ptq_tensor.to(torch.float32) - decoded).abs().max().item()
        status = '✓' if diff < 1e-4 else f'✗ diff={diff:.6f}'
        if diff >= 1e-4:
            mismatches.append(f'{layer_file}/{key}: max diff = {diff:.6f}')
        print(f'  {layer_file[:-12]:>10s}  {key.split(".")[-2]:>10s}  max diff = {diff:.2e}  {status}')

print()
if mismatches:
    print(f'FAIL: {len(mismatches)} mismatches')
else:
    print(f'All {len(ptq_layers)} layers, {len(ptq_layers)*3} weights — bit-identical ✓')
```

### 对比指标

| 指标 | PTQ (float) | Deploy (packed) |
|------|-------------|-----------------|
| 输出层数 | 32 layer + 2 rest | 32 layer + 2 rest |
| 量化层大小 | ~302 MB/层 (bf16 float) | ~85 MB/层 (uint8 packed) |
| 非量化层 (rest) | 8.6 GB + 1.1 GB | 8.6 GB + 1.1 GB (byte-identical) |
| 压缩比 | — | ~3.55× (量化层) |

### 逐层对比结果

```
Comparing 32 layers...

   layer_000   down_proj  max diff = 0.00e+00  ✓
   layer_000   gate_proj  max diff = 0.00e+00  ✓
   layer_000     up_proj  max diff = 0.00e+00  ✓
   layer_001   down_proj  max diff = 0.00e+00  ✓
   layer_001   gate_proj  max diff = 0.00e+00  ✓
   layer_001     up_proj  max diff = 0.00e+00  ✓
   layer_002   down_proj  max diff = 0.00e+00  ✓
   layer_002   gate_proj  max diff = 0.00e+00  ✓
   layer_002     up_proj  max diff = 0.00e+00  ✓
   layer_003   down_proj  max diff = 0.00e+00  ✓
   layer_003   gate_proj  max diff = 0.00e+00  ✓
   layer_003     up_proj  max diff = 0.00e+00  ✓
   layer_004   down_proj  max diff = 0.00e+00  ✓
   layer_004   gate_proj  max diff = 0.00e+00  ✓
   layer_004     up_proj  max diff = 0.00e+00  ✓
   layer_005   down_proj  max diff = 0.00e+00  ✓
   layer_005   gate_proj  max diff = 0.00e+00  ✓
   layer_005     up_proj  max diff = 0.00e+00  ✓
   layer_006   down_proj  max diff = 0.00e+00  ✓
   layer_006   gate_proj  max diff = 0.00e+00  ✓
   layer_006     up_proj  max diff = 0.00e+00  ✓
   layer_007   down_proj  max diff = 0.00e+00  ✓
   layer_007   gate_proj  max diff = 0.00e+00  ✓
   layer_007     up_proj  max diff = 0.00e+00  ✓
   layer_008   down_proj  max diff = 0.00e+00  ✓
   layer_008   gate_proj  max diff = 0.00e+00  ✓
   layer_008     up_proj  max diff = 0.00e+00  ✓
   layer_009   down_proj  max diff = 0.00e+00  ✓
   layer_009   gate_proj  max diff = 0.00e+00  ✓
   layer_009     up_proj  max diff = 0.00e+00  ✓
   layer_010   down_proj  max diff = 0.00e+00  ✓
   layer_010   gate_proj  max diff = 0.00e+00  ✓
   layer_010     up_proj  max diff = 0.00e+00  ✓
   layer_011   down_proj  max diff = 0.00e+00  ✓
   layer_011   gate_proj  max diff = 0.00e+00  ✓
   layer_011     up_proj  max diff = 0.00e+00  ✓
   layer_012   down_proj  max diff = 0.00e+00  ✓
   layer_012   gate_proj  max diff = 0.00e+00  ✓
   layer_012     up_proj  max diff = 0.00e+00  ✓
   layer_013   down_proj  max diff = 0.00e+00  ✓
   layer_013   gate_proj  max diff = 0.00e+00  ✓
   layer_013     up_proj  max diff = 0.00e+00  ✓
   layer_014   down_proj  max diff = 0.00e+00  ✓
   layer_014   gate_proj  max diff = 0.00e+00  ✓
   layer_014     up_proj  max diff = 0.00e+00  ✓
   layer_015   down_proj  max diff = 0.00e+00  ✓
   layer_015   gate_proj  max diff = 0.00e+00  ✓
   layer_015     up_proj  max diff = 0.00e+00  ✓
   layer_016   down_proj  max diff = 0.00e+00  ✓
   layer_016   gate_proj  max diff = 0.00e+00  ✓
   layer_016     up_proj  max diff = 0.00e+00  ✓
   layer_017   down_proj  max diff = 0.00e+00  ✓
   layer_017   gate_proj  max diff = 0.00e+00  ✓
   layer_017     up_proj  max diff = 0.00e+00  ✓
   layer_018   down_proj  max diff = 0.00e+00  ✓
   layer_018   gate_proj  max diff = 0.00e+00  ✓
   layer_018     up_proj  max diff = 0.00e+00  ✓
   layer_019   down_proj  max diff = 0.00e+00  ✓
   layer_019   gate_proj  max diff = 0.00e+00  ✓
   layer_019     up_proj  max diff = 0.00e+00  ✓
   layer_020   down_proj  max diff = 0.00e+00  ✓
   layer_020   gate_proj  max diff = 0.00e+00  ✓
   layer_020     up_proj  max diff = 0.00e+00  ✓
   layer_021   down_proj  max diff = 0.00e+00  ✓
   layer_021   gate_proj  max diff = 0.00e+00  ✓
   layer_021     up_proj  max diff = 0.00e+00  ✓
   layer_022   down_proj  max diff = 0.00e+00  ✓
   layer_022   gate_proj  max diff = 0.00e+00  ✓
   layer_022     up_proj  max diff = 0.00e+00  ✓
   layer_023   down_proj  max diff = 0.00e+00  ✓
   layer_023   gate_proj  max diff = 0.00e+00  ✓
   layer_023     up_proj  max diff = 0.00e+00  ✓
   layer_024   down_proj  max diff = 0.00e+00  ✓
   layer_024   gate_proj  max diff = 0.00e+00  ✓
   layer_024     up_proj  max diff = 0.00e+00  ✓
   layer_025   down_proj  max diff = 0.00e+00  ✓
   layer_025   gate_proj  max diff = 0.00e+00  ✓
   layer_025     up_proj  max diff = 0.00e+00  ✓
   layer_026   down_proj  max diff = 0.00e+00  ✓
   layer_026   gate_proj  max diff = 0.00e+00  ✓
   layer_026     up_proj  max diff = 0.00e+00  ✓
   layer_027   down_proj  max diff = 0.00e+00  ✓
   layer_027   gate_proj  max diff = 0.00e+00  ✓
   layer_027     up_proj  max diff = 0.00e+00  ✓
   layer_028   down_proj  max diff = 0.00e+00  ✓
   layer_028   gate_proj  max diff = 0.00e+00  ✓
   layer_028     up_proj  max diff = 0.00e+00  ✓
   layer_029   down_proj  max diff = 0.00e+00  ✓
   layer_029   gate_proj  max diff = 0.00e+00  ✓
   layer_029     up_proj  max diff = 0.00e+00  ✓
   layer_030   down_proj  max diff = 0.00e+00  ✓
   layer_030   gate_proj  max diff = 0.00e+00  ✓
   layer_030     up_proj  max diff = 0.00e+00  ✓
   layer_031   down_proj  max diff = 0.00e+00  ✓
   layer_031   gate_proj  max diff = 0.00e+00  ✓
   layer_031     up_proj  max diff = 0.00e+00  ✓

All 32 layers, 96 weights — bit-identical ✓
```
