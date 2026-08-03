# HiFloat4 Cast — Test Results

Qwen3.5-9B (bf16, 32 层 dense), Ascend 910B NPU.

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

## 4. 交叉验证: PTQ (伪量化) vs Deploy (打包→解码) 逐比特一致

| 指标 | PTQ (float) | Deploy (packed) |
|------|-------------|-----------------|
| 输出层数 | 32 layer + 2 rest | 32 layer + 2 rest |
| 量化层大小 | ~302 MB/层 (bf16 float) | ~85 MB/层 (uint8 packed) |
| 非量化层 (rest) | 8.6 GB + 1.1 GB | 8.6 GB + 1.1 GB (byte-identical) |
| 压缩比 | — | ~3.55× (量化层) |

逐层对比 (PTQ float vs Deploy pack→unpack→float): **max diff = 0.00** — 逐比特一致，证明算法正确。

```
layer_000: down_proj  max diff = 0.00e+00 ✓
layer_000: gate_proj   max diff = 0.00e+00 ✓
layer_000: up_proj     max diff = 0.00e+00 ✓
layer_001: down_proj  max diff = 0.00e+00 ✓
... (32 层全部一致)
```
