# MXFP4 Ascend-C Kernel Optimisation

## Objective

Optimise `csrc/mxfp4_kernel.cpp` so that the Ascend-C kernel **beats torch_npu on all shapes**, especially the large ones (≥256K elements).  The kernel implements MXFP4 quant-dequant: per-block FP4 E2M1 quantization with E8M0 power-of-two scales, block_size=32, scale_factor=6.0.

## Current baseline (Ascend 910B3, CANN 8.2)

```
         shape     elems   torch_npu     ascendc   speedup
         (1, 32)        32     0.714ms     0.039ms    18.43x  ✓
        (4, 128)       512     0.714ms     0.039ms    18.16x  ✓
      (16, 1024)     16384     0.712ms     0.039ms    18.12x  ✓
      (64, 4096)    262144     0.694ms     0.038ms    18.10x  ✓
     (256, 4096)   1048576     0.719ms     0.059ms    12.26x  ✓
    (1024, 4096)   4194304     0.720ms     0.219ms     3.28x  ✓
```

torch_npu ≈ 0.72 ms constant (ACLNN launch overhead dominates)。已完成的优化：Phase 1a BlockReduceMax；Phase 1b/1c 8x unroll；Phase 1c int32 位运算推导 scale*0.5。Phase 1b（标量 GetValue + 位操作）和 Phase 1c（Duplicate 循环）仍是 per-block 循环。

## Success criterion — Phase 3 目标（深度向量化）

| Shape | Target speedup vs torch_npu |
|---|---|
| (256, 4096) | ≥ 15x |
| (1024, 4096) | **≥ 5x** |

Correctness must stay: `max_err ≤ 1e-6` against the PyTorch reference.

## What you may change

**Only this file:**

```
csrc/mxfp4_kernel.cpp
```

Do NOT modify `pybind_wrapper.cpp`, `CMakeLists.txt`, `build.sh`, `mxfp4_npu.py`.  The kernel entry point signature must stay:

```cpp
extern "C" __global__ __aicore__ void mxfp4_quant_dequant(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
```

Tiling layout (int32 array on GM): `[totalLen, numCores, scaleBits, 0]`.

## Algorithm reference (PyTorch, see `../mxfp4.py`)

For each block of 32 float32 elements:

1. `max_abs = amax(abs(block))`
2. `raw_scale = clamp(max_abs / 6.0, min=2^-30)`
3. `scale = exp2(round(log2(raw_scale)))`       — E8M0 power-of-two
4. `y = block / scale`
5. Quantise `abs(y)` to FP4 E2M1 codebook `{0, 0.5, 1, 1.5, 2, 3, 4, 6}` using midpoint thresholds `{0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0}`
6. `output = sign(block) * q_abs * scale`

## 瓶颈分析（当前状态）

Phase 1a 已用 BlockReduceMax 融合（~5 次 API 调用代替 192 次）。Phase 0/2/3/4 都是 tile-wide 向量操作。**当前的两个瓶颈**：

### 瓶颈 A：Phase 1b — 192 次标量 GetValue + 位操作（4x unrolled 但仍是标量循环）
```cpp
for (b=0; b<192; b+=4) {
    float m0 = tmpBuf.GetValue(b*32);  // 标量读取
    // ... 4 组标量位操作算 invScale ...
}
```
每个 tile 仍有 48 次循环迭代 × 4 次 GetValue = 192 次标量 UB 读取。

### 瓶颈 B：Phase 1c — 192 次 Duplicate（4x unrolled 但仍是 per-block 循环）
```cpp
for (b=0; b<192; b+=4) {
    Duplicate(tmpBuf[b*32], invScaleArr[b], 32);     // 每次只填 32 个 float
    Duplicate(tmpBuf[(b+1)*32], invScaleArr[b+1], 32);
    ...
}
```
每个 tile 发起 192 次 Duplicate API 调用。

## 优化方向（按优先级）

### 优先级 1（已尝试，Ln/Exp 方案失败）：向量化 Phase 1b

**Ln/Exp 方案已验证不可行：** gather 208 个 maxAbs 到紧凑向量 → Muls/Maxs/Ln/Muls/Adds/Cast(TRUNC)/Cast/Muls/Exp 计算 invScale → 读回。结果：max_err=1.0（Exp 精度不足导致 E8M0 rounding 在 sqrt(2) 边界出错）。同时性能也回退（0.219→0.360ms）因为 gather/scatter + pipeline sync 开销抵消了向量化收益。

**正确方案需要整数位操作向量化**（不用 Ln/Exp），但 Ascend-C 缺少向量整数右移和 AND 操作。

### 优先级 2（必做）：消除 Phase 1c 的 Duplicate 循环

如果优先级 1 产出了紧凑的 `invScaleVec[192]`，广播到 `[192×32]` 可以用：
- `Duplicate` 的 RepeatParams 重载（一次 API 调用设 repeat=192, dstStride=32）
- 或 `DataCopy` 的 stride 模式
- 搜索 `Duplicate` 在 `kernel_operator.h` 中是否有 `(dst, src, repeat, dstBlkStride, srcBlkStride, dstRepStride, srcRepStride)` 形式的重载

### 优先级 3：增大 BLOCKS_PER_TILE

当前 BPT=192，tile = 6144 floats = 24KB。910B3 的 UB 有 256KB，当前总 buffer 占用约 (2×24KB input/output queue × 2 stage) + (3×24KB + 128B calcBuf) ≈ 170KB。还有约 80KB 余量，可以尝试 BPT=256 甚至 320，减少 tile 迭代次数。

### 优先级 4：减少 Phase 2 的 tile-wide 操作数

当前 20 次 tile-wide op。可能的方向：
- 把 3 个 step function (threshold 2.5/3.5/5.0) 也用 Cast-floor 技巧合并
- 如果 Ascend-C 支持 LUT (lookup table) 或 Gather 操作，直接查表

## Environment

```
Hardware:   Ascend 910B3
CANN:       8.2.RC1
Toolkit:    /usr/local/Ascend/ascend-toolkit/latest
Compiler:   ccec (clang-15 bisheng)
Python:     /user/lishangzhan/miniforge3/envs/triton-ascend/bin/python
torch:      2.6.0+cpu
torch_npu:  2.6.0.post4
```

## Ascend-C API quick reference

Key headers: `#include "kernel_operator.h"` → `using namespace AscendC;`

### Data types & buffers
```
TPipe, TQue<QuePosition::VECIN, depth>, TQue<QuePosition::VECOUT, depth>
TBuf<QuePosition::VECCALC>
GlobalTensor<T>, LocalTensor<T>
```

### Data movement
```
DataCopy(dst_local, src_gm, count)
DataCopy(dst_gm, src_local, count)
```

### Vector arithmetic (element-wise, count elements)
```
Abs(dst, src, count)
Add(dst, src0, src1, count)    Sub  Mul  Div
Adds(dst, src, scalar, count)  Muls  Mins  Maxs
Duplicate(dst, scalar, count)
```

### Reduction
```
ReduceMax(dst, src, workLocal, count)  // result in dst[0]
ReduceMin  ReduceSum
```

### Compare + Select
```
// Compare populates a bitmask in BinaryRepeatParams / uint64
Compare(dst, src0, src1, CMPMODE::GE, count)
Select(dst, sel, src_true, src_false, SELMODE::VSEL_TENSOR_TENSOR_MODE, count)
```

### Math
```
Exp(dst, src, count)   Ln(dst, src, count)
Reciprocal(dst, src, count)
```

### Scalar access (slow — avoid in hot path)
```
tensor.GetValue(idx)   tensor.SetValue(idx, val)
```

## Build & test

```bash
PY=/user/lishangzhan/miniforge3/envs/triton-ascend/bin/python

# rebuild (after editing csrc/mxfp4_kernel.cpp)
cd /user/lishangzhan/mxfp4_ascendc && bash build.sh

# test correctness + benchmark
$PY test_mxfp4.py
```

Build takes ~5 minutes.  If ccec reports errors, the message usually points to the exact Ascend-C API misuse.

## Git 规范

仓库位于 `/user/lishangzhan/mxfp4_ascendc`，local config 已设好 user.name/email。

- 每次优化迭代，无论成功或回退，都必须产出一个 git commit
- commit message 格式：`opt: <做了什么>` + 空行 + benchmark 表格
- 如果优化失败需要回退：`git checkout -- csrc/mxfp4_kernel.cpp` 然后 commit message 写 `revert: <原因>`
- 更新 `program.md` 的 baseline 表格后，一起包含在同一个 commit 中

示例 commit message：
```
opt: vectorise max-abs reduction with Abs() + ReduceMax()

         shape     elems   torch_npu     ascendc   speedup
      (64, 4096)    262144     0.71ms     0.35ms     2.03x
     (256, 4096)   1048576     0.72ms     0.50ms     1.44x
    (1024, 4096)   4194304     0.74ms     0.80ms     0.93x
```

## Iteration protocol

1. `git -C /user/lishangzhan/mxfp4_ascendc log --oneline -5` — 了解历史
2. Read `csrc/mxfp4_kernel.cpp` — 理解当前实现
3. Edit `csrc/mxfp4_kernel.cpp`
4. `cd /user/lishangzhan/mxfp4_ascendc && bash build.sh` — must exit 0
5. `$PY test_mxfp4.py` — all shapes PASS, check speedup column
6. 若性能提升 → 更新 `program.md` baseline 表格 → `git add -A && git commit`
7. 若编译失败或正确性不通过 → 修复重试，或 `git checkout -- csrc/` 回退 → `git commit` 说明原因
