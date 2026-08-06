/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cmath>
#include "vf_quantize.h"

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#define SPLIT_CORE_CUBE
#define ASCENDC_CUBE_ONLY
#include "lib/matmul_intf.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace AscendC;

__aicore__ inline constexpr MatmulConfig GetCustomMDLCFG_DP() {
    auto mmCfg = CFG_MDL;
    mmCfg.enUnitFlag = true;
    return mmCfg;
}

__aicore__ inline constexpr MatmulConfig GetCustomMDLCFG_UP() {
    auto mmCfg = CFG_MDL;
    mmCfg.enUnitFlag = true;
    mmCfg.doMTE2Preload = 1;
    return mmCfg;
}

enum class MmStage : int32_t { DP = 0, UP = 1 };

using MmTilingType = decltype(SvdQuantTilingData::downProjectionTilingData);

template <typename T, MmStage Stage>
struct MmStageSpec;

// Down Projection
template <typename T>
struct MmStageSpec<T, MmStage::DP> {
    constexpr static MatmulConfig CUSTOM_CFG_MDL = GetCustomMDLCFG_DP();
    using MatmulT = MatmulImpl<MatmulType<TPosition::GM, CubeFormat::ND, T, false>,
        MatmulType<TPosition::GM, CubeFormat::ND, T, false>, MatmulType<TPosition::GM, CubeFormat::ND, T, false>,
        MatmulType<TPosition::GM, CubeFormat::ND, T>, CUSTOM_CFG_MDL>;

    __aicore__ static inline void GetDims(const SvdQuantTilingData &td, int32_t &m, int32_t &k, int32_t &n) {
        m = td.M;
        k = td.K;
        n = td.R;
    }

    __aicore__ static inline const MmTilingType &GetTiling(const SvdQuantTilingData &td) {
        return td.downProjectionTilingData;
    }

    __aicore__ static inline uint32_t GetGBufElemCount(const SvdQuantTilingData &td, int32_t k, int32_t n) {
        return td.batchSize * k * n;
    }

    __aicore__ static inline uint32_t GetBatchA(const SvdQuantTilingData &td) { return td.batchSize; }
    __aicore__ static inline uint32_t GetBatchB(const SvdQuantTilingData &td) { return td.batchSize; }
    __aicore__ static inline uint32_t isTransA() { return false; }
    __aicore__ static inline uint32_t isTransB() { return false; }

    template <typename ALocal, typename BLocal>
    __aicore__ static inline void BindInputs(MatmulT &mm, ALocal aLocal, BLocal bLocal) {
        mm.SetTensorA(aLocal);
        mm.SetTensorB(bLocal);
    }
};

// Up Projection
template <typename T>
struct MmStageSpec<T, MmStage::UP> {
    constexpr static MatmulConfig CUSTOM_CFG_MDL = GetCustomMDLCFG_UP();
    using MatmulT = MatmulImpl<MatmulType<TPosition::GM, CubeFormat::ND, T, false>,
        MatmulType<TPosition::GM, CubeFormat::ND, T, false>, MatmulType<TPosition::GM, CubeFormat::ND, T, false>,
        MatmulType<TPosition::GM, CubeFormat::ND, T>, CUSTOM_CFG_MDL>;

    __aicore__ static inline void GetDims(const SvdQuantTilingData &td, int32_t &m, int32_t &k, int32_t &n) {
        m = td.M;
        k = td.R;
        n = td.N;
    }

    __aicore__ static inline const MmTilingType &GetTiling(const SvdQuantTilingData &td) {
        return td.upProjectionTilingData;
    }

    __aicore__ static inline uint32_t GetGBufElemCount(const SvdQuantTilingData &td, int64_t k, int64_t n) {
        return td.batchSize * k * n;
    }

    __aicore__ static inline uint32_t isTransA() { return false; }
    __aicore__ static inline uint32_t isTransB() { return false; }
    __aicore__ static inline uint32_t GetBatchA(const SvdQuantTilingData &td) { return td.batchSize; }
    __aicore__ static inline uint32_t GetBatchB(const SvdQuantTilingData &td) { return td.batchSize; }

    template <typename ALocal, typename BLocal>
    __aicore__ static inline void BindInputs(MatmulT &mm, ALocal aLocal, BLocal bLocal) {
        mm.SetTensorA(aLocal);
        mm.SetTensorB(bLocal);
    }
};

template <typename T, MmStage Stage>
class SvdMatmulStageKernel {
public:
    using Spec = MmStageSpec<T, Stage>;
    using MatmulT = typename Spec::MatmulT;

public:
    __aicore__ inline SvdMatmulStageKernel() {}

    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, const SvdQuantTilingData &tilingData, TPipe *tpipe) {
        tiling_ = tilingData;
        pipe_ = tpipe;
        batchSize_ = tilingData.batchSize;

        Spec::GetDims(tilingData, m_, k_, n_);

        aGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(a), batchSize_ * m_ * k_);
        bGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(b), k_ * n_);
        cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(c), batchSize_ * m_ * n_);
    }

    __aicore__ inline void Process() {
        if ASCEND_IS_AIC {
            const auto &t = Spec::GetTiling(tiling_);

            if (GetBlockIdx() >= t.usedCoreNum) {
                return;
            }

            size_t offsetA = 0;
            size_t offsetB = 0;
            size_t offsetC = 0;
            bool isTransA = false;
            bool isTransB = false;

            size_t tailM = 0;
            size_t tailN = 0;
            CalcOffset(GetBlockIdx(), t, offsetA, offsetB, offsetC, tailM, tailN, Spec::isTransA(), Spec::isTransB());

            auto aLocal = aGlobal_[offsetA];
            auto bLocal = bGlobal_[offsetB];
            auto cLocal = cGlobal_[offsetC];

            mm_.SetSubBlockIdx(0);
            mm_.Init(&t, pipe_);
            mm_.SetSingleShape(tailM, tailN, t.Ka);

            for (int32_t batch = 0; batch < batchSize_; batch++) {
                Spec::BindInputs(mm_, aLocal[static_cast<size_t>(batch) * m_ * k_], bLocal);
                mm_.IterateAll(cLocal[static_cast<size_t>(batch) * m_ * n_]);
            }

            mm_.End();
        }
    }

private:
    __aicore__ inline void CalcOffset(int blockIdx, const TCubeTiling &tiling, size_t &offsetA, size_t &offsetB,
        size_t &offsetC, size_t &tailM, size_t &tailN, bool isTransA, bool isTransB) {
        size_t mSingleBlocks = tiling.M / tiling.singleCoreM;
        size_t nSingleBlocks = tiling.N / tiling.singleCoreN;
        size_t mCoreIndx = blockIdx % mSingleBlocks;
        size_t nCoreIndx = blockIdx / mSingleBlocks;

        offsetA = mCoreIndx * tiling.Ka * tiling.singleCoreM;
        if (isTransA) {
            offsetA = mCoreIndx * tiling.singleCoreM;
        }
        offsetB = nCoreIndx * tiling.singleCoreN;
        if (isTransB) {
            offsetB = nCoreIndx * tiling.Kb * tiling.singleCoreN;
        }
        offsetC = mCoreIndx * tiling.N * tiling.singleCoreM + nCoreIndx * tiling.singleCoreN;

        tailM =
            mCoreIndx == (mSingleBlocks - 1) ? tiling.M - (mSingleBlocks - 1) * tiling.singleCoreM : tiling.singleCoreM;
        tailN =
            nCoreIndx == (nSingleBlocks - 1) ? tiling.N - (nSingleBlocks - 1) * tiling.singleCoreN : tiling.singleCoreN;
    }

private:
    GlobalTensor<T> aGlobal_;
    GlobalTensor<T> bGlobal_;
    GlobalTensor<T> cGlobal_;

    int32_t m_ = 0;
    int32_t k_ = 0;
    int32_t n_ = 0;
    int32_t r_ = 0;
    int32_t batchSize_ = 1;

    SvdQuantTilingData tiling_;
    TPipe *pipe_ = nullptr;
    MatmulT mm_;
};

static constexpr uint32_t SCALE_CEIL_NUMBER = 64;
static constexpr uint32_t SCALE_NUMBER = 2;

class Fp4MatmulKernel {
public:
    __aicore__ inline Fp4MatmulKernel(){};

    using aType = MatmulTypeWithScale<TPosition::GM, TPosition::GM, CubeFormat::ND, fp4x2_e2m1_t, false>;
    using bType = MatmulTypeWithScale<TPosition::GM, TPosition::GM, CubeFormat::ND, fp4x2_e2m1_t, true>;
    using cType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
    constexpr static MatmulConfig CUSTOM_CFG_MDL = GetCustomMDLCFG_DP();

    AscendC::Matmul<aType, bType, cType, cType, CUSTOM_CFG_MDL, AscendC::MatmulCallBackFunc<nullptr, nullptr, nullptr>,
        AscendC::Impl::Detail::MatmulWithScalePolicy>
        matmulObj;

    __aicore__ inline void Init(
        GM_ADDR a, GM_ADDR b, GM_ADDR as, GM_ADDR bs, GM_ADDR c, const TCubeTiling &tiling, int32_t batchSize) {
        this->tiling = tiling;
        batchSize_ = batchSize;
        kM = tiling.M;
        kN = tiling.N;
        kK = tiling.Ka;
        kScaleK = (tiling.Ka + SCALE_CEIL_NUMBER - 1) / SCALE_CEIL_NUMBER * SCALE_NUMBER;
        kSingleM = tiling.singleCoreM;
        kSingleN = tiling.singleCoreN;
        kScaleNumber = SCALE_NUMBER;
        kCoreBlocksM = tiling.M / tiling.singleCoreM;
        kCoreNum = tiling.usedCoreNum;

        aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ fp4x2_e2m1_t *>(a), kM * kK);
        bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ fp4x2_e2m1_t *>(b), kK * kN);
        cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(c), kM * kN);
        asGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ fp8_e8m0_t *>(as), kM * kScaleK);
        bsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ fp8_e8m0_t *>(bs), kScaleK * kN);

        int32_t offsetA = 0;
        int32_t offsetB = 0;
        int32_t offsetC = 0;
        int32_t offsetAscale = 0;
        int32_t offsetBscale = 0;
        CalcOffset(GetBlockIdx(), offsetA, offsetB, offsetAscale, offsetBscale, offsetC);

        aGlobal = aGlobal[offsetA];
        bGlobal = bGlobal[offsetB];
        cGlobal = cGlobal[offsetC];
        asGlobal = asGlobal[offsetAscale];
        bsGlobal = bsGlobal[offsetBscale];

        if (GetSysWorkSpacePtr() == nullptr) {
            return;
        }
    }

    __aicore__ inline void Process(AscendC::TPipe *pipe) {
        if ASCEND_IS_AIC {
            if (GetBlockIdx() >= kCoreNum) {
                return;
            }

            matmulObj.Init(&this->tiling, pipe);
            int32_t mSingleBlocks = tiling.M / tiling.singleCoreM;
            int32_t nSingleBlocks = tiling.N / tiling.singleCoreN;
            int32_t tailM = mCoreIndex == (mSingleBlocks - 1) ? tiling.M - (mSingleBlocks - 1) * tiling.singleCoreM :
                                                                tiling.singleCoreM;
            int32_t tailN = nCoreIndex == (nSingleBlocks - 1) ? tiling.N - (nSingleBlocks - 1) * tiling.singleCoreN :
                                                                tiling.singleCoreN;

            matmulObj.SetSingleShape(tailM, tailN, kK);
            matmulObj.SetTensorB(bGlobal, true);
            matmulObj.SetTensorScaleB(bsGlobal, true);
            for (int32_t batch = 0; batch < batchSize_; batch++) {
                matmulObj.SetTensorA(aGlobal[batch * kM * kK], false);
                matmulObj.SetTensorScaleA(asGlobal[batch * kM * kScaleK], false);
                matmulObj.IterateAll(cGlobal[batch * kM * kN], 1);
            }
            matmulObj.End();
        }
    }

private:
    uint32_t kM;
    uint32_t kN;
    uint32_t kK;
    uint32_t kScaleK;
    uint32_t kSingleM;
    uint32_t kSingleN;
    uint32_t kScaleNumber;
    uint32_t kCoreBlocksM;
    uint32_t kCoreNum;
    int32_t batchSize_;

    __aicore__ inline void CalcOffset(int32_t blockIdx, int32_t &offsetA, int32_t &offsetB, int32_t &offsetAscale,
        int32_t &offsetBscale, int32_t &offsetC) {
        mCoreIndex = blockIdx % kCoreBlocksM;
        nCoreIndex = blockIdx / kCoreBlocksM;

        offsetA = mCoreIndex * kK * kSingleM;
        offsetB = nCoreIndex * kK * kSingleN;
        offsetAscale = mCoreIndex * kScaleK * kSingleM;
        offsetBscale = nCoreIndex * kSingleN * kScaleK;
        offsetC = mCoreIndex * kN * kSingleM + nCoreIndex * kSingleN;
    }

    GlobalTensor<fp4x2_e2m1_t> aGlobal;
    GlobalTensor<fp4x2_e2m1_t> bGlobal;
    GlobalTensor<bfloat16_t> cGlobal;
    AscendC::GlobalTensor<AscendC::fp8_e8m0_t> asGlobal;
    AscendC::GlobalTensor<AscendC::fp8_e8m0_t> bsGlobal;
    TCubeTiling tiling;
    int32_t mCoreIndex;
    int32_t nCoreIndex;
};

extern "C" __global__ __aicore__ void svd_quant(
    GM_ADDR A, GM_ADDR W, GM_ADDR SC, GM_ADDR DP, GM_ADDR UP, GM_ADDR O, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    GM_ADDR userWorkspace = GetUserWorkspace(workspace);
    AscendC::TPipe pipe;

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    int32_t inputSize = tilingData.batchSize * tilingData.M * tilingData.K;
    GM_ADDR quantOutputPtr = userWorkspace + tilingData.batchSize * tilingData.M * tilingData.R * sizeof(bfloat16_t);
    GM_ADDR scaleOutputPtr = quantOutputPtr + (inputSize / 2);

    // Quantization
    if ASCEND_IS_AIV {
        QuantizeFP4<bfloat16_t> quantOp;
        quantOp.Init(A, quantOutputPtr, scaleOutputPtr, static_cast<uint32_t>(inputSize), &pipe);
        quantOp.Process();
        pipe.Reset();
    }

    // Down Projection
    {
        SvdMatmulStageKernel<bfloat16_t, MmStage::DP> mmDP;
        mmDP.Init(A, DP, userWorkspace, tilingData, &pipe);
        mmDP.Process();
    }

    if ASCEND_IS_AIC {
        CrossCoreSetFlag<0x0, PIPE_FIX>(0x0);
        CrossCoreWaitFlag(0x0);
    }

    // Up Projection
    {
        SvdMatmulStageKernel<bfloat16_t, MmStage::UP> mmUP;
        mmUP.Init(userWorkspace, UP, O, tilingData, &pipe);
        mmUP.Process();
    }

    SyncAll<false>();

    // Fp4 Gate Projection
    {
        int32_t batchSize = tilingData.batchSize;
        Fp4MatmulKernel mmFP4;
        mmFP4.Init(quantOutputPtr, W, scaleOutputPtr, SC, O, tilingData.fp4MMTilingData, batchSize);
        mmFP4.Process(&pipe);
    }
}
