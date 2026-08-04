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
#include <utility>
#include "svd_quant_tiling.h"

using namespace ge;

#define RETURN_IF_FAILED(X)        \
    if ((X) == ge::GRAPH_FAILED) { \
        return ge::GRAPH_FAILED;   \
    }
#define RETURN_ON_ERROR(X, MESSAGE)        \
    if (X) {                               \
        std::cout << MESSAGE << std::endl; \
        return ge::GRAPH_FAILED;           \
    }
#define ALIGN_UP(x, y) (((x) + (y)-1) / (y) * (y))
#define CEIL_DIV(x, y) (((x) + (y)-1) / (y))

namespace svd_quant {
constexpr uint64_t INPUT_A_IDX = 0;
constexpr uint64_t INPUT_W_IDX = 1;
constexpr uint64_t INPUT_S_IDX = 2;
constexpr uint64_t INPUT_DP_IDX = 3;
constexpr uint64_t INPUT_UP_IDX = 4;
} // namespace svd_quant

namespace optiling {

static constexpr uint32_t SCALE_CEIL_NUMBER = 64;
static constexpr uint32_t SCALE_NUMBER = 2;

static constexpr int32_t MAX_BASE_MN = 256;
static constexpr int32_t MIN_BASE_MN = 16;

// mxTypePara bit layout (MatmulApiStaticTiling/TCubeTiling):
// [0:6]   scaleFactorKa
// [8:14]  scaleFactorKb
// [16:22] scaleFactorM
// [24:30] scaleFactorN
struct ScaleFactors {
    uint32_t ka;
    uint32_t kb;
    uint32_t m;
    uint32_t n;
};
uint32_t BuildMxTypePara(ScaleFactors &scale) {
    return ((scale.ka & 0x7FU) << 0) | ((scale.kb & 0x7FU) << 8) | ((scale.m & 0x7FU) << 16) |
           ((scale.n & 0x7FU) << 24);
}

ge::graphStatus SvdQuantTiling::ValidateShapes() {
    RETURN_ON_ERROR(context_ == nullptr, "SvdQuantTiling::ValidateShapes: context_ is null");

    auto inputDescAPtr = context_->GetInputDesc(svd_quant::INPUT_A_IDX);
    auto inputDescWPtr = context_->GetInputDesc(svd_quant::INPUT_W_IDX);
    auto inputDescSPtr = context_->GetInputDesc(svd_quant::INPUT_S_IDX);
    auto inputDescDPPtr = context_->GetInputDesc(svd_quant::INPUT_DP_IDX);
    auto inputDescUPPtr = context_->GetInputDesc(svd_quant::INPUT_UP_IDX);
    RETURN_ON_ERROR(inputDescAPtr == nullptr || inputDescWPtr == nullptr || inputDescSPtr == nullptr ||
                        inputDescDPPtr == nullptr || inputDescUPPtr == nullptr,
        "SvdQuantTiling::ValidateShapes: input desc are null");

    auto dtypeA = inputDescAPtr->GetDataType();
    auto dtypeW = inputDescWPtr->GetDataType();
    auto dtypeS = inputDescSPtr->GetDataType();
    auto dtypeDP = inputDescDPPtr->GetDataType();
    auto dtypeUP = inputDescUPPtr->GetDataType();
    RETURN_ON_ERROR(dtypeA != ge::DT_BF16 || dtypeW != ge::DT_UINT8 || dtypeS != ge::DT_UINT8 ||
                        dtypeDP != ge::DT_BF16 || dtypeUP != ge::DT_BF16,
        "SvdQuantTiling::ValidateShapes: Data Type is wrong");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SvdQuantTiling::ReadShapes() {
    auto inputShapeActivation = context_->GetInputShape(svd_quant::INPUT_A_IDX);
    auto inputShapeW = context_->GetInputShape(svd_quant::INPUT_W_IDX);
    auto inputShapeS = context_->GetInputShape(svd_quant::INPUT_S_IDX);
    auto inputShapeDP = context_->GetInputShape(svd_quant::INPUT_DP_IDX);
    auto inputShapeUP = context_->GetInputShape(svd_quant::INPUT_UP_IDX);
    RETURN_ON_ERROR(inputShapeActivation == nullptr || inputShapeW == nullptr || inputShapeS == nullptr ||
                        inputShapeDP == nullptr || inputShapeUP == nullptr,
        "SvdQuantTiling::ReadShapes: input shapes are null");

    int32_t dimIdxA = static_cast<int32_t>(inputShapeActivation->GetOriginShape().GetDimNum());
    int32_t dimIdxW = static_cast<int32_t>(inputShapeW->GetOriginShape().GetDimNum());
    int32_t dimIdxS = static_cast<int32_t>(inputShapeS->GetOriginShape().GetDimNum());
    int32_t dimIdxDP = static_cast<int32_t>(inputShapeDP->GetOriginShape().GetDimNum());
    int32_t dimIdxUP = static_cast<int32_t>(inputShapeUP->GetOriginShape().GetDimNum());
    RETURN_ON_ERROR(dimIdxA < 2 || dimIdxW != 2 || dimIdxS != 3 || dimIdxDP != 2 || dimIdxUP != 2,
        "SvdQuantTiling::ReadShapes: invalid input shapes");

    const int32_t K = static_cast<int32_t>(inputShapeActivation->GetStorageShape().GetDim(--dimIdxA));
    RETURN_ON_ERROR(K % 32, "SvdQuantTiling::ReadShapes: K dimension should be a multiple of 32");

    int32_t M = static_cast<int32_t>(inputShapeActivation->GetStorageShape().GetDim(--dimIdxA));
    int32_t batchSize = 1;
    while (dimIdxA > 0) {
        int32_t dim = static_cast<int32_t>(inputShapeActivation->GetStorageShape().GetDim(--dimIdxA));
        batchSize *= (dim > 0 ? dim : 1);
    }

    const int32_t N = static_cast<int32_t>(inputShapeUP->GetStorageShape().GetDim(1));
    const int32_t R = static_cast<int32_t>(inputShapeDP->GetStorageShape().GetDim(1));
    int32_t ScaleK = (K + SCALE_CEIL_NUMBER - 1) / SCALE_CEIL_NUMBER;
    RETURN_ON_ERROR(static_cast<int32_t>(inputShapeUP->GetStorageShape().GetDim(0)) != R ||
                        static_cast<int32_t>(inputShapeDP->GetStorageShape().GetDim(0)) != K ||
                        static_cast<int32_t>(inputShapeW->GetStorageShape().GetDim(1)) != K / 2 ||
                        static_cast<int32_t>(inputShapeW->GetStorageShape().GetDim(0)) != N ||
                        static_cast<int32_t>(inputShapeS->GetStorageShape().GetDim(2)) != SCALE_NUMBER ||
                        static_cast<int32_t>(inputShapeS->GetStorageShape().GetDim(1)) != ScaleK ||
                        static_cast<int32_t>(inputShapeS->GetStorageShape().GetDim(0)) != N,
        "SvdQuantTiling::ReadShapes: Input shapes are incompatible");

    if (M == 1) {
        // Decode mode optimization
        M *= batchSize;
        batchSize = 1;
    }
    tilingData.set_batchSize(batchSize);
    tilingData.set_M(M);
    tilingData.set_K(K);
    tilingData.set_N(N);
    tilingData.set_R(R);
    return ge::GRAPH_SUCCESS;
}

bool SvdQuantTiling::CalcB16MatmulTiling(matmul_tiling::MultiCoreMatmulTiling &mmTiling, TCubeTiling &cubeTiling,
    int32_t M, int32_t N, int32_t K, int32_t coreNum) {
    int32_t baseM, baseN, baseK = 64 < K ? 64 : ALIGN_UP(K, 16);

    CommonMatmulTiling(mmTiling, M, N, K, coreNum, baseM, baseN);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    uint64_t l1Size, l0cSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0cSize);

    int32_t l1TotalSize = static_cast<int32_t>(l1Size / 2) / sizeof(int16_t);
    int32_t l0BlockSize = std::max(baseK * baseM, baseK * baseN);
    l0BlockSize = l0BlockSize == 0 ? 1 : l0BlockSize;
    int32_t maxDepthVal = l1TotalSize / l0BlockSize;

    if (mmTiling.GetTiling(cubeTiling) == -1) {
        std::cout << "The BF16 Matmul tiling data is None" << std::endl;
        return false;
    }
    SetMatmulCubeTiling(cubeTiling, K, l0cSize, baseM * baseN, baseK, maxDepthVal);
    return true;
}

void SvdQuantTiling::SetMatmulCubeTiling(
    TCubeTiling &cubeTiling, int32_t K, uint64_t l0cSize, int32_t baseMN, int32_t baseK, int32_t maxDepthVal) {
    int32_t depthVal = 1;
    while (depthVal * 2 <= maxDepthVal) {
        depthVal *= 2;
    }
    int32_t dbFactor = 2;
    int32_t stepK = depthVal / dbFactor;
    baseK = baseK == 0 ? 16 : baseK;
    while ((stepK > CEIL_DIV(K, baseK))) {
        stepK /= 2;
    }
    int32_t stepMN = (depthVal / dbFactor) / stepK;

    int32_t dbL0C = 1;
    if ((l0cSize / (baseMN * sizeof(float))) >= 2) {
        dbL0C = 2;
    }

    cubeTiling.set_baseK(baseK);
    cubeTiling.set_stepM(stepMN);
    cubeTiling.set_stepN(stepMN);
    cubeTiling.set_stepKa(stepK);
    cubeTiling.set_stepKb(stepK);
    cubeTiling.set_depthA1(depthVal);
    cubeTiling.set_depthB1(depthVal);
    cubeTiling.set_dbL0A(2);
    cubeTiling.set_dbL0B(2);
    cubeTiling.set_dbL0C(dbL0C);
}

void SvdQuantTiling::CommonMatmulTiling(matmul_tiling::MultiCoreMatmulTiling &mmTiling, int32_t M, int32_t N, int32_t K,
    int32_t coreNum, int32_t &baseM, int32_t &baseN) {
    baseM = std::min(std::max(ALIGN_UP(M, 16), MIN_BASE_MN), MAX_BASE_MN);
    baseN = std::min(std::max(ALIGN_UP(N, 16), MIN_BASE_MN), MAX_BASE_MN);
    baseM = baseM == 0 ? MIN_BASE_MN : baseM;
    int32_t mLoops = std::min(std::max(M / baseM, 1), coreNum);
    mLoops = mLoops == 0 ? 1 : mLoops;
    while (coreNum % mLoops != 0 && (mLoops != 1)) {
        mLoops--;
    }
    int32_t nLoops = coreNum / mLoops;
    while ((N / nLoops < baseN) && (nLoops != 1)) {
        nLoops--;
    }

    nLoops = nLoops == 0 ? 1 : nLoops;
    int32_t coreNumVal = mLoops * nLoops;
    int32_t singleM = M / mLoops;
    int32_t singleN = N / nLoops;

    mmTiling.SetDim(coreNumVal);
    mmTiling.SetSingleShape(singleM, singleN, K);
    mmTiling.SetFixSplit(baseM, baseN);
    mmTiling.SetTraverse(matmul_tiling::MatrixTraverse::FIRSTM);
}

bool SvdQuantTiling::CalcMxMatmulTiling(matmul_tiling::MultiCoreMatmulTiling &mmTiling, TCubeTiling &cubeTiling,
    int32_t M, int32_t N, int32_t K, int32_t coreNum) {
    int32_t baseM, baseN, baseK = 256 < K ? 256 : ALIGN_UP(K, 64);

    CommonMatmulTiling(mmTiling, M, N, K, coreNum, baseM, baseN);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    uint64_t l1Size, l0cSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0cSize);
    int32_t sizeOfFp4Factor = 2;
    float scaleFactor = 1.25;
    int32_t l1TotalSize = static_cast<int32_t>(l1Size) * sizeOfFp4Factor;
    int32_t l0BlockSize = std::max(baseK * baseM, baseK * baseN);
    float fpDepthVal = static_cast<float>(l1TotalSize / 2) / (static_cast<float>(l0BlockSize) * scaleFactor);
    int32_t maxDepthVal = static_cast<int32_t>(fpDepthVal);

    if (mmTiling.GetTiling(cubeTiling) == -1) {
        std::cout << "The Fp4 Matmul tiling data is None" << std::endl;
        return false;
    }
    SetMatmulCubeTiling(cubeTiling, K, l0cSize, baseM * baseN, baseK, maxDepthVal);
    ScaleFactors scaleCase = {4U, 4U, 1U, 1U};
    cubeTiling.set_mxTypePara(BuildMxTypePara(scaleCase));
    return true;
}

bool SvdQuantTiling::GetFp4MMTiling(matmul_tiling::MultiCoreMatmulTiling &fp4MMTiling, int coreNum) {
    int32_t M = tilingData.get_M();
    int32_t N = tilingData.get_N();
    int32_t K = tilingData.get_K();
    fp4MMTiling.SetAType(
        matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT4_E2M1, false);
    fp4MMTiling.SetBType(
        matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT4_E2M1, true);
    fp4MMTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BF16);
    fp4MMTiling.SetBias(false);
    fp4MMTiling.SetScaleAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, false);
    fp4MMTiling.SetScaleBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, true);
    fp4MMTiling.SetShape(M, N, K);
    fp4MMTiling.SetOrgShape(M, N, K);
    fp4MMTiling.SetBufferSpace(-1, -1, -1);
    fp4MMTiling.SetMadType(matmul_tiling::MatrixMadType::MXMODE);
    return CalcMxMatmulTiling(fp4MMTiling, tilingData.fp4MMTilingData, M, N, K, coreNum);
}

bool SvdQuantTiling::GetDownProjectionTiling(matmul_tiling::MultiCoreMatmulTiling &downProjectionTiling, int coreNum) {
    int32_t M = tilingData.get_M();
    int32_t N = tilingData.get_R();
    int32_t K = tilingData.get_K();
    downProjectionTiling.SetAType(
        matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BF16, false);
    downProjectionTiling.SetBType(
        matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BF16, false);
    downProjectionTiling.SetCType(
        matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BF16);
    downProjectionTiling.SetBias(false);
    downProjectionTiling.SetShape(M, N, K);
    downProjectionTiling.SetOrgShape(M, N, K);
    downProjectionTiling.SetBufferSpace(-1, -1, 0, -1);
    return CalcB16MatmulTiling(downProjectionTiling, tilingData.downProjectionTilingData, M, N, K, coreNum);
}

bool SvdQuantTiling::GetUpProjectionTiling(matmul_tiling::MultiCoreMatmulTiling &upProjectionTiling, int coreNum) {
    int32_t M = tilingData.get_M();
    int32_t N = tilingData.get_N();
    int32_t K = tilingData.get_R();
    upProjectionTiling.SetAType(
        matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BF16, false);
    upProjectionTiling.SetBType(
        matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BF16, false);
    upProjectionTiling.SetCType(
        matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BF16);
    upProjectionTiling.SetBias(false);
    upProjectionTiling.SetShape(M, N, K);
    upProjectionTiling.SetOrgShape(M, N, K);
    upProjectionTiling.SetBufferSpace(-1, -1, 0, -1);
    return CalcB16MatmulTiling(upProjectionTiling, tilingData.upProjectionTilingData, M, N, K, coreNum);
}

ge::graphStatus SvdQuantTiling::GetTiling() {
    RETURN_ON_ERROR(context_ == nullptr, "SvdQuantTiling::RunBigKernelTiling: context_ is null");

    RETURN_IF_FAILED(ValidateShapes());
    RETURN_IF_FAILED(ReadShapes());
    auto platformInfo = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    int32_t coreNum = platformInfo.GetCoreNumAic();
    matmul_tiling::MultiCoreMatmulTiling fp4MMTiling(platformInfo), downProjectionTiling(platformInfo),
        upProjectionTiling(platformInfo);
    RETURN_ON_ERROR(!GetFp4MMTiling(fp4MMTiling, coreNum), "The MxFp4 Matmul tiling data is None");

    RETURN_ON_ERROR(
        !GetDownProjectionTiling(downProjectionTiling, coreNum), "The Down Projection Matmul tiling data is None");

    RETURN_ON_ERROR(
        !GetUpProjectionTiling(upProjectionTiling, coreNum), "The Up Projection Matmul tiling data is None");

    context_->SetBlockDim(coreNum);
    context_->SetTilingKey(0);

    int32_t M = tilingData.get_M();
    int32_t K = tilingData.get_K();
    int32_t R = tilingData.get_R();
    int32_t batchSize = tilingData.get_batchSize();

    const size_t downProjectionWorkspaceElem = batchSize * M * R * sizeof(float) / 2;
    const size_t activationQuantWorkspaceElem =
        batchSize * (M * K * sizeof(float) / 8 + ((M * K) / 32) * sizeof(float) / 4);

    const size_t userWorkspaceSize = downProjectionWorkspaceElem + activationQuantWorkspaceElem;
    size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
    RETURN_ON_ERROR(currentWorkspace == nullptr, "SvdQuantTiling: workspace pointer is null");

    currentWorkspace[0] = userWorkspaceSize + static_cast<size_t>(platformInfo.GetLibApiWorkSpaceSize());

    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingSvdQuant(gert::TilingContext *context_) {
    SvdQuantTiling tiling(context_);
    return tiling.GetTiling();
}

ge::graphStatus TilingPrepareForSvdQuant(gert::TilingParseContext * /* context_ */) {
    return ge::GRAPH_SUCCESS;
}

// --------------------------Registering the Tiling and TilingPrepare Functions--------

IMPL_OP(SvdQuant).Tiling(TilingSvdQuant).TilingParse<SvdQuantCompileInfo>(TilingPrepareForSvdQuant);

} // namespace optiling
