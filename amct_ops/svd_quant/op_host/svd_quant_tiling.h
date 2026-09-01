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

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "tiling/matrix/matmul_tiling_base.h"
#include <iostream>

#ifndef SVD_QUANT_TILING_H_
#define SVD_QUANT_TILING_H_

namespace optiling {

BEGIN_TILING_DATA_DEF(SvdQuantTilingData)
TILING_DATA_FIELD_DEF(int32_t, batchSize);
TILING_DATA_FIELD_DEF(int32_t, M);
TILING_DATA_FIELD_DEF(int32_t, K);
TILING_DATA_FIELD_DEF(int32_t, N);
TILING_DATA_FIELD_DEF(int32_t, R);

TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, fp4MMTilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, downProjectionTilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, upProjectionTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SvdQuant, SvdQuantTilingData)

struct SvdQuantCompileInfo {};

class SvdQuantTiling {
public:
    explicit SvdQuantTiling(gert::TilingContext *context) : context_(context){};
    ~SvdQuantTiling() = default;

    ge::graphStatus GetTiling();

private:
    SvdQuantTilingData tilingData;
    gert::TilingContext *context_ = nullptr;
    ge::graphStatus ValidateShapes();
    ge::graphStatus ReadShapes();
    bool GetFp4MMTiling(matmul_tiling::MultiCoreMatmulTiling &, int);
    bool GetDownProjectionTiling(matmul_tiling::MultiCoreMatmulTiling &, int);
    bool GetUpProjectionTiling(matmul_tiling::MultiCoreMatmulTiling &, int);
    void CommonMatmulTiling(matmul_tiling::MultiCoreMatmulTiling &mmTiling, int32_t M, int32_t N, int32_t K,
        int32_t coreNum, int32_t &baseM, int32_t &baseN);
    bool CalcB16MatmulTiling(matmul_tiling::MultiCoreMatmulTiling &mmTiling, TCubeTiling &cubeTiling, int32_t M,
        int32_t N, int32_t K, int32_t coreNum);
    bool CalcMxMatmulTiling(matmul_tiling::MultiCoreMatmulTiling &mmTiling, TCubeTiling &cubeTiling, int32_t M,
        int32_t N, int32_t K, int32_t coreNum);
    void SetMatmulCubeTiling(
        TCubeTiling &cubeTiling, int32_t K, uint64_t l0cSize, int32_t baseMN, int32_t baseK, int32_t maxDepthVal);
};

} // namespace optiling

#endif // SVD_QUANT_TILING_H_
