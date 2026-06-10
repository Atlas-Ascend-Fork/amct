/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
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
    explicit SvdQuantTiling(gert::TilingContext* context) : context_(context){};
    ~SvdQuantTiling() = default;

    ge::graphStatus GetTiling();
private:
    SvdQuantTilingData tilingData;
    gert::TilingContext *context_ = nullptr;
    ge::graphStatus ReadShapes();
    bool GetFp4MMTiling(matmul_tiling::MultiCoreMatmulTiling&, int);
    bool GetDownProjectionTiling(matmul_tiling::MultiCoreMatmulTiling&, int);
    bool GetUpProjectionTiling(matmul_tiling::MultiCoreMatmulTiling&, int);
    bool CalcB16MatmulTiling(matmul_tiling::MultiCoreMatmulTiling& mmTiling, TCubeTiling& cubeTiling,
        int32_t M, int32_t N, int32_t K, int32_t coreNum);
    bool CalcMxMatmulTiling(matmul_tiling::MultiCoreMatmulTiling& mmTiling, TCubeTiling& cubeTiling,
        int32_t M, int32_t N, int32_t K, int32_t coreNum);
};

} // namespace optiling

#endif // SVD_QUANT_TILING_H_
