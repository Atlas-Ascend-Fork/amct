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

/*
 * HiFloat4 (hifx4) FP -> HiF4 -> FP fake-quant kernel — generated ccec port.
 *
 * The kernel bodies live in hifx_quant_op.h, auto-migrated from the EasyASC DSL
 * (kernels/a2/vec_only/hifx.py in the HiFloat4 reference tree) and validated
 * bit-exact against the numpy golden on Ascend A2/A3 hardware. This translation
 * unit only instantiates them so the ASC compiler emits the vector code and the
 * `extern "C"` host launchers (run_hifx_kernel / run_hifx_kernel_bf16) that the
 * CXX torch binding in op_extension/ links against.
 *
 * Only the 1D plain-hifx entries are used by the host:
 *   run_hifx_kernel(blockDim, stream, x, out, M, N, mant_bit=3)         fp32  (fp16 input
 *   is up-converted by the host)                                        path)
 *   run_hifx_kernel_bf16(blockDim, stream, x, out, M, N, mant_bit=3)    bf16
 * The header is trimmed to these two kernels only (no sr2 / lthr / psr variants).
 *
 * Contract: x, out are contiguous [M, N] with N % 64 == 0 (the host pads N to a
 * multiple of BATCH=512), kernel blocks are 40. mant_bit=3 -> S1P2 in-group codes.
 */

#include "hifx_quant_op.h"
