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

#include "ops_common.h"

const std::vector<std::string> g_custom_lib_path = get_custom_lib_path();
const std::vector<std::string> g_default_custom_lib_path = get_default_custom_lib_path();

void *GetOpApiFuncAddrFromFeatureLib(const char *api_name) {
    static auto ops_infer_handler = GetOpApiLibHandler("libaclnn_ops_infer.so");
    auto ops_infer_handler_func_addr = GetOpApiFuncFromFeatureLib(ops_infer_handler, "libaclnn_ops_infer.so", api_name);
    if IS_VALID_FUNC_ADDR (ops_infer_handler_func_addr)
        return ops_infer_handler_func_addr;

    static auto ops_train_handler = GetOpApiLibHandler("libaclnn_ops_train.so");
    auto ops_train_handler_func_addr = GetOpApiFuncFromFeatureLib(ops_train_handler, "libaclnn_ops_train.so", api_name);
    if IS_VALID_FUNC_ADDR (ops_train_handler_func_addr)
        return ops_train_handler_func_addr;

    static auto math_handler = GetOpApiLibHandler("libaclnn_math.so");
    auto math_handler_func_addr = GetOpApiFuncFromFeatureLib(math_handler, "libaclnn_math.so", api_name);
    if IS_VALID_FUNC_ADDR (math_handler_func_addr)
        return math_handler_func_addr;

    static auto sparse_handler = GetOpApiLibHandler("libaclnn_sparse.so");
    auto sparse_handler_func_addr = GetOpApiFuncFromFeatureLib(sparse_handler, "libaclnn_sparse.so", api_name);
    if IS_VALID_FUNC_ADDR (sparse_handler_func_addr)
        return sparse_handler_func_addr;

    static auto fft_handler = GetOpApiLibHandler("libaclnn_fft.so");
    auto fft_handler_func_addr = GetOpApiFuncFromFeatureLib(fft_handler, "libaclnn_fft.so", api_name);
    if IS_VALID_FUNC_ADDR (fft_handler_func_addr)
        return fft_handler_func_addr;

    static auto rand_handler = GetOpApiLibHandler("libaclnn_rand.so");
    auto rand_handler_func_addr = GetOpApiFuncFromFeatureLib(rand_handler, "libaclnn_rand.so", api_name);
    if IS_VALID_FUNC_ADDR (rand_handler_func_addr)
        return rand_handler_func_addr;

    return nullptr;
}

c10::SmallVector<int64_t, SIZE> array_to_small_vector(c10::IntArrayRef shape) {
    c10::SmallVector<int64_t, SIZE> shape_small_vec;
    for (uint64_t i = 0; i < shape.size(); i++) {
        shape_small_vec.emplace_back(shape[i]);
    }

    return shape_small_vec;
}
