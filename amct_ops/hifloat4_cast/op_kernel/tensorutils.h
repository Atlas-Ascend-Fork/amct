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

// Lightweight Ascend C helpers used by the HiFloat4 cast kernel: global/UB tensor
// handles with manual UB-offset allocation, a double-buffer wrapper and a double
// event. Only the pieces the kernel needs are kept here.

#pragma once
#include "kernel_operator.h"

using namespace AscendC;

#define aifunc __aicore__ inline

typedef enum {
    PGM = 0,
    PL1 = 1,
    PL0A = 2,
    PL0B = 3,
    PL0C = 4,
    PUB = 5,
} pos_t;

/* ------------- Tensor ------------- */

template <typename T, pos_t pos>
class Tensor {};

template <typename T>
class Tensor<T, PGM> {
public:
    aifunc Tensor() {}
    aifunc Tensor(__gm__ uint8_t *ptr) { m_ptr = (__gm__ T *)ptr; }
    aifunc __gm__ T *ptr() { return m_ptr; }
    template <typename U>
    aifunc operator Tensor<U, PGM>() {
        return Tensor<U, PGM>((__gm__ uint8_t *)m_ptr);
    }
    aifunc __gm__ void *vptr() { return (__gm__ void *)m_ptr; }
    aifunc Tensor<T, PGM> operator[](int off) { return Tensor<T, PGM>((__gm__ uint8_t *)(m_ptr + off)); }

private:
    __gm__ T *m_ptr;
};

template <typename T>
class Tensor<T, PUB> {
public:
    aifunc Tensor() {}
    aifunc Tensor(uint64_t offset) { m_ptr = (__ubuf__ T *)offset; }
    aifunc Tensor(__ubuf__ uint8_t *ptr) { m_ptr = (__ubuf__ T *)ptr; }
    aifunc Tensor(__ubuf__ uint8_t *ptr, int size, int &offset) {
        m_ptr = (__ubuf__ T *)(ptr + offset);
        offset += size * sizeof(T);
    }
    aifunc __ubuf__ T *ptr() { return m_ptr; }
    aifunc __ubuf__ void *vptr() { return (__ubuf__ void *)m_ptr; }
    aifunc Tensor<T, PUB> operator[](int off) { return Tensor<T, PUB>((__ubuf__ uint8_t *)(m_ptr + off)); }
    template <typename U>
    aifunc operator Tensor<U, PUB>() {
        return Tensor<U, PUB>((__ubuf__ uint8_t *)m_ptr);
    }

private:
    __ubuf__ T *m_ptr;
};

/* ------------- Double Buffer ------------- */

template <typename T, pos_t pos>
class DBuff {};

template <typename T>
class DBuff<T, PUB> {
public:
    aifunc DBuff() {}
    aifunc DBuff(int base, int size, int &offset) {
        tsr1 = Tensor<T, PUB>(base + offset);
        tsr2 = Tensor<T, PUB>(base + offset + size * sizeof(T));
        offset += 2 * size * sizeof(T);
    }
    aifunc Tensor<T, PUB> get(int i) {
        if (i % 2 == 0) {
            return tsr1;
        } else {
            return tsr2;
        }
    }

private:
    Tensor<T, PUB> tsr1, tsr2;
};

/* ------------- Events ------------- */

template <pipe_t p1, pipe_t p2>
class DEvent {
public:
    aifunc DEvent() {}
    aifunc DEvent(int e_id1, int e_id2) {
        id1 = (event_t)e_id1;
        id2 = (event_t)e_id2;
    }
    aifunc void wait() {
        if (wait_cnt % 2 == 0) {
            wait_flag(p1, p2, id1);
        } else {
            wait_flag(p1, p2, id2);
        }
        wait_cnt++;
    }
    aifunc void set() {
        if (set_cnt % 2 == 0) {
            set_flag(p1, p2, id1);
        } else {
            set_flag(p1, p2, id2);
        }
        set_cnt++;
    }
    aifunc void setall() {
        set();
        set();
    }
    aifunc void release() {
        for (int i = wait_cnt; i < set_cnt; ++i) {
            wait();
        }
    }

private:
    event_t id1 = (event_t)0, id2 = (event_t)1;
    int wait_cnt = 0;
    int set_cnt = 0;
};
