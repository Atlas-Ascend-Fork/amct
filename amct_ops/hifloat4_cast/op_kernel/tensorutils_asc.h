#pragma once

#include "kernel_operator.h"

namespace hifloat4_asc_utils {
using namespace AscendC;

constexpr uint64_t VECTORFULLMASK[2] = {(uint64_t)-1, (uint64_t)-1};

__aicore__ constexpr HardEvent GetHardEventByPipe(pipe_t src, pipe_t dst) {
    if (src == PIPE_MTE2) {
        if (dst == PIPE_V) {
            return HardEvent::MTE2_V;
        } else if (dst == PIPE_MTE3) {
            return HardEvent::MTE2_MTE3;
        }
    } else if (src == PIPE_V) {
        if (dst == PIPE_MTE2) {
            return HardEvent::V_MTE2;
        } else if (dst == PIPE_MTE3) {
            return HardEvent::V_MTE3;
        }
    } else if (src == PIPE_MTE3) {
        if (dst == PIPE_V) {
            return HardEvent::MTE3_V;
        } else if (dst == PIPE_MTE2) {
            return HardEvent::MTE3_MTE2;
        }
    }
    return HardEvent::MTE3_MTE2;
}

__aicore__ inline int CeilDiv(int a, int b) {
    return b == 0 ? 0 : (a + b - 1) / b;
}

template <typename T1, typename T2>
__aicore__ inline T1 Min(T1 a, T2 b) {
    return a < b ? a : static_cast<T1>(b);
}

template <typename T1, typename T2>
__aicore__ inline T1 Max(T1 a, T2 b) {
    return a < b ? static_cast<T1>(b) : a;
}

__aicore__ inline void SetVectorMaskByCount(int count) {
    uint64_t mask_low;
    uint64_t mask_high;
    if (count <= 0) {
        mask_low = 0;
        mask_high = 0;
    } else if (count < 64) {
        mask_low = ((uint64_t)1 << count) - 1;
        mask_high = 0;
    } else if (count == 64) {
        mask_low = (uint64_t)-1;
        mask_high = 0;
    } else if (count < 128) {
        mask_low = (uint64_t)-1;
        mask_high = ((uint64_t)1 << (count - 64)) - 1;
    } else {
        mask_low = (uint64_t)-1;
        mask_high = (uint64_t)-1;
    }
    SetVectorMask<half, MaskMode::NORMAL>(mask_high, mask_low);
}

template <TPosition pos, typename T>
__aicore__ inline LocalTensor<T> AllocateLocalTensor(int len) {
    TBuf<pos> tbuf;
    TPipe* pipe_ptr = GetTPipePtr();
    pipe_ptr->InitBuffer(tbuf, len * sizeof(T));
    return tbuf.template Get<T>();
}

template <typename T, TPosition pos>
class DBuff {
public:
    __aicore__ inline DBuff() {}

    __aicore__ inline void Init(int len) {
        TPipe* pipe_ptr = GetTPipePtr();
        pipe_ptr->InitBuffer(buf1, len * sizeof(T));
        pipe_ptr->InitBuffer(buf2, len * sizeof(T));
        tsr1 = buf1.template Get<T>();
        tsr2 = buf2.template Get<T>();
    }

    __aicore__ inline LocalTensor<T> get(int i) {
        return (i % 2 == 0) ? tsr1 : tsr2;
    }

private:
    TBuf<pos> buf1;
    TBuf<pos> buf2;
    LocalTensor<T> tsr1;
    LocalTensor<T> tsr2;
};

template <pipe_t p1, pipe_t p2, bool preset>
class SEvent {
public:
    __aicore__ inline SEvent() {
        TPipe* pipe_ptr = GetTPipePtr();
        id1 = (event_t)pipe_ptr->AllocEventID<GetHardEventByPipe(p1, p2)>();
        if constexpr (preset) {
            setall();
        }
    }

    __aicore__ inline ~SEvent() {
        if constexpr (preset) {
            release();
        }
    }

    __aicore__ inline void wait() {
        wait_flag(p1, p2, id1);
    }

    __aicore__ inline void set() {
        set_flag(p1, p2, id1);
    }

    __aicore__ inline void setall() {
        set();
    }

    __aicore__ inline void release() {
        wait();
    }

private:
    event_t id1;
};

template <pipe_t p1, pipe_t p2, bool preset>
class DEvent {
public:
    __aicore__ inline DEvent() {
        TPipe* pipe_ptr = GetTPipePtr();
        id1 = (event_t)pipe_ptr->AllocEventID<GetHardEventByPipe(p1, p2)>();
        id2 = (event_t)pipe_ptr->AllocEventID<GetHardEventByPipe(p1, p2)>();
        if constexpr (preset) {
            setall();
        }
    }

    __aicore__ inline ~DEvent() {
        if constexpr (preset) {
            release();
        }
    }

    __aicore__ inline void wait() {
        if (wait_cnt % 2 == 0) {
            wait_flag(p1, p2, id1);
        } else {
            wait_flag(p1, p2, id2);
        }
        wait_cnt++;
    }

    __aicore__ inline void set() {
        if (set_cnt % 2 == 0) {
            set_flag(p1, p2, id1);
        } else {
            set_flag(p1, p2, id2);
        }
        set_cnt++;
    }

    __aicore__ inline void setall() {
        set();
        set();
    }

    __aicore__ inline void release() {
        for (int i = wait_cnt; i < set_cnt; ++i) {
            wait();
        }
    }

private:
    event_t id1;
    event_t id2;
    int wait_cnt = 0;
    int set_cnt = 0;
};

template <typename T>
__aicore__ inline void GM2UBPAD(LocalTensor<T> dst, GlobalTensor<T> src, int n_burst, int burst_len_byte, int src_stride_byte, int dst_stride) {
    DataCopyExtParams param;
    param.blockCount = n_burst;
    param.blockLen = burst_len_byte;
    param.srcStride = src_stride_byte;
    param.dstStride = dst_stride;

    DataCopyPadExtParams<T> pad_param;
    pad_param.isPad = false;
    pad_param.leftPadding = 0;
    pad_param.rightPadding = 0;
    DataCopyPad(dst, src, param, pad_param);
}

template <typename T>
__aicore__ inline void UB2GMPAD(GlobalTensor<T> dst, LocalTensor<T> src, int n_burst, int burst_len_byte, int src_stride, int dst_stride_byte) {
    DataCopyExtParams param;
    param.blockCount = n_burst;
    param.blockLen = burst_len_byte;
    param.srcStride = src_stride;
    param.dstStride = dst_stride_byte;
    DataCopyPad(dst, src, param);
}

template <typename T>
__aicore__ inline void UB2UB(LocalTensor<T> dst, LocalTensor<T> src, int n_burst, int burst_len, int src_stride, int dst_stride) {
    DataCopyParams param;
    param.blockCount = n_burst;
    param.blockLen = burst_len;
    param.srcStride = src_stride;
    param.dstStride = dst_stride;
    DataCopy(dst, src, param);
}
}  // namespace hifloat4_asc_utils
