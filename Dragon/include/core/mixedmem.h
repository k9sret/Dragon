// ------------------------------------------------------------
// Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
//
// Licensed under the BSD 2-Clause License.
// You should have received a copy of the BSD 2-Clause License
// along with the software. If not, See,
//
//      <https://opensource.org/licenses/BSD-2-Clause>
//
// ------------------------------------------------------------

#ifndef DRAGON_CORE_MIXEDMEM_H_
#define DRAGON_CORE_MIXEDMEM_H_

#include "core/context.h"
#include "core/context_cuda.h"
#include "core/context_cnml.h"

namespace dragon {

class MixedMemory {
 public:
    enum State {
        UNINITIALIZED,
        STATE_AT_CPU,
        STATE_AT_CUDA,
        STATE_AT_CNML,
        SWITCHED,
        SYNCED 
    };

    MixedMemory() : cpu_ptr_(nullptr),
          cuda_ptr_(nullptr), cnml_ptr_(nullptr) {}
    MixedMemory(const TypeMeta& meta, const size_t nbytes)
        : meta_(meta), nbytes_(nbytes), cpu_ptr_(nullptr),
          cuda_ptr_(nullptr), cnml_ptr_(nullptr) {}
    ~MixedMemory();

    const void* cpu_data();
    const void* cuda_data();
    const void* cnml_data();

    void* mutable_cpu_data();
    void* mutable_cuda_data();
    void* mutable_cnml_data();

    void* malloc_cnml_data();

    cnmlCpuTensor_t& cnml_cpu_tensor();
    cnmlTensor_t& cnml_mlu_tensor();

    void set_cpu_data(void* cpu_ptr, size_t nbytes);

    void SwitchToDevice();
    void SwitchToCUDADevice(int device_id);

    inline size_t nbytes() const { return nbytes_; }
    inline State state() const { return state_; }
    const Map<string, string> info() const;

    void ToCPU();
    void ToCUDA();

 private:
    void* cpu_ptr_, *cuda_ptr_, *cnml_ptr_;
    cnmlCpuTensor_t cnml_cpu_tensor_ = nullptr;
    cnmlTensor_t cnml_mlu_tensor_ = nullptr;
    int own_cpu_ptr_ = 1, ptr_device_ = 0;
    State state_ = UNINITIALIZED;
    size_t nbytes_ = 0;
    TypeMeta meta_;
};

}    // namespace dragon

#endif    // DRAGON_CORE_MIXEDMEM_H_