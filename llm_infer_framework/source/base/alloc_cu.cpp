#include "alloc.h"
#include <cuda_runtime_api.h>

// CUDADeviceAllocator 函数实现

void * CUDADeviceAllocator::allocate(size_t size) const {
    if(size == 0){
        return nullptr;
    }
    void * ptr = nullptr;
    cudaError_t err = cudaMalloc(&data, size);
    CHECK_EQ(err, cudaSuccess);
    return ptr;
}

void * CUDADeviceAllocator::release(void * ptr) const{
    if(ptr){
        cudaFree(ptr);
    }
}