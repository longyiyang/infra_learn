#include "alloc.h"
#include <cuda_runtime_api.h>

// 基类方法实现

void DeviceAllocator::memcpy(const void * src_ptr, void * dest_ptr,
    size_t size, MemcpyKind memcpy_kind, void * stream, bool need_sync) const {
        CHECK_NE(src_ptr,nullptr);
        CHECK_NE(dest_ptr,nullptr);

        if(!byte_size) return;

    cudaStream_t stream_ = nullptr;
    if(stream) stream_ = static_cast<CUstream_st*>(stream);

    if(memcpy_kind == MemcpyKind::kmemcpyCPU2CPU)
    {
        std::memcpy(dest_ptr, src_ptr, size);
    }  
    else if(memcpy_kind == MemcpyKind::kmemcpyCPU2CUDA)
    {
        if(!stream_)
        {
            cudaMemcpy(dest_ptr,src_ptr,size,cudaMemcpyHostToDevice);
        }
        else
        {
            cudaMemcpy(dest_ptr,src_ptr,size,cudaMemcpyHostToDevice,stream_);
        }
    }

    if(need_sync){
        cudaDeviceSynchronize(); // ?
    }
}