#include "alloc.h"
#include <cuda_runtime_api.h>

namespace base{

// CUDADeviceAllocator 函数实现

// void * CUDADeviceAllocator::allocate(size_t size) const {
//     if(size == 0){
//         return nullptr;
//     }
//     void * ptr = nullptr;
//     cudaError_t err = cudaMalloc(&data, size);
//     CHECK_EQ(err, cudaSuccess);
//     return ptr;
// }

void * CUDADeviceAllocator::allocate(size_t byte_size) const{
    int id = -1;
    cudaError_t state = cudaGetDevice(&id);
    CHECK(state == cudaSuccess);

    // 大块显存情况下
    if(byte_size > 1024 * 1024){
        CudaMemoryBuffer big_buffers = big_buffers_map_[id]; // 申请的字节数大于1MB
        int sel_id = -1;
        for(int i = 0;i < big_buffers.size();i++){
            if(big_buffers[i].byte_size >= byte_size && !big_buffers[i].busy
                && big_buffers[i].byte_size - byte_size < 1024 * 1024 )
            {
                if(sel_id == -1 || big_buffers[sel_id].byte_size > big_buffers[i].byte_size)
                {
                    sel_id = i;
                }
            }
        }

        if(sel_id != -1)
        {
            big_buffers[sel_id].busy = true;
            return big_buffers[sel_id].data;
        }

        // 如果没找到，再调用cudaMalloc
        void * ptr = nullptr;
        state = cudaMalloc(&ptr,byte_size);
        if(cudaSuccess != state)
        {
            char buf[256];
            snprintf( buf,256,
                "Error...",
                byte_size >> 20;
            )
            LOG(ERROR) << buf;
            return nullptr;
        }
        big_buffers.emplace_back(ptr,byte_size,true);
        return ptr;
    }

    // 小块显存申请情况下
    // 同理用 cuda_bufers_map_
}

// void * CUDADeviceAllocator::release(void * ptr) const{
//     if(ptr){
//         cudaFree(ptr);
//     }
// }

void * CUDADeviceAllocator::release(void * ptr) const{
    
    for(auto & it : cuda_buffers_map_)
    {
        auto & cuda_buffers = it.second;
        for(int i = 0;i < cuda_buffers.size(); i++)
        {
            if(cuda_buffers[i].data == ptr)
            {
                no_busy_cnt_[it.first] += cuda_buffers[i].byte_size;
                cuda_buffers[i].busy = false;
                return;
            }
        }
    }

    // 上面是空闲块busy位改变，相关记录改变
    // 往下是某个条件下的释放
    for(auto & it: cuda_buffers_map_)
    {
        if(no_busy_cnt_[it.first] > 1024 * 1024 * 1024) // 如果空闲块大于1GB
        {
            auto & cuda_buffers = it.second;
            std::vector<CudaMemoryBuffer> temp;
            for(int i = 0; i < cuda_buffers.size(); i++)
            {
                if(!cuda_buffers[i].busy)
                {
                    state = cudaSetDevice(it.first);
                    state = cudaFree(cuda_buffers[i].data);
                    // check
                }
                else{
                    temp.push_back(cuda_buffers[i]);
                }
            }

            cuda_buffers.clear();
            it.second = temp;
            no_busy_cnt_[it.first] = 0;
        }
    }

    // 大块
    for(auto & it : big_buffers_map_)
    {
        auto & big_buffers = it.second;
        for(int i = 0;i < big_buffers.size(); i++)
        {
            if(big_buffers[i].data == ptr)
            {
                big_buffers[i].busy = false;
                return;
            }
        }

        state = cudaFree(ptr);
    }

}

}