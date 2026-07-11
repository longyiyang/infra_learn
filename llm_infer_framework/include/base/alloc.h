#pragma once

#include "base.h"
#include <map>
#include <cstddef>
#include <memory>

namespace base {

class DeviceAllocator {
public:
    explicit DeviceAllocator(DeviceType device_type) : device_type_(device_type) {}

    virtual DeviceType device_type() const {return device_type_;}

    virtual void release(void *ptr) const = 0;
    virtual void * allocate(size_t size) const = 0;

    virtual void memcpy(const void * src_ptr, void * dest_ptr,
        size_t size, MemcpyKind memcpy_kind = MemcpyKind::kmemcpyCPU2CPU,
        void * stream = nullptr,
        bool sync = false) const;

private:
    DeviceType device_type_ = DeviceType::kDeviceUnkonwn;
};

// cpu 资源分配器
class CPUDeviceAllocator : public DeviceAllocator{
public:
    explicit CPUDeviceAllocator();

    void * allocate(size_t size) const;
    void release(void *ptr) const;
    void memcpy(const void * src_ptr, void * dest_ptr,size_t size) const;

};

struct CudaMemoryBuffer{
    void * data;
    size_t byte_size;
    bool busy;

    CudaMemoryBuffer() = default;
    CudaMemoryBuffer(void *data,size_t byte_size,bool busy)
                    : data(data),byte_size(byte_size),busy(busy)
                    {}
};

// gpu 资源分配器
class CUDADeviceAllocator : public DeviceAllocator{
public:
    explicit CUDADeviceAllocator();

    void * allocate(size_t size) const override;
    void release(void *ptr) const override;
private:
    mutable std::map<int,size_t> no_busy_cnt_;
    // 对应大块显存的管理,注意值是一个vector
    mutable std::map<int,std::vector<CudaMemoryBuffer>> big_buffers_map_;
    // 对应小块显存的管理
    mutable std::map<int,std::vector<CudaMemoryBuffer>> cuda_bufers_map_;
};

class CPUDeviceAllocatorFactoty{
public:
    static std::shared_ptr<CPUDeviceAllocator> get_instance(){
        if(instance == nullptr){
            instance = std::make_shared<CPUDeviceAllocator>();
        }
        return instance;
    }
private:
    static std::shared_ptr<CPUDeviceAllocator> instance;
};

}
