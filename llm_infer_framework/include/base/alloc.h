#pragma once

#include "base.h"

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

// gpu 资源分配器
class CUDADeviceAllocator : public DeviceAllocator{
public:
    explicit CUDADeviceAllocator();

    void * allocate(size_t size) const;
    void release(void *ptr) const;
  
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
