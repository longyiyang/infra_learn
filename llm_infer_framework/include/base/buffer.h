#pragma once

#include "alloc.h"
// #include <iostream>
#include <cstddef>
#include <memory>

namespace base {

class Buffer : public std::enable_shared_from_this<Buffer>
{
public:
    Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator,
           void * ptr = nullptr, bool use_external = false);
    ~Buffer();
    void * ptr() const { return ptr_; }
    bool allocate();
private:
    size_t byte_size_ = 0;
    void * ptr_ = nullptr;
    bool use_external_ = false;
    DeviceType device_type_ = DeviceType::kDeviceUnkonwn;
    std::shared_ptr<DeviceAllocator> allocator_;
    
};

}
