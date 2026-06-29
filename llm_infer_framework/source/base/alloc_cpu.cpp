#include "alloc.h"
#include <cstdlib>
#include <cstring>

// CPUDeviceAllocator 函数实现

namespace base {

std::shared_ptr<CPUDeviceAllocator> CPUDeviceAllocatorFactoty::instance = nullptr;

CPUDeviceAllocator::CPUDeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCPU) {}

void DeviceAllocator::memcpy(const void * src_ptr, void * dest_ptr,
        size_t size, MemcpyKind memcpy_kind, void * stream, bool sync) const {
    (void)stream;
    (void)sync;

    if (!src_ptr || !dest_ptr || size == 0) {
        return;
    }

    if (memcpy_kind == MemcpyKind::kmemcpyCPU2CPU) {
        std::memcpy(dest_ptr, src_ptr, size);
    }
}

void * CPUDeviceAllocator::allocate(size_t size) const {
    if(size == 0){
        return nullptr;
    }
    void * ptr = malloc(size);
    return ptr; // 内存泄露问题
}

void CPUDeviceAllocator::release(void * ptr) const{
    if(ptr){
        free(ptr);
    }
}

}
