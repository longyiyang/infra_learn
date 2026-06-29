#include "base/buffer.h"
#include <gtest/gtest.h>

TEST(test_buffer,allocate){
    using namespace base;
    auto alloc = base::CPUDeviceAllocatorFactoty::get_instance(); // 单例模式
    Buffer buffer(32,alloc);
    EXPECT_NE(buffer.ptr(),nullptr);
}
