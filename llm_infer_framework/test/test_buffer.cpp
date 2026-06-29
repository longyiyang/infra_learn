#include "base/buffer.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <memory>

TEST(test_buffer,allocate){
    using namespace base;
    auto alloc = base::CPUDeviceAllocatorFactoty::get_instance(); // 单例模式
    Buffer buffer(32,alloc);
    EXPECT_NE(buffer.ptr(),nullptr);
}

TEST(test_buffer,use_external){
    using namespace base;
    float * ptr = new float[32];
    Buffer buffer(sizeof(float) * 32, nullptr, ptr, true);
    EXPECT_EQ(buffer.ptr(), ptr);
    delete[] ptr;
}

TEST(test_buffer, allocate_scope) {
  using namespace base;
  auto alloc = base::CPUDeviceAllocatorFactoty::get_instance();
  {
    Buffer buffer(32, alloc);
    ASSERT_NE(buffer.ptr(), nullptr);
    LOG(INFO) << "buffer allocated inside scope";
  }
  LOG(INFO) << "buffer released after scope";
}

TEST(test_buffer, allocate_shared) {
  using namespace base;
  auto alloc = base::CPUDeviceAllocatorFactoty::get_instance();
  std::shared_ptr<Buffer> buffer;
  { buffer = std::make_shared<Buffer>(32, alloc); }
  LOG(INFO) << "shared buffer allocated";
  ASSERT_NE(buffer->ptr(), nullptr);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();
  int result = RUN_ALL_TESTS();
  google::ShutdownGoogleLogging();
  return result;
}
