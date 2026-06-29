#include <cstdint>

namespace base{

enum class DeviceType : uint8_t{
    kDeviceUnkonwn = 0,
    kDeviceCPU = 1,
    kDeviceGPU = 2,
};

enum class MemcpyKind {
    kmemcpyCPU2CPU = 0,
    kmemcpyCPU2CUDA = 1,
    kmemcpyCUDA2CPU = 2,
    kmemcpyCUDA2CUDA = 3,
};

}