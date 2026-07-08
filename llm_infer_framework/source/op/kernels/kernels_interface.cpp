#include "base/base.h"
#include "kernels_interface.h"
#include "cpu/add_kernel.h"
#include "cuda/add_kernel.cuh"
#include "cpu/rmsnorm_kernel.h"
#include "cuda/rmsnorm_kernel.cuh"

namespace kernel{

// AddKernel 类型是函数指针？
AddKernel get_add_kernel(base::DeviceType device_type)
{
    if(device_type == base::DeviceType::kDeviceCPU)
        return add_kernel_cpu;
    else if(device_type == base::DeviceType::kDeviceGPU)
        return add_kernel_cu;
    else{
        //LOG(FATAL) << "Unknown device type for get a kernel.";
        return nullptr;
    }
}

RMSNormKernel get_rmsnorm_kernel(base::DeviceType device_type)
{
    if(device_type == base::DeviceType::kDeviceCPU){
        return rmsnorm_kernel_cpu;
    }
    else if(device_type ==  base::DeviceType::kDeviceGPU){
        return row_rmsnorm_f32;
    }
    else{
        return nullptr;
    }
    
}

}