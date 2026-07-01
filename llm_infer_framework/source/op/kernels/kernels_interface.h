#include "tensor/tensor.h"

namespace kernel {

// 定义 void * (~) 这个类型的函数指针为 AddKernel
typedef void (*AddKernel)(const tensor::Tensor& input1, const tensor::Tensor& input2,
                const tensor::Tensor& output,void * stream); 


AddKernel get_add_kernel(base::DeviceType device_type);


}

