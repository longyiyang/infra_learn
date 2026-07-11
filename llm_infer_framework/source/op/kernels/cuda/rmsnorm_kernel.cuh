#include "tensor/tensor.h"
namespace kernel {
    
void row_rmsnorm_f32_v0(const float * in,const float * wei,float *out,const int size,const float eps);

void rmsnorm_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                    const tensor::Tensor& output, void* stream = nullptr);

}  