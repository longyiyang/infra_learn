#include "cpu/rmsnorm_kernel.h"

namespace  kernel {
void rmsnorm_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                    const tensor::Tensor& output, void* stream = nullptr)
{
    float * in_ptr = const_cast<float*>(input.ptr<float>());
    float * out_ptr = const_cast<float*>(output.ptr<float>());

    int size = static_cast<int32_t>(input.size());
    float sum = 0.f;

    for(int i = 0;i < size;i++){
        float input_value = input.index<float>(i);
        sum += input_value * input_value;
    }

    const float eps = 1e-5f;
    float mean = sum / float(size) + eps;

    const float rsqrt = 1.f / std::sqrt(mean);
    for(int i = 0;i < size;i++){
        *(out_ptr + i) = weight.index<float>(i) * (rsqrt * (*(in_ptr + i)));
    }

}


}