#include "op/layer.h"
#include "tensor/tensor.h"
#include <glog/logging.h>

namespace op{

void Layer::set_input(int32_t idx, const tensor::Tensor & input){
    CHECK_GE(idx,0);
    CHECK_LT(idx,inputs_.size());
    if(!input.is_empty()){
        CHECK(input.device_type() == device_type_);
    }
    this->inputs_.at(idx) = input;
}

base::Status LayerParam::set_weight(int32_ idx,const std::vector<int32_t> &dims,
                                    const void * weight_ptr,base::DeviceType device_type)
{
    // 计算权重的数量（维度依次相乘）
    size_t size = std::accumulate(dims.begin(),dims.end(),sizeof(float),std::multiplies<>());
    // 权重指针赋值给一个buffer,管理内存资源
    std::shared_ptr<base::Buffer> buffer = std::make_shared<base::Buffer>(size,nullptr,(void*)(weight_ptr),true);
    if(device_type != base::DeviceType::kDeviceUnkonwn){
        buffer->set_device_type(device_type);
    }
    // 创建每个算子关联的权重
    tensor::Tensor weight(base::DataType::kDataTypeFp32,dims);
    weight.set_device_type(device_type);
    CHECK(weight.assign(buffer));
    weights_.at(idx) = weight;
    return base::error::Success();
}

}