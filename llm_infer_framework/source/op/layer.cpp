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



}