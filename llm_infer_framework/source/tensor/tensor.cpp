#include "tensor/tensor.h"
#include <glog/logging.h>

namespace tensor{

Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1,
        bool need_alloc,std::shared_ptr<base::DeviceAllocator> alloc,
        void * ptr) : data_type_(data_type)
{
    dims_.push_back(dim0);
    dims_.push_back(dim1);
    size_ = dim0 * dim1;

    if(need_alloc && alloc){
        allocate(alloc);
    }
    else{
        init_buffer(alloc,data_type_,need_alloc,ptr);
    }
    
}

bool Tensor::allocate(std::shared_ptr<base::DeviceAllocator> allocator,bool need_realloc)
{
    if(!allocator){
        LOG(ERROR) << "The allocator parameter in the allocate function is null pointer!";
        return false;
    }
    
    size_t byte_size = this->byte_size();
    if(!byte_size)
    {
        LOG(ERROR) << "The byte_size parameter in allocate function is equal to zero!";
        return false;
    }

    if(buffer_ && byte_size <= buffer_->byte_size()){
        if(!need_realloc)
            return true;
    }

    buffer_ = std::make_shared<base::Buffer>(byte_size,allocator,nullptr);
    if(!buffer_ -> ptr()){
        LOG(ERROE) << "The memory allocated is a null pointer!";
        return false;
    }
    return ture;
}

void Tensor::init_buffer(std::shared_ptr<base::DeviceAllocator> alloc,base::DataType data_type,
        bool need_alloc,void * ptr)
{
    if(!alloc && !need_alloc){ //  用已有 ptr 指向的资源？
        std::make_shared<base::Buffer>(data_type)(size_ * base::DataTypeSize(data_type),nullptr,ptr,ture);
        this->buffer_ = buffer;
    }
    else{
        allocate(alloc,ture);
    }
}

}