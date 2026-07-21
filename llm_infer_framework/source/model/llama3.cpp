#include "model/llama3.h"

namespace model{

void LLama2Model::create_param_layers()
{
    // 创建 embedding layer
    llama_layers_ -> embedding_layer_ = std::make_shared<op::EmbeddingLayer>(device_type_,config_->dim_,
    config_->seq_len_,std::abs(config_->vocab_size_));

    const void * weight_embedding = raw_model_data_->weight(0);
    llama_layers_->embedding_layer_->set_weight(0,{std::abs(config_->vocab_size_),
                                                config_->dim_},weight_embedding,cpu_device_type);

    
    // 创建 matmul layer
    int32_t dim = config_->dim_;
    size_t pos = dim * std::abs(config_->vocab_size_) + dim * config_->layer_num_;
    
    // query matrix
    for(int32_t i = 0;i < config_->layer_num_;++i)
    {
        auto wq = std::make_shared<op::MatmulLayer>(device_type_,dim,dim);
        wq->set_weight(0,{dim,dim},this->raw_model_data_->weight(pos),cpu_device_type);
        llama_layers_->wq_layers_.push_back(wq);
        pos += dim * dim;
    }
                    
}

}