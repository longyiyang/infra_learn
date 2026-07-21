#include "model/model.h"
#include "model/config.h"
namespace model{

base::Status Model::read_model_file(){
    using namespace base;
    if(model_path_.empty())
    {
        return error::PathNotValid("Fail to open the weight file, the model path is empty!");
    }

    int32_t fd = open(model_path_.data(),O_RDONLY);
    if(fd == -1)
    {
        return error::PathNotValid("Fail to open the weight file" + model_path_ +
                                    " may be the path does not exist!");
    }

    FILE * file = fopen(model_path_.data(),"rb");
    if(!file){
        return error::PathNotValid("文件打开失败...");
    }

    auto config = ModelConfig{};
    if(fread(&config,sizeof(ModelConfig),1,file) != 1){
        return error::ModelParseError("未能检索到模型配置信息...");
    }

    if(id_quant_model_){
        if(fread(&group_size_,sizeof(int32_t),1,file) != 1){
            return error::ModelParseError("未能检索到 group_size 信息...");
        }
    }

    auto gen_status = generate_model_infos(config){
        if(!gen_status)
        {
            return gen_status;
        }
    }
}

base::Status Model::generate_model_infos(const ModelConfig& config) const{
    config_->dim_ = config.dim;
    config_->hidden_dim_ = config.hidden_dim;
    config_->layer_num_ = config.layer_num;
    config_->head_num_ = config.head_num;
    config_->kv_head_num_ = config.kv_head_num;
    config_->seq_len_ = config.seq_len;
    
    // 用到了分组注意力 GQA
    // 多个 Query 头会共享同一个 K/V 头

    // K/V 的总维度 = 每个头的维度 × KV 头数量，但是为什么要计算 kv的总维度
    config_->kv_dim_ = (config.dim * config.kv_head_num) / config.head_num;

    // 每个 KV 头对应多少个 Query 头
    config_->kv_mul_ = config.head_num / config.kv_head_num;
    config_->head_size_ = config.dim / config.head_num;
#if defined(QWEN3_SUPPORT)
    config_->immediate_dim_ = config.immediate_dim_;
#endif
    if(config.vocab_size > 0){
        config_->is_shared_weight_ = true;
    }
    else{
        config_->is_shared_weight_ = false;
    }

    config_ -> vocab_size_ = std::abs(config.vocab_size);
    return base::error::Success();

}

}