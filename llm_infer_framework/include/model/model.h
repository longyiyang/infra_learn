#include "base.h"
#include <memory>

namespace model{

class Model{
public:
private:
protected:
    virtual base::Status read_model_file();
    virtual base::Status generate_model_infos(const ModelConfig& config) const;

protected:
    int32_t group_size_ = 1; // 量化的组数？
    bool id_quant_model_ = false;
    std::string model_path_;

    // 固定生命周期：成员对象和宿主对象同生共死
    // 动态生命周期：对象可以晚（按需）创建、早销毁，存在时间由程序运行时决定
    std::unique_ptr<TransformerConfig> config_; // 为什么这里要用智能指针？
    std::shared_ptr<

};











}