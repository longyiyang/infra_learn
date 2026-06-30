#include "base/base.h"
#include "tensor/tensor.h"
#include <string>

namespace op{

enum class  LayerType : uint8_t{  // 算子基类
    kLayerUnkonwn = 0,
    kLayerLinear = 1,
    kLayerEncode = 2,
    kLayerEmbedding = 3,
    kLayerRMSNorm = 4,
    kLayerMatmul = 5,
    kLayerRoPe = 6,
    kLayerMHA = 7,
    kLayerSoftmax = 8,
    kLayerAdd = 9,
    kLayerSwiGLU = 10,
};

class Baselayer{
public:
    explicit Baselayer(base::DeviceType device_type, LayerType layer_type,
        base::DataType data_type, std::string layer_name =  "");

    base::DataType data_type() const;
    LayerType layer_type() const;

    const std::string & get_layer_name() const; 
    base::DeviceType device_type() const;

    void set_device_type(base::DeviceType device_type);

    // 输入输出相关
    virtual void set_input(int32_t idx,const tensor::Tensor& input) = 0;
    virtual void set_output(int32_t idx,const tensor::Tensor& input) = 0;

    virtual size_t input_size() const = 0;
    virtual size_t output_size() const = 0;

    virtual const tensor::Tensor & get_input(int32_t idx) const = 0;
    virtual const tensor::Tensor & get_output(int32_t idx) const = 0;

    virtual base::Status set_weight(int32_t idx,const tensor::Tensor& weight);
    virtual base::Status set_weight(int32_t idx,const std::vector<int32_t> & dims,
                                    const void* weight_ptr,
                                    base::DeviceType device_type = base::DeviceType::kDeviceUnkonwn);
    

    
                                
protected:
    std::string layer_name_;
    LayerType layer_type_ = LayerType::kLayerUnkonwn; // 层类型
    base::DataType  data_type_ = base::DataType::kDataTypeUnknown; // 层数据类型 
    base::DeviceType device_type_ = base::DeviceType::kDeviceUnkonwn;  
};

// 不带参数（权重）的算子类
class Layer :  public Baselayer{
public:
    explicit Layer(base::DeviceType device_type, LayerType layer_type,
        std::string layer_name =  "");
    
    void set_input(int32_t idx,const tensor::Tensor & input) override;
    void set_output(int32_t idx,const tensor::Tensor & input) override;
    
    const tensor::Tensor & get_input(int32_t idx) const override; // 常引用
    const tensor::Tensor & get_output(int32_t idx) const override;    
    size_t input_size() const override;
    size_t output_size() const override;

    void reset_input_size(size_t size);
    void reset_output_size(size_t size);

    virtual void to_cuda();

private:
    std::vector<tensor::Tensor> inputs_;
    std::vector<tensor::Tensor> outputs_;    
};


}

