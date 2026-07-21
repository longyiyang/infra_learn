#include "layer.h"

namespace op{

class MatmulLayer: public LayerParam{
public:
    explicit MatmulLayer(base::DeviceType device_type,int32_t dim0,int32_t dim1,
                        bool is_quant_layer = false,bool has_bias = false);

    base::Status forward() override;

private:
    int32_t dim0_ = 0;
    int32_t dim1_ = 0;
    bool has_bias_ = false;
    std::vector<tensor::Tensor> bias_;

};

}