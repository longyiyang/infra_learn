#include "layer.h"

namespace op{
class EmbeddingLayer : public LayerParam{
public:
    explicit EmbeddingLayer(base::DeviceType device_type,int32_t dim,int32_t seq_len,
                            int32_t vocab_size);
    base::Status check() const override;
    base::Status forward() override;

private:
    int32_t dim_ = 0;
    int32_t seq_len_ = 0;
    int32_t vocab_size_ = 0;
};

}