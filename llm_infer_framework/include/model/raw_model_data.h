#include <cstdint>

namespace model{
struct RawModelData{
    ~RawModelData();
    int32_t fd = -1;
    size_t file_size = 0;
    void * data = nullptr;
    void * weight_data = nullptr;

    virtual const void * weight(size_t offset) const = 0;
};

}