#include <vector>
#include "base/buffer.h"

namespace tensor {
class Tensor{
public:
    explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1,
        bool need_alloc,std::shared_ptr<base::DeviceAllocator> alloc, void * ptr);
    
    void init_buffer(std::shared_ptr<base::DeviceAllocator> alloc,base::DataType data_type,
        bool need_alloc,void * ptr);

    bool Tensor::allocate(std::shared_ptr<base::DeviceAllocator> allocator,bool need_realloc = false);
    
    size_t size() const {return this->size_;}
    size_t byte_size() const {return this->size() * DataTypeSize(data_type_);}

    template<typename T> // 函数模板声明，定义也写在头文件里
    T * ptr();

    template<typename T>
    const T* ptr() const;
    
private:
    size_t size_ = 0;
    std::vector<int32_t> dims_;
    std::shared_ptr<base::Buffer> buffer_;
    base::DataType data_type_ = base::DataType::kDataTypeUnknown;
};

template<typename T>
T* Tensor::ptr(){
    if(!buffer_) return nullptr;
    return reinterpret_cast<T*>(buffer_->ptr());
}


}