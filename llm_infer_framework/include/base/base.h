#include <cstdint>
#include <string>

namespace base{

enum class DeviceType : uint8_t{
    kDeviceUnkonwn = 0,
    kDeviceCPU = 1,
    kDeviceGPU = 2,
};

enum class MemcpyKind {
    kmemcpyCPU2CPU = 0,
    kmemcpyCPU2CUDA = 1,
    kmemcpyCUDA2CPU = 2,
    kmemcpyCUDA2CUDA = 3,
};

enum class DataType : uint8_t{
    kDataTypeUnknown = 0,
    kDataTypeFp32 = 1,
    kDataTypeInt8 = 2,
    kDataTypeInt32 = 3,
};

inline size_t DataTypeSize(DataType data_type){ // 内联函数可写在头文件里，多个.cpp可以包含
    if(data_type == DataType::kDataTypeFp32) return sizeof(float);
    else if(data_type == DataType::kDataTypeInt8) return sizeof(int8_t);
    else if(data_type == DataType::kDataTypeInt8) return sizeof(int8_t);
    else if(data_type == DataType::kDataTypeInt32) return sizeof(int32_t);
    else return 0;
}

enum StatusCode : uint8_t{
    kSuccess = 0,
    kFunctionImplement = 1,
    kPathNotValid = 2,
    kModeParseError = 3,
    kInternalError = 5,
    kKeyValueHasExist = 6,
    kInvalidArgument = 7,
};

class Status{
private:
    int code_ = StatusCode::kSuccess;
    std::string message_;
public:
    Status(int code = StatusCode::kSuccess, std::string err_message = "");
    Status(const Status& other) = default; // 拷贝构造

    Status & operator=(const Status& other) = default; // 赋值构造，之前已定义 
    Status & operator=(int code); 

};

namespace error{

Status Success(const std::string & err_msg = "");
Status PathNotValid(const std::string& err_msg = "");
Status ModelParseError(const std::string& err_msg = "");
}

}