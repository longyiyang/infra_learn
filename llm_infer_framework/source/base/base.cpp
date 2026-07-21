#include "base/base.h"
#include <string>
namespace base{

Status::Status(int code,std::string err_msg)
    : code_(code),message_(std::move(err_msg)) {} // 为什么这里要用move?

namespace Error{
Status PathNotValid(const std::string & err_msg)
{
    return Status{kPathNotValid,err_msg};
}

Status ModelParseError(const std::string& err_msg)
{
    return Status{kModeParseError,err_msg};
}

} // namespace error


}