#include "op/add.h"
#include "kernels/kernels_interface.h"

namespace op {
    base::Status VecAddLayer::forward()
    {
        auto status = this->check();
        if(!status) return status;

        auto input1 = this->get_input(0);
        auto input2 = this->get_input(1);
        auto output = this->get_output(0);
        kernel::get_add_kernel(device_type_)(input1,input2,output,nullptr);
    }
}