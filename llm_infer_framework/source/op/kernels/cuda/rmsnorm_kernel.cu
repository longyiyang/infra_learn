
#include <cub/block/block_reduce.cuh>
#include "cuda/rmsnorm_kernel.cuh"

#define WARP_SIZE 32

template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum(float val){
    #pragma unroll
    for(int mask = kWarpSize >> 1;mask >=1;mask >> 1){
        val += __shfl_xor_sync(0xFFFFFFFF,val,mask);
    }

    return val;
}

static __global__ void row_rmsnorm_f32_v0(const float * in,const float * wei,float *out,
                                        const int size,const float eps){
    const int tid = threadIdx.x;
    const int lane_id = tid % warpSize;

    float sum = 0.0f;
    for(int i = lane_id;i < size;i += warpSize){
        sum += in[i] * in[i];
    }

    using WarpReduce = cub::WarpReduce<float,32>;
    __shared__ typename WarpReduce::TempStorage temp;
    //__shared__ float shared_val;

    sum = WarpReduce(temp).Reduce(sum,cub::Sum());

    const float scale = rsqrtf(sum / static_cast<float>(size) + eps);
    for(int i = lane_id;i < size; i += warpSize){
        out[i] = scale * in[i] * wei[i];
    }

}

// warp 规约改为 block 之间的规约
template<int32_t BLOCK_DIM>
static __global__ void row_rmsnorm_f32_v1(const float*in,const float* wei,float * out,
                                            const int size,const float eps)
{
    const int tid = threadIdx.x;
    float sum =  0.0f;
    for(int i = tid; i < size; i +=  blockDim.x){
        sum += in[i] * in[i];
    }

    using BlockReduce = cub::BlockReduce<float,BLOCK_DIM>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float shared_val;
    sum = BlockReduce(temp).Sum(sum);
    if(threadIdx.x == 0){
        shared_val = sum;
    }
    __syncthreads();
    sum = shared_val;

    const float scale = rsqrtf(sum / static_cast<float>(size) + eps);
    for(int i = tid;i < size;i += blockDim.x){
        out[i] = scale * in[i] * wei[i];
    }
}

template<const int NUM_THREADS=128>
__device__ __forceinline__ float block_redude_sum(float val){
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SZIE - 1) / WARP_SZIE;
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    static __shared__ float shared[NUM_WARPS];

    val = warp_reduce_sum<WARP_SIZE>(val);
    if(lane = 0)
        shared[warp] = val;
    __syncthreads();

    val = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
    val = warp_reduce_sum<NUM_WARPS>(val);
    return val;
}

void rmsnorm_kernel_cu(const tensor::Tensor&  input,const tensor::Tensor& weight,
                        const tensor::Tensor& output, void * stream)
{

    const float eps = 1e-5f;
    // 各种检查先略过
    const int32_t size = static_cast<int32_t>(input.size());
    float * input_ptr = const_cast<float*>(input.ptr<float>());
    float * wei_ptr = const_cast<float*>(weight.ptr<float>());
    float * out_ptr = const_cast<float*>(output.ptr<float>());
    constexpr int threads_num = 128;
    if(stream){
        cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
        row_rmsnorm_f32_v0<<<1,threads_num,0,stream_>>>(input_ptr,wei_ptr,out_ptr,size,eps);
    }
    else{
        return row_rmsnorm_f32_v0<<<1,threads_num>>>(input_ptr,wei_ptr,out_ptr,size,eps);
    }

}

int main(){

    /*
    constexpr int threads_num = 128;
    if(stream){
        cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
        row_rmsnorm_f32<<<1,threads_num,0,stream>>>(in_ptr,wei_ptr,out_ptr,size,eps);
    }
    else{
        row_rmsnorm_f32<<<1,threads_num>>>(in_ptr,wei_ptr,out_ptr,size,eps);
    }

    */
    
    return 0;
}