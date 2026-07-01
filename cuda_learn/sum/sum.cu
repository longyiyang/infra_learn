
const int BLOCK_SIZE = 1024;

// 每个线程读一个元素到共享显存,一个线程块直接处理一个数据向量的子集（数据向量非常长，需要多个线程块）
// 规约方式，步长每次乘2

__global__ void reduce_v0(float *g_idata,float *g_odata)
{
    __shared__ float sdata[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int i =  blockIdx.x * blockDim.x + tid;
    sdata[tid] = g_idata[i];
    __syncthreads();

    for(unsigned int s = 1;s < blockDim.x;s *= 2){
        if(tid % (2 * s)){
            sdata[tid] += sdata[tid + s];
        }

        __syncthreads();
    }

    if(tid == 0) g_odata[blockIdx.x] = sdata[0];

}

// 解决线程闲置的问题


__global__ void reduce_v1(float *g_idata,float *g_odata)
{
    __shared__ float sdata[BLOCK_SIZE];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
    __syncthreads();

    for(unsigned int s = 1;s < blockDim.x;s *= 2){
        if(tid % (2 * s) == 0)
            sdata[tid] += sdata[tid + s];

        __syncthreads();
    }

    if(tid == 0) g_odata[blockIdx.x] = sdata[0];

}

// 规约方式（步长逐渐变小）减少线程束分化频繁出现

__global__ void reduce_v2(float *g_idata,float *g_odata)
{
    __shared__ float sdata[BLOCK_SIZE];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
    __syncthreads();

    for(unsigned int s = blockDim.x / 2;s > 0;s >> 1){
        if(tid < s)  // s >= 32 时，一个或多个 warp 里的线程都满足 tid < s，减少了warp分化
            sdata[tid] += sdata[tid + s];

        __syncthreads();
    }

    if(tid == 0) g_odata[blockIdx.x] = sdata[0];

}

// 上一种方式的再次优化，s == 32 时对最后满足判断条件的32个线程进行优化处理（tid < 32）
// 用 warp 通信原语 __shfl_dwon_sync

#define MASK 0xFFFFFFFF

__device__ void warpReduce(float * cache,unsigned int tid)
{
    float v = cache[tid] + cache[tid + 32];
    v += __shfl_down_sync(MASK,v,16);  // __shfl_down_sync 某个线程如果超过范围则直接保留自身的值
    v += __shfl_down_sync(MASK,v,8);
    v += __shfl_down_sync(MASK,v,4);
    v += __shfl_down_sync(MASK,v,2);
    v += __shfl_down_sync(MASK,v,1);

    cache[tid] = v;
}

__global__ void reduce_v2(float *g_idata,float *g_odata)
{
    __shared__ float sdata[BLOCK_SIZE];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
    __syncthreads();

    for(unsigned int s = blockDim.x / 2;s > 32;s >> 1){
        if(tid < s)  // s >= 32 时，一个或多个 warp 里的线程都满足 tid < s，减少了warp分化
            sdata[tid] += sdata[tid + s];

        __syncthreads();
    }

    if(tid < 32) warpReduce(sdata,tid);
    if(tid == 0) g_odata[blockIdx.x] = sdata[0];

}

// wrap 得到的结果，之后 warp 再规约
__inline__ __device__ float block_reduce(float val)
{
    const int tid = threadIdx.x;
    const int warpSize = 32;

    for(int offset = warpSize / 2; offset > 0;offset /= 2){
        val += __shfl_down_sync(0xFFFFFFFF,val,offset);
    }

    __shared__ float warpSum[32];
    if(tid % warpSize == 0) warpSum[tid / warpSize] = val;
    __syncthreads();

    int warp_id = tid / warpSize;
    if(warp_id == 0){
        val = (tid < blockDim.x / warpSize) ? warpSum[tid] : 0.0f;
        for(int offset = warpSize / 2; offset > 0;offset /= 2)
            val += __shfl_down_sync(0xFFFFFFFF,val,offset);
    }

    return val;
}

int main(){
    /*
    v0 版本
    reduce_v0<<<num_blocks,BLOCK_SIZE>>>(d_data,d_result);
    reduce_v0<<<1,num_blocks>>>(d_result,d_final_result);

    int num_blocks = ((N + BLOCK_SIZE - 1) / BLOCK_SIZE) / 2;  预处理后只需要原先一半的线程块数量
    reduce_v1<<<num_blocks,BLOCK_SIZE>>>(d_data,d_result);
    reduce_v1<<<1,num_blocks>>>(d_result,d_final_result);
    
    */
}





