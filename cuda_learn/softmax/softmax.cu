

__global__ void softmax_forward_kernel1(float *out, const float *inp,
    int N,int C){
    // 用一个线程处理所有矩阵的一行元素
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(i < N){
        const float * inp_row = inp + i * C;
        float *out_row = out + i * C;

        float maxval = -INFINITY;
        for(int j = 0;j < C;j++){
            if(inp_row[j] > maxval){
                maxval = inp_row[j];
            }
        }

        float sum = 0.f;
        for(int j = 0;j < C;j++){
            out_row[j] = expf(inp_row[j] - maxval);
            sum += out_row[j];
        }

        for(int j = 0;j < C; j++){
            out_row[j] /= (float)sum;
        }
    }
}


// 共享显存存中间计算结果 + 普通规约
// 一个线程块处理一行数据

__global__ void softmax_forward_kernel2(float * out,const float* inp,
                                        int N,int C){
    extern __shared__ float shared[]; // 注意每个线程块各自含有共享显存
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x; // 一个线程块内的线程数
    const float * x = inp + idx * C;

    float maxval = -INFINITY;
    for(int i = tid;i < C;i += block_size){
        maxval =  fmaxf(maxval,x[i]);
    }
    shared[tid] = maxval;
    __syncthreads();

    // 最大值局部结果规约
    for(int stride = block_size / 2;stride >= 1; stride /= 2){
        __syncthreads();
        if(tid < stride){
            shared[tid] = fmaxf(shared[tid],shared[tid + stride]);
        }
    }

    __syncthreads();
    float offset = shared[0];

    for(int i = tid; i < C; i += block_size){
        out[idx * C + i] = expf(x[i] - offset);
    }
    __syncthreads();

    x = out + idx * C;
    float sumval = 0.0f;
    for(int i = tid; i < C; i += block_size){
        sumval += x[i];
    }

    shared[tid] = sumval;
    __syncthreads();

    // 总和局部结果规约
    for(int stride = block_size / 2;stride >= 1; stride /= 2){
        __syncthreads();
        if(tid < stride){
            shared[tid] += shared[tid + stride];
        }
    }

    __syncthreads();
    float sum = shared[0];

    for(int i = tid; i < C; i += block_size){
        out[idx * C + i] = x[i] / sum;
    }
}


// warp 洗牌指令（一个 warp 32 个线程） 
// 一个线程块用一个warp

__device__ float warpReduceMax(float val){
    for(int offset =  16; offset > 0; offset /= 2){
        val = fmaxf(val,__shfl_down_sync(0xFFFFFFFF,val,offset));
    }
    return val;
}

__device__ float warpReduceSum(float val){
    for(int offset =  16; offset > 0; offset /= 2){
        val += __shfl_down_sync(0xFFFFFFFF,val,offset);
    }
    return val;
}

__global__ void softmax_forward_kernel3(float * out,const float* inp,
                                        int N,int C)
{
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_size =  blockDim.x;
    const float * x = inp + idx * C;

    float maxval = -INFINITY;
    for(int i = tid; i < C; i += block_size){
        maxval = fmaxf(maxval,x[i]);
    }

    maxval = warpReduceMax(maxval);
    float offset = __shfl_sync(0xFFFFFFFF,maxval,0);

    for(int i = tid;i < C;i += block_size){
        out[idx * C + i] = expf(x[i] - offset);
    }
    x = out + idx * C;
    float sumval = 0.0f;
    for(int i = tid; i < C; i += block_size){
        sumval += x[i]; 
    }

    // 求完局部和到块内共享显存，再规约
    sumval = warpReduceSum(sumval);
    float sum = __shfl_sync(0xFFFFFFFF,sumval,0);

    for(int i = tid;i < C;i+=block_size){
        out[idx * C + i] = x[i] / sum;
    }

}

// 各个warp 结果之间的规约，中间结果存共享显存
// 一个线程块用多个 warp

__global__ void softmax_forward_kernel4(float * out,const float* inp,
                                        int N,int C)
{
    extern __shared__ float shared[];
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int warpId =  threadIdx.x / 32;
    int laneId = threadIdx.x % 32;
    int block_size = blockDim.x;
    int warpsPerBlock =  blockDim.x / 32;

    // 两个块内共享显存
    float * maxvals = shared;
    float * sumvals = &shared[warpsPerBlock];
    
    const float * x = inp + idx * C;

    float maxval = -INFINITY;
    for(int i = tid; i < C; i += block_size){
        maxval = fmaxf(maxval,x[i]);
    }

    maxval = warpReduceMax(maxval);
    if(laneId == 0) maxvals[warpId] = maxval;
    __syncthreads();

    if(tid == 0){
        float val = maxvals[tid];
        for(int i = 1; i < warpsPerBlock;i++){
            val = fmaxf(val,maxvals[i]);
        }
        maxvals[0] = val;
    }

    __syncthreads();
    float offset =  maxvals[0];

    for(int i = tid;i < C;i += block_size){
        out[idx * C + i] = exp(x[i] - offset);
    }

    x = out + idx * C;
    float sumval =  0.0f;
    for(int i = tid; i < C; i += block_size){
        sumval += x[i]; 
    }

    sumval = warpReduceSum(sumval);
    if(laneId == 0) sumvals[warpId] = sumval;
    __syncthreads();

    if(tid ==  0){
        float val =  sumvals[tid];
        for(int i = 1;i < warpsPerBlock;i++){
            val += sumvals[i];
        }
        sumvals[0] = val;
    }

    __syncthreads();

    float sum = sumvals[0];
    for(int i = tid;i < C;i++){
        out[idx * C + i] = x[i] / sum;
    }

}

int main(){

    // 各个算子启动代码
    /*
    int blockSize = 1;
    int numBlocks = N;
    softmax_forward_kernel1<<<numBlocks,blockSize>>>(d_out,d_inp,N,C);

    int blockSize = 128;
    int numBlocks = N;
    softmax_forward_kernel2<<<numBlocks,blockSize>>>(d_out,d_inp,N,C);

    int blockSize = 32;
    int numBlocks = N;
    softmax_forward_kernel3<<<numBlocks,32>>>(d_out,d_inp,N,C);

    int blockSize = 128;
    int numBlocks = N;
    softmax_forward_kernel4<<<numBlocks,blockSize>>>(d_out,d_inp,N,C);

    */

    return 0;
}