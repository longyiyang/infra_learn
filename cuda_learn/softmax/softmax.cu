

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

// 共享显存存中间计算结果 + 规约


// warp 洗牌指令 


// warp 结果之间的规约