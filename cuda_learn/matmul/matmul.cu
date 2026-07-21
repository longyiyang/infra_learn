

// 一个线程处理矩阵的一个元素

__global__ void mysgemm_v1(int M,int N,int K,float alpha,float *A,float *B,
                            float beta,float *C)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    if(gx >= N || gy >= M) return;

    float tmp = 0.0f;
    for(int i = 0;i < K;i++)
    {
        tmp += A[gy * K + i] * B[i * N + gx];
    }

    C[gy * N + gx] = alpha * tmp + beta * C[gy * N + gx];
}

// 基于共享显存的优化（减少全局显存的重复读，只需读一次，就可完成一次计算）

template <const int BLOCK_SIZE>
__global__ void mysgemm_v2(int M,int N,int K,float alpha,float *A,float *B,
                            float beta,float *C)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;

    const int BM = BLOCK_SIZE;
    const int BN = BLOCK_SIZE;
    const int BK = BLOCK_SIZE;

    int tx = threadIdx.x % BN;
    int ty = threadIdx.x / BN;

    __shared__ float As[BM * Bk];
    __shared__ float Bs[BK * BN];

    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    float tmp = 0;
    for(int k = 0;k < K;k += BK)
    {
        As[ty * BK + tx] = A[ty * K + tx];
        Bs[ty * BN + tx] = B[ty * N + tx];
        __syncthreads();

        // 读完对应区域的数据到共享显存才开始下面的计算

        A += BK;
        B += BK * N;
        for(int i = 0;i < BK;i++){
            tmp += As[ty * BK + i] * Bs[i * BN + tx];
        }
        __syncthreads();
    }

    C[ty * N + tx] = alpha * tmp + beta * C[ty * N + tx];
}


// 降低共享显存读写带来的问题，一个线程一次读多个数据，一个线程处理一个子块（tile）的数据

template <const int BM,const int BN,const int BK,const int TM,const int TN>
__global__ void mysgemm_v4(int M,int N,int K,float alpha,float *A,float *B,
                            float beta,float *C)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int block_row_thread = BN / TN;
    int block_col_thread = BM / TM;
    int thread_num = block_row_thread * block_col_thread;

    // 求一个tile（TM * TN）最左上角的元素的坐标
    int tx = (threadIdx.x % block_row_thread) * TN;
    int ty = (threadIdx.x / block_row_thread) * TM;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    int a_tile_row = threadIdx.x / BK;
    int a_tile_col = threadIdx.x % BK;
    int a_tile_stride = thread_num / BK;

    int b_tile_row = threadIdx.x / BN;
    int b_tile_col = threadIdx.x % BN;
    int b_tile_stride = thread_num / BN;

    float tmp[TM][TN] = {0.};

#pragma unroll
    for(int k = 0;k < K;k += BK)
    {
#pragma unroll
        // 先完成共享显存的数据读入
        for(int i = 0; i < BM;i += a_tile_stride)
        {
            As[(a_tile_row + i) * BK + a_tile_col] = 
                A[(a_tile_row + i) * K + a_tile_col];
        }
#pragma unroll
        for(int i = 0; i < BK;i += b_tile_stride)
        {
            Bs[(b_tile_row + i) * BN + b_tile_col] = 
                B[(b_tile_row + i) * N + b_tile_col];
        }
        __syncthreads();

        A += BK;
        B += BK * N;
#pragma unroll
        for(int i = 0;i < BK;i++)
        {
#pragma unroll
            for(int j = 0;j < TM;j++)
            {
                for(int l = 0;l < TN;l++)
                    tmp[j][l] += As[(ty + j) * BK + i] * Bs[tx + l + i * BN];
            }
        }
        __syncthreads();
    }

// 遍历完后，结果汇总
#pragma unroll 
    for(int j = 0;j < TM;j++)
    {
        for(int l = 0;l < TN;l++)
            C[(ty + j) * N + tx + l] = 
                alpha * tmp[j][l] + beta * C[(ty + j) * N + tx + l];
    }

}


int main(){
    const int N = 1024;
    const int BLOCK_SIZE = 128;
    dim3 threads(BLOCK_SIZE,BLOCK_SIZE);
    dim3 blocks((N + threads.x - 1)/threads.x,
                (N + threads.y - 1)/threads.y);

    //mysgemm_v1<<<blocks,threads>>>(N,N,N,alpha,d_A,d_B,beta,d_C_v1);
}










