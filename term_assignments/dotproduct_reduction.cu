//##########################################################//
//   Name: Kirtan Mali                                      //
//   Roll no: 18AG10016                                     //
//   Question 4: DotProduct using Reduction Kernel          //
//##########################################################//

#include <stdio.h>
#include <stdlib.h>

// Cuda Libraries
#include <cuda.h>
#include <cuda_runtime.h>

// Macro for error checking and debugging
#define CHECK(call) {                                                        \
    const cudaError_t error = call;                                          \
    if (error != cudaSuccess) {                                              \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                        \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));  \
        exit(1);                                                             \
    }                                                                        \
}

typedef long long int lli;
#define MAX_VAL 100
// Works with Block Size 1024 for other blocksizes add if else in kernels
#define BLOCKSIZE 1024

// Function Prototypes
double *createvector(lli n, int isempty, int seed);
void printvector(double *vector, lli n);
double dotproduct_CPU(double *A, double *B, lli n);

// Reduction Kernels
__global__ void reducefinal(double *input, double *output, int n)
{
	unsigned int tid = threadIdx.x;

    extern __shared__ double tile[];

    int blockId = blockIdx.x;
    int i = blockId * (blockDim.x*2) + threadIdx.x;
    unsigned int blockSize = blockDim.x;

    if (i < n)
    {
        tile[tid] = input[i] + input[i+blockDim.x];
    }
    else
    {
        tile[tid] = 0.0f;
    }

    __syncthreads();


    if (tid < 512)
        tile[tid] += tile[tid + 512];
    __syncthreads();

    if (tid < 256)
        tile[tid] += tile[tid + 256];
    __syncthreads();

    if (tid < 128)
        tile[tid] += tile[tid + 128];
    __syncthreads();

    if (tid < 64)
        tile[tid] += tile[tid + 64];
    __syncthreads();

    if (tid < 32)
        tile[tid] += tile[tid + 32];
    __syncthreads();

    if (tid < 16)
        tile[tid] += tile[tid + 16];
    __syncthreads();

    if (tid < 8)
        tile[tid] += tile[tid + 8];
    __syncthreads();

    if (tid < 4)
        tile[tid] += tile[tid + 4];
    __syncthreads();

    if (tid < 2)
        tile[tid] += tile[tid + 2];
    __syncthreads();

    if (tid < 1)
    {
        tile[tid] += tile[tid + 1];
        output[blockId] = tile[tid];
    }
}

// DotProduct Kernel
__global__ void dotproduct(double *A, double *B, double *output, lli n)
{
    int tid = threadIdx.x;

    extern __shared__ double tile[];

    int blockId = blockIdx.x;
    int i = blockId * blockDim.x + threadIdx.x;

    if (i < n)
    {
        tile[tid] = A[i] * B[i];
    }
    else
    {
        tile[tid] = 0.0f;
    }

    __syncthreads();

    for (int s=blockDim.x/2; s>0; s>>=1)
    {
        if (tid < s)
        {
            tile[tid] += tile[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        output[blockId] = tile[tid];
    }
}

double reduce_CPU(double *vector, lli n)
{
    double sum = 0;

    for (int i=0; i<n; i++)
    {
        sum += vector[i];
    }

    return sum;
}

int main(int argc, char **argv)
{
    cudaError_t err = cudaSuccess;
    lli t;

    int isprint = 1;
    if (argc > 1)
    {
        printf("\n\nDisabling Printing ...\n\n");
        isprint = 0;
    }

    scanf("%lld", &t);

    while(t--)
    {
        srand(t);
        lli n;

        scanf("%lld", &n);

        size_t size = sizeof(double) * n;

        double *h_A = createvector(n, 0, t);
        double *h_B = createvector(n, 0, rand() * t);

        double *d_A = NULL;
        CHECK(cudaMalloc((void **)&d_A, size));
        CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));


        double *d_B = NULL;
        CHECK(cudaMalloc((void **)&d_B, size));
        CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

        double *h_dotproduct_output = createvector(n/BLOCKSIZE, 1, t);

        double *d_dotproduct_output = NULL;
        CHECK(cudaMalloc((void **)&d_dotproduct_output, size/BLOCKSIZE));

        dotproduct<<<n/BLOCKSIZE, BLOCKSIZE, BLOCKSIZE*sizeof(double)>>>(d_A, d_B, d_dotproduct_output, n);

        err = cudaGetLastError();

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch dotproduct kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        double *d_input = NULL;
        CHECK(cudaMalloc((void **)&d_input, size/BLOCKSIZE));

        CHECK(cudaMemcpy(d_input, d_dotproduct_output, size/BLOCKSIZE, cudaMemcpyDeviceToDevice));

        double *d_final_output = NULL;
        CHECK(cudaMalloc((void **)&d_final_output, size/BLOCKSIZE));

        lli tempn = n/BLOCKSIZE;

        while (tempn > 1024)
        {
            reducefinal<<<tempn/BLOCKSIZE, BLOCKSIZE, BLOCKSIZE*sizeof(double)>>>(d_input, d_final_output, tempn);

            err = cudaGetLastError();

            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to launch reducefinal kernel (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

            tempn /= BLOCKSIZE;

            CHECK(cudaMemcpy(d_input, d_final_output, tempn*sizeof(double), cudaMemcpyDeviceToDevice));
        }

        double *h_final_output = (double *)malloc(sizeof(double) * tempn);

        CHECK(cudaMemcpy(h_final_output, d_input, tempn*sizeof(double), cudaMemcpyDeviceToHost));

        if (isprint == 0)
        {
            printf("\n\n***** Final Output of GPU Reduction Kernal *****\n\n");
            printvector(h_final_output, tempn);
        }

        double final_dotproduct = reduce_CPU(h_final_output, tempn);

        printf("\nDot Product => %0.2f", final_dotproduct);
        printf("Dot Product from CPU => %0.2f\n\n", dotproduct_CPU(h_A, h_B, n));
    }

    return 0;
}

// Utility Functions
double dotproduct_CPU(double *A, double *B, lli n)
{
    double dp = 0;

    for (lli i=0; i<n; i++)
    {
        dp += A[i] * B[i];
    }

    return dp;
}

void printvector(double *vector, lli n)
{
    for (lli i=0; i<n; i++)
    {
        printf("% 6.2f ", vector[i]);
    }
    printf("\n");
}

double *createvector(lli n, int isempty, int seed)
{
    srand(seed+1);
    double *vector = (double *)malloc(n * sizeof(double));

    for (lli i=0;i<n;i++)
    {
        if (isempty == 0)
        {
            vector[i] = (double)rand()/((double)RAND_MAX/MAX_VAL);
        }
        else
        {
            vector[i] = 0.0f;
        }
    }

    return vector;
}