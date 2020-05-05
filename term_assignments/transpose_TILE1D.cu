//##########################################################//
//   Name: Kirtan Mali                                      //
//   Roll no: 18AG10016                                     //
//   Question 3: Matrix Transpose using Dynamic Shared Mem  //
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

#define TILEDIM 32
#define BLOCK_ROWS 32
#define PAD 1

void printMat(float *matrix, lli n);
void transpose_CPU(float *matrix, float *output, int n);
float *createMat(lli n, int isempty, int seed);

__global__ void transposeCoalesced(float *matrix, float *output, lli n)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * TILEDIM + tx;
    int y = blockIdx.y * TILEDIM + ty;

    extern __shared__ float tile[];

    for (int j=0; j<TILEDIM; j+=BLOCK_ROWS)
    {
        tile[(ty+j)*(TILEDIM+PAD) + tx] = matrix[(y+j)*n + x];
    }

    __syncthreads();

    x = blockIdx.y * TILEDIM + tx;
    y = blockIdx.x * TILEDIM + ty;

    for (int j=0; j<TILEDIM; j+=BLOCK_ROWS)
    {
        output[(y+j)*n + x] = tile[tx*(TILEDIM+PAD) + ty+j];
    }
}

int main(int argc, char **argv)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    int isprint = 1;
    if (argc > 1)
        isprint = 0;

    lli t;
    scanf("%lld", &t);

    while (t--)
    {
        lli n;
        scanf("%lld", &n);

        size_t size = sizeof(float) * n * n;

        float *h_matrix = createMat(n, 0, t);
        float *h_output = createMat(n, 1, t);
        float *h_output_check = createMat(n, 1, t);

        float *d_matrix = NULL;
        float *d_output = NULL;

        CHECK(cudaMalloc((void **)&d_matrix, size));

        CHECK(cudaMalloc((void **)&d_output, size));

        CHECK(cudaMemcpy(d_matrix, h_matrix, size, cudaMemcpyHostToDevice));

        dim3 dimGrid(ceil((float)n/BLOCK_ROWS), ceil((float)n/BLOCK_ROWS), 1);
        dim3 dimBlock(BLOCK_ROWS, BLOCK_ROWS, 1);

        transposeCoalesced<<<dimGrid, dimBlock, TILEDIM*(TILEDIM+PAD)*sizeof(float)>>>(d_matrix, d_output, n);

        err = cudaGetLastError();

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch transposeCoalesced kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

        // transpose_CPU(h_matrix, h_output_check, n);

        if (isprint == 1)
        {
            printf("\n\n***** Original Matrix *****\n\n");
            printMat(h_matrix, n);
            printf("\n\n***** Transposed Matrix using GPU *****\n\n");
            printMat(h_output, n);
            // printf("\n\n***** Transposed Matrix using CPU *****\n\n");
            // printMat(h_output_check, n);
        }
    }

    return 0;
}

// Utility Functions
float *createMat(lli n, int isempty, int seed)
{
    srand(seed+1);
    size_t size = sizeof(float) * n * n;
    float *matrix = (float *)malloc(size);

    for (int i=0; i<n*n; i++)
    {
        if (isempty == 1)
            matrix[i] = 0.0f;
        else
            matrix[i] = (float)rand()/((float)RAND_MAX/MAX_VAL);
    }

    return matrix;
}

void printMat(float *matrix, lli n)
{
    for (lli i=0; i<n*n; i++)
    {
        printf("% 6.2f ", matrix[i]);
        if (i % n == n-1)
            printf("\n");
    }
}

void transpose_CPU(float *matrix, float *output, int n)
{
    for (int i=0; i<n; i++)
    {
        for (int j=0; j<n; j++)
        {
            output[i*n+j] = matrix[j*n+i];
        }
    }
}