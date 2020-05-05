//##########################################################//
//   Name: Kirtan Mali                                      //
//   Roll no: 18AG10016                                     //
//   Question 2: Matrix Transpose using Rect Tiles          //
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

#define TILEX 32
#define TILEY 16
#define BLOCKX 32
#define BLOCKY 16

void printMat(float *matrix, lli n);
void transpose_CPU(float *matrix, float *output, int n);
float *createMat(lli n, int isempty, int seed);

__global__ void transposeCoalesced_RECTTILES(float *matrix, float *output, int n)
{
    // shared memory
    __shared__ float tile[BLOCKY][BLOCKX];

    // global memory index for original matrix
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;

    // transposed index in shared memory
    int irow = (threadIdx.y * blockDim.x + threadIdx.x) % blockDim.y;
    int icol = (threadIdx.y * blockDim.x + threadIdx.x) / blockDim.y;

    // global memory index for transposed matrix
    int ox = blockDim.y * blockIdx.y + irow;
    int oy = blockDim.x * blockIdx.x + icol;

    if (ix < n && iy < n) {
        tile[threadIdx.y][threadIdx.x] = matrix[iy * n + ix];
        __syncthreads();
        output[oy * n + ox] = tile[irow][icol];
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
        srand(t);
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

        dim3 dimGrid((n + BLOCKX - 1) / BLOCKX, (n + BLOCKY - 1) / BLOCKY);
        dim3 dimBlock(BLOCKX, BLOCKY);

        transposeCoalesced_RECTTILES<<<dimGrid, dimBlock>>>(d_matrix, d_output, n);
     
        err = cudaGetLastError();
 
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch convolution_2D_DEVICE kernel (error code %s)!\n", cudaGetErrorString(err));
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
            // printf("\n\n***** Transposed Matrix using CPU *****\n\n")
            //printMat(h_output_check, n);
        }
     
        free(h_matrix);
        free(h_output);
        free(h_output_check);
        cudaFree(d_matrix);
        cudaFree(d_output);
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
