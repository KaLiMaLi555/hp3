//##########################################################//
//   Name: Kirtan Mali                                      //
//   Roll no: 18AG10016                                     //
//   Question 1: 2D Convolution Matrix                      //
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

// Important parameters
#define KERNEL_SIZE 3
#define KERNEL_HALF (KERNEL_SIZE >> 1)
#define BLOCK_SIZE 32
#define TILE_SIZE (BLOCK_SIZE - KERNEL_SIZE + 1)

// Function prototypes
void printMat(float *matrix, lli n);
void convolution_2D_HOST(float *matrix, float *output, int n, int kernelH, int kernelW);
float *createMat(lli n, int isempty, int seed);

// Convolution Kernel
__global__
void convolution_2D_DEVICE(float *matrix, float *output, int n)
{
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE];

    // get thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // get the output indices
    int row_o = ty + blockIdx.y * TILE_SIZE;
    int col_o = tx + blockIdx.x * TILE_SIZE;

    // shift to obtain input indices
    int row_i = row_o - KERNEL_HALF;
    int col_i = col_o - KERNEL_HALF;

    // Load tile elements
    if(row_i >= 0 && row_i < n && col_i >= 0 && col_i < n)
        tile[ty][tx] = matrix[row_i*n + col_i];
    else
        tile[ty][tx] = 0.0f;

    __syncthreads();

    if(tx < TILE_SIZE && ty < TILE_SIZE){
        float pValue = 0.0f;
        for(int y=0; y<KERNEL_SIZE; y++)
            for(int x=0; x<KERNEL_SIZE; x++)
                pValue += tile[y+ty][x+tx] / 9.0;

        if(row_o < n && col_o < n)
        {
            output[row_o*n + col_o] = pValue;
        }
    }

}

int main(int argc, char **argv)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    int isprint = 1;
    if (argc > 1)
    {
        printf("\n\nDisabling Printing ...\n\n");
        isprint = 0;
    }

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

        float *d_matrix = NULL;
        float *d_output = NULL;

        CHECK(cudaMalloc((void **)&d_matrix, size));

        CHECK(cudaMalloc((void **)&d_output, size));

        CHECK(cudaMemcpy(d_matrix, h_matrix, size, cudaMemcpyHostToDevice));

        dim3 blockSize, gridSize;
        blockSize.x = BLOCK_SIZE, blockSize.y = BLOCK_SIZE, blockSize.z = 1;
        gridSize.x = ceil((float)n/TILE_SIZE),
        gridSize.y = ceil((float)n/TILE_SIZE),
        gridSize.z = 1;

        convolution_2D_DEVICE<<<gridSize, blockSize>>>(d_matrix, d_output, n);

        err = cudaGetLastError();

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch convolution_2D_DEVICE kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // convolution_2D_HOST(h_matrix, h_output, n, 3, 3);

        CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

        if (isprint == 1)
        {
            printf("\n\n***** Original Matrix *****\n\n");
            printMat(h_matrix, n);
            printf("\n\n***** Convolved Matrix Output *****\n\n");
            printMat(h_output, n);
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
        printf("%0.2f ", matrix[i]);
        if (i % n == n-1)
            printf("\n");
    }
}

void convolution_2D_HOST(float *matrix, float *output, int n, int kernelH, int kernelW)
{
    for (lli i=0; i<n; i++)
    {
        for (lli j=0; j<n; j++)
        {
            lli startx = i - (kernelH/2);
            lli starty = j - (kernelW/2);

            float newval = 0.0;
            for (lli a=0; a<kernelH; a++)
            {
                for (lli b=0; b<kernelW; b++)
                {
                    if (startx + a >= 0 && startx + a < n && starty + b >= 0 && starty + b < n)
                    {
                        newval += matrix[(startx+a)*n + (starty+b)] / (float)(kernelH*kernelW);
                    }
                }
            }

            output[i*n + j] = newval;
        }
    }
}