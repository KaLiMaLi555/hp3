/* ###############################################
#   Kernel1 -> out = sin(input1) + cos(input2)   #
#   Kernel2 -> out = log(input)                  #
#   Kernel3 -> out = sqrt(input)                 #
#   Input data in test.txt                       #
#   Kirtan Mali                                  #
############################################### */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__
void process_kernel1(float *input1, float *input2, float *output, int datasize)
{
    int blockId = blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;
    int i = blockId * blockDim.x * blockDim.y * blockDim.z + threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

    if (i < datasize)
    {
        output[i] = sin(input1[i]) + cos(input2[i]);
    }
}

__global__
void process_kernel2(float *input, float* output, int datasize)
{
    int blockId = blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;
    int i = blockId * blockDim.x * blockDim.y * blockDim.z + threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

    if (i < datasize)
    {
        output[i] = log(input[i]);
    }
}

__global__
void process_kernel3(float *input, float *output, int datasize)
{
    int blockId = blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;
    int i = blockId * blockDim.x * blockDim.y * blockDim.z + threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

    if (i < datasize)
    {
        output[i] = sqrt(input[i]);
    }
}

int main()
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // The vector length to be used, and compute its size
    // ===============================
    int numElements = 16384;
    // ===============================

    size_t size = numElements * sizeof(float);

    // Allocate the host input vector input1
    float *h_input1 = (float *)malloc(size);

    // Allocate the host input vector input2
    float *h_input2 = (float *)malloc(size);

    // Allocate the host input vector output
    float *h_output = (float *)malloc(size);

    // Verify that allocations succeeded
    if (h_input1 == NULL || h_input2 == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        scanf("%f", &h_input1[i]);
    }

    for (int i = 0; i < numElements; ++i)
    {
        scanf("%f", &h_input2[i]);
    }

    // Allocate the device input vector input1
    float *d_input1 = NULL;
    err = cudaMalloc((void **)&d_input1, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector input2
    float *d_input2 = NULL;
    err = cudaMalloc((void **)&d_input2, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector output1
    float *d_output1 = NULL;
    err = cudaMalloc((void **)&d_output1, size);

    // Allocate the device output vector output2
    float *d_output2 = NULL;
    err = cudaMalloc((void **)&d_output2, size);

    // Allocate the device output vector output
    float *d_output = NULL;
    err = cudaMalloc((void **)&d_output, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors input1 and input2 in host memory to the device input vectors in
    // device memory

    err = cudaMemcpy(d_input1, h_input1, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_input2, h_input2, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Vector Add CUDA Kernel
    // According to question
    dim3 gridsize1(4,2,2);
    dim3 blocksize1(32,32,1);

    dim3 gridsize2(2,8,1);
    dim3 blocksize2(8,8,16);

    dim3 gridsize3(16,1,1);
    dim3 blocksize3(128,8,1);

    process_kernel1<<<gridsize1, blocksize1>>>(d_input1, d_input2, d_output1, numElements);

    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch process_kernel1 kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    process_kernel2<<<gridsize2, blocksize2>>>(d_output1, d_output2, numElements);

    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch process_kernel2 kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    process_kernel3<<<gridsize3, blocksize3>>>(d_output2, d_output, numElements);

    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch process_kernel3 kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.

    err = cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i)
    {
        if (fabs(sqrt(log(sin(h_input1[i]) + cos(h_input2[i]))) - h_output[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
        printf("%0.2f ", h_output[i]);
    }

    printf("\n");
    return 0;
}