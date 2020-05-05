/* ###############################################
#   Basic reduction kernel without optimization  #
#   							                 #
#   Kirtan Mali                                  #
############################################### */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>


__global__ void reduce1(float *input, float *output, float K)
{
	unsigned int tid = threadIdx.x;

    int blockId = blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;
    int i = blockId * blockDim.x * blockDim.y * blockDim.z + threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

	for (unsigned int s=blockDim.x/2; s>0; s>>=1)
	{
		if (tid < s)
		{
			input[i] += input[i + s];
		}
		__syncthreads();
	}

	// write result for this block to global mem32
	if (tid == 0)
	{
		output[blockId] = input[i-threadIdx.x] / K;
	}
}

int main()
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Number of test cases
    int t;
    scanf("%d", &t);

    while (t--)
    {
        // Inputing p and q
        double p, q;
        scanf("%lf %lf", &p, &q);

        double N = pow(2, p);
		double K = pow(2, q);

        size_t size = N * sizeof(float);

		// Allocate the host input vector input1
		float *h_input = (float *)malloc(size);

		// Allocate the host input vector output
		float *h_output = (float *)malloc(size);

		// Verify that allocations succeeded
		if (h_input == NULL || h_output == NULL)
		{
			fprintf(stderr, "Failed to allocate host vectors!\n");
			exit(EXIT_FAILURE);
		}

		// Initialize the host input vectors
		for (int i = 0; i < N; ++i)
		{
			scanf("%f", &h_input[i]);
		}

		// Allocate the device input vector input1
		float *d_input = NULL;
		err = cudaMalloc((void **)&d_input, size);

		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		// Allocate the device output vector output1
		float *d_output = NULL;
		err = cudaMalloc((void **)&d_output, size);

		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		// Copy the host input vectors input1 and input2 in host memory to the device input vectors in
		// device memory

		err = cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

        while (N >= K)
		{
			dim3 gridsize((int)sqrt(N/K),(int)sqrt(N/K),1);
        	dim3 blocksize((int)K,1,1);
			reduce1<<<gridsize, blocksize>>>(d_input, d_output, K);

			N = N / K;

			err = cudaMemcpy(d_input, d_output, size, cudaMemcpyDeviceToDevice);
		}

		err = cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

        for (int i=0;i<N;i++)
        {
            printf("%0.3f ", h_output[i]);
        }

        printf("\n");
    }

    return 0;
}