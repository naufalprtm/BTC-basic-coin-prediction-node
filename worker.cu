#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>  // For printf
#include <stdlib.h> // For exit

// Macro for checking CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Function to check CUDA runtime version
void checkCudaVersion() {
    int driverVersion = 0;
    int runtimeVersion = 0;

    CUDA_CHECK(cudaDriverGetVersion(&driverVersion));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtimeVersion));

    printf("CUDA Driver Version: %d\n", driverVersion);
    printf("CUDA Runtime Version: %d\n", runtimeVersion);
}

// CUDA kernel for matrix multiplication with shared memory
extern "C" __global__ void matrixMul(float* A, float* B, float* C, int N) {
    __shared__ float shared_A[16][16];
    __shared__ float shared_B[16][16];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float value = 0;

    for (int i = 0; i < (N + 15) / 16; ++i) {
        if (i * 16 + threadIdx.x < N && row < N)
            shared_A[threadIdx.y][threadIdx.x] = A[row * N + i * 16 + threadIdx.x];
        else
            shared_A[threadIdx.y][threadIdx.x] = 0.0;

        if (i * 16 + threadIdx.y < N && col < N)
            shared_B[threadIdx.y][threadIdx.x] = B[(i * 16 + threadIdx.y) * N + col];
        else
            shared_B[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        for (int j = 0; j < 16; ++j) {
            value += shared_A[threadIdx.y][j] * shared_B[j][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = value;
    }
}

// Function to run matrix multiplication on GPU
extern "C" void runMatrixMul(float* A, float* B, float* C, int N) {
    // Validate input
    if (N <= 0) {
        fprintf(stderr, "Invalid matrix size: %d. Size must be positive.\n", N);
        exit(EXIT_FAILURE);
    }

    size_t size = N * N * sizeof(float);

    // Check CUDA runtime version
    checkCudaVersion();

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, size));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size));

    // Check for memory allocation errors
    if (!d_A || !d_B || !d_C) {
        fprintf(stderr, "Memory allocation failed.\n");
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        exit(EXIT_FAILURE);
    }

    // Copy matrices from host to device
    CUDA_CHECK(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice));

    // Define block and grid size
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    // Run kernel
    matrixMul<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        exit(EXIT_FAILURE);
    }

    // Ensure kernel execution is complete
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost));

    // Check for memory copy errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA memory copy error: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        exit(EXIT_FAILURE);
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}
