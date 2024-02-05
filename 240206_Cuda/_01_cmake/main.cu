#include <cuda_runtime.h>
#include <iostream>

__global__ void hello_cuda()
{
    printf("Hello from CUDA-1!\n");
}

int main()
{
    std::cout << "Hello from CPU-1!" << std::endl;
    hello_cuda<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}