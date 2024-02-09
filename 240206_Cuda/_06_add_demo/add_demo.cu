/*



在CUDA中，host和device是两个重要的概念，
我们用host指代CPU及其内存，而用device指代GPU及其内存。
CUDA程序中既包含host程序，又包含device程序，它们分别在CPU和GPU上运行。
同时，host与device之间可以进行通信，这样它们之间可以进行数据拷贝。
典型的CUDA程序的执行流程如下：

1、分配host内存，并进行数据初始化；
2、分配device内存，并从host将数据拷贝到device上；
3、调用CUDA的核函数在device上完成指定的运算；
4、将device上的运算结果拷贝到host上；
5、释放device和host上分配的内存。


计算两个长度为100000的一维向量相加
*/
#include <stdio.h>
#include <iostream>

// CUDA内核函数，用于在GPU上执行向量加法
__global__ void vecAdd(float *A, float *B, float *C, int n)
{
    // 获取当前线程的全局索引
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 如果索引在向量长度内，执行加法操作
    if (i < n)
    {
        C[i] = A[i] + B[i];
    }
}

int main()
{
    int n = 100000;               // 向量长度
    float *h_A, *h_B, *h_C;       // 主机端指针
    float *d_A, *d_B, *d_C;       // 设备端指针
    int size = n * sizeof(float); // 分配的内存大小

    // 分配主机端内存
    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size);
    h_C = (float *)malloc(size);

    // 初始化向量数据
    for (int i = 0; i < n; i++)
    {
        h_A[i] = i;
        h_B[i] = i;
    }

    // 分配设备端内存
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // 将向量数据从主机复制到设备
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 定义CUDA内核的执行配置
    int blockSize = 256;
    int gridSize = (int)ceil((float)n / blockSize);

    // 在GPU上执行内核
    vecAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);

    // 将结果从设备复制回主机
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 验证结果
    for (int i = 0; i < n; i++)
    {
        if (i < 100) {
            printf("h_A[i] + h_B[i] = h_C[i] : %f + %f = %f !\n", h_A[i], h_B[i], h_C[i]);
        }
        if (h_C[i] != h_A[i] + h_B[i])
        {
            printf("Error at index %d!\n", i);
            break;

        }
    }

    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // 释放主机内存
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Vector addition completed successfully!\n");
    return 0;
}