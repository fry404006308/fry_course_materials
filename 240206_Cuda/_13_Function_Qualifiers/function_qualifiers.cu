#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// __device__ 函数只能在设备(GPU)上调用
__device__ int deviceAdd(int a, int b)
{
    // 在设备上执行加法操作
    return a + b;
}

// __host__ __device__ 函数可以在主机(CPU)和设备(GPU)上调用
__host__ __device__ int hostDeviceMultiply(int a, int b)
{
    // 在主机或设备上执行乘法操作
    return a * b;
}

// __global__ 函数在设备(GPU)上执行,从主机(CPU)调用
__global__ void kernelFunc(int* data, int size)
{
    // 计算当前线程的唯一索引
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 检查索引是否在有效范围内
    if (tid < size)
    {
        // 在设备上调用 deviceAdd 函数,将当前元素与其索引相加
        data[tid] = deviceAdd(data[tid], tid);

        // 在设备上调用 hostDeviceMultiply 函数,将当前元素乘以2
        data[tid] = hostDeviceMultiply(data[tid], 2);
    }
}

// 打印vector的函数
void printVector(const std::vector<int>& vec)
{
    // 使用基于范围的for循环遍历vector并打印每个元素
    for (const auto& val : vec)
    {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

int main()
{
    const int size = 10;

    // 创建一个大小为 size 的 vector,并初始化为0
    std::vector<int> data(size);

    // 初始化 vector 的元素为其对应的索引值
    for (int i = 0; i < size; ++i)
    {
        data[i] = i;
    }

    std::cout << "Data init: ";
    // 打印初始化后的 vector
    printVector(data);

    int* deviceData;
    // 在设备(GPU)上分配内存
    cudaMalloc(&deviceData, size * sizeof(int));

    // 将数据从主机内存复制到设备内存
    cudaMemcpy(deviceData, data.data(), size * sizeof(int), cudaMemcpyHostToDevice);

    // 启动内核函数,在设备上执行
    kernelFunc << <1, size >> > (deviceData, size);

    // 将数据从设备内存复制回主机内存
    cudaMemcpy(data.data(), deviceData, size * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Result: ";
    // 打印处理后的 vector
    printVector(data);

    // 释放设备内存
    cudaFree(deviceData);

    std::cout << "hostDeviceMultiply: ";
    // 在主机上调用 hostDeviceMultiply 函数并打印结果
    std::cout << hostDeviceMultiply(3, 7) << std::endl;

    return 0;
}


/*

Data init: 0 1 2 3 4 5 6 7 8 9
Result: 0 4 8 12 16 20 24 28 32 36
hostDeviceMultiply: 21

D:\_230711_learnArchive\fry_course_materials\240206_Cuda\_13_Function_Type_Qualifiers\build\Debug\fry13FunctionQualifiers.exe (进程 2228)已退出，代码为 0。
要在调试停止时自动关闭控制台，请启用“工具”->“选项”->“调试”->“调试停止时自动关闭控制台”。
按任意键关闭此窗口. . .



*/

