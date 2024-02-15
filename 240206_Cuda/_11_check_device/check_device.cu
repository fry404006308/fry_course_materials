

#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>

// 定义一个宏，用于检查 CUDA 操作的返回状态。如果状态不为零（表示出现错误），则打印错误信息并终止程序
#define CHECK(status)                                          \
    do                                                         \
    {                                                          \
        auto ret = (status);                                   \
        if (ret != 0)                                          \
        {                                                      \
            std::cerr << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)



int info2()
{
    int nDevices;

    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("Device name: %s\n", prop.name);
        printf("Memory Clock Rate (KHz): %d\n",
               prop.memoryClockRate);
        printf("Memory Bus Width (bits): %d\n",
               prop.memoryBusWidth);
        printf("Peak Memory Bandwidth (GB/s): %f\n\n",
               2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
        printf("Shared mem per block: %lu\n", prop.sharedMemPerBlock);
        printf("Registers per block: %d\n", prop.regsPerBlock);
        printf("Warp size: %d\n", prop.warpSize);
        printf("Max threads per block: %d\n",
               prop.maxThreadsPerBlock);
        printf("Total constant memory: %lu\n", prop.totalConstMem);
        printf("CUDA version: %d.%d\n", prop.major, prop.minor);
        printf("Clock rate: %d\n", prop.clockRate);
        printf("MultiProcessor Count: %d\n", prop.multiProcessorCount);
        printf("Integrated: %d\n", prop.integrated);
        printf("Can Map Host Memory: %d\n", prop.canMapHostMemory);
        printf("Compute mode: %d\n", prop.computeMode);
        printf("Concurrent Kernels: %d\n", prop.concurrentKernels);
        printf("ECCEnabled: %d\n", prop.ECCEnabled);
        printf("Kernel Exec Timeout Enabled: %d\n",
               prop.kernelExecTimeoutEnabled);
        printf("Total memory (bytes): %lu\n\n", prop.totalGlobalMem);
    }
    return 0;
}

int main(void)
{
    int dev = 0;
    cudaDeviceProp devProp;
    CHECK(cudaGetDeviceProperties(&devProp, dev));
    std::cout << "GPU device: " << dev << ": " << devProp.name << std::endl;
    std::cout << "SM number: " << devProp.multiProcessorCount << std::endl;

    info2();

    return 0;
}

/*
GPU device: 0: NVIDIA GeForce RTX 4060 Laptop GPU
SM number: 24
Device Number: 0
Device name: NVIDIA GeForce RTX 4060 Laptop GPU
Memory Clock Rate (KHz): 8001000
Memory Bus Width (bits): 128
Peak Memory Bandwidth (GB/s): 256.032000

Total global mem:  4290248704
Shared mem per block: 49152
Registers per block: 65536
Warp size: 32
Max threads per block: 1024
Total constant memory: 65536
CUDA version: 8.9
Clock rate: 2100000
MultiProcessor Count: 24
Integrated: 0
Can Map Host Memory: 1
Compute mode: 0
Concurrent Kernels: 1
ECCEnabled: 0
Kernel Exec Timeout Enabled: 1
Total memory (bytes): 4290248704


D:\_230711_learnArchive\LA_code_300_backend\_code_720_cpp\084_CMake\440_Cuda\_240125_1736_findCudaToolkitDemo\build\_03_check_device\Debug\Fry03CheckDevice.exe (进程 6220)已退出，代码为 0。
要在调试停止时自动关闭控制台，请启用“工具”->“选项”->“调试”->“调试停止时自动关闭控制台”。
按任意键关闭此窗口. . .


下面是对上述CUDA设备信息每个字段的中文解释：

- `GPU device: 0: NVIDIA GeForce RTX 4060 Laptop GPU`：表示当前的GPU设备ID为0，设备型号是NVIDIA GeForce RTX 4060 Laptop GPU，专为笔记本电脑设计的显卡。
- `SM number: 24`：SM数量，指的是流多处理器（Streaming Multiprocessors）的数量，这是NVIDIA GPU架构中的一个核心组件，用于执行计算任务。
- `Device Number: 0`：设备号，通常用于多GPU系统中区分不同的GPU。
- `Device name: NVIDIA GeForce RTX 4060 Laptop GPU`：设备名称，再次说明了GPU的型号。
- `Memory Clock Rate (KHz): 8001000`：内存时钟频率，以千赫兹（KHz）计，这里是8001000 KHz，即8001 MHz。
- `Memory Bus Width (bits): 128`：内存总线宽度，以位（bits）计，这里是128位，这个参数影响着内存带宽。
- `Peak Memory Bandwidth (GB/s): 256.032000`：峰值内存带宽，以GB/s（吉字节每秒）计，这里是256.032 GB/s。

- `Total global mem: 4290248704`：总的全局内存大小，以字节计，这里是4290248704字节，大约为4GB。
- `Shared mem per block: 49152`：每个块的共享内存大小，以字节计，这里是49152字节，也就是48KB。
- `Registers per block: 65536`：每个块的寄存器数量，这里是65536个。
- `Warp size: 32`：Warp大小，每个Warp包含的线程数，这里是32个。
- `Max threads per block: 1024`：每个块的最大线程数，这里是1024个。
- `Total constant memory: 65536`：总的常量内存大小，以字节计，这里是65536字节，也就是64KB。
- `CUDA version: 8.9`：CUDA的版本号，这里是8.9。
- `Clock rate: 2100000`：GPU核心的时钟频率，以赫兹（Hz）计，这里是2100000 Hz，即2100 MHz。
- `MultiProcessor Count: 24`：多处理器数量，再次提到了SM的数量，这里是24个。
- `Integrated: 0`：是否为集成GPU，0表示不是，1表示是。
- `Can Map Host Memory: 1`：是否能映射主机内存，1表示可以，0表示不可以。
- `Compute mode: 0`：计算模式，0通常表示没有限制的计算模式。
- `Concurrent Kernels: 1`：是否支持并行执行多个内核，1表示支持，0表示不支持。
- `ECCEnabled: 0`：是否启用错误校正码（ECC）内存，0表示没有启用，1表示启用。
- `Kernel Exec Timeout Enabled: 1`：内核执行超时监控是否启用，1表示启用，用于检测长时间运行的内核，0表示没有启用。
- `Total memory (bytes): 4290248704`：总内存大小，以字节为单位，这里是4290248704字节，再次表示大约为4GB的显存容量。


*/
