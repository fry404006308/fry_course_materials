#pragma once

// cuda 编程
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cuda_runtime_api.h"
#include <nvtx3/nvToolsExt.h>

#include <cassert>

#include <stdio.h>

inline int DivUp(int a, int b)
{
    assert(b > 0);
    return static_cast<int>(ceil(static_cast<float>(a) / b));
};



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

#define CHECK_CUDA_ERROR(call)                                                \
    do                                                                        \
    {                                                                         \
        const cudaError_t error = call;                                       \
        if (error != cudaSuccess)                                             \
        {                                                                     \
            printf("ERROR: %s:%d,", __FILE__, __LINE__);                      \
            printf("code:%d, reason:%s\n", error, cudaGetErrorString(error)); \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

#define CHECK_RUN()                                                \
    do                                                             \
    {                                                              \
        cudaError_t error = cudaGetLastError();                    \
        if (error != cudaSuccess)                                  \
        {                                                          \
            printf("CUDA error: %s\n", cudaGetErrorString(error)); \
        }                                                          \
    } while (0)

void GpuResize(uint8_t *image,
               uint8_t *out_image,
               uint32_t src_width,
               uint32_t src_height,
               uint32_t out_width,
               uint32_t out_height,
               uint32_t channels); // 添加 channels 参数







