

// cuda 编程
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cuda_runtime_api.h"
#include <nvtx3/nvToolsExt.h>

#include <cassert>
#include <stdio.h>

#include "resize.cuh"



__global__ void ResizeOperate_kernel(uint8_t *src, uint8_t *dst,
                                     float scale_x, float scale_y, int src_width,
                                     int src_height, int out_width, int out_height,
                                     int channels) // 添加 channels 参数
{
    const int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((dst_x < out_width) && (dst_y < out_height))
    {
        float fy = (dst_y + 0.5f) * scale_y - 0.5f;
        int sy = floor(fy);
        fy -= sy;
        sy = fmaxf(0, fminf(sy, src_height - 2));

        const uint8_t *aPtr = src + sy * src_width * channels;       // 乘以 channels
        const uint8_t *bPtr = src + (sy + 1) * src_width * channels; // 乘以 channels

        float fx = (dst_x + 0.5f) * scale_x - 0.5f;
        int sx = floor(fx);
        fx -= sx;
        fx *= ((sx >= 0) && (sx < src_width - 1));
        sx = fmaxf(0, fminf(sx, src_width - 2));

        uint32_t sp = sx * channels;                                   // 乘以 channels
        uint32_t sp_right = (sx + 1) * channels;                       // 乘以 channels
        uint32_t dp = dst_y * out_width * channels + dst_x * channels; // 乘以 channels

        for (int i = 0; i < channels; i++)
        {
            // 循环处理每个通道
            dst[dp + i] = static_cast<uint8_t>((1.0f - fx) * (aPtr[sp + i] * (1.0f - fy) + bPtr[sp + i] * fy) + fx * (aPtr[sp_right + i] * (1.0f - fy) + bPtr[sp_right + i] * fy));
        }
    }
}

void GpuResize(uint8_t *image,
               uint8_t *out_image,
               uint32_t src_width,
               uint32_t src_height,
               uint32_t out_width,
               uint32_t out_height,
               uint32_t channels) // 添加 channels 参数
{
    float scale_x = static_cast<float>(src_width) / out_width;
    float scale_y = static_cast<float>(src_height) / out_height;

    constexpr int batch_size = 1;
    constexpr int THREADS_PER_BLOCK = 1024;
    constexpr int BLOCK_WIDTH = 32;

    const dim3 blockSize(BLOCK_WIDTH, THREADS_PER_BLOCK / BLOCK_WIDTH, 1);
    const dim3 gridSize(DivUp(out_width, blockSize.x), DivUp(out_height, blockSize.y), batch_size);

    ResizeOperate_kernel<<<gridSize, blockSize, 0>>>(image, out_image,
                                                     scale_x, scale_y, src_width,
                                                     src_height, out_width, out_height, channels); // 传递 channels 参数

    CHECK_RUN();
}

