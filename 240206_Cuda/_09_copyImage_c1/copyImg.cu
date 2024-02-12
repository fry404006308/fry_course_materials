#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

// CUDA内核定义
__global__ void copyImageToPosition(unsigned char *src, int srcWidth, int srcHeight,
                                    unsigned char *dst, int dstWidth, int dstHeight,
                                    int dstX, int dstY)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < srcWidth && y < srcHeight)
    {
        int dstIndex = (y + dstY) * dstWidth + (x + dstX);
        int srcIndex = y * srcWidth + x;
        if (y + dstY < dstHeight && x + dstX < dstWidth)
        {
            dst[dstIndex] = src[srcIndex];
        }
    }
}

int main()
{
    // 图像尺寸和位置
    int srcWidth = 640;
    int srcHeight = 480;
    int dstWidth = 1024;
    int dstHeight = 768;
    int dstX = 200; // 目标位置的X坐标
    int dstY = 100; // 目标位置的Y坐标

    // 创建源图像和目标图像
    cv::Mat srcImage(srcHeight, srcWidth, CV_8UC1, cv::Scalar(255)); // 白色图像
    cv::Mat dstImage(dstHeight, dstWidth, CV_8UC1, cv::Scalar(0));   // 黑色图像

    std::string saveDir = std::string(OUTPUT_IMAGE_DIR);

    // 使用OpenCV显示和保存图像
    cv::namedWindow("Source Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Source Image", srcImage);
    cv::imwrite(saveDir + "/source_image.png", srcImage);

    cv::namedWindow("Destination Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Destination Image", dstImage);
    cv::imwrite(saveDir + "/destination_image.png", dstImage);

    // 分配设备内存
    unsigned char *d_src, *d_dst;
    cudaMalloc((void **)&d_src, srcWidth * srcHeight * sizeof(unsigned char));
    cudaMalloc((void **)&d_dst, dstWidth * dstHeight * sizeof(unsigned char));

    // 将初始化的数据拷贝到设备上
    cudaMemcpy(d_src, srcImage.data, srcWidth * srcHeight * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst, dstImage.data, dstWidth * dstHeight * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // 定义线程块的大小和网格的大小
    // 每个线程处理图像的一个像素点，
    // 图像的宽方向基本单位是16个线程（也就是每次能处理16个像素点），高方向基础单位是16个线程
    // 所以宽方向需要的总的blockk数为  (srcWidth + threadsPerBlock.x - 1) / threadsPerBlock.x
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((srcWidth + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (srcHeight + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // 调用内核函数
    copyImageToPosition<<<numBlocks, threadsPerBlock>>>(d_src, srcWidth, srcHeight,
                                                        d_dst, dstWidth, dstHeight,
                                                        dstX, dstY);

    // 等待CUDA操作完成
    cudaDeviceSynchronize();

    // 检查CUDA操作是否有错误
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
        // 在这里可以进行错误处理
    }

    // 将结果拷贝回主机内存
    cudaMemcpy(dstImage.data, d_dst, dstWidth * dstHeight, cudaMemcpyDeviceToHost);

    // 清理设备内存
    cudaFree(d_src);
    cudaFree(d_dst);

    cv::namedWindow("Result Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Result Image", dstImage);
    cv::imwrite(saveDir + "/result_image.png", dstImage);

    // 等待按键然后退出
    cv::waitKey(0);

    return 0;
}