#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>



// CUDA内核定义用于复制图像的每个通道
__global__ void copyImageToPosition(unsigned char *src, int srcWidth, int srcHeight,
                                    unsigned char *dst, int dstWidth, int dstHeight,
                                    int dstX, int dstY, int channels)
{
    // 计算当前像素点（线程）对应的宽高坐标
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;


    // 因为这里是小于src的宽和高，而src和dst的图像大小是不一样大的 
    if (x < srcWidth && y < srcHeight)
    {
        // 循环计算BGR
        // opencv 中 同一个像素的不同 channel 是连续的
        for (int c = 0; c < channels; ++c)
        {
            // opencv 中 同一个像素的不同 channel 是连续的
            // 所以算位置的时候，要乘上 channel
            int dstIndex = ((y + dstY) * dstWidth + (x + dstX)) * channels + c;
            int srcIndex = (y * srcWidth + x) * channels + c;

            // 检查目标位置是否在目标图像的边界内
            if ((x + dstX) < dstWidth && (y + dstY) < dstHeight)
            {
                dst[dstIndex] = src[srcIndex];
            }
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

    // 创建源图像和目标图像（假设都是3通道的彩色图像）
    cv::Mat srcImage(srcHeight, srcWidth, CV_8UC3, cv::Scalar(255, 198, 0)); // 图像
    cv::Mat dstImage(dstHeight, dstWidth, CV_8UC3, cv::Scalar(0, 128, 255));       // 图像

    std::string saveDir = std::string(OUTPUT_IMAGE_DIR);

    // 使用OpenCV显示和保存图像
    cv::namedWindow("Source Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Source Image", srcImage);
    cv::imwrite(saveDir + "/source_image.png", srcImage);

    cv::namedWindow("Destination Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Destination Image", dstImage);
    cv::imwrite(saveDir + "/destination_image.png", dstImage);

    // 获取图像的通道数
    assert(srcImage.channels() == dstImage.channels());
    int channels = srcImage.channels();

    // 分配设备内存
    unsigned char *d_src, *d_dst;
    cudaMalloc((void **)&d_src, srcWidth * srcHeight * channels * sizeof(unsigned char));
    cudaMalloc((void **)&d_dst, dstWidth * dstHeight * channels * sizeof(unsigned char));

    // 将初始化的数据拷贝到设备上
    cudaMemcpy(d_src, srcImage.data, srcWidth * srcHeight * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst, dstImage.data, dstWidth * dstHeight * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // 定义线程块的大小和网格的大小
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((srcWidth + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (srcHeight + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // 调用内核函数
    copyImageToPosition<<<numBlocks, threadsPerBlock>>>(d_src, srcWidth, srcHeight,
                                                        d_dst, dstWidth, dstHeight,
                                                        dstX, dstY, channels);

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
    // 结果不正确是这边拷贝结果数据的时候，只拷贝了部分
    cudaMemcpy(dstImage.data, d_dst, dstWidth * dstHeight * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

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