
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

// opencv
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/types_c.h"


# include "resize.cuh"

void CreateC3ColorfulImg(cv::Mat &ans_img, int width, int height)
{

    assert(width > 0 && "创建图片的宽必须大于0");
    assert(height > 0 && "创建图片的高必须大于0");
    assert(ans_img.empty() && "初始图片必须为空");

    ans_img = cv::Mat(height, width, CV_8UC3);

    // 创建颜色渐变背景
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            ans_img.at<cv::Vec3b>(y, x)[0] = (x / 3) % 255;       // B通道
            ans_img.at<cv::Vec3b>(y, x)[1] = (y / 2) % 255;       // G通道
            ans_img.at<cv::Vec3b>(y, x)[2] = ((x + y) / 4) % 255; // R通道
        }
    }

    int width_1_2 = static_cast<int>(width / 2);
    int width_1_3 = static_cast<int>(width / 3);
    int width_1_4 = static_cast<int>(width / 4);
    int width_1_5 = static_cast<int>(width / 5);
    int width_1_6 = static_cast<int>(width / 6);
    int width_1_7 = static_cast<int>(width / 7);
    int width_1_8 = static_cast<int>(width / 8);

    int height_1_2 = static_cast<int>(height / 2);
    int height_1_3 = static_cast<int>(height / 3);
    int height_1_4 = static_cast<int>(height / 4);
    int height_1_5 = static_cast<int>(height / 5);
    int height_1_6 = static_cast<int>(height / 6);
    int height_1_7 = static_cast<int>(height / 7);
    int height_1_8 = static_cast<int>(height / 8);

    // 画矩形
    cv::rectangle(ans_img, cv::Rect(width_1_8, height_1_8, width_1_2, height_1_2), cv::Scalar(255, 0, 255), -1);

    // 画圆形
    cv::circle(ans_img, cv::Point(width_1_2, height_1_2), std::min(width_1_3, height_1_3), cv::Scalar(0, 255, 255), 2);

    // 画三角形
    std::vector<cv::Point> triangle;
    triangle.push_back(cv::Point(width_1_3, height_1_3));
    triangle.push_back(cv::Point(width_1_3 * 2, height_1_3 * 2));
    triangle.push_back(cv::Point(width_1_3, height_1_3 * 2));
    cv::fillConvexPoly(ans_img, triangle, cv::Scalar(255, 255, 0));

    // 在图片中间写字
    cv::putText(ans_img, "Hello, FanRenyi OpenCV!", cv::Point(width_1_5, height_1_5), cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(255, 255, 255), 2);
}

void GpuUchar2MatUchar(uint8_t *gpu_data, cv::Mat &ans_mat, int dst_w, int dst_h, int dst_channel)
{
    assert(ans_mat.empty()), "ans_mat 必须为空";
    if (dst_channel == 3)
    {
        ans_mat = cv::Mat(dst_h, dst_w, CV_8UC3);
    }
    else if (dst_channel == 1)
    {
        ans_mat = cv::Mat(dst_h, dst_w, CV_8UC1);
    }
    else
    {
        throw std::runtime_error("channels can only be 1 or 3");
    }
    int total_size = dst_w * dst_h * dst_channel * sizeof(uint8_t);
    CHECK_CUDA_ERROR(cudaMemcpyAsync(ans_mat.data, gpu_data, total_size, cudaMemcpyDeviceToHost));
}

void CompareTwoMats(const cv::Mat& mat1, const cv::Mat& mat2) {
    // 比较宽、高、通道数的差异
    if (mat1.cols != mat2.cols || mat1.rows != mat2.rows || mat1.channels() != mat2.channels()) {
        std::cout << "两个Mat的尺寸或通道数不同!" << std::endl;
        std::cout << "Mat1: 宽: " << mat1.cols << ", 高: " << mat1.rows << ", 通道数: " << mat1.channels() << std::endl;
        std::cout << "Mat2: 宽: " << mat2.cols << ", 高: " << mat2.rows << ", 通道数: " << mat2.channels() << std::endl;
        return;
    }

    // 如果宽、高、通道数都一模一样,则比较每个像素点的差异
    cv::Mat diff;
    cv::absdiff(mat1, mat2, diff);
    double max_val, min_val;
    cv::minMaxLoc(diff, &min_val, &max_val);
    std::cout << "像素点差异的最小值: " << min_val << ", 最大值: " << max_val << std::endl;

    // 统计每个通道的均值,并计算两个Mat每个通道均值的差异
    std::vector<cv::Mat> channels1, channels2;
    cv::split(mat1, channels1);
    cv::split(mat2, channels2);
    for (int i = 0; i < mat1.channels(); ++i) {
        double mean1 = cv::mean(channels1[i])[0];
        double mean2 = cv::mean(channels2[i])[0];
        double mean_diff = std::abs(mean1 - mean2);
        std::cout << "通道 " << i << " 的均值差异: " << mean_diff << std::endl;
        std::cout << "mean1: " << mean1 << std::endl;
        std::cout << "mean2: " << mean2 << std::endl;
    }

    // 算出每个Mat像素点值的总和,然后比较两个Mat的总和差异并求出差异的均值
    double sum1 = cv::sum(mat1)[0];
    double sum2 = cv::sum(mat2)[0];
    double sum_diff = std::abs(sum1 - sum2);
    double mean_sum_diff = sum_diff / (mat1.cols * mat1.rows * mat1.channels());
    std::cout << "sum1: " << sum1 << std::endl;
    std::cout << "sum2: " << sum2 << std::endl;
    std::cout << "像素点值总和的差异: " << sum_diff << ", 差异的均值: " << mean_sum_diff << std::endl;
}

int main()
{

    std::string saveDir = std::string(OUTPUT_IMAGE_DIR);


    // 原始图像的宽度、高度和通道数
    const uint32_t srcWidth = 1920;
    const uint32_t srcHeight = 1080;
    const uint32_t channels = 3;

    // 调整后图像的宽度和高度
    const uint32_t outWidth = 800;
    const uint32_t outHeight = 600;

    // 创建SRC图像
    cv::Mat srcImg;
    CreateC3ColorfulImg(srcImg, srcWidth, srcHeight);

    // 使用OpenCV显示和保存图像
    cv::namedWindow("Source Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Source Image", srcImg);
    cv::imwrite(saveDir + "/source_image.png", srcImg);




    // ==================================================
    // Part1、 GPU Resize
    // ==================================================

    // 分配 GPU 内存
    uint8_t *gpuSrcImage;
    uint8_t *gpuOutImage;
    CHECK_CUDA_ERROR(cudaMalloc(&gpuSrcImage, srcWidth * srcHeight * channels * sizeof(uint8_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&gpuOutImage, outWidth * outHeight * channels * sizeof(uint8_t)));

    // 将原始图像数据从 CPU 复制到 GPU
    CHECK_CUDA_ERROR(cudaMemcpy(gpuSrcImage, srcImg.data, srcWidth * srcHeight * channels * sizeof(uint8_t), cudaMemcpyHostToDevice));

    // 调用 GpuResize 函数进行图像缩放
    GpuResize(gpuSrcImage, gpuOutImage, srcWidth, srcHeight, outWidth, outHeight, channels);

    // GPU转回CPU
    cv::Mat gpuResizedMat;
    GpuUchar2MatUchar(gpuOutImage, gpuResizedMat, outWidth, outHeight, channels);

    cv::namedWindow("gpuResizedMat", cv::WINDOW_AUTOSIZE);
    cv::imshow("gpuResizedMat", gpuResizedMat);
    cv::imwrite(saveDir + "/gpu_resized_mat.png", gpuResizedMat);

    // 释放 GPU 内存
    cudaFree(gpuSrcImage);
    cudaFree(gpuOutImage);


    // ==================================================
    // Part2、 比较GPU和CPU的结果
    // ==================================================
    cv::Mat cpuResizedMat;
    cv::resize(srcImg, cpuResizedMat, cv::Size(outWidth, outHeight), 0, 0,
        cv::INTER_LINEAR);
    CompareTwoMats(cpuResizedMat, gpuResizedMat);
    cv::namedWindow("cpuResizedMat", cv::WINDOW_AUTOSIZE);
    cv::imshow("cpuResizedMat", cpuResizedMat);
    cv::imwrite(saveDir + "/cpu_resized_mat.png", cpuResizedMat);


    // 显示结果
    cv::waitKey(0);


    return 0;
}
