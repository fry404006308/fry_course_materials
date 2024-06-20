// matrix_multiply.cu
#include <iostream>
// #include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

// opencv
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/types_c.h"

// CUDA kernel for matrix multiplication
// 用于矩阵乘法的 CUDA 内核函数
// 参数:
//   A: 输入矩阵 A 的设备内存指针
//   B: 输入矩阵 B 的设备内存指针
//   C: 输出矩阵 C 的设备内存指针
//   rowsA: 矩阵 A 的行数
//   colsA: 矩阵 A 的列数
//   colsB: 矩阵 B 的列数
__global__ void matrixMultiplyKernel(float *A, float *B, float *C, int rowsA, int colsA, int colsB)
{
    // 当前核函数是以结果矩阵C矩阵为基础来实现的
    // 相当于我们在结果矩阵上面找到要计算的元素
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 检查当前线程所处理的矩阵 C 的元素是否在有效范围内
    if (row < rowsA && col < colsB)
    {
        // 初始化当前线程所计算的矩阵 C 的元素的值为 0
        float value = 0;
        // 对矩阵 A 的列和矩阵 B 的行进行点积计算
        // 这里是一个核函数计算了A矩阵的一行 乘以 B 矩阵的一列 的结果
        // 因为结果矩阵 C 的每个位置需要的计算就是A一行乘以B一列
        for (int k = 0; k < colsA; ++k)
        {
            // 累加矩阵 A 的第 row 行和矩阵 B 的第 col 列的对应元素的乘积
            // row * colsA + k 代表矩阵 A 的第 row 行的第 k 列元素的索引
            // k * colsB + col 代表矩阵 B 的第 col 列的第 k 行元素的索引
            value += A[row * colsA + k] * B[k * colsB + col];
        }
        // 将计算得到的值存储到输出矩阵 C 的对应位置
        C[row * colsB + col] = value;
    }
}

// Function to perform matrix multiplication using CUDA
void matrixMultiplyCUDA(cv::Mat &A, cv::Mat &B, cv::Mat &C)
{
    int rowsA = A.rows;
    int colsA = A.cols;
    int rowsB = B.rows;
    int colsB = B.cols;

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, rowsA * colsA * sizeof(float));
    cudaMalloc(&d_B, rowsB * colsB * sizeof(float));
    cudaMalloc(&d_C, rowsA * colsB * sizeof(float));

    // Copy input matrices from host to device
    cudaMemcpy(d_A, A.ptr<float>(), rowsA * colsA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.ptr<float>(), rowsB * colsB * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockDim(16, 16);
    // 对于AxB矩阵乘法而言，结果C矩阵的宽高分别为colsB（B的列数）和rowsA（A的行数）
    // A（3行2列）x B（2行4列）=C（3行4列）（B的列数colsB x A的行数rowsA）
    // x 方向 对象的是矩阵的列
    // y 方向 对象的是矩阵的行
    dim3 gridDim((colsB + blockDim.x - 1) / blockDim.x, (rowsA + blockDim.y - 1) / blockDim.y);

    // Launch CUDA kernel
    matrixMultiplyKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, rowsA, colsA, colsB);

    // Copy result matrix from device to host
    cudaMemcpy(C.ptr<float>(), d_C, rowsA * colsB * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main()
{
    // Create input matrices with random values
    cv::Mat A(3, 2, CV_32F); // 3是行，2是列
    cv::Mat B(2, 4, CV_32F);
    cv::randu(A, cv::Scalar(0), cv::Scalar(10));
    cv::randu(B, cv::Scalar(0), cv::Scalar(10));

    std::cout << "A:" << std::endl;
    std::cout << A << std::endl;
    std::cout << "B:" << std::endl;
    std::cout << B << std::endl;
    std::cout << std::endl;

    // Perform matrix multiplication using CUDA
    cv::Mat C_cuda(3, 4, CV_32F);
    matrixMultiplyCUDA(A, B, C_cuda);

    // Print the contents of C_cuda
    std::cout << "C_cuda:" << std::endl;
    std::cout << C_cuda << std::endl;
    std::cout << std::endl;

    // Perform matrix multiplication using OpenCV
    cv::Mat C_opencv = A * B;

    // Print the contents of C_opencv
    std::cout << "C_opencv:" << std::endl;
    std::cout << C_opencv << std::endl;
    std::cout << std::endl;

    // Compare results
    double diff = cv::norm(C_cuda - C_opencv);
    std::cout << "Difference between CUDA and OpenCV results: " << diff << std::endl;

    return 0;
}


/*

A:
[5.3028278, 1.9925919;
 4.0105944, 8.1438503;
 4.3713298, 2.4878969]
B:
[7.7310505, 7.6209373, 3.0779448, 7.0243168;
 4.784472, 7.9219003, 0.85843134, 0.75060272]

C_cuda:
[50.52993, 56.197632, 18.032314, 38.744385;
 69.970131, 95.079254, 19.335325, 34.284481;
 45.698246, 53.022499, 15.590401, 32.573029]

C_opencv:
[50.52993, 56.197632, 18.032314, 38.744389;
 69.970131, 95.079262, 19.335323, 34.284481;
 45.698246, 53.022503, 15.590401, 32.573029]

Difference between CUDA and OpenCV results: 9.53674e-06

D:\_230711_learnArchive\fry_course_materials\240206_Cuda\_14_matrix_multiplication\build\Debug\fry14MatrixMultiplication.exe (进程 35676)已退出，代码为 0。
要在调试停止时自动关闭控制台，请启用“工具”->“选项”->“调试”->“调试停止时自动关闭控制台”。
按任意键关闭此窗口. . .

*/



/*

GPU 和 CPU 计算结果之间的轻微差异通常是由于浮点数表示和运算的特性导致的。

在计算机中,浮点数是以近似值的形式存储和处理的。浮点数的表示方式遵循 IEEE 754 标准,但是在不同的硬件架构(如 CPU 和 GPU)上,浮点数的实现可能会有细微的差异。这些差异可能来自以下几个方面:

1. 舍入误差: 浮点数运算通常涉及舍入操作,将结果舍入到最接近的可表示数。不同的硬件架构可能使用略有不同的舍入策略,导致结果的轻微差异。

2. 运算顺序: 浮点数运算不满足结合律,这意味着运算的顺序可能会影响结果。在并行计算环境中,如 GPU,运算的顺序可能与 CPU 上的顺序不同,从而导致结果的差异。

3. 优化和近似: 为了提高性能,一些硬件架构可能会对某些浮点数运算进行优化或近似处理。这可能导致结果与严格的数学计算略有不同。

4. 编译器优化: 不同的编译器或编译选项可能会对浮点数运算应用不同的优化策略,导致结果的差异。

在你的示例中,GPU 和 CPU 计算结果之间的差异非常小,大约为 9.53674e-06。这个差异在大多数实际应用中是可以接受的,因为它远远小于浮点数的有效位数。

然而,在某些对精度要求非常高的应用中,如科学计算或金融领域,即使是这样小的差异也可能是重要的。在这种情况下,可以采取以下措施来最小化差异:

1. 使用更高精度的浮点数类型,如 double 或 long double。
2. 仔细检查并调整算法,尽量减少舍入误差的累积。
3. 在 GPU 和 CPU 上使用相同的编译器和编译选项,以确保一致性。
4. 必要时,可以使用软件实现的高精度数学库,以确保结果的一致性。

总的来说,GPU 和 CPU 之间的轻微差异是由硬件和软件实现的差异所导致的,在大多数情况下是可以接受的。但是,在对精度要求严格的应用中,需要采取适当的措施来最小化这些差异。
*/







