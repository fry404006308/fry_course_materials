#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <ctime>

int main()
{
    // 图片的宽度和高度
    int width = 640;
    int height = 480;

    // 创建一个空白的Mat对象，用于存储图片
    cv::Mat image = cv::Mat::zeros(height, width, CV_8UC3);

    // 生成随机颜色的背景
    cv::RNG rng(time(0)); // 随机数生成器
    image.setTo(cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)));

    // 在图片上绘制一个矩形
    cv::rectangle(image,
                  cv::Point(50, 50),     // 矩形的左上角坐标
                  cv::Point(200, 200),   // 矩形的右下角坐标
                  cv::Scalar(0, 255, 0), // 绿色
                  3,                     // 线的粗细
                  cv::LINE_8);

    // 在图片上绘制一个圆
    cv::circle(image,
               cv::Point(320, 240),   // 圆心坐标
               100,                   // 半径
               cv::Scalar(255, 0, 0), // 蓝色
               3,                     // 线的粗细
               cv::LINE_AA);

    // 在图片上绘制一条线
    cv::line(image,
             cv::Point(400, 100),   // 起点坐标
             cv::Point(550, 100),   // 终点坐标
             cv::Scalar(0, 0, 255), // 红色
             3,                     // 线的粗细
             cv::LINE_8);

    // 在图片上添加文字
    std::string text = "FRY OpenCV with C++";
    int fontFace = cv::FONT_HERSHEY_COMPLEX;
    double fontScale = 1.0;
    int thickness = 2;

    // 获取文本框的大小
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);
    // 将文本框置于图片中央
    cv::Point textOrg((image.cols - textSize.width) / 2, (image.rows + textSize.height) / 2);

    // 绘制文字
    cv::putText(image, text, textOrg, fontFace, fontScale, cv::Scalar(255, 255, 255), thickness);

    // 保存图片
    std::string saveDir = std::string(OUTPUT_IMAGE_DIR);
    std::string filename = saveDir + "/output_image.jpg";
    cv::imwrite(filename, image);

    // 也可以展示图片
    cv::imshow("Generated Image", image);
    cv::waitKey(0); // 等待任意键按下

    return 0;
}