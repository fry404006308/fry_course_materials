#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <random>

void CreateC3ColorfulImg_random(cv::Mat &ans_img, int width, int height)
{
    assert(width > 0 && "创建图片的宽必须大于0");
    assert(height > 0 && "创建图片的高必须大于0");
    assert(ans_img.empty() && "初始图片必须为空");

    // 创建随机数生成器
    std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr)));
    std::uniform_int_distribution<int> color_dist(0, 255);
    std::uniform_real_distribution<float> pos_dist(0.1f, 0.9f);

    ans_img = cv::Mat(height, width, CV_8UC3);

    // 创建随机颜色渐变背景
    cv::Scalar start_color(color_dist(rng), color_dist(rng), color_dist(rng));
    cv::Scalar end_color(color_dist(rng), color_dist(rng), color_dist(rng));

    for (int y = 0; y < height; y++)
    {
        float t = static_cast<float>(y) / height;
        cv::Scalar color = start_color * (1 - t) + end_color * t;
        for (int x = 0; x < width; x++)
        {
            ans_img.at<cv::Vec3b>(y, x) = cv::Vec3b(color[0], color[1], color[2]);
        }
    }

    // 随机生成形状的位置和大小
    int rect_x = static_cast<int>(width * pos_dist(rng));
    int rect_y = static_cast<int>(height * pos_dist(rng));
    int rect_width = static_cast<int>(width * pos_dist(rng) / 2);
    int rect_height = static_cast<int>(height * pos_dist(rng) / 2);

    int circle_x = static_cast<int>(width * pos_dist(rng));
    int circle_y = static_cast<int>(height * pos_dist(rng));
    int circle_radius = std::min(width, height) / 6 + color_dist(rng) % (std::min(width, height) / 6);

    std::vector<cv::Point> triangle;
    for (int i = 0; i < 3; i++)
    {
        triangle.push_back(cv::Point(static_cast<int>(width * pos_dist(rng)),
                                     static_cast<int>(height * pos_dist(rng))));
    }

    // 绘制随机颜色的形状
    cv::rectangle(ans_img, cv::Rect(rect_x, rect_y, rect_width, rect_height),
                  cv::Scalar(color_dist(rng), color_dist(rng), color_dist(rng)), -1);

    cv::circle(ans_img, cv::Point(circle_x, circle_y), circle_radius,
               cv::Scalar(color_dist(rng), color_dist(rng), color_dist(rng)), 2);

    cv::fillConvexPoly(ans_img, triangle,
                       cv::Scalar(color_dist(rng), color_dist(rng), color_dist(rng)));

    // 随机生成文本内容和位置
    std::vector<std::string> texts = {"Hello, OpenCV!", "Random Image", "AI Generated", "Unique Design", "Creative Art"};
    std::string text = texts[rng() % texts.size()];
    int text_x = static_cast<int>(width * pos_dist(rng));
    int text_y = static_cast<int>(height * pos_dist(rng));

    cv::putText(ans_img, text, cv::Point(text_x, text_y),
                cv::FONT_HERSHEY_SIMPLEX, 1.0,
                cv::Scalar(color_dist(rng), color_dist(rng), color_dist(rng)), 2);
}

int main()
{
    try
    {
        // 检查CUDA是否可用
        std::cout << "CUDA是否可用: " << (torch::cuda::is_available() ? "是" : "否") << std::endl;

        // 设置设备(如果CUDA可用,使用GPU,否则使用CPU)
        torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

        // 加载图像
        // std::string image_path = "path/to/your/image.jpg"; // 请替换为您的图像路径
        // cv::Mat image = cv::imread(image_path);
        cv::Mat image;

        CreateC3ColorfulImg_random(image, 400, 300);

        // 保存原图
        std::string input_path = OUTPUT_IMAGE_DIR + std::string("/input_image.jpg");
        cv::imwrite(input_path, image);

        std::cout << "输入图像已保存到: " << input_path << std::endl;

        if (image.empty())
        {
            std::cerr << "无法加载图像" << std::endl;
            return -1;
        }

        // 将OpenCV Mat转换为LibTorch张量
        torch::Tensor tensor_image = torch::from_blob(image.data, {image.rows, image.cols, 3}, torch::kByte);
        tensor_image = tensor_image.permute({2, 0, 1}).to(torch::kFloat).div(255).to(device);

        // 转换为灰度图
        torch::Tensor gray_image = 0.299 * tensor_image[0] + 0.587 * tensor_image[1] + 0.114 * tensor_image[2];

        // 将张量转回CPU(如果在GPU上)并转换为OpenCV Mat
        gray_image = gray_image.to(torch::kCPU).mul(255).to(torch::kByte);
        cv::Mat output_image(gray_image.size(0), gray_image.size(1), CV_8UC1, gray_image.data_ptr());

        // 保存结果图像
        std::string output_path = OUTPUT_IMAGE_DIR + std::string("/output_gray_image.jpg");
        cv::imwrite(output_path, output_image);

        std::cout << "灰度图像已保存到: " << output_path << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "发生错误: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}