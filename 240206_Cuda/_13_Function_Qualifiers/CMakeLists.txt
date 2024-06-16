cmake_minimum_required(VERSION 3.23)


# 这句话要放到project指令前，否则不生效
# 设置CUDA目标架构，这句话需要在 add_executable 之前
set(CMAKE_CUDA_ARCHITECTURES "60;61;62;70;72;75;80;86;89")

#增加-std=c++17
set(CMAKE_CXX_STANDARD 17)


# 显式设置 CMP0144 策略：你可以在 CMakeLists.txt 文件中显式设置这个策略，
# 来告诉 CMake 应该如何处理 <PACKAGENAME>_ROOT 变量。
cmake_policy(SET CMP0144 NEW)


# 表示支持 C++和cuda语言
project(fry13FunctionQualifiers LANGUAGES CXX CUDA)




# =================== Cuda ===================
# 查找CUDA Toolkit
find_package(CUDAToolkit REQUIRED)
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5")
# =================== End Cuda ===================

# ================== 设置输出目录相关 ==================
# 设置输出目录的变量
set(OUTPUT_IMAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output_images")
# 创建输出目录
file(MAKE_DIRECTORY ${OUTPUT_IMAGE_DIR})
# 在CMakeLists.txt文件中，直接将路径添加为宏定义：
add_definitions(-DOUTPUT_IMAGE_DIR="${OUTPUT_IMAGE_DIR}")

# 代码中使用： std::string saveDir = std::string(OUTPUT_IMAGE_DIR);
# ================== END 设置输出目录相关 ==================

# =================== Opencv ===================
# 寻找OpenCV库
# 设置 OpenCV_DIR 为后面的find_package指定了opencv的查找路径
set(OpenCV_DIR "D:/_200_software/code/_240617_Opencv4_10/opencv/build")
# find_package(OpenCV REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS calib3d core features2d flann gapi highgui imgcodecs imgproc world)
get_target_property(__dll_dbg opencv_world IMPORTED_LOCATION_DEBUG)
get_target_property(__dll_release opencv_world  IMPORTED_LOCATION_RELEASE)

# include_directories 就是头文件路径
include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_BINARY_DIR}
	${OpenCV_INCLUDE_DIRS} 
)
# =================== Opencv ===================



# aux_source_directory(./ SRC)
file(GLOB SOURCE_CODE ${CMAKE_CURRENT_SOURCE_DIR}/*.c*)

add_executable(fry13FunctionQualifiers ${SOURCE_CODE})

# target_link_libraries 可以链接动态库和静态库
# 链接OpenCV库
target_link_libraries(
	fry13FunctionQualifiers
	${OpenCV_LIBS}
    ${OpenCV_LIBRARIES}
)

















