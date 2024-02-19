cmake_minimum_required(VERSION 3.23)
project(Fry02OpencvTest)


# 使用 file(GLOB ...) 来查找所有的 .cpp 和 .cu 源文件
file(GLOB SOURCE_CODE ${CMAKE_CURRENT_SOURCE_DIR}/*.c* ${CMAKE_CURRENT_SOURCE_DIR}/*.h)



add_executable(Fry02OpencvTest ${SOURCE_CODE})


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
set(OpenCV_DIR C:/230912_software/_300_code/231024_opencv470/opencv/build)
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











