cmake_minimum_required(VERSION 3.23)
project(Fry02Opencv)


# 使用 file(GLOB ...) 来查找所有的 .c* 和 .h 源文件
file(GLOB SOURCE_CODE ${CMAKE_CURRENT_SOURCE_DIR}/*.c* ${CMAKE_CURRENT_SOURCE_DIR}/*.h)



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
# 用find_package命令指定CMake需要寻找OpenCV，
# 并且必须包含指定的组件
# （calib3d, core, features2d, flann, gapi, highgui, imgcodecs, imgproc, world）。
# 如果这些组件中有任何一个未找到，CMake将会报错。
find_package(OpenCV REQUIRED COMPONENTS calib3d core features2d flann gapi highgui imgcodecs imgproc world)
# 用get_target_property命令获取opencv_world目标的调试版本DLL的位置，
# 并将其存储在变量__dll_dbg中。
get_target_property(__dll_dbg opencv_world IMPORTED_LOCATION_DEBUG)
# 获取opencv_world目标的发布版本DLL的位置，并将其存储在变量__dll_release中。
get_target_property(__dll_release opencv_world  IMPORTED_LOCATION_RELEASE)

# include_directories 就是用来包含头文件路径
# 用include_directories命令为项目添加包含目录
# ${PROJECT_NAME}是当前项目的名字，
# ${CMAKE_BINARY_DIR}是当前项目的二进制目录（通常是构建目录）
# ${OpenCV_INCLUDE_DIRS}是OpenCV头文件的目录。
# PRIVATE指定了这些包含目录的范围，这里只对${PROJECT_NAME}目标有效。
include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_BINARY_DIR}
	${OpenCV_INCLUDE_DIRS} 
)
# =================== Opencv ===================


add_executable(Fry02Opencv ${SOURCE_CODE})


# target_link_libraries 可以链接动态库和静态库
# 链接OpenCV库
target_link_libraries(
	Fry02Opencv
	${OpenCV_LIBS}
    ${OpenCV_LIBRARIES}
)








