cmake_minimum_required(VERSION 3.23)

# 设置CUDA目标架构
set(CMAKE_CUDA_ARCHITECTURES "60;61;62;70;72;75;80;86;89")

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)

# 设置CMP0144策略
cmake_policy(SET CMP0144 NEW)

# 项目名称和支持的语言
project(fry_libtorch_02 LANGUAGES CXX CUDA)

# 设置输出目录
set(OUTPUT_IMAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output_images")
file(MAKE_DIRECTORY ${OUTPUT_IMAGE_DIR})
add_definitions(-DOUTPUT_IMAGE_DIR="${OUTPUT_IMAGE_DIR}")

# CUDA设置
find_package(CUDAToolkit REQUIRED)
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8")

# OpenCV设置
set(OpenCV_DIR "C:/_240813_software/320_code/_240818_Opencv_410/opencv/build")
find_package(OpenCV REQUIRED COMPONENTS calib3d core features2d flann gapi highgui imgcodecs imgproc world)

# LibTorch设置
set(CMAKE_PREFIX_PATH "C:/_240813_software/320_code/_240818_LibTorch/libtorch-win-shared-with-deps-debug-2.4.0+cu118/libtorch")
find_package(Torch REQUIRED)

# 包含目录
include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_BINARY_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

# 源文件
file(GLOB SOURCE_CODE ${CMAKE_CURRENT_SOURCE_DIR}/*.c*)

# 可执行文件
add_executable(fry_libtorch_02 ${SOURCE_CODE})

# 链接库
target_link_libraries(fry_libtorch_02
    ${OpenCV_LIBS}
    ${OpenCV_LIBRARIES}
    ${TORCH_LIBRARIES}
)
 
# 为CUDA设置特定的编译选项
set_property(TARGET fry_libtorch_02 PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# 如果使用MSVC编译器，设置额外的编译选项
if(MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET fry_libtorch_02
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:fry_libtorch_02>)
endif(MSVC)







