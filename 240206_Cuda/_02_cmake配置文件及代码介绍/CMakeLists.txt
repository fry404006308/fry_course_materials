
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
project(fry01Cuda LANGUAGES CXX CUDA)


# =================== Cuda ===================
# 查找CUDA Toolkit
# set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8")
# 手动指定CUDA编译器的路径
# set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc.exe)
find_package(CUDAToolkit REQUIRED)
# =================== End Cuda ===================



# 使用 file(GLOB ...) 来查找所有的 .cpp 和 .cu 源文件
file(GLOB SOURCE_CODE ${CMAKE_CURRENT_SOURCE_DIR}/*.c* ${CMAKE_CURRENT_SOURCE_DIR}/*.h)


# 定义项目中的可执行文件
add_executable(fry01Cuda ${SOURCE_CODE})


# 为可执行文件链接CUDA库，例如cudart
target_link_libraries(fry01Cuda PUBLIC CUDA::cudart)







