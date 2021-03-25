cmake_minimum_required(VERSION 3.17)
project(LearnCUDA3)

#other libs
set(LIBS libnvinfer.so libcudart.so.10.2)
link_directories(/usr/lib/x86_64-linux-gnu/)
link_directories(/usr/local/cuda-10.1/targets/x86_64-linux/lib)
set(CMAKE_CXX_STANDARD 14)
#cuda
include_directories(/usr/local/cuda-10.1/targets/x86_64-linux/include/)
include_directories(/usr/local/cuda/lib64 include)
#tensorRT
include_directories(/usr/include/x86_64-linux-gnu)
#src files
file(GLOB src "src/*.cpp")
add_executable(vgg main.cpp ${src})
target_link_libraries(vgg ${LIBS})
