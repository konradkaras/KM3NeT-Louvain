cmake_minimum_required(VERSION 3.6)
find_package(CUDA REQUIRED)
find_package(PythonInterp 3.6 REQUIRED)
find_package(PythonLibs 3.6 REQUIRED)

include_directories(
    ${PYTHON_INCLUDE_DIRS}
    ~/pybind11/include
    )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

cuda_add_library(gpu_louvain SHARED  gpu_louvain.cpp louvain_kernel.cu)
target_link_libraries(gpu_louvain ${PYTHON_LIBRARIES} ${CUDA_LIBRARIES})
set_target_properties(gpu_louvain PROPERTIES PREFIX "")