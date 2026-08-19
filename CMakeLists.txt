cmake_minimum_required(VERSION 3.18)
project(xfeat_lightglue LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 61 75 86 CACHE STRING "CUDA architectures")
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs calib3d)
find_package(CUDAToolkit REQUIRED)
find_package(yaml-cpp REQUIRED)

set(TENSORRT_ROOT "" CACHE PATH "TensorRT installation prefix")
set(_TENSORRT_HINTS
  ${TENSORRT_ROOT}
  $ENV{TENSORRT_ROOT}
  /usr/local/TensorRT-8.6.1.6
  /usr/local/TensorRT
  /home/xudong/thirdparty/TensorRT-8.6.1.6
)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS ${_TENSORRT_HINTS} PATH_SUFFIXES include)
find_library(TENSORRT_NVINFER_LIBRARY nvinfer HINTS ${_TENSORRT_HINTS} PATH_SUFFIXES lib lib64)
find_library(TENSORRT_PLUGIN_LIBRARY nvinfer_plugin HINTS ${_TENSORRT_HINTS} PATH_SUFFIXES lib lib64)
if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_NVINFER_LIBRARY OR NOT TENSORRT_PLUGIN_LIBRARY)
  message(FATAL_ERROR "TensorRT was not found. Set -DTENSORRT_ROOT=/path/to/TensorRT")
endif()

add_library(xfeat_libs SHARED
  src/xfeat.cpp
  src/xfeat_cuda.cu
)
target_include_directories(xfeat_libs
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${TENSORRT_INCLUDE_DIR}
)
target_compile_options(xfeat_libs PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)
target_link_libraries(xfeat_libs
  PUBLIC ${OpenCV_LIBS}
  PRIVATE CUDA::cudart yaml-cpp::yaml-cpp ${TENSORRT_NVINFER_LIBRARY} ${TENSORRT_PLUGIN_LIBRARY}
)

add_library(lightglue_libs SHARED src/lightglue.cpp)
target_include_directories(lightglue_libs
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${TENSORRT_INCLUDE_DIR}
)
target_compile_options(lightglue_libs PRIVATE -O3 -Wall -Wextra)
target_link_libraries(lightglue_libs
  PUBLIC ${OpenCV_LIBS}
  PRIVATE CUDA::cudart yaml-cpp::yaml-cpp ${TENSORRT_NVINFER_LIBRARY} ${TENSORRT_PLUGIN_LIBRARY}
)

add_executable(match_test src/main.cpp)
target_compile_options(match_test PRIVATE -O3 -Wall -Wextra)
target_link_libraries(match_test PRIVATE xfeat_libs lightglue_libs)

get_filename_component(TENSORRT_LIBRARY_DIR "${TENSORRT_NVINFER_LIBRARY}" DIRECTORY)
set_target_properties(xfeat_libs lightglue_libs match_test PROPERTIES
  BUILD_RPATH "${TENSORRT_LIBRARY_DIR}"
)
