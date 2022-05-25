project(PlayGround)
cmake_minimum_required(VERSION 3.23)



find_path(ONNX_RUNTIME_SESSION_INCLUDE_DIRS onnxruntime_cxx_api.h)
find_library(ONNX_RUNTIME_LIB onnxruntime)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable(playground main.cpp)
set_property(TARGET playground PROPERTY CXX_STANDARD 17)
target_include_directories(playground PRIVATE ${OpenCV_INCLUDE_DIRS} ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS}) # 
target_link_libraries(playground PRIVATE opencv_core opencv_dnn ${OpenCV_LIBRARIES} ${ONNX_RUNTIME_LIB}) # 

