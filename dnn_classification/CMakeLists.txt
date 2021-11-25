cmake_minimum_required(VERSION 2.8)

project( DNN_CLASSIFICATION )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( DNN_CLASSIFICATION main.cpp )
target_link_libraries( DNN_CLASSIFICATION ${OpenCV_LIBS} )