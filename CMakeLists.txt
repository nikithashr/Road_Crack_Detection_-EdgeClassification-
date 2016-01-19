cmake_minimum_required(VERSION 2.8)
project( BoW_Slic )
if(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++0x")
endif()
 
find_package( OpenCV REQUIRED )
add_executable( test_bow_slic src/test_bow_slic.cpp src/bow_slic.cpp src/oversegmentation.cpp src/debugging_functions.cpp)
target_link_libraries( test_bow_slic ${OpenCV_LIBS} )
