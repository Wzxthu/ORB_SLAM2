cmake_minimum_required(VERSION 2.8)
project(ORB_SLAM2)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

ADD_DEFINITIONS(-DOPENCV)

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -march=native")

if (CMAKE_BUILD_TYPE MATCHES Release)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV 3.0 HINTS /usr/local/)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(OpenCL REQUIRED)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/include
${PROJECT_SOURCE_DIR}/Thirdparty/DLib/include
${PROJECT_SOURCE_DIR}/Thirdparty/darknet/include
${PROJECT_SOURCE_DIR}/Thirdparty/g2o
${PROJECT_BINARY_DIR}/Thirdparty/g2o
${OpenCV_INCLUDE_DIRS}
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
${OpenCL_INCLUDE_DIR}
)

add_subdirectory(Thirdparty/g2o)
add_subdirectory(Thirdparty/DBoW2)
add_subdirectory(Thirdparty/DLib)
add_subdirectory(Thirdparty/darknet)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
src/System.cc
src/Tracking.cc
src/LocalMapping.cc
src/LoopClosing.cc
src/ORBextractor.cc
src/ORBmatcher.cc
src/FrameDrawer.cc
src/Converter.cc
src/MapPoint.cc
src/KeyFrame.cc
src/Map.cc
src/MapDrawer.cc
src/Optimizer.cc
src/PnPsolver.cc
src/Frame.cc
src/KeyFrameDatabase.cc
src/Sim3Solver.cc
src/Initializer.cc
src/Viewer.cc
src/ObjectDetector.cc
src/Landmark.cc include/LineSegmentDetector.h src/LineSegmentDetector.cc src/CubeSLAM.cc)

link_directories(${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${OpenCL_LIBRARY}
DBoW2
DLib
types_sim3
darknet
)

# Build examples

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D)

add_executable(rgbd_tum
Examples/RGB-D/rgbd_tum.cc)
target_link_libraries(rgbd_tum ${PROJECT_NAME})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Stereo)

add_executable(stereo_kitti
Examples/Stereo/stereo_kitti.cc)
target_link_libraries(stereo_kitti ${PROJECT_NAME})

add_executable(stereo_euroc
Examples/Stereo/stereo_euroc.cc)
target_link_libraries(stereo_euroc ${PROJECT_NAME})


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular)

add_executable(mono_tum
Examples/Monocular/mono_tum.cc)
target_link_libraries(mono_tum ${PROJECT_NAME})

add_executable(mono_kitti
Examples/Monocular/mono_kitti.cc)
target_link_libraries(mono_kitti ${PROJECT_NAME})

add_executable(mono_euroc
Examples/Monocular/mono_euroc.cc)
target_link_libraries(mono_euroc ${PROJECT_NAME})

add_executable(cubeslam_test test/cubeslam_test.cc)
target_link_libraries(cubeslam_test ${PROJECT_NAME})