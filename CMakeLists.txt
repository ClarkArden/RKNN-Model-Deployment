cmake_minimum_required(VERSION 3.4.1)

set(PROJECT_NAME "rknn_model")
set(SDK_ROOT_PATH "/home/clark/rk3588/rk3588_linux_release")
set(RKNPU_PATH ${SDK_ROOT_PATH}/external/rknpu2)

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/include)

project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# skip 3rd-party lib dependencies
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/${PROJECT_NAME})
set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

# rknn api
set(RKNN_API_PATH ${RKNPU_PATH}/runtime/${CMAKE_SYSTEM_NAME}/librknn_api)
set(LIB_ARCH aarch64)
set(RKNN_RT_LIB ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so)
include_directories(${RKNN_API_PATH}/include)
include_directories(${RKNPU_PATH}/examples/3rdparty)


# opencv
set(OpenCV_DIR ${RKNPU_PATH}/examples/3rdparty/opencv/opencv-linux-aarch64/share/OpenCV)
find_package(OpenCV REQUIRED)

#rga
set(RGA_PATH ${RKNPU_PATH}/examples/3rdparty/rga)
set(LIB_ARCH gcc-aarch64)
set(RGA_LIB ${RGA_PATH}/libs/Linux//${LIB_ARCH}/librga.so)
include_directories(${RGA_PATH}/include)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME}
    src/main.cc
    src/logger.cc
    src/rknn_model.cc
    src/yolo11.cc
    src/utils.cc
)

target_link_libraries(${PROJECT_NAME}
  ${RKNN_RT_LIB}
  ${RGA_LIB}
  ${OpenCV_LIBS}
)

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/${PROJECT_NAME})
install(TARGETS ${PROJECT_NAME} DESTINATION ./)

install(PROGRAMS model/car.jpg DESTINATION ./model)
install(PROGRAMS model/coco_80_labels_list.txt DESTINATION ./model)
install(PROGRAMS model/yolo11.rknn DESTINATION ./model)
