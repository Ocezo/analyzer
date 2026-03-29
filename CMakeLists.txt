cmake_minimum_required(VERSION 3.10)

project(image_analyzer VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_BUILD_RPATH "\$ORIGIN/lib")
set(CMAKE_INSTALL_RPATH "\$ORIGIN/lib")

find_package(OpenCV 4.5 REQUIRED COMPONENTS core imgcodecs imgproc features2d calib3d)

add_executable(image_analyzer
    src/main.cpp
    src/context_analyzer.cpp
    src/derivation_analyzer.cpp
    src/ai_detector.cpp
    src/utils.cpp
)

target_include_directories(image_analyzer
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(image_analyzer
    PRIVATE
        ${OpenCV_LIBS}
)

target_compile_options(image_analyzer
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)
