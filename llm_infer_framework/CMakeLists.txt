cmake_minimum_required(VERSION 3.16)

project(llm_infer_framework LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(llm_infer_framework STATIC
    source/base/buffer.cpp
    source/base/alloc_cpu.cpp
)

target_include_directories(llm_infer_framework
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/base
)

set(GOOGLETEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../other/googletest-main")
if(NOT EXISTS "${GOOGLETEST_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "googletest not found: ${GOOGLETEST_DIR}")
endif()

add_subdirectory("${GOOGLETEST_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/googletest")
find_package(glog REQUIRED)

enable_testing()

add_executable(test_buffer
    test/test_buffer.cpp
)

target_link_libraries(test_buffer
    PRIVATE
        llm_infer_framework
        gtest
        glog::glog
)

include(GoogleTest)
gtest_discover_tests(test_buffer)
