# Examples CMakeLists.txt for FORGE

cmake_minimum_required(VERSION 3.20)

# Basic gradient example
add_executable(basic_gradient basic_gradient.cpp)
target_link_libraries(basic_gradient PRIVATE forge::forge)
target_include_directories(basic_gradient PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src ${CMAKE_CURRENT_SOURCE_DIR}/../tools)

# Multi-variable example
add_executable(multi_variable multi_variable.cpp)
target_link_libraries(multi_variable PRIVATE forge::forge)
target_include_directories(multi_variable PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src ${CMAKE_CURRENT_SOURCE_DIR}/../tools)

# Performance benchmark example
add_executable(performance_demo performance_demo.cpp)
target_link_libraries(performance_demo PRIVATE forge::forge)
target_include_directories(performance_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src ${CMAKE_CURRENT_SOURCE_DIR}/../tools)

# Set output directory for examples
set_target_properties(basic_gradient multi_variable performance_demo
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)