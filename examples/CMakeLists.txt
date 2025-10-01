# Examples for Forge library

# Basic example
add_executable(basic_example basic_example.cpp)
target_link_libraries(basic_example forge)

# Automatic differentiation example
add_executable(autodiff_example autodiff_example.cpp)
target_link_libraries(autodiff_example forge)

# JSON loading example
add_executable(json_loading_example json_loading_example.cpp)
target_link_libraries(json_loading_example forge)

# Graph from JSON example (comprehensive)
add_executable(graph_from_json graph_from_json.cpp)
target_link_libraries(graph_from_json forge)

# Copy JSON data files to build directory
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ops10_function_graph.json
               ${CMAKE_CURRENT_BINARY_DIR}/ops10_function_graph.json COPYONLY)

# Set output directory for all examples
set_target_properties(basic_example autodiff_example json_loading_example graph_from_json
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Add examples to CTest
add_test(NAME basic_example_test COMMAND basic_example)
add_test(NAME autodiff_example_test COMMAND autodiff_example)
add_test(NAME json_loading_example_test 
         COMMAND json_loading_example
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
add_test(NAME graph_from_json_test
         COMMAND graph_from_json
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})