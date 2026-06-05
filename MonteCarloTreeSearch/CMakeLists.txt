cmake_minimum_required(VERSION 3.12)
project(pymcts)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(pybind11 REQUIRED)

# Add threading support
find_package(Threads REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/mcts/include)

# Create the MCTS library
add_library(mcts_lib STATIC
    mcts/src/mcts.cpp
    mcts/src/JobScheduler.cpp
    examples/TicTacToe/TicTacToe.cpp
)

# Set compiler flags for the library
target_compile_options(mcts_lib PRIVATE -O2 -g3 -pedantic)
target_link_libraries(mcts_lib Threads::Threads)

# Create the pybind11 module
pybind11_add_module(pymcts 
    pybind/pymcts.cpp
    pybind/py_wrappers.cpp
)

# Link libraries
target_link_libraries(pymcts PRIVATE mcts_lib Threads::Threads)

# Set properties for the Python module
target_compile_definitions(pymcts PRIVATE VERSION_INFO="${EXAMPLE_VERSION_INFO}")
target_compile_options(pymcts PRIVATE -O2 -g3 -pedantic)

# Disable parallel rollouts for initial Python bindings (simpler)
target_compile_definitions(pymcts PRIVATE -UPARALLEL_ROLLOUTS)