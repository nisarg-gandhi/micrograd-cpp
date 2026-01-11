cmake_minimum_required(VERSION 3.10)

# Project Name
project(MicrogradCPP)

# Standard C++17 (Good for modern features like shared_ptr)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Source files
set(SOURCES src/main.cpp)

# Output executable name
add_executable(micrograd ${SOURCES})