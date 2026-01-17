cmake_minimum_required(VERSION 3.14)
project(micrograd_cpp)

set(CMAKE_CXX_STANDARD 14)


if(POLICY CMP0148)
  cmake_policy(SET CMP0148 NEW)
endif()


find_package(Python COMPONENTS Interpreter Development REQUIRED)


include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11
  GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)


add_executable(demo src/main.cpp)


pybind11_add_module(micrograd_cpp src/bindings.cpp)


if(MSVC)
  target_compile_options(micrograd_cpp PRIVATE /O2)
else()
  target_compile_options(micrograd_cpp PRIVATE -O3)
endif()