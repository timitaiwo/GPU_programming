cmake_minimum_required(VERSION 4.0)
project(hip_kernels LANGUAGES CXX HIP)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}") # sensible since this is a sample usecase

add_executable("kernel"
                addition.hip
)
