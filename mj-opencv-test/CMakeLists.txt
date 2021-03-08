
# require CMake 2.8 or greater
cmake_minimum_required(VERSION 2.8)

# declare my-recognition project
project(imagenet)

# import jetson-inference and jetson-utils packages.
# note that if you didn't do "sudo make install"
# while building jetson-inference, this will error.
find_package(jetson-utils)
find_package(jetson-inference)

# CUDA is required
find_package(CUDA)

# compile the my-recognition program
cuda_add_executable(imagenet imagenet.cpp)

# link my-recognition to jetson-inference library
target_link_libraries(imagenet jetson-inference)
