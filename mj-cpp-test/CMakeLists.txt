
# require CMake 2.8 or greater
cmake_minimum_required(VERSION 2.8)

file(GLOB imagenetSources *.cpp *.cu)
file(GLOB imagenetIncludes *.h )

# declare my-recognition project
project(mj-cpp)

# import jetson-inference and jetson-utils packages.
# note that if you didn't do "sudo make install"
# while building jetson-inference, this will error.
find_package(jetson-utils)
find_package(jetson-inference)

# CUDA is required
find_package(CUDA)

include_directories()

# compile the my-recognition program
cuda_add_executable(mj-cpp ${imagenetSources})



# link my-recognition to jetson-inference library
target_link_libraries(mj-cpp jetson-inference)
