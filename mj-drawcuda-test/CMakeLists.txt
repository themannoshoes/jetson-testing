
# require CMake 2.8 or greater
cmake_minimum_required(VERSION 3.0.2)

file(GLOB imagenetSources *.cpp *.cu)
file(GLOB imagenetIncludes *.h ./ros_msg_include/*.h)

# declare my-recognition project
project(imagenet)

# import jetson-inference and jetson-utils packages.
# note that if you didn't do "sudo make install"
# while building jetson-inference, this will error.
find_package(jetson-utils)
find_package(jetson-inference)
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  rospy
  std_msgs
  turtlesim
  message_generation
)

# CUDA is required
find_package(CUDA)

include_directories()

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)


# compile the my-recognition program
cuda_add_executable(imagenet ${imagenetSources})



# link my-recognition to jetson-inference library
target_link_libraries(imagenet jetson-inference ${catkin_LIBRARIES})
