cmake_minimum_required(VERSION 3.8)
project(ur_sim)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

############################################ CUSTOM
find_package(ament_cmake REQUIRED)
find_package(controller_manager REQUIRED)
find_package(rclcpp REQUIRED)
find_package(ros_gz REQUIRED)
find_package(ros_gz_bridge REQUIRED)
find_package(ros_gz_sim REQUIRED)
find_package(ros_gz_image REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(xacro REQUIRED)

find_package(Python REQUIRED COMPONENTS Interpreter)


# Define a CMake variable named dependencies that lists all
# ROS 2 packages and other dependencies the project requires.
set(dependencies
  rclcpp
  ros_gz
  ros_gz_bridge
  ros_gz_sim
  ros_gz_image
  std_msgs
  trajectory_msgs
)

# Add the specified directories to the list of paths that the compiler
# uses to search for header files. This is important for C++
# projects where you have custom header files that are not located
# in the standard system include paths.
include_directories(include)

# # Tells CMake to create executable targets. Make sure CMake
# # knows about the program's dependencies.
# add_executable(example_joint_trajectory_publisher_cpp src/example_joint_trajectory_publisher.cpp)
# ament_target_dependencies(example_joint_trajectory_publisher_cpp ${dependencies})

# Copy necessary files to designated locations in the project
install (
  DIRECTORY config launch models worlds
  DESTINATION share/${PROJECT_NAME}
)

# start: build urdf and sdf
# set robot source xacro
# if(NOT ROBOT_BUILD_SRC)
#   set(ROBOT_BUILD_SRC ${CMAKE_CURRENT_SOURCE_DIR}/urdf/ur.control_gz.urdf.xacro)
#   message(STATUS "No robot source file set. Proceeding with default.")
# endif()
# message("Using robot source file: ${ROBOT_BUILD_SRC}")

# # set robot name
# if(NOT ROBOT_BUILD__NAME)
#   set(ROBOT_BUILD__NAME "ur")
#   message(STATUS "No robot name set. Proceeding with default.")
# endif()
# message("Using robot name: ${ROBOT_BUILD__NAME}")

#set (robot type)
############################################################################################################
# if(NOT ROBOT_BUILD__TYPE)
#   set(ROBOT_BUILD__TYPE "ur5e")
#   message(STATUS "No robot type set. Proceeding with default.")
# endif()
# message("Using robot type: ${ROBOT_BUILD__TYPE}")

# set(ROBOT_BUILD_WS ${CMAKE_CURRENT_BINARY_DIR}/robot_build_ws)
# make_directory(${ROBOT_BUILD_WS})

# set(ROBOT_BUILD_OUTPUT_URDF ${ROBOT_BUILD_WS}/robot.urdf)
# set(ROBOT_BUILD_OUTPUT_SDF ${ROBOT_BUILD_WS}/robot.sdf)

# add_custom_command(
#   OUTPUT ${ROBOT_BUILD_OUTPUT_URDF} ${ROBOT_BUILD_OUTPUT_SDF}
#   COMMAND xacro ${ROBOT_BUILD_SRC} name:=${ROBOT_BUILD__NAME} ur_type:=${ROBOT_BUILD__TYPE} path_to_config_dir:=${CMAKE_CURRENT_SOURCE_DIR}/config -o ${ROBOT_BUILD_OUTPUT_URDF}  # this command must be first
#   COMMAND gz sdf -p ${ROBOT_BUILD_OUTPUT_URDF} > ${ROBOT_BUILD_OUTPUT_SDF}
#   WORKING_DIRECTORY ${ROBOT_BUILD_WS}
#   COMMENT "Build urdf and sdf from xacro"
# )

# add_custom_target(build_urdf_and_sdf_with_sensors ALL
#   DEPENDS ${ROBOT_BUILD_OUTPUT_URDF} ${ROBOT_BUILD_OUTPUT_SDF}
# )

# install(
#   FILES ${ROBOT_BUILD_OUTPUT_URDF} ${ROBOT_BUILD_OUTPUT_SDF}
#   DESTINATION share/${PROJECT_NAME}/urdf
# )
#########################################################################################
#end: build urdf and sdf


#############################################################################
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/models/robot_sensor.sdf          
        DESTINATION share/${PROJECT_NAME}/models)   

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/urdf/robot.urdf          
        DESTINATION share/${PROJECT_NAME}/urdf)                    
#############################################################################
install(
  DIRECTORY include/
  DESTINATION include
)

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()