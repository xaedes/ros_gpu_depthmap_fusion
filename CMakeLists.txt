cmake_minimum_required(VERSION 3.9)
project(gpu_depthmap_fusion)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_BUILD_TYPE Debug)


find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(glfw3 3.2 REQUIRED)

# catkin packages
find_package(
    catkin 
REQUIRED COMPONENTS 
    roscpp 
    std_msgs
    geometry_msgs
    sensor_msgs
    tf2
    tf2_ros
    tf2_geometry_msgs
    ros_component
    ros_cmake_generate_dependency
    ros_topic_sync
    ros_viz
    glad_ros
    glm_ros
    cv_bridge
)


# create catkin package for project
set(LIBRARY_NAME "${PROJECT_NAME}")
set(COMPONENT_NAME "${LIBRARY_NAME}_component")
set(NODELET_NAME "${LIBRARY_NAME}_nodelet")
set(NODE_NAME "${LIBRARY_NAME}_node")


# catkin_python_setup()
catkin_package(
LIBRARIES 
    ${LIBRARY_NAME}
    ${COMPONENT_NAME}
    ${NODELET_NAME}
CATKIN_DEPENDS 
    roscpp
    std_msgs
    geometry_msgs
    sensor_msgs
    tf2
    tf2_ros
    tf2_geometry_msgs    
    ros_component
    ros_cmake_generate_dependency
    ros_topic_sync
    ros_viz
    glad_ros
    glm_ros    
    cv_bridge    
DEPENDS
    OpenCV
    # glfw
    # OpenMP
)  


add_library(
    ${LIBRARY_NAME} 
    src/gpu_depthmap_fusion.cpp
    src/gl_compute_test.cpp
)
add_library(${COMPONENT_NAME} src/gpu_depthmap_fusion_component.cpp)
add_library(${NODELET_NAME} src/gpu_depthmap_fusion_nodelet.cpp)
add_executable(${NODE_NAME} src/gpu_depthmap_fusion_node.cpp)

include(GenerateDependency)

target_include_directories(${LIBRARY_NAME} PUBLIC include/)

generate_dependency(${LIBRARY_NAME} catkin)
generate_dependency(${LIBRARY_NAME} glfw)
generate_dependency(${LIBRARY_NAME} OpenCV)

generate_dependency(${COMPONENT_NAME} ${LIBRARY_NAME})
generate_dependency(${NODELET_NAME} ${COMPONENT_NAME})
generate_dependency(${NODE_NAME}    ${COMPONENT_NAME})

target_link_libraries(${LIBRARY_NAME} PUBLIC OpenMP::OpenMP_CXX)

add_dependencies(${LIBRARY_NAME} ${catkin_EXPORTED_TARGETS}) 



## Mark the library for installations
install(TARGETS 
    ${LIBRARY_NAME}
    ${COMPONENT_NAME}
    ${NODELET_NAME}
    ${NODE_NAME}    
    ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# install header files
install(
    DIRECTORY 
    include/${PROJECT_NAME}/
    DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
    FILES_MATCHING PATTERN "*.h"
)


# install other files
install(
    FILES 
    nodelet_plugins.xml 
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
