cmake_minimum_required(VERSION 3.14)

project(projetile)

set(MUJOCO_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
set(MUJOCO_LIBRARIES  ${CMAKE_CURRENT_SOURCE_DIR}/../../lib)

include_directories(${MUJOCO_INCLUDE_DIR})
link_directories(${MUJOCO_LIBRARIES})

# include_directories(/home/nara/Documents/mujoco_test/mujoco-3.1.1/include)
# link_directories(/home/nara/Documents/mujoco_test/mujoco-3.1.1/lib)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} mujoco glfw)