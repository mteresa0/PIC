##
#  CMake script for the dummy program:
##

# Find the deal.II package
find_package(deal.II 9.5.0 REQUIRED
  HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR})

# Set the name of the project and target:
set(TARGET "main")

# Include 'include' directories
INCLUDE_DIRECTORIES(
  ${CMAKE_SOURCE_DIR}/inc
)

# Include all .cc files in the src directory
file(GLOB_RECURSE TARGET_SRC src/*.cc src/*.cpp)

# Declare the CMake minimum version required
cmake_minimum_required(VERSION 3.13.4)

# Initialize deal.II variables
deal_ii_initialize_cached_variables()

# Set the project
project(my_project)

# Invoke autopilot to help with deal.II configuration
deal_ii_invoke_autopilot()

