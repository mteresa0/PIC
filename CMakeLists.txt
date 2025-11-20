##
#  CMake script for the dummy program:
##

# Declare the CMake minimum version required
cmake_minimum_required(VERSION 3.13.4)
cmake_policy(SET CMP0003 NEW) # setting improved behaviour of cmake

# Find the deal.II package
find_package(deal.II 9.5.0 REQUIRED
  HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR})

# Set the name of the project and target:
set(TARGET "main")

# Set outputs to clean
set(CLEAN_UP_FILES
  outputs/*.vtk outputs/*.vtu outputs/*.csv
)

# Include 'include' directories
INCLUDE_DIRECTORIES(
  ${CMAKE_SOURCE_DIR}/inc
)

# Include all .cc files in the src directory
file(GLOB_RECURSE TARGET_SRC src/*.cc src/*.cpp)

# Option to enable or disable logging
option(ENABLE_LOG "Enable debug logging" ON)

# Pass LOG_ENABLED macro to compiler
if(ENABLE_LOG)
    add_compile_definitions(LOG_ENABLED=1)
else()
    add_compile_definitions(LOG_ENABLED=0)
endif()

# Option to enable or disable logging
option(OUTPUTS_EXTRA "Enable extra outputs" OFF)

# Pass LOG_ENABLED macro to compiler
if(OUTPUTS_EXTRA)
    add_compile_definitions(EXTRA_OUTPUTS=1)
else()
    add_compile_definitions(EXTRA_OUTPUTS=0)
endif()

# Initialize deal.II variables
deal_ii_initialize_cached_variables()

# Set the project
project(my_project)

# Invoke autopilot to help with deal.II configuration
deal_ii_invoke_autopilot()

add_custom_target(clean_outputs
    COMMAND ${CMAKE_COMMAND} -E rm -f ${CLEAN_UP_FILES}
)