cmake_minimum_required (VERSION 3.12)

project ("ArchProbe" LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE PATH "Build Type" FORCE)
endif()

message("-- Looking for libopencl-stub")
if (EXISTS "${PROJECT_SOURCE_DIR}/third/libopencl-stub")
    add_compile_definitions(CL_VERSION_2_0)
    add_subdirectory(third/libopencl-stub)
    set(OpenCL_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/third/libopencl-stub/include")
    message("-- OpenCL context enabled")
else()
    message("")
    message("-- OpenCL not found! OpenCL context is disabled")
    message("")
endif()

set(LINK_LIBS
    OpenCL
)
set(INC_DIRS
    "${PROJECT_SOURCE_DIR}/include"
    ${OpenCL_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/third/OpenCL-SDK/external/OpenCL-CLHPP/include
)

# The testbench library.
include_directories(${INC_DIRS})
file(GLOB SRCS "${PROJECT_SOURCE_DIR}/src/*")
file(GLOB INCS "${PROJECT_SOURCE_DIR}/include/*")
add_library(ArchProbeCore STATIC ${SRCS} ${INCS})
target_link_libraries(ArchProbeCore ${LINK_LIBS})
target_compile_definitions(ArchProbeCore PUBLIC CL_TARGET_OPENCL_VERSION=200)

# Testbench apps.
make_directory("${CMAKE_BINARY_DIR}/assets/")
add_subdirectory("${PROJECT_SOURCE_DIR}/apps")
