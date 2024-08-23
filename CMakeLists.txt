project                     (diy-work-stealing)
cmake_minimum_required      (VERSION 3.12)

option                      (wrapped_mpi    "MPI compiler wrapper requires no further MPI libs"     OFF)
option                      (log            "Build DIY with logging"                                OFF)

# C++11
if                          (NOT DEFINED CMAKE_CXX_STANDARD)
    set                     (CMAKE_CXX_STANDARD 11)
endif                       ()

set                         (CMAKE_POSITION_INDEPENDENT_CODE ON)

# ignore variadic args warning (no other good solution)
set                         (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-varargs")

# Default to Release
if                          (NOT CMAKE_BUILD_TYPE)
    set                     (CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
    set_property            (CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif                       (NOT CMAKE_BUILD_TYPE)

# MPI
if                          (NOT wrapped_mpi)
    find_package            (MPI REQUIRED)
    set                     (libraries MPI::MPI_C)
endif                       ()

# Threads
find_package                (Threads)

# DIY
if                          (NOT DEFINED DIY_PATH AND DEFINED ENV{DIY_PATH})
    set                     (DIY_PATH $ENV{DIY_PATH})
endif                       ()

if                          (DIY_PATH)
    message                 ("Found DIY_PATH")
    set                     (DIY_INCLUDE_DIR ${DIY_PATH}/include)
else                        ()
    message                 ("Looking for DIY")
    find_package            (diy REQUIRED)
endif                       ()
message                     ("DIY_INCLUDE_DIR = ${DIY_INCLUDE_DIR}")
include_directories         (SYSTEM ${DIY_INCLUDE_DIR})

# Debugging
if                          (${CMAKE_BUILD_TYPE} STREQUAL "Debug" OR
                             ${CMAKE_BUILD_TYPE} STREQUAL "RelWithDebInfo")
    add_definitions         ("-DDEBUG")
endif                       ()

# Logging
if                          (log)
    add_definitions         ("-DDIY_USE_SPDLOG")
    find_package            (spdlog REQUIRED)
endif()

# fmt
if                          (NOT DEFINED FMT_PATH AND DEFINED ENV{FMT_PATH})
    set                     (FMT_PATH $ENV{FMT_PATH})
endif                       ()

if                          (FMT_PATH)
    message                 ("Found FMT_PATH")
    set                     (FMT_INCLUDE_DIR ${FMT_PATH}/include)
else                        ()
    message                 ("Looking for fmt")
    find_package            (fmt REQUIRED)
endif                       ()
message                     ("FMT_INCLUDE_DIR = ${FMT_INCLUDE_DIR}")
include_directories         (SYSTEM ${FMT_INCLUDE_DIR})

# Include dirs
set                         (CMAKE_INCLUDE_SYSTEM_FLAG_CXX "-isystem")
include_directories         (${CMAKE_CURRENT_SOURCE_DIR}/include
                            SYSTEM ${MPI_C_INCLUDE_PATH})

# Libraries
set                         (libraries
                             ${libraries}
                             ${CMAKE_THREAD_LIBS_INIT}
                             ${CMAKE_DL_LIBS})

add_subdirectory            (src)
