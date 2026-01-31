cmake_minimum_required(VERSION 3.18)
project(drone_detection_core LANGUAGES CXX)

# C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /arch:AVX2")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
endif()

# Options
option(USE_CUDA "Enable CUDA support" ON)
option(USE_OPENMP "Enable OpenMP support" ON)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)

# Find packages
find_package(Eigen3 REQUIRED)

if(USE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
endif()

# CUDA support
if(USE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        find_package(CUDAToolkit REQUIRED)
        add_definitions(-DUSE_CUDA)
        message(STATUS "CUDA enabled: ${CMAKE_CUDA_COMPILER_VERSION}")
    else()
        message(WARNING "CUDA compiler not found, building without GPU support")
        set(USE_CUDA OFF)
    endif()
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
    ${CMAKE_BINARY_DIR}/_deps/eigen3-src/unsupported
)

# Source files
set(CORE_SOURCES
    src/signal_processor.cpp
    src/beamforming.cpp
    src/tdoa_engine.cpp
    src/audio_buffer.cpp
    src/fft_processor.cpp
)

set(CORE_HEADERS
    include/signal_processor.hpp
    include/beamforming.hpp
    include/tdoa_engine.hpp
    include/audio_buffer.hpp
    include/fft_processor.hpp
    include/common.hpp
)

# CUDA sources
if(USE_CUDA)
    set(CUDA_SOURCES
        cuda/beamforming_kernels.cu
        cuda/fft_kernels.cu
        cuda/signal_kernels.cu
    )
endif()

# Core library
if(USE_CUDA)
    add_library(drone_core STATIC ${CORE_SOURCES} ${CUDA_SOURCES})
    target_link_libraries(drone_core PRIVATE CUDA::cudart CUDA::cufft)
else()
    add_library(drone_core STATIC ${CORE_SOURCES})
endif()

target_link_libraries(drone_core PUBLIC Eigen3::Eigen)

if(OpenMP_CXX_FOUND)
    target_link_libraries(drone_core PRIVATE OpenMP::OpenMP_CXX)
endif()

# Python bindings
if(BUILD_PYTHON_BINDINGS)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

    # Find or fetch pybind11
    find_package(pybind11 CONFIG)
    if(NOT pybind11_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v2.11.1
        )
        FetchContent_MakeAvailable(pybind11)
    endif()

    pybind11_add_module(drone_core_py
        bindings/python_bindings.cpp
    )

    target_link_libraries(drone_core_py PRIVATE drone_core)

    # Install Python module
    install(TARGETS drone_core_py
        LIBRARY DESTINATION ${Python3_SITELIB}
    )
endif()

# Install
install(TARGETS drone_core
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(FILES ${CORE_HEADERS}
    DESTINATION include/drone_detection
)
