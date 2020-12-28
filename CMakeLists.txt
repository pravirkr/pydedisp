cmake_minimum_required(VERSION 3.15)

project(dedisp VERSION 1.0.0 LANGUAGES C CXX)

option(ENABLE_CLANG_TIDY "Run Clang Tidy to get static analysis" OFF)
option(BUILD_DOCS        "Build documentation" OFF)
option(BUILD_SRC         "Build dedisp" ON)
option(BUILD_PYTHON      "Build python bindings" ON)

#set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-Wall")

set(ENV{CUDACXX} $ENV{CUDA_PATH}/bin/nvcc)
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER AND BUILD_SRC)
    enable_language(CUDA)
    if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS_EQUAL 10.0.0)
        message(FATAL_ERROR "Found CUDA ${CMAKE_CUDA_COMPILER_VERSION}. Need CUDA 10+")
    else()
        message(STATUS "Found CUDA ${CMAKE_CUDA_COMPILER_VERSION}.")
    endif()

    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED True)
    set(CMAKE_CUDA_FLAGS "-c -Xcompiler=-Wall -O3 -arch=sm_60 -Wno-deprecated-declarations")

    add_subdirectory("src")
    add_subdirectory("example")
elseif(NOT CMAKE_CUDA_COMPILER)
    message(STATUS "Not building src. No CUDA compiler found...")
endif(CMAKE_CUDA_COMPILER AND BUILD_SRC)

if(BUILD_DOCS)
	message(STATUS "Building Docs...")
	# Add the cmake folder so the FindSphinx module is found
	set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

    add_subdirectory("docs")
endif(BUILD_DOCS)

if(BUILD_PYTHON)
    message(STATUS "Building Python bindings...")

    find_package(Python COMPONENTS Interpreter Development)

    # pybind11 hint
    execute_process(
        COMMAND
          "${Python_EXECUTABLE}" -c
          "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE _tmp_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT
    )

    list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")

    # Now we can find pybind11
    find_package(pybind11 CONFIG REQUIRED)

    # pybind11 method:
    pybind11_add_module(libdedisp MODULE ${PROJECT_SOURCE_DIR}/pydedisp/bindings.cpp)
    target_include_directories(libdedisp PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(libdedisp PRIVATE ${PROJECT_NAME})

    install (TARGETS libdedisp DESTINATION ${PROJECT_SOURCE_DIR}/lib)

endif(BUILD_PYTHON)

# ------------------------------------------------------------------------------
# Clang Tidy
# ------------------------------------------------------------------------------
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(ENABLE_CLANG_TIDY)
    find_program(CLANG_TIDY_BIN clang-tidy)
    find_program(RUN_CLANG_TIDY_BIN run-clang-tidy)

    if(CLANG_TIDY_BIN STREQUAL "CLANG_TIDY_BIN-NOTFOUND")
        message(FATAL_ERROR "unable to locate clang-tidy")
    endif()

    if(RUN_CLANG_TIDY_BIN STREQUAL "RUN_CLANG_TIDY_BIN-NOTFOUND")
        message(FATAL_ERROR "unable to locate run-clang-tidy")
    endif()

    list(APPEND RUN_CLANG_TIDY_BIN_ARGS
        -clang-tidy-binary ${CLANG_TIDY_BIN}
        -header-filter=.*
        -checks=clan*,cert*,misc*,perf*,cppc*,read*,mode*,-cert-err58-cpp,-misc-noexcept-move-constructor
    )

    add_custom_target(
        tidy
        COMMAND ${RUN_CLANG_TIDY_BIN} ${RUN_CLANG_TIDY_BIN_ARGS}
        COMMENT "running clang tidy"
    )

endif()
