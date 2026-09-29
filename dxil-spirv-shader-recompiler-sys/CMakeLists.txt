cmake_minimum_required(VERSION 3.16)
project(dxil-spirv-shader-recompiler-native C CXX)

set(DXIL_SPIRV_CLI ON CACHE BOOL "" FORCE)
set(DXIL_SPIRV_NATIVE_LLVM OFF CACHE BOOL "" FORCE)
set(SPIRV_SKIP_TESTS ON CACHE BOOL "" FORCE)
set(SPIRV_SKIP_EXECUTABLES ON CACHE BOOL "" FORCE)
add_subdirectory(dxil-spirv EXCLUDE_FROM_ALL)

add_custom_target(dxil-spirv-shader-recompiler-native
    DEPENDS dxil-spirv-c-static SPIRV-Tools-static SPIRV-Tools-opt SPIRV-Tools-link)
