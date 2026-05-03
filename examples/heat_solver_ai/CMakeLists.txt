# =====================================================================
# heat_solver_ai —— 用 LibTorch 加载 PyTorch 模型，从 C++ 选最优求解器
#
# 用法：
#   cd examples/heat_solver_ai
#   mkdir build && cd build
#   cmake -DCMAKE_PREFIX_PATH="<libtorch_dir>" \
#         -DMFEM_DIR=<mfem_install_dir> ..
#   cmake --build . -j
#
# 依赖：
#   - LibTorch（C++ ABI 与训练时的 PyTorch 版本一致）
#       https://pytorch.org/cppdocs/installing.html
#   - MFEM（与 examples/ 同级父目录或显式指定 MFEM_DIR）
# =====================================================================

cmake_minimum_required(VERSION 3.18)
project(heat_solver_ai LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------
# Torch
# ---------------------------------------------------------------------
find_package(Torch REQUIRED CONFIG)
message(STATUS "Found Torch: ${TORCH_INSTALL_PREFIX}")

# ---------------------------------------------------------------------
# MFEM —— 如未安装为 cmake package，可改为手动指定 include/lib 路径
# ---------------------------------------------------------------------
find_package(MFEM CONFIG QUIET)
if(NOT MFEM_FOUND)
   if(NOT DEFINED MFEM_DIR)
      set(MFEM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../..")
   endif()
   message(STATUS "Falling back to MFEM_DIR=${MFEM_DIR}")
   set(MFEM_INCLUDE_DIRS "${MFEM_DIR}")
   find_library(MFEM_LIB mfem
      HINTS "${MFEM_DIR}" "${MFEM_DIR}/build"
      REQUIRED)
   set(MFEM_LIBRARIES ${MFEM_LIB})
endif()

# ---------------------------------------------------------------------
# 静态库：solver_predictor
# ---------------------------------------------------------------------
add_library(solver_predictor STATIC solver_predictor.cpp)
target_include_directories(solver_predictor
   PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}
   PRIVATE ${MFEM_INCLUDE_DIRS})
target_link_libraries(solver_predictor
   PUBLIC  "${TORCH_LIBRARIES}"
   PRIVATE ${MFEM_LIBRARIES})
target_compile_options(solver_predictor PRIVATE -Wall -Wextra)

# ---------------------------------------------------------------------
# 可执行：heat_solve_with_ai
# ---------------------------------------------------------------------
add_executable(heat_solve_with_ai heat_solve_with_ai.cpp)
target_include_directories(heat_solve_with_ai PRIVATE ${MFEM_INCLUDE_DIRS})
target_link_libraries(heat_solve_with_ai
   PRIVATE solver_predictor ${MFEM_LIBRARIES} "${TORCH_LIBRARIES}")
target_compile_options(heat_solve_with_ai PRIVATE -Wall -Wextra)

# 复制 model & meta 到构建目录便于直接运行
foreach(fn IN ITEMS solver_selector.pt solver_selector_meta.json)
   if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${fn}")
      configure_file("${CMAKE_CURRENT_SOURCE_DIR}/${fn}"
                     "${CMAKE_CURRENT_BINARY_DIR}/${fn}" COPYONLY)
   endif()
endforeach()
