cmake_minimum_required(VERSION 3.20)

# Standalone CMake project: builds the temhelk/yngine library + a small
# stdin/stdout driver that the Python bridge in `yinsh_ml.yngine` talks to.
# The upstream library is vendored as a git submodule at ../yngine. Build it
# in-tree so a single `cmake --build` produces the `yngine_driver` binary.

project(YngineDriver LANGUAGES CXX)

# Reach into the temhelk/yngine subtree as a subdirectory. We point CMake at
# the library subdir directly (`yngine/yngine`) rather than the project root
# so we skip its top-level CTest/benchmarks scaffolding — we only want the
# library target. We also need its CMake glue: bitboard, generate_tables, etc.
# So just add the project root and rely on its `Yngine` library target.
set(YNGINE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../yngine")
if(NOT EXISTS "${YNGINE_ROOT}/CMakeLists.txt")
    message(FATAL_ERROR
        "yngine submodule missing at ${YNGINE_ROOT}. Run:\n"
        "  git submodule update --init --recursive")
endif()

# yngine declares cmake_minimum_required(3.30). cmake 4.x policy CMP0181 makes
# that a hard error if our top-level cmake_min is lower, so we re-declare.
# Apple Homebrew ships cmake 4.x; if you're on <3.30, install a newer cmake.
add_subdirectory("${YNGINE_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}/yngine_lib")

add_executable(yngine_driver yngine_driver.cpp)
target_link_libraries(yngine_driver PRIVATE Yngine::Yngine)
target_compile_features(yngine_driver PRIVATE cxx_std_20)
set_target_properties(yngine_driver PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION $<$<CONFIG:Debug>:FALSE:TRUE>)
