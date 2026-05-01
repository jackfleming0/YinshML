import os

from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

# Apple Silicon dev needs a deployment target that supports the C++17
# stdlib pybind11 expects; cloud Linux ignores this env var.
os.environ.setdefault("MACOSX_DEPLOYMENT_TARGET", "11.0")

ext_modules = [
    Pybind11Extension(
        "yinsh_ml.game_cpp._engine",
        sources=["yinsh_ml/game_cpp/src/bindings.cpp"],
        depends=[
            "yinsh_ml/game_cpp/src/bitboard.hpp",
            "yinsh_ml/game_cpp/src/tables.hpp",
            "yinsh_ml/game_cpp/src/zobrist.hpp",
            "yinsh_ml/game_cpp/src/state.hpp",
            "yinsh_ml/game_cpp/src/moves.hpp",
            "yinsh_ml/game_cpp/src/apply.hpp",
            "yinsh_ml/game_cpp/src/movegen.hpp",
        ],
        include_dirs=["yinsh_ml/game_cpp/src"],
        cxx_std=17,
        # __uint128_t is a GCC/Clang extension, not C++ standard. Disable
        # the resulting -Wpedantic noise so we don't drown the build log.
        extra_compile_args=["-Wno-pedantic", "-O3"],
    ),
]

setup(
    name="yinsh_ml",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.3",
        "scipy>=1.10.1",
        "torch>=2.1.0",
        "coremltools>=6.3.0",
        "matplotlib>=3.7.1",
        "seaborn>=0.12.2",
        "tqdm>=4.65.0",
        "tensorboard>=2.14.0",
        "pandas>=2.0.3",
        "pytest>=7.4.0",
        "click>=8.0.0",
    ],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    entry_points={
        'console_scripts': [
            'yinsh-track=yinsh_ml.cli.main:cli',
        ],
    },
    python_requires=">=3.9",
)
