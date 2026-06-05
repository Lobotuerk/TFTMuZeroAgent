from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
from setuptools import setup, Extension
import os

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "pymcts",
        [
            "pybind/pymcts.cpp",
            "pybind/py_wrappers.cpp",
            "pybind/mcts_python.cpp",  # Use Python-specific MCTS implementation
            "examples/TicTacToe/TicTacToe.cpp",
        ],
        include_dirs=[
            # Path to pybind11 headers
            pybind11.get_include(),
            "mcts/include",
            "examples/TicTacToe",
            "pybind",  # For our Python-specific headers
        ],
        cxx_std=11,
        # Don't define PARALLEL_ROLLOUTS to disable parallel rollouts
        extra_compile_args=[
            "/O2" if os.name == 'nt' else "-O2",  # Use MSVC syntax on Windows
            "/UPARALLEL_ROLLOUTS" if os.name == 'nt' else "-UPARALLEL_ROLLOUTS",  # Explicitly undefine
        ],
        extra_link_args=[
        ],
    ),
]

setup(
    name="pymcts",
    version="0.1.0",
    author="MCTS Library Python Bindings",
    author_email="",
    url="",
    description="Python bindings for Monte Carlo Tree Search C++ library",
    long_description="",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.6",
    install_requires=[
        "pybind11>=2.6.0",
    ],
)