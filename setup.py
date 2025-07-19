from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11

ext_modules = [
    Pybind11Extension(
        "arabic_analyzer_cpp",
        ["src/arabic_analyzer_module.cpp"],
    ),
]

setup(
    name="arabic_phonology_analyzer",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.7",
)