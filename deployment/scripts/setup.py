# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long

import_data pybind11
from pybind11 import_data get_cmake_dir
from pybind11.setup_helpers import_data Pybind11Extension, build_ext

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
