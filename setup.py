# SPDX-FileCopyrightText: 2025 contributors to the kvcached project
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from typing import List

from setuptools import find_packages, setup

try:
    import torch
    from torch.utils.cpp_extension import (
        BuildExtension,
        CUDAExtension,
        include_paths,
        library_paths,
    )
except ImportError:
    raise ImportError("Torch not found, please install torch>=2.6.0 first.")

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
ROOT_PATH = SCRIPT_PATH
CSRC_PATH = os.path.join(ROOT_PATH, "csrc")


def get_csrc_files(path) -> List[str]:
    src_dir = Path(path)
    # setuptools requires relative paths
    cpp_files = [
        str(f.relative_to(SCRIPT_PATH)) for f in src_dir.rglob("*.cpp")
    ]
    return cpp_files


def get_extensions():
    csrc_files = get_csrc_files(CSRC_PATH)

    # Get the C++ ABI flag from PyTorch
    cxx_abi = torch._C._GLIBCXX_USE_CXX11_ABI

    extra_compile_args = [
        "-std=c++17", f"-D_GLIBCXX_USE_CXX11_ABI={int(cxx_abi)}"
    ]

    vmm_ops_module = CUDAExtension(
        "kvcached.vmm_ops",
        csrc_files,
        include_dirs=include_paths() + [os.path.join(CSRC_PATH, "inc")],
        library_dirs=library_paths(),
        libraries=["torch", "torch_cpu", "torch_python", "cuda"],
        extra_compile_args={
            "cxx": extra_compile_args,
            "nvcc": extra_compile_args
        },
    )
    return [vmm_ops_module], {"build_ext": BuildExtension}


ext_modules, cmdclass = get_extensions()

setup(
    packages=find_packages(),
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    data_files=[("", ["kvcached_autopatch.pth"])],
)
