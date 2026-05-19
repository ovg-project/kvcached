# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import sys
from pathlib import Path
from typing import List

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.install import install

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

# torch-ROCm sets torch.version.hip and leaves torch.version.cuda None.
IS_ROCM = torch.version.hip is not None
ROCM_PATH = os.environ.get("ROCM_PATH", "/opt/rocm")


def get_csrc_files(path) -> List[str]:
    src_dir = Path(path)
    # setuptools requires relative paths.
    # Exclusions:
    #  - "._*": macOS AppleDouble metadata files.
    #  - "*_hip.cpp": hipify-generated artifacts.
    cpp_files = [
        str(f.relative_to(SCRIPT_PATH)) for f in src_dir.rglob("*.cpp")
        if not f.name.startswith("._") and not f.name.endswith("_hip.cpp")
    ]
    return cpp_files


def get_extensions():
    csrc_files = get_csrc_files(CSRC_PATH)

    # Get the C++ ABI flag from PyTorch
    cxx_abi = torch._C._GLIBCXX_USE_CXX11_ABI

    extra_compile_args = [
        "-std=c++17", f"-D_GLIBCXX_USE_CXX11_ABI={int(cxx_abi)}"
    ]

    gpu_lib = "amdhip64" if IS_ROCM else "cuda"
    inc_dirs = include_paths() + [os.path.join(CSRC_PATH, "inc")]
    lib_dirs = library_paths()
    if IS_ROCM:
        inc_dirs.append(os.path.join(ROCM_PATH, "include"))
        lib_dirs.append(os.path.join(ROCM_PATH, "lib"))

    vmm_ops_module = CUDAExtension(
        "kvcached.vmm_ops",
        csrc_files,
        include_dirs=inc_dirs,
        library_dirs=lib_dirs,
        libraries=["torch", "torch_cpu", "torch_python", gpu_lib],
        extra_compile_args={
            "cxx": extra_compile_args,
            "nvcc": extra_compile_args
        },
    )
    return [vmm_ops_module], {"build_ext": BuildExtension}


ext_modules, cmdclass = get_extensions()

PTH_FILE = "kvcached_autopatch.pth"


# Custom build_py to copy .pth file to build directory
# This ensures it gets included in wheels and direct installs
class BuildPyWithPth(build_py):
    def run(self):
        build_py.run(self)
        # Copy .pth file to the build lib directory (root level)
        # This makes it part of the build output that gets installed
        pth_src = os.path.join(SCRIPT_PATH, PTH_FILE)
        pth_dst = os.path.join(self.build_lib, PTH_FILE)
        self.copy_file(pth_src, pth_dst)
        print(f"Copied {PTH_FILE} to build directory: {pth_dst}")


# Custom install command to ensure .pth file goes to site-packages root
class InstallWithPth(install):
    def run(self):
        install.run(self)
        # After standard install, copy .pth file to install_lib (site-packages)
        pth_src = os.path.join(SCRIPT_PATH, PTH_FILE)
        pth_dst = os.path.join(self.install_lib, PTH_FILE)
        self.copy_file(pth_src, pth_dst)
        print(f"Installed {PTH_FILE} to: {pth_dst}")


# Custom develop command for editable installs
class DevelopWithPth(develop):
    def run(self):
        develop.run(self)
        # For editable installs, copy .pth file to site-packages
        import site

        if "--user" in sys.argv:
            target_dir = site.getusersitepackages()
        else:
            site_dirs = site.getsitepackages()
            target_dir = site_dirs[0] if site_dirs else self.install_lib

        pth_src = os.path.join(SCRIPT_PATH, PTH_FILE)
        pth_dst = os.path.join(target_dir, PTH_FILE)

        os.makedirs(target_dir, exist_ok=True)
        shutil.copy2(pth_src, pth_dst)
        print(f"Installed {PTH_FILE} for editable install to: {pth_dst}")


cmdclass["build_py"] = BuildPyWithPth
cmdclass["install"] = InstallWithPth
cmdclass["develop"] = DevelopWithPth

setup(
    packages=find_packages(),
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
