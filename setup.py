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
        CppExtension,
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
    # Filter out macOS AppleDouble metadata files (._* prefix)
    cpp_files = [
        str(f.relative_to(SCRIPT_PATH)) for f in src_dir.rglob("*.cpp")
        if not f.name.startswith("._")
    ]
    return cpp_files


def get_extensions():
    csrc_files = get_csrc_files(CSRC_PATH)

    # Get the C++ ABI flag from PyTorch
    cxx_abi = torch._C._GLIBCXX_USE_CXX11_ABI

    # KVCACHED_BACKEND forces a backend for cross-builds; otherwise infer it
    # from the installed PyTorch, in the order hip > cuda > xpu. That precedence
    # is intentional: torch.version.hip and torch.version.cuda are both set on
    # PyTorch-ROCm, which presents AMD GPUs as CUDA devices. kvcached/utils.py's
    # _detect_accelerator_backend() repeats the same order so the runtime and
    # the build agree.
    forced_backend = os.getenv("KVCACHED_BACKEND", "").strip().lower()
    if forced_backend not in ("", "hip", "cuda", "xpu"):
        raise RuntimeError(
            f"Invalid KVCACHED_BACKEND={forced_backend!r}. "
            "Expected one of: hip, cuda, xpu."
        )

    is_hip_build = forced_backend == "hip" or (
        not forced_backend and bool(getattr(torch.version, "hip", None))
    )
    is_cuda_build = not is_hip_build and (
        forced_backend == "cuda"
        or (not forced_backend and bool(getattr(torch.version, "cuda", None)))
    )
    is_xpu_build = not is_hip_build and not is_cuda_build and (
        forced_backend == "xpu"
        or (not forced_backend and bool(getattr(torch.version, "xpu", None)))
    )
    if is_hip_build:
        backend_define = "-DKVCACHED_USE_HIP"
        backend_name = "HIP/ROCm"
    elif is_cuda_build:
        backend_define = "-DKVCACHED_USE_CUDA"
        backend_name = "CUDA"
    elif is_xpu_build:
        backend_define = "-DKVCACHED_USE_XPU"
        backend_name = "Intel XPU (Level Zero via SYCL)"
    else:
        raise RuntimeError(
            "Unable to determine GPU backend from PyTorch. "
            "Expected one of torch.version.hip, torch.version.cuda or "
            "torch.version.xpu, or an explicit KVCACHED_BACKEND."
        )

    extra_compile_args = [
        "-std=c++17",
        f"-D_GLIBCXX_USE_CXX11_ABI={int(cxx_abi)}",
        backend_define,
    ]

    # HIP resolves its headers through the CUDA device type, matching how
    # PyTorch-ROCm presents itself; XPU has its own include/library roots.
    torch_device_type = "xpu" if is_xpu_build else "cuda"
    ext_include_dirs = include_paths(device_type=torch_device_type) + [
        os.path.join(CSRC_PATH, "inc")
    ]
    ext_library_dirs = library_paths(device_type=torch_device_type)

    if is_xpu_build:
        # XPU builds use CppExtension with a plain host compiler for the same
        # reason HIP does: kvcached emits no device kernels, so there is nothing
        # for icpx/hipcc to compile. gpu_vmm.hpp reaches Level Zero through the
        # sycl_ext_oneapi_virtual_mem extension, which is host-side library code.
        extra_compile_args.append("-DSYCL_DISABLE_FSYCL_SYCLHPP_WARNING=1")
        # sycl: the virtual-memory extension entry points.
        # c10_xpu: PyTorch's SYCL context/device accessors (see xpu_runtime.cpp),
        #          which is what keeps kvcached mappings addressable by engine
        #          kernels.
        ext_libraries = ["sycl", "c10_xpu"]
        vmm_ops_module = CppExtension(
            "kvcached.vmm_ops",
            csrc_files,
            include_dirs=ext_include_dirs,
            library_dirs=ext_library_dirs,
            libraries=ext_libraries,
            extra_compile_args={"cxx": extra_compile_args},
        )
    elif is_hip_build:
        # HIP builds: use CppExtension to avoid PyTorch's hipify step.
        # Our code already handles HIP natively via gpu_vmm.hpp conditional
        # compilation, so hipify is unnecessary and breaks torch headers.
        extra_compile_args.extend([
            "-D__HIP_PLATFORM_AMD__=1",
            "-DUSE_ROCM=1",
        ])
        ext_libraries = ["amdhip64"]
        vmm_ops_module = CppExtension(
            "kvcached.vmm_ops",
            csrc_files,
            include_dirs=ext_include_dirs,
            library_dirs=ext_library_dirs,
            libraries=ext_libraries,
            extra_compile_args={"cxx": extra_compile_args},
        )
    else:
        # CUDA driver APIs require libcuda for cuMem* symbols.
        ext_libraries = ["cuda"]
        vmm_ops_module = CUDAExtension(
            "kvcached.vmm_ops",
            csrc_files,
            include_dirs=ext_include_dirs,
            library_dirs=ext_library_dirs,
            libraries=ext_libraries,
            extra_compile_args={
                "cxx": extra_compile_args,
                "nvcc": extra_compile_args,
            },
        )
    print(f"Building kvcached.vmm_ops with backend: {backend_name}")
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
