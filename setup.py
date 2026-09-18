# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import base64
import hashlib
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import List

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.editable_wheel import editable_wheel
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

    is_hip_build = bool(getattr(torch.version, "hip", None))
    is_cuda_build = bool(getattr(torch.version, "cuda", None))
    if is_hip_build:
        backend_define = "-DKVCACHED_USE_HIP"
        backend_name = "HIP/ROCm"
    elif is_cuda_build:
        backend_define = "-DKVCACHED_USE_CUDA"
        backend_name = "CUDA"
    else:
        raise RuntimeError(
            "Unable to determine GPU backend from PyTorch. "
            "Expected either torch.version.hip or torch.version.cuda."
        )

    extra_compile_args = [
        "-std=c++17",
        f"-D_GLIBCXX_USE_CXX11_ABI={int(cxx_abi)}",
        backend_define,
    ]

    ext_include_dirs = include_paths(device_type="cuda") + [
        os.path.join(CSRC_PATH, "inc")
    ]
    ext_library_dirs = library_paths(device_type="cuda")

    if is_hip_build:
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


def add_pth_to_wheel(wheel_path: str) -> None:
    """Append the .pth (and its RECORD entry) to the root of a built wheel.

    Files at the root of a wheel are installed into site-packages, which is
    the only place the interpreter executes .pth files.
    """
    pth_src = os.path.join(SCRIPT_PATH, PTH_FILE)
    with open(pth_src, "rb") as f:
        data = f.read()
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest())
    record_line = f"{PTH_FILE},sha256={digest.rstrip(b'=').decode()},{len(data)}\n"

    tmp_path = wheel_path + ".tmp"
    with zipfile.ZipFile(wheel_path) as src, zipfile.ZipFile(
            tmp_path, "w", zipfile.ZIP_DEFLATED) as dst:
        record_found = False
        for item in src.infolist():
            payload = src.read(item.filename)
            if item.filename.endswith(".dist-info/RECORD"):
                record_found = True
                payload += record_line.encode()
            dst.writestr(item, payload)
        if not record_found:
            raise RuntimeError(f"no RECORD in {wheel_path}")
        dst.writestr(PTH_FILE, data)
    os.replace(tmp_path, wheel_path)


# PEP 660 editable installs (pip install -e . with setuptools>=64) build an
# editable wheel and never run the legacy develop command, so DevelopWithPth
# does not fire for them. Ship the .pth inside the editable wheel instead.
class EditableWheelWithPth(editable_wheel):
    def run(self):
        editable_wheel.run(self)
        wheels = sorted(Path(self.dist_dir).glob("*.whl"), key=os.path.getmtime)
        if not wheels:
            raise RuntimeError(f"no editable wheel found in {self.dist_dir}")
        add_pth_to_wheel(str(wheels[-1]))
        print(f"Added {PTH_FILE} to editable wheel: {wheels[-1]}")


cmdclass["build_py"] = BuildPyWithPth
cmdclass["install"] = InstallWithPth
cmdclass["develop"] = DevelopWithPth
cmdclass["editable_wheel"] = EditableWheelWithPth

setup(
    packages=find_packages(),
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
