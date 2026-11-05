INC_DIR = ../../csrc/inc

# Auto-detect backend: set KVCACHED_BACKEND=hip for AMD or xpu for Intel,
# defaults to cuda.
KVCACHED_BACKEND ?= cuda

SRCS = bench_vmm.cpp

ifeq ($(KVCACHED_BACKEND),hip)
  CXX := hipcc
  CXXFLAGS = -DKVCACHED_USE_HIP -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1
  LDLIBS = -lamdhip64
else ifeq ($(KVCACHED_BACKEND),xpu)
  # A plain host compiler suffices: the benchmark emits no device kernels and
  # the virtual-memory extension is host-side library code. Override with
  # `make KVCACHED_BACKEND=xpu CXX=icpx` if you prefer the oneAPI compiler.
  CXX := g++
  CXXFLAGS = -DKVCACHED_USE_XPU -DSYCL_DISABLE_FSYCL_SYCLHPP_WARNING=1
  LDLIBS = -lsycl
  # gpu_vmm.hpp's XPU arm calls kvcached::xpu_runtime. The extension satisfies
  # that from csrc/xpu_runtime.cpp via PyTorch's SYCL context; this build is
  # torch-free, so it supplies its own standalone context instead.
  SRCS += xpu_runtime_standalone.cpp
  # Where to find <sycl/*.hpp> and libsycl. Not needed when a oneAPI
  # environment is active (setvars.sh exports CPATH/LIBRARY_PATH). Otherwise
  # point SYCL_ROOT at an oneAPI install or a Python env that ships the SYCL
  # runtime, e.g.
  #   make KVCACHED_BACKEND=xpu SYCL_ROOT=/opt/intel/oneapi/compiler/latest
  #   make KVCACHED_BACKEND=xpu SYCL_ROOT=$$CONDA_PREFIX
  ifneq ($(SYCL_ROOT),)
    CXXFLAGS += -I$(SYCL_ROOT)/include
    # rpath so the binary runs without LD_LIBRARY_PATH.
    LDFLAGS += -L$(SYCL_ROOT)/lib -Wl,-rpath,$(SYCL_ROOT)/lib
  endif
else
  CXX := nvcc
  CXXFLAGS = -DKVCACHED_USE_CUDA
  LDLIBS = -lcuda
endif

CXXFLAGS += -O2 -g -std=c++17 -I$(INC_DIR)

all: bench_vmm.bin

bench_vmm.bin: $(SRCS)
	$(CXX) $^ -o $@ $(CXXFLAGS) $(LDFLAGS) $(LDLIBS)

clean:
	$(RM) bench_vmm.bin
