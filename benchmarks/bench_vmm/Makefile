INC_DIR = ../../csrc/inc

# Auto-detect backend: set KVCACHED_BACKEND=hip for AMD, defaults to cuda.
KVCACHED_BACKEND ?= cuda

ifeq ($(KVCACHED_BACKEND),hip)
  CXX := hipcc
  CXXFLAGS = -DKVCACHED_USE_HIP -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1
  LDLIBS = -lamdhip64
else
  CXX := nvcc
  CXXFLAGS = -DKVCACHED_USE_CUDA
  LDLIBS = -lcuda
endif

CXXFLAGS += -O2 -g -std=c++17 -I$(INC_DIR)

all: bench_vmm.bin

bench_vmm.bin: bench_vmm.cpp
	$(CXX) $^ -o $@ $(CXXFLAGS) $(LDLIBS)

clean:
	$(RM) bench_vmm.bin
