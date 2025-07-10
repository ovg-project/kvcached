NVCC ?= nvcc

all: bench_vmm.bin

bench_vmm.bin: bench_vmm.cpp
	$(NVCC) $^ -o $@ -O2 -g -lcuda -std=c++17

clean:
	$(RM) bench_vmm.bin