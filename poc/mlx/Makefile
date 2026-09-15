# KVCached MLX PoC — Build System
#
# Targets:
#   make test-vmm     — Build & run C VMM test (Linux/macOS)
#   make test-metal   — Build & run Metal GPU test (macOS only)
#   make test-python  — Run Python VMM test (Linux/macOS)
#   make test         — Run all available tests
#   make clean

CC = gcc
CFLAGS = -O2 -Wall -Wextra -std=c11

# Detect platform
UNAME := $(shell uname -s)

.PHONY: all test test-vmm test-metal test-python clean

all: test

# ---- C VMM test (Linux/macOS) ----

test_vmm_remap: test_vmm_remap.c
	$(CC) $(CFLAGS) -o $@ $< -lrt 2>/dev/null || $(CC) $(CFLAGS) -o $@ $<

test-vmm: test_vmm_remap
	@echo "========================================"
	@echo "Running C VMM remap test..."
	@echo "========================================"
	./test_vmm_remap

# ---- Metal VMM test (macOS only) ----

ifeq ($(UNAME),Darwin)
test_metal_vmm: metal_vmm.c metal_vmm_metal.m test_metal_vmm.m metal_vmm.h
	clang -O2 -framework Metal -framework Foundation \
		metal_vmm.c metal_vmm_metal.m test_metal_vmm.m \
		-o test_metal_vmm

test-metal: test_metal_vmm
	@echo "========================================"
	@echo "Running Metal VMM GPU test..."
	@echo "========================================"
	./test_metal_vmm
else
test-metal:
	@echo "Metal test requires macOS — skipping"
endif

# ---- Python VMM test ----

test-python:
	@echo "========================================"
	@echo "Running Python VMM test..."
	@echo "========================================"
	python3 test_mlx_vmm.py

# ---- Run all ----

test: test-vmm test-python
ifeq ($(UNAME),Darwin)
test: test-metal
endif

# ---- Clean ----

clean:
	rm -f test_vmm_remap test_metal_vmm *.o
