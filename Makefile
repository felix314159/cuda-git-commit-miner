SHELL := /bin/bash

NVCC ?= nvcc
CXX ?= c++
TARGET ?= gitminer-head
CPU_TARGET ?= gitminer-cpu
CUDA_TARGET ?= gitminer-cuda
SRC := main.cu
CUDA_ARCH ?= sm_89
# Ubuntu 24.04 commonly pairs GCC 13 with CUDA 12.0, which nvcc marks as
# unsupported even though it often still works for small projects like this.
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(CUDA_ARCH) -allow-unsupported-compiler -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=2
UNAME_M := $(shell uname -m)
UNAME_S := $(shell uname -s)
CPU_DEFAULT_ARCHES := aarch64 arm64
CPU_CXXFLAGS ?= -O3 -std=c++17

ifneq ($(filter $(UNAME_M),$(CPU_DEFAULT_ARCHES)),)
DEFAULT_BACKEND := cpu
else
DEFAULT_BACKEND := cuda
endif

ifeq ($(UNAME_S),Darwin)
ifeq ($(UNAME_M),arm64)
CPU_CXXFLAGS += -mcpu=apple-m1
endif
else
CPU_CXXFLAGS += -march=native
endif

# nvcc 12.0 on Ubuntu 24.04 aarch64 trips over newer glibc ARM vector-math
# declarations. Lowering the GNU version macros for the nvcc frontend avoids
# that toolkit bug while still using the system compiler underneath.
ifeq ($(UNAME_M),aarch64)
NVCCFLAGS += -D__GNUC__=8 -D__GNUC_MINOR__=0 -D__GNUC_PATCHLEVEL__=0
NVCC_STDERR_FILTER = 2> >(sed \
	-e '/^<command-line>: warning: "__GNUC__" redefined$$/d' \
	-e '/^<command-line>: warning: "__GNUC_MINOR__" redefined$$/d' \
	-e '/^<built-in>: note: this is the location of the previous definition$$/d' \
	>&2)
else
NVCC_STDERR_FILTER =
endif

.PHONY: all clean cpu cuda run run-cpu run-cuda

all: $(TARGET)

ifeq ($(DEFAULT_BACKEND),cpu)
$(TARGET): $(SRC) Makefile
	$(CXX) $(CPU_CXXFLAGS) -DGITMINER_CPU_ONLY -x c++ $< -o $@
else
$(TARGET): $(SRC) Makefile
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(NVCC_STDERR_FILTER)
endif

cpu: $(CPU_TARGET)

$(CPU_TARGET): $(SRC) Makefile
	$(CXX) $(CPU_CXXFLAGS) -DGITMINER_CPU_ONLY -x c++ $< -o $@

cuda: $(CUDA_TARGET)

$(CUDA_TARGET): $(SRC) Makefile
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(NVCC_STDERR_FILTER)

run: $(TARGET)
	./$(TARGET)

run-cpu: $(CPU_TARGET)
	./$(CPU_TARGET)

run-cuda: $(CUDA_TARGET)
	./$(CUDA_TARGET)

clean:
	rm -f $(TARGET) $(CPU_TARGET) $(CUDA_TARGET)
