SHELL := /bin/bash

NVCC ?= nvcc
TARGET ?= gitminer-head
SRC := main.cu
CUDA_ARCH ?= sm_89
# Ubuntu 24.04 commonly pairs GCC 13 with CUDA 12.0, which nvcc marks as
# unsupported even though it often still works for small projects like this.
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(CUDA_ARCH) -allow-unsupported-compiler -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=2
UNAME_M := $(shell uname -m)

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

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(NVCC_STDERR_FILTER)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
