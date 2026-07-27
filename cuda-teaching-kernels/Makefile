NVCC ?= nvcc
ARCH ?= -arch=native
FLAGS := -O3 $(ARCH) -std=c++17
SOURCES := reduction_01_naive_atomic.cu reduction_02_shared_tree.cu reduction_03_warp_shuffle.cu reduction_04_grid_stride.cu transpose_01_naive.cu transpose_02_shared_tile.cu transpose_03_padded_tile.cu
TARGETS := $(SOURCES:.cu=.exe)

.PHONY: all run clean
all: $(TARGETS)

%.exe: %.cu cuda_common.cuh
	$(NVCC) $(FLAGS) $< -o $@

run: all
	@rm -f results.md
	@$(foreach t,$(TARGETS),./$(t);)

clean:
	@rm -f $(TARGETS) results.md
