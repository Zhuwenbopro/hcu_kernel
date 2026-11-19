HIPCC ?= hipcc
.DELETE_ON_ERROR:

build/libgemma_int8_5120.so: gemma_fused_add_rmsnorm_int8_5120.cu
	mkdir -p build
	$(HIPCC) -x hip -O3 -std=c++17 --offload-arch=gfx936 -shared -fPIC $< -o $@
