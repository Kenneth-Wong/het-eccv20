all: co_nms_kernel.cu co_nms_kernel.h
	/usr/local/cuda/bin/nvcc -c -o co_nms.cu.o co_nms_kernel.cu --compiler-options -fPIC -gencode arch=compute_61,code=sm_61
clean:
	rm co_nms.cu.o