all: src/cuda/co_nms.cu.o
	python build.py

src/cuda/co_nms.cu.o: src/cuda/co_nms_kernel.cu
	$(MAKE) -C src/cuda

clean:
	$(MAKE) -C src/cuda clean
