NVCC = nvcc

CFLAGS = -std=c++17 -I/usr/include -I/lusr/cuda-12.2.2/include -L/lusr/cuda-12.2.2/lib64
NVCCFLAGS = -std=c++17 -Xcompiler -fPIC -arch=sm_75 -I/lusr/cuda-12.2.2/include -L/lusr/cuda-12.2.2/lib64

CUDASRC = main.cu
CUDA_OBJS = $(CUDASRC:.cu=.o)

EXE = fine_grain_synch

all: $(EXE)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<

$(EXE): $(CUDA_OBJS)
	$(NVCC) -o $(EXE) $(CUDA_OBJS)

test:
	./fine_grain_synch inputs/test0