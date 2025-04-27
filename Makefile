NVCC = nvcc
CXX = g++

CFLAGS = -std=c++17 -I/usr/include -I/lusr/cuda-12.2.2/include -L/lusr/cuda-12.2.2/lib64
NVCCFLAGS = -std=c++17 -Xcompiler -fPIC -arch=sm_75 -I/lusr/cuda-12.2.2/include -L/lusr/cuda-12.2.2/lib64

LIBS = -lboost_program_options -lcudart

CUDASRC = main.cu
CUDA_OBJS = $(CUDASRC:.cu=.o)

CPPSRC = arg_parser.cpp
CPP_OBJS = $(CPPSRC:.cpp=.o)

EXE = fine_grain_synch

all: clean $(EXE)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<

%.o: %.cpp
	$(CXX) $(CFLAGS) -c $< -o $@

$(EXE): $(CUDA_OBJS) $(CPP_OBJS)
	$(NVCC) -o $(EXE) $(LIBS) $(CUDA_OBJS) $(CPP_OBJS)

test:
	./fine_grain_synch inputs/test0 -s
	./fine_grain_synch inputs/test0 -g
	./fine_grain_synch inputs/test0 -f

clean:
	rm -f main.o
	rm -f arg_parser.o
	rm -f $(EXE)