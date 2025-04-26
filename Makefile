NVCC = nvcc

CFLAGS = -std=c++17 -I/usr/include -I/lusr/cuda-12.2.2/include -L/lusr/cuda-12.2.2/lib64
NVCCFLAGS = -std=c++17 -Xcompiler -fPIC -arch=sm_75 -I/lusr/cuda-12.2.2/include -L/lusr/cuda-12.2.2/lib64

MAIN_SRC = main.cu
MAIN_OBJ = $(MAIN_SRC:.cu=.o)
HASH_TABLE_SRC = hash_table_benchmark.cu
HASH_TABLE_OBJ = $(HASH_TABLE_SRC:.cu=.o)

MAIN_EXE = fine_grain_synch
HASH_TABLE_EXE = hash_table_bench

all: clean $(MAIN_EXE) $(HASH_TABLE_EXE)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<

$(MAIN_EXE): $(MAIN_OBJ)
	$(NVCC) -o $(MAIN_EXE) $(MAIN_OBJ)

$(HASH_TABLE_EXE): $(HASH_TABLE_OBJ)
	$(NVCC) -o $(HASH_TABLE_EXE) $(HASH_TABLE_OBJ)

test:
	./$(MAIN_EXE) inputs/test0

hash_table_test:
	./$(HASH_TABLE_EXE) -cf 1024 -n 1000000 -s 4 -c 16

clean:
	rm -f *.o $(MAIN_EXE) $(HASH_TABLE_EXE)