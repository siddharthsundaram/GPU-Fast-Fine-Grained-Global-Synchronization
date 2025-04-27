NVCC = nvcc

CFLAGS = -std=c++17 -I/usr/include -I/lusr/cuda-12.2.2/include -L/lusr/cuda-12.2.2/lib64
NVCCFLAGS = -std=c++17 -Xcompiler -fPIC -arch=sm_75 -I/lusr/cuda-12.2.2/include -L/lusr/cuda-12.2.2/lib64

# Add boost libraries and standard C++ libraries
LIBS = -lboost_program_options -lboost_system -lstdc++

MAIN_SRC = main.cu
MAIN_OBJ = $(MAIN_SRC:.cu=.o)
HASH_TABLE_SRC = hash_table.cu
HASH_TABLE_OBJ = $(HASH_TABLE_SRC:.cu=.o)

MAIN_EXE = fine_grain_synch
HASH_TABLE_EXE = hash_table_benchmark

all: clean $(MAIN_EXE) $(HASH_TABLE_EXE)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<

$(MAIN_EXE): $(MAIN_OBJ)
	$(NVCC) -o $(MAIN_EXE) $(MAIN_OBJ)

$(HASH_TABLE_EXE): $(HASH_TABLE_OBJ)
	$(NVCC) -o $(HASH_TABLE_EXE) $(HASH_TABLE_OBJ) $(LIBS)

test:
	./$(MAIN_EXE) inputs/test0

# Updated to only use collision factor and server blocks parameters
hash_table_test:
	./$(HASH_TABLE_EXE) --cf 1024 --servers 4

# Additional benchmark targets for different collision factors
hash_table_256:
	./$(HASH_TABLE_EXE) --cf 256 --servers 4

hash_table_1k:
	./$(HASH_TABLE_EXE) --cf 1024 --servers 4

hash_table_32k:
	./$(HASH_TABLE_EXE) --cf 32768 --servers 4

hash_table_128k:
	./$(HASH_TABLE_EXE) --cf 131072 --servers 4

# Run all collision factor benchmarks sequentially
hash_table_all: $(HASH_TABLE_EXE)
	@echo "Running all hash table benchmarks..."
	@echo "CF=256:"
	@./$(HASH_TABLE_EXE) --cf 256 --servers 4
	@echo "\nCF=1024:"
	@./$(HASH_TABLE_EXE) --cf 1024 --servers 4
	@echo "\nCF=32768:"
	@./$(HASH_TABLE_EXE) --cf 32768 --servers 4
	@echo "\nCF=131072:"
	@./$(HASH_TABLE_EXE) --cf 131072 --servers 4
	@echo "\nBenchmark complete."

clean:
	rm -f *.o $(MAIN_EXE) $(HASH_TABLE_EXE)