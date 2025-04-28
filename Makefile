NVCC = nvcc
CXX = g++

CFLAGS = -std=c++17 -I/usr/include -I/lusr/cuda-12.2.2/include -L/lusr/cuda-12.2.2/lib64
NVCCFLAGS = -std=c++17 -Xcompiler -fPIC -arch=sm_75 -I/lusr/cuda-12.2.2/include -L/lusr/cuda-12.2.2/lib64

# Add boost libraries and standard C++ libraries
LIBS = -lboost_program_options -lboost_system -lstdc++ -lcudart

MAIN_SRC = main.cu
MAIN_OBJ = $(MAIN_SRC:.cu=.o)
HASH_TABLE_SRC = hash_table.cu
HASH_TABLE_OBJ = $(HASH_TABLE_SRC:.cu=.o)
CPPSRC = arg_parser.cpp
CPP_OBJS = $(CPPSRC:.cpp=.o)

MAIN_EXE = fine_grain_synch
HASH_TABLE_EXE = hash_table_benchmark

all: clean $(MAIN_EXE) $(HASH_TABLE_EXE)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<

%.o: %.cpp
	$(CXX) $(CFLAGS) -c $< -o $@

$(MAIN_EXE): $(MAIN_OBJ)  $(CPP_OBJS)
	$(NVCC) -o $(MAIN_EXE) $(LIBS) $(MAIN_OBJ) $(CPP_OBJS)

$(HASH_TABLE_EXE): $(HASH_TABLE_OBJ) $(CPP_OBJS)
	$(NVCC) -o $(HASH_TABLE_EXE) $(LIBS) $(HASH_TABLE_OBJ) $(LIBS) $(CPP_OBJS)

test:
	./$(MAIN_EXE) inputs/test0 -s
	./$(MAIN_EXE) inputs/test0 -g
	./$(MAIN_EXE) inputs/test0 -f

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