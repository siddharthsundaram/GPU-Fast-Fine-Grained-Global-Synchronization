# Compiler
NVCC = nvcc
RM = rm -f

# Directories
SRC_DIR = src
OBJ_DIR = obj

# CUDA flags
NVCCFLAGS = -std=c++11 -O3 -arch=sm_75
LDFLAGS = -lcudart

# Source files
CPP_SRC = $(wildcard $(SRC_DIR)/*.cpp)
CU_SRC = $(wildcard $(SRC_DIR)/*.cu)

# Object files
CPP_OBJ = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CPP_SRC))
CU_OBJ = $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(CU_SRC))
ALL_OBJ = $(CPP_OBJ) $(CU_OBJ)

# Executable at top level
TARGET = em_gmm

# Default target
all: directories $(TARGET)

# Create necessary directories
directories:
	@mkdir -p $(OBJ_DIR)

# Link object files
$(TARGET): $(ALL_OBJ)
	$(NVCC) -o $@ $^ $(LDFLAGS)
	@echo "Linking complete: $(TARGET)"

# Compile C++ sources
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Compile CUDA sources
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Clean up artifacts
clean:
	$(RM) $(OBJ_DIR)/*.o $(TARGET)
	@echo "Cleanup complete!"

.PHONY: all clean directories
