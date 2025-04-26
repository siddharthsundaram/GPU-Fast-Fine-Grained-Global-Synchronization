NVCC = nvcc
RM = rm -f

SRC_DIR = src
OBJ_DIR = obj

NVCCFLAGS = -std=c++11 -O3 -arch=sm_75
LDFLAGS = -lcudart

CPP_SRC = $(wildcard $(SRC_DIR)/*.cpp)
CU_SRC = $(wildcard $(SRC_DIR)/*.cu)

CPP_OBJ = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CPP_SRC))
CU_OBJ = $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(CU_SRC))
ALL_OBJ = $(CPP_OBJ) $(CU_OBJ)

TARGET = em_gmm

all: directories $(TARGET)

directories:
	@mkdir -p $(OBJ_DIR)

$(TARGET): $(ALL_OBJ)
	$(NVCC) -o $@ $^ $(LDFLAGS)
	@echo "Linking complete: $(TARGET)"

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	$(RM) $(OBJ_DIR)/*.o $(TARGET)
	@echo "Cleanup complete!"

.PHONY: all clean directories
