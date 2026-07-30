###########################################################

CUDA_ROOT_DIR=/usr/local/cuda

###########################################################

CC=gcc
CC_FLAGS=
CC_LIBS=

###########################################################

NVCC=nvcc
NVCC_FLAGS=-arch=native -O3 -Xptxas -O3 -Xcompiler -O3
NVCC_LIBS=

CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
CUDA_LINK_LIBS= -lcudart -lstdc++

###########################################################

SRC_DIR = src
OBJ_DIR = bin
INC_DIR = include

###########################################################

EXE = mayo-2-gpu-fine

OBJS = $(OBJ_DIR)/main.o \
       $(OBJ_DIR)/mayo.o \
       $(OBJ_DIR)/inputs.o \
       $(OBJ_DIR)/fips202.o \
       $(OBJ_DIR)/aes_c.o 

###########################################################

$(EXE) : $(OBJS)
	$(NVCC) $(NVCC_FLAGS) $(OBJS) -o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(OBJ_DIR)/main.o : main.cu | $(OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) -I$(INC_DIR) $(CUDA_INC_DIR) -c $< -o $@

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) -I$(INC_DIR) $(CUDA_INC_DIR) -c $< -o $@

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CC_FLAGS) -I$(INC_DIR) $(CUDA_INC_DIR) -c $< -o $@

clean:
	$(RM) bin/* *.o $(EXE)