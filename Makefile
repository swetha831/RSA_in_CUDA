.SUFFIXES:  .cpp .cu .o
CUDA_HOME := /usr/local/cuda
INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib -lcudart
CC	:= nvcc
DEP	:=  
ifeq ($(def), )
DEF := 
else
DEF := -D$(def)
endif

NVCCFLAGS	:= -lineinfo -arch=sm_53 --ptxas-options=-v -g -rdc=true --use_fast_math

all:	optimized rsa

optimized:	rsamodified.o $(DEP)
	$(CC) $(INC) $(NVCCFLAGS) -o rsamodified rsamodified.o $(LIB)

rsa:	rsa.o $(DEP)
	$(CC) $(INC) $(NVCCFLAGS) -o rsa rsa.o $(LIB)

.cpp.o:
	$(CC) $(INC) $(NVCCFLAGS) $(DEF) -c $< -o $@ 

.cu.o:
	$(CC) $(INC) $(NVCCFLAGS) $(DEF) -c $< -o $@
	

clean:
	rm -f *.o rsamodified rsa


