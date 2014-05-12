GCC_FLAGS:= -std=c++0x -Wall -g -DDEBUG -O0
TARGET   := fastmatrix

CFILES  = $(wildcard *.cpp)
OBJS1   = $(notdir $(CFILES))
OBJS2   = $(patsubst %.cpp,%.o, $(OBJS1))
OBJS    = $(addprefix build/,$(OBJS2))	

INTEL:=1
NVIDIA:=1

ifdef INTEL
	OCL_LIB += -L/usr/lib64 -lOpenCL
	OCL_INC += -I/usr/include
endif

ifdef NVIDIA
	#OCL_LIB += -L/usr/lib64 -lOpenCL
	OCL_LIB += -L/usr/lib64/nvidia -lOpenCL
	OCL_INC += -I/usr/local/cuda/include
endif

ifdef AMD
	OCL_LIB += -L/usr/lib64 -lOpenCL
	OCL_INC += -I/opt/AMDAPP/include
endif

LIBS     := -L../../OpenCL-Wrapper/Code/lib/ -lOclWrapper -lGL \
-lpthread $(OCL_LIB)
INCS     := -I./matrix -I./util -I../../OpenCL-Wrapper/Code/inc $(OCL_INC)

default: $(TARGET)

all: clean $(TARGET)

run: $(TARGET)
	./$(TARGET)

objdir:
	@mkdir -p build

$(TARGET): objdir $(OBJS)
	g++ $(GCC_FLAGS) $(OBJS) -o $(TARGET) $(LIBS)

build/%.o : %.cpp
	g++ -c $(INCS) $(GCC_FLAGS) $< -o $@

.PHONY : clean

clean:
	rm -f build/*  $(TARGET)

