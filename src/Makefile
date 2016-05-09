NVCC        = nvcc
CLANG       = clang++
NVCCFLAGS = -arch=sm_30 -ccbin=$(CLANG) -O3 -lglfw3 -lSDL2 -lSDL2_image -lSDLmain -lcufft \
	-Xlinker -framework,Cocoa -Xlinker -framework,OpenGL -Xlinker -framework,IOKit -Xlinker -framework,CoreVideo \
	-I/usr/local/Cellar/sdl/1.2.15/include/SDL -I/usr/local/Cellar/sdl2/2.0.4/include -I/usr/local/Cellar/sdl2_image/2.0.1_1/include \
	-L/usr/local/Cellar/sdl/1.2.15/lib -L/usr/local/Cellar/sdl2/2.0.4/lib -L/usr/local/Cellar/sdl2_image/2.0.1_1/lib

OBJECTS = main.o \
	Waves.o \
	SkyBox.o \
	Registry.o \
	OceanSimulation.o \
	Waves_kernel.o

all: $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) $(OBJECTS) -o waves_simulation

%.o: %.cpp
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm *.o
