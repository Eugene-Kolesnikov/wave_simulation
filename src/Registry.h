#ifndef WaveSimulation_Registry_h
#define WaveSimulation_Registry_h

#ifdef __APPLE__
# define __gl_h_
# define GL_DO_NOT_WARN_IF_MULTI_GL_VERSION_HEADERS_INCLUDED
#endif

#include <glm/glm.hpp>

#define checkCudaErrors(call) {										             \
    cudaError err = call;												         \
    if(err != cudaSuccess) {											         \
        fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",	         \
            __FILE__, __LINE__, cudaGetErrorString(err));				         \
        exit(1);														         \
    }																	         \
}

class Registry
{
public:
    static glm::vec2 pMouse;
    static glm::vec3 cameraPos;
    static float pitch;
    static float yaw;
    static int width;
    static int height;
};

char* loadFile(const char *filename);

#endif
