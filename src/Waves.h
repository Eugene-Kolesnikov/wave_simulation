#ifndef __WaveSimulation__Waves__
#define __WaveSimulation__Waves__

//#define __DEBUG__
#include <OpenGL/gl3.h>
#include <glm/glm.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cufft.h>
#include <math_constants.h>



class Waves
{
public:
    Waves();
    ~Waves();

public:
    void initGL();
    void render(const glm::mat4& PV);

private:
    float phillips(float Kx, float Ky);
    void generateH0();
    float gauss();
    void createMeshPositionVBO(GLuint *id, int w, int h);
    void createMeshIndexBuffer(GLuint *id, int w, int h);
    void computeHt();

private:
    unsigned int meshSize;
    unsigned int spectrum;

    GLuint glShaderV, glShaderF;
    GLuint glProgram;

    float g;
    float A;
    float patchSize;
    float2 windDir;
    float windSpeed;
    float timeStep;
    float curTime;
    float dirDepend;

    // OpenGL vertex buffers
    GLuint vao;
    GLuint posVertexBuffer;
    GLuint heightVertexBuffer, slopeVertexBuffer;
    struct cudaGraphicsResource *cuda_posVB_resource, *cuda_heightVB_resource, *cuda_slopeVB_resource; // handles OpenGL-CUDA exchange
    GLuint indexBuffer;

    // FFT data
    cufftHandle fftPlan;
    float2 *d_h0;   // heightfield at time 0
    float2 *d_ht;   // heightfield at time t
    float2 *d_slope;
    float2* h_h0;

#ifdef __DEBUG_HEIGHT__
    float2* h_ht;
#endif

    // pointers to device object
    float *g_hptr;
    float2 *g_sptr;
};

#endif /* defined(__WaveSimulation__Waves__) */
