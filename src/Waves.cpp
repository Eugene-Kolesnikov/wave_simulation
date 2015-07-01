#include "Waves.h"
#include <cstdio>
#include "Registry.h"

#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

extern "C" {
	void cudaGenerateSpectrumKernel(float2 *d_h0, float2 *d_ht, unsigned int in_width, unsigned int out_width, unsigned int out_height, float animTime, float patchSize);
	void cudaUpdateHeightmapKernel(float* d_heightMap, float2 *d_ht, unsigned int width, unsigned int height);
	void cudaCalculateSlopeKernel(float2 *slopeOut, float *hptr, unsigned int width, unsigned int height);
}

Waves::Waves()
{
	meshSize = 256;
	spectrum = meshSize + 1;
	patchSize = 200;
	g = 9.81f;
	A = 1e-7f;
	windSpeed = 100.0f;
	timeStep = 1.0f / 25.0f;
	curTime = 0.0f;
	dirDepend = 0.4f;
	windDir.x = 1.0f;
	windDir.y = 0.2f;
}

Waves::~Waves()
{
}

void Waves::initGL()
{
	// create FFT plan
	checkCudaErrors(cufftPlan2d(&fftPlan, meshSize, meshSize, CUFFT_C2C));

	// allocate memory
	int spectrumSize = spectrum*spectrum*sizeof(float2);
	int outputSize =  meshSize*meshSize*sizeof(float2);
	checkCudaErrors(cudaMalloc((void **)&d_h0, spectrumSize));
	h_h0 = (float2 *) malloc(spectrumSize);
	generateH0();
	checkCudaErrors(cudaMemcpy(d_h0, h_h0, spectrumSize, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void **)&d_ht, outputSize));
	checkCudaErrors(cudaMalloc((void **)&d_slope, outputSize));

#ifdef __DEBUG_HEIGHT__
	h_ht = (float2 *) malloc(outputSize);
#endif

	// Create buffer of verticies
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	// create vertex buffers and register with CUDA
	glGenBuffers(1, &heightVertexBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, heightVertexBuffer);
	glBufferData(GL_ARRAY_BUFFER, meshSize*meshSize*sizeof(float), 0, GL_DYNAMIC_DRAW);
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_heightVB_resource, heightVertexBuffer, cudaGraphicsMapFlagsWriteDiscard));
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &slopeVertexBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, slopeVertexBuffer);
	glBufferData(GL_ARRAY_BUFFER, meshSize*meshSize*sizeof(float2), 0, GL_DYNAMIC_DRAW);
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_slopeVB_resource, slopeVertexBuffer, cudaGraphicsMapFlagsWriteDiscard));
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// create vertex and index buffer for mesh
	createMeshPositionVBO(&posVertexBuffer, meshSize, meshSize);
	createMeshIndexBuffer(&indexBuffer, meshSize, meshSize);

	glBindVertexArray(0);

	glShaderV = glCreateShader(GL_VERTEX_SHADER);
	glShaderF = glCreateShader(GL_FRAGMENT_SHADER);
	const GLchar* vShaderSource = loadFile("/Users/eugene/Desktop/cuda-workspace/WaterSimulation/waves.vert.glsl");
	const GLchar* fShaderSource = loadFile("/Users/eugene/Desktop/cuda-workspace/WaterSimulation/waves.frag.glsl");
	glShaderSource(glShaderV, 1, &vShaderSource, NULL);
	glShaderSource(glShaderF, 1, &fShaderSource, NULL);
	delete [] vShaderSource;
	delete [] fShaderSource;
	glCompileShader(glShaderV);
	glCompileShader(glShaderF);
	glProgram = glCreateProgram();
	glAttachShader(glProgram, glShaderV);
	glAttachShader(glProgram, glShaderF);
	glLinkProgram(glProgram);
}


void Waves::render(const glm::mat4& PV)
{
#ifdef __DEBUG_H0__
	static int k = 0;
	if(!k) {
		computeHt();
		++k;
	}
#else
	computeHt();
	cudaDeviceSynchronize();
#endif

	curTime += timeStep;

	glUseProgram(glProgram);

	GLint PVM = glGetUniformLocation(glProgram, "PVM");
	glm::mat4 TranslationMat = glm::translate(glm::mat4(1.0f), glm::vec3(0.0, -500.0, 0.0));
	glUniformMatrix4fv(PVM, 1, GL_FALSE, glm::value_ptr(PV * TranslationMat));

	glBindVertexArray(vao);

	glBindBuffer(GL_ARRAY_BUFFER, heightVertexBuffer);
	GLuint height = glGetAttribLocation(glProgram, "height");
	glVertexAttribPointer(height, 1, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(height);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindBuffer(GL_ARRAY_BUFFER, posVertexBuffer);
	GLuint meshPos = glGetAttribLocation(glProgram, "meshPos");
	glVertexAttribPointer(meshPos, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(meshPos);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindBuffer(GL_ARRAY_BUFFER, slopeVertexBuffer);
	GLuint slope = glGetAttribLocation(glProgram, "slope");
	glVertexAttribPointer(slope, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(slope);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glm::vec3 eyePos = Registry::cameraPos;
	glm::vec3 lightPos(-1000.0, -500.0, 300.0);

	glm::vec3 sourceColor(0.0, 0.027, 0.0623);
	glm::vec3 diffColor(0.165, 0.298, 0.337);
	glm::vec3 specColor(0.325, 0.4588, 0.498);

	glm::vec3 Ka(1,1,1);
	glm::vec3 Kd(0.5, 0.6, 0.7);
	glm::vec3 Ks(0.2,0.2,0.3);
	float alpha = 40.0;

	glUniform3fv( glGetUniformLocation(glProgram, "eyePos"), 		1, glm::value_ptr(eyePos) );
	glUniform3fv( glGetUniformLocation(glProgram, "lightPos"), 		1, glm::value_ptr(lightPos) );
	glUniform3fv( glGetUniformLocation(glProgram, "sourceColor"), 	1, glm::value_ptr(sourceColor) );
	glUniform3fv( glGetUniformLocation(glProgram, "diffColor"), 	1, glm::value_ptr(diffColor) );
	glUniform3fv( glGetUniformLocation(glProgram, "specColor"), 	1, glm::value_ptr(specColor) );
	glUniform3fv( glGetUniformLocation(glProgram, "Ka"),			1, glm::value_ptr(Ka) );
	glUniform3fv( glGetUniformLocation(glProgram, "Kd"), 			1, glm::value_ptr(Kd) );
	glUniform3fv( glGetUniformLocation(glProgram, "Ks"), 			1, glm::value_ptr(Ks) );
	glUniform1f(glGetUniformLocation(glProgram, "alpha"), 			   alpha);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);
//#define __DEBUG_WATER__
#ifdef __DEBUG_WATER__
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
#endif
	glDrawElements(GL_TRIANGLE_STRIP, (meshSize-1)*(meshSize-1) * 2, GL_UNSIGNED_INT, 0);
#ifdef __DEBUG_WATER__
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
#endif

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
	glUseProgram(0);
}

float Waves::phillips(float Kx, float Ky)
{
	float k_squared = Kx * Kx + Ky * Ky;

	if (k_squared == 0.0f) {
		return 0.0f;
	}

	float L = windSpeed * windSpeed / g;
	float k_x = Kx / sqrtf(k_squared);
	float k_y = Ky / sqrtf(k_squared);
	float w_dot_k = k_x * windDir.x + k_y * windDir.y;
	float phillips = A * expf(-1.0f / (k_squared * L * L)) /
			(k_squared * k_squared) * w_dot_k * w_dot_k;

	// filter out waves moving opposite to wind
	if (w_dot_k < 0.0f) {
		phillips *= dirDepend; // dir_depend;
	}

	return phillips;
}

void Waves::generateH0()
{
	for (unsigned int y = 0; y < spectrum; ++y) {
		for (unsigned int x = 0; x < spectrum; ++x) {
			float kx = (-(int)meshSize / 2.0f + x) * (2.0f * CUDART_PI_F / patchSize);
			float ky = (-(int)meshSize / 2.0f + y) * (2.0f * CUDART_PI_F / patchSize);

			float P = sqrtf(phillips(kx, ky));

			float Er = gauss();
			float Ei = gauss();

			float h0_re = Er * P * CUDART_SQRT_HALF_F;
			float h0_im = Ei * P * CUDART_SQRT_HALF_F;

			int i = y * spectrum + x;
			h_h0[i].x = h0_re;
			h_h0[i].y = h0_im;

#ifdef __DEBUG_HEIGHT__
			//printf("%f + 1i * %f\n", h0_re, h0_im);
#endif
		}
	}
}

float Waves::gauss()
{
	// BoxÐMuller transform.
	// Generates Gaussian random number with mean 0 and standard deviation 1.
    float u1 = rand() / (float)RAND_MAX;
    float u2 = rand() / (float)RAND_MAX;

    if (u1 < 1e-6f) {
        u1 = 1e-6f;
    }

    return sqrtf(-2 * logf(u1)) * cosf(2*CUDART_PI_F * u2);
}

void Waves::createMeshPositionVBO(GLuint *id, int w, int h)
{
	glGenBuffers(1, id);
	glBindBuffer(GL_ARRAY_BUFFER, *id);
	glBufferData(GL_ARRAY_BUFFER, w * h * 3 * sizeof(float), 0, GL_STATIC_DRAW);
	float *pos = (float *)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	if (!pos) {
		printf("Error: createMeshPositionVBO\n");
		return;
	}

	for (int y = 0; y < h; ++y) {
		for (int x = 0; x < w; ++x) {
			*pos++ = -1030.0f + 2060.0f / meshSize * x;
	        *pos++ = 0.0f;
	        *pos++ = -1030.0f + 2060.0f / meshSize * y;
	    }
	}

	glUnmapBuffer(GL_ARRAY_BUFFER);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Waves::createMeshIndexBuffer(GLuint *id, int w, int h)
{
	int size = (w-1) * (h-1) * 6 * sizeof(GLuint);

	// create index buffer
	glGenBuffers(1, id);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *id);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, size, 0, GL_STATIC_DRAW);

	// fill with indices for rendering mesh as triangle strips
	GLuint *indices = (GLuint *) glMapBuffer(GL_ELEMENT_ARRAY_BUFFER, GL_WRITE_ONLY);
	if (!indices) {
		printf("Error: createMeshIndexBuffer\n");
		return;
	}

	bool forward = true;

	for (int y = 0; y < h - 1; ++y) {
		if(forward) {
			for (int x = 0; x < w; ++x) {
				*indices++ = y * w + x;
				*indices++ = (y + 1) * w + x;
	    	}
			if(y == h - 2) {
				continue;
			}
			*indices++ = (y+1) * w + (w-2);
			*indices++ = (y+2) * w + (w-1);
			*indices++ = (y+2) * w + (w-2);
			forward = false;
		} else {
			for (int x = w-2; x >= 0; --x) {
				*indices++ = (y + 1) * w + x;
				*indices++ = y * w + x;
			}
			if(y == h - 2) {
				continue;
			}
			*indices++ = (y + 1) * w + 1;
			*indices++ = (y + 1) * w + 0;
			forward = true;
		}
	}

	glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void Waves::computeHt()
{
	size_t num_bytes = meshSize*meshSize*sizeof(float2);

	// generate wave spectrum in frequency domain
	cudaGenerateSpectrumKernel(d_h0, d_ht, spectrum, meshSize, meshSize, curTime, patchSize);
	int spectrumSize = meshSize * meshSize * sizeof(float2);

#ifdef __DEBUG_HEIGHT__
	checkCudaErrors(cudaMemcpy(h_ht, d_ht, spectrumSize, cudaMemcpyDeviceToHost));
	    for(int i = 0; i < meshSize; ++i) {
	    	for(int j = 0; j < meshSize; ++j) {
	    	//	printf("%f + 1i * %f,\n", h_ht[i * meshSize + j].x, h_ht[i * meshSize + j].y);
	    	}
	    }
#endif

	// execute inverse FFT to convert to spatial domain
	checkCudaErrors(cufftExecC2C(fftPlan, d_ht, d_ht, CUFFT_INVERSE));

	// update heightmap values in vertex buffer
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_heightVB_resource, 0));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&g_hptr, &num_bytes, cuda_heightVB_resource));
	cudaUpdateHeightmapKernel(g_hptr, d_ht, meshSize, meshSize);
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_heightVB_resource, 0));

	cudaDeviceSynchronize();

	// calculate slope for shading
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_slopeVB_resource, 0));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&g_sptr, &num_bytes, cuda_slopeVB_resource));
	cudaCalculateSlopeKernel(g_sptr, g_hptr, meshSize, meshSize);
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_slopeVB_resource, 0));

	cudaDeviceSynchronize();

#ifdef __DEBUG_H0__
	checkCudaErrors(cudaMemcpy(h_ht, g_sptr, meshSize*meshSize, cudaMemcpyDeviceToHost));
		    for(int i = 0; i < meshSize; ++i) {
		    	for(int j = 0; j < meshSize; ++j) {
		    		printf("(%f  |  %f) ", h_ht[i * meshSize + j].x, h_ht[i * meshSize + j].y);
		    	}
		    	printf("\n");
		    }
#endif
}
