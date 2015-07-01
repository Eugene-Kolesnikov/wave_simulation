#ifndef __WaveSimulation__SkyBox__
#define __WaveSimulation__SkyBox__

#include <OpenGL/gl3.h>
#include <glm/glm.hpp>
#include <SDL2_image/SDL_image.h>

class SkyBox
{
public:
    SkyBox();
    ~SkyBox();

public:
    void initGL();
    void render(const glm::mat4& PV);

private:
    GLuint vao;
    GLuint vbo;
    GLuint ebo;
    GLuint cubeTexture;
    GLuint glShaderV, glShaderF;
    GLuint glProgram;

private:
    static GLfloat vertices[8 * 3];
    static GLushort indices[36];

private:
    void setupCubeMap(GLuint& texture, SDL_Surface *xpos, SDL_Surface *xneg, SDL_Surface *ypos, SDL_Surface *yneg, SDL_Surface *zpos, SDL_Surface *zneg);
};

#endif /* defined(__WaveSimulation__SkyBox__) */
