#include "SkyBox.h"
#include "Registry.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

GLfloat SkyBox::vertices[8 * 3] = {
    -1000.0,  800.0,  1000.0,
    -1000.0, -800.0,  1000.0,
    1000.0, -800.0,  1000.0,
    1000.0,  800.0,  1000.0,
    -1000.0,  800.0, -1000.0,
    -1000.0, -800.0, -1000.0,
    1000.0, -800.0, -1000.0,
    1000.0,  800.0, -1000.0
};

GLushort SkyBox::indices[36] = {
    0, 1, 2, 2, 0, 3,
    3, 2, 6, 6, 3, 7,
    7, 6, 5, 5, 7, 4,
    4, 5, 1, 1, 4, 0,
    0, 3, 7, 7, 0, 4,
    1, 2, 6, 6, 1, 5
};

SkyBox::SkyBox()
{
}

SkyBox::~SkyBox()
{
}

void SkyBox::initGL()
{
    // Create buffer of indices
    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // Create buffer of verticies
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // Load texture
    glActiveTexture(GL_TEXTURE0);
    glEnable(GL_TEXTURE_CUBE_MAP);
    glGenTextures(1, &cubeTexture);
    glBindTexture(GL_TEXTURE_CUBE_MAP, cubeTexture);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    SDL_Surface *xpos = IMG_Load("../cube/front.png");
    SDL_Surface *xneg = IMG_Load("../cube/back.png");
    SDL_Surface *ypos = IMG_Load("../cube/top.png");
    SDL_Surface *yneg = IMG_Load("../cube/bottom.png");
    SDL_Surface *zpos = IMG_Load("../cube/left.png");
    SDL_Surface *zneg = IMG_Load("../cube/right.png");
    setupCubeMap(cubeTexture, xpos, xneg, ypos, yneg, zpos, zneg);

    glShaderV = glCreateShader(GL_VERTEX_SHADER);
    glShaderF = glCreateShader(GL_FRAGMENT_SHADER);
    const GLchar* vShaderSource = loadFile("../skybox.vert.glsl");
    const GLchar* fShaderSource = loadFile("../skybox.frag.glsl");
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
    glUseProgram(glProgram);
    GLuint vPosition = glGetAttribLocation(glProgram, "vPosition");
    glVertexAttribPointer(vPosition, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glEnableVertexAttribArray(vPosition);

    glUseProgram(0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
    glBindVertexArray(0);
}

void SkyBox::render(const glm::mat4 &PV)
{
    glUseProgram(glProgram);
    glBindVertexArray(vao);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBindTexture(GL_TEXTURE_CUBE_MAP, cubeTexture);

    GLint PVM = glGetUniformLocation(glProgram, "PVM");
    glm::mat4 Rotation = glm::rotate(glm::mat4(1.0f), 0.0f, glm::vec3(0,1,0));
    glm::mat4 TranslationMat = glm::translate(glm::mat4(1.0f), glm::vec3(0.0,0.0,0.0));
    glUniformMatrix4fv(PVM, 1, GL_FALSE, glm::value_ptr(PV * TranslationMat * Rotation));
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_SHORT, NULL);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
    glBindVertexArray(0);
    glUseProgram(0);
}

void SkyBox::setupCubeMap(GLuint& texture, SDL_Surface *xpos, SDL_Surface *xneg, SDL_Surface *ypos, SDL_Surface *yneg, SDL_Surface *zpos, SDL_Surface *zneg) {
    glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X, 0, GL_RGBA, xpos->w, xpos->h, 0, xpos->format->BytesPerPixel == 4 ? GL_BGRA : GL_RGBA, GL_UNSIGNED_BYTE, xpos->pixels);
    glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, 0, GL_RGBA, xneg->w, xneg->h, 0, xneg->format->BytesPerPixel == 4 ? GL_BGRA : GL_RGBA, GL_UNSIGNED_BYTE, xneg->pixels);
    glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Y, 0, GL_RGBA, ypos->w, ypos->h, 0, ypos->format->BytesPerPixel == 4 ? GL_BGRA : GL_RGBA, GL_UNSIGNED_BYTE, ypos->pixels);
    glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, 0, GL_RGBA, yneg->w, yneg->h, 0, yneg->format->BytesPerPixel == 4 ? GL_BGRA : GL_RGBA, GL_UNSIGNED_BYTE, yneg->pixels);
    glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Z, 0, GL_RGBA, zpos->w, zpos->h, 0, zpos->format->BytesPerPixel == 4 ? GL_BGRA : GL_RGBA, GL_UNSIGNED_BYTE, zpos->pixels);
    glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, 0, GL_RGBA, zneg->w, zneg->h, 0, zneg->format->BytesPerPixel == 4 ? GL_BGRA : GL_RGBA, GL_UNSIGNED_BYTE, zneg->pixels);
}
