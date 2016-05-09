#ifndef __WaveSimulation__OceanSimulation__
#define __WaveSimulation__OceanSimulation__

#include <OpenGL/gl3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <SDL2/SDL_image.h>

#include "Registry.h"
#include "SkyBox.h"
#include "Waves.h"

class OceanSimulation
{
public:
    OceanSimulation();
    ~OceanSimulation();

public:
    void initGL();
    void render();

private:
    SkyBox skybox;
    Waves waves;

private:
    glm::mat4 ViewMatrix();
};


#endif /* defined(__WaveSimulation__OceanSimulation__) */
