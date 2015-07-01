#include "OceanSimulation.h"

OceanSimulation::OceanSimulation()
{
}

OceanSimulation::~OceanSimulation()
{
}

//#define __DEBUG_WATER__

void OceanSimulation::initGL()
{
#ifdef __DEBUG_WATER__
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
#else
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
#endif
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

#ifndef __DEBUG_WATER__
    skybox.initGL();
#endif
    waves.initGL();
}

void OceanSimulation::render()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glm::mat4 Projection = glm::perspective(45.5f, (float)Registry::width / Registry::height, 0.1f, 3000.0f);
    glm::mat4 RotationPitch = glm::rotate(glm::mat4(1.0f), -Registry::pitch, glm::vec3(1,0,0));
    glm::mat4 RotationYaw = glm::rotate(glm::mat4(1.0f), -Registry::yaw, glm::vec3(0,1,0));
    glm::mat4 Translate = glm::translate(glm::mat4(1.0f),Registry::cameraPos);
    glm::mat4 PV = Projection * RotationPitch * RotationYaw * Translate;

#ifndef __DEBUG_WATER__
    skybox.render(PV);
#endif
    waves.render(PV);

    glFlush();
}
