#ifndef WaveSimulation_Registry_h
#define WaveSimulation_Registry_h

#include <glm/glm.hpp>

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
