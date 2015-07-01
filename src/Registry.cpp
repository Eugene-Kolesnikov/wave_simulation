#include "Registry.h"
#include <fstream>
#include <sstream>

glm::vec2 Registry::pMouse = glm::vec2(0.0f, 0.0f);
glm::vec3 Registry::cameraPos = glm::vec3(0.0f, +100.0f, 0.0f);
float Registry::pitch = 0.0;
float Registry::yaw = M_PI_2;
int Registry::width = 0;
int Registry::height = 0;

char* loadFile(const char *filename) {
    char* data;
    int len;
    std::ifstream ifs(filename, std::ifstream::in);
    if(ifs.is_open() == false) {
        printf("File not open!\n");
    }
    ifs.seekg(0, std::ios::end);
    len = (int)ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    data = new char[len + 1];
    ifs.read(data, len);
    data[len] = 0;
    ifs.close();
    return data;
}
