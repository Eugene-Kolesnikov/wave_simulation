//#define __DEBUG__
#define GLFW_INCLUDE_GLCOREARB

#include <GLFW/glfw3.h>
#include "OceanSimulation.h"
#include "Registry.h"
#include <cstdio>

double cursorX;
double cursorY;

void error_callback(int error, const char* description);
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void cursor_position_callback(GLFWwindow* window, double x, double y);

int main(void)
{
    GLFWwindow* window;

    if(!glfwInit()) {
        printf("Error: glfwInit\n");
        exit(EXIT_FAILURE);
    }

    glfwSetErrorCallback(error_callback);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(1024, 1024, "Wave Simulation", NULL, NULL);
    if(!window) {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glfwSetKeyCallback(window, key_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

#ifdef __DEBUG__
    int major, minor, rev;
    major = glfwGetWindowAttrib(window, GLFW_CONTEXT_VERSION_MAJOR);
    minor = glfwGetWindowAttrib(window, GLFW_CONTEXT_VERSION_MINOR);
    rev = glfwGetWindowAttrib(window, GLFW_CONTEXT_REVISION);
    printf("OpenGL version recieved: %d.%d.%d\n", major, minor, rev);
    printf("Supported OpenGL is %s\n", (const char*)glGetString(GL_VERSION));
    printf("Supported GLSL is %s\n\n", (const char*)glGetString(GL_SHADING_LANGUAGE_VERSION));
#endif

    OceanSimulation simulation;
    simulation.initGL();

    while (!glfwWindowShouldClose(window))
    {
        glfwGetFramebufferSize(window, &Registry::width, &Registry::height);

        simulation.render();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    cudaDeviceReset();
    return EXIT_SUCCESS;
}

void error_callback(int error, const char* description)
{
    fprintf(stderr, "Error: %s\n", description);
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (action != GLFW_PRESS && action != GLFW_REPEAT)
        return;

    glm::vec3 e(sin(Registry::yaw) * cos(Registry::pitch), sin(Registry::yaw) * sin(Registry::pitch), cos(Registry::yaw));
    e = glm::normalize(e)*20.0f;
    switch (key) {
        case GLFW_KEY_ESCAPE: glfwSetWindowShouldClose(window, GL_TRUE); break;
        case GLFW_KEY_W: Registry::cameraPos += e; break;
        case GLFW_KEY_S: Registry::cameraPos -= e; break;
        case GLFW_KEY_A:
        case GLFW_KEY_F:
        default:
            break;
    }
#ifdef __DEBUG__
    printf("%f %f %f\n", Registry::cameraPos.x, Registry::cameraPos.y, Registry::cameraPos.z);
#endif
}

bool cursor = false;

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if (button != GLFW_MOUSE_BUTTON_LEFT)
        return;

    if(action == GLFW_PRESS) {
        cursor = true;
        glfwGetCursorPos(window, &cursorX, &cursorY);
    } else {
        cursor = false;
    }
}

float clamp(float x, float a, float b)
{
    if(a > x)
        return a;
    if(b < x)
        return b;
    return x;
}

void cursor_position_callback(GLFWwindow* window, double x, double y)
{
    if(cursor) {
        Registry::pitch = clamp(Registry::pitch - (y - cursorY) * 0.01f, - M_PI_2 + 1e-4f, M_PI_2 - 1e-4f);
        Registry::yaw -= (x - cursorX) * 0.01f;
        cursorX = x;
        cursorY = y;
    }
}

