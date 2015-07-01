#version 410 core

layout(location = 0) in vec4 vPosition;
out vec3 texCoord;
uniform mat4 PVM;

void main() {
    gl_Position = PVM * vPosition;
    texCoord = vPosition.xyz;
}