#version 410 core

in vec3 texCoord;
out vec4 fColor;
uniform samplerCube cubemap;

void main (void) {
    fColor = texture(cubemap, texCoord);
}