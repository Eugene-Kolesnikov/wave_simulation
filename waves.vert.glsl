#version 410 core

layout(location = 0) in vec4 meshPos;
layout(location = 1) in float height;
layout(location = 2) in vec2 slope;

uniform mat4 PVM;
uniform vec3 lightPos;
uniform vec3 eyePos;

out vec3 l;
out vec3 h;
out vec3 n;
out vec3 r;

out vec4 pos;

void main() {
    vec3 lp = abs(lightPos);
    vec3 p = vec3(meshPos.x, 1e+2 * height, meshPos.z);
    gl_Position = PVM * vec4(p, 1.0);
    p.x = p.x - 1000; p.z = p.z - 1000;
    p.y = p.y - 500;
    pos = vec4(p, 1.0);
    l = normalize(lp - p);
    vec3 v = normalize(eyePos - p);
    h = normalize((v + l) / length(v + l));
    n = normalize(cross( vec3(0.0, slope.y, 1.0 / 256), vec3(1.0 / 256, slope.x, 0.0)));
    r = reflect(-l, n);
}