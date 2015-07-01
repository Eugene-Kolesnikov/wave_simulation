#version 410 core

in vec4 pos;
out vec4 fColor;

uniform vec3 sourceColor;
uniform vec3 diffColor;
uniform vec3 specColor;
uniform vec3 lightPos;
uniform vec3 eyePos;

in vec3 l;
in vec3 h;
in vec3 n;
in vec3 r;

uniform vec3 Ka;
uniform vec3 Kd;
uniform vec3 Ks;
uniform float alpha;

vec3 BlinnPhongModel()
{
    return  Ka * sourceColor +
            Kd * max(dot(n, -l), 0.0) * diffColor +
            Ks * max(pow(dot(n, h), alpha), 0.0) * specColor;
}


void main (void) {
    //fColor = vec4(exp(-abs(pos.x)/2000.0), exp(-abs(pos.y) / 200.0), exp(-abs(pos.z)/2000.0), 1.0);
    vec3 BlinnPhong = exp(-0.8 + 1.2*abs(pos.x/3000+pos.z/3000)) * BlinnPhongModel();
    fColor = vec4(BlinnPhong, 0.9);
}