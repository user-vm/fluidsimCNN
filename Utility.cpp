#include "Utility.h"
#include "pez.h"
#include <string.h>
#include <cmath>
#include <cstdio>

using namespace vmath;

static struct {
    GLuint Advect;
    GLuint AdvectVelocity;
    GLuint Jacobi;
    GLuint SubtractGradient;
    GLuint ComputeDivergence;
    GLuint ApplyImpulse;
    GLuint ApplyBuoyancy;
    GLuint UpdateObstacles;
    GLuint UpdateObstacleSpeeds;
    GLuint ApplyObstacleSpeeds;
    GLuint VorticityConfinement;
    GLuint CopyVelocity;
} Programs;

const float JitterTemperature = 200.0;
const float JitterDensity = 5.0;

//these values set the resolution of the simulation, and other important values
//the fact
const float CellSize = 1.25f;
const int GridWidth = 64;
const int GridHeight = 64;
const int GridDepth = 64;
const float SplatRadius = GridWidth / 16.0f;
const float AmbientTemperature = -10.0f;
const float ImpulseTemperature = 100.0f;
const float ImpulseDensity = 1.25f;
const int NumJacobiIterations = 40;
float TimeStep = 0.25f;
const float SmokeBuoyancy = 10.0f;
const float SmokeWeight = 1.0f;
const float GradientScale = 1.0f/CellSize;
const float TemperatureDissipation = 1.0f;
const float VelocityDissipation = 1.0f;
const float DensityDissipation = 1.0f;
const float VorticityBoost = 0.0f;
Vector4 XYImpulsePosition(0,0,0,0);
Vector4 XYImpulsePositionLast(0,0,0,0);
Vector3 ImpulsePosition( GridWidth / 2.0f, (GridHeight - (int) SplatRadius) / 2.0f, GridDepth / 2.0f);

const bool DrawMovingCube = true;
const dim3 CubeSize = dim3(GridWidth/8, GridHeight/8, GridDepth/16);

extern const float inverseBeta = 0.1666f;
extern const float obstacleSpeedMultiplier = 1.0f;

//destSpeed is NULL by default
void CreateObstacles(SurfacePod dest, SurfacePod destSpeed)
{
    glBindFramebuffer(GL_FRAMEBUFFER, dest.FboHandle);
    glViewport(0, 0, dest.Width, dest.Height);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    GLuint program = LoadProgram("Fluid.Vertex", 0, "Fluid.Fill");
    glUseProgram(program);

    GLuint cubeVbo;
    glGenBuffers(1, &cubeVbo);
    GLuint lineVbo;
    glGenBuffers(1, &lineVbo);
    GLuint circleVbo;
    glGenBuffers(1, &circleVbo);

    glEnableVertexAttribArray(SlotPosition);

    // Cleanup
    glDeleteProgram(program);
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &cubeVbo);
    glDeleteBuffers(1, &lineVbo);
    glDeleteBuffers(1, &circleVbo);
}

GLuint LoadProgram(const char* vsKey, const char* gsKey, const char* fsKey)
{
    const char* vsSource = pezGetShader(vsKey);
    const char* gsSource = pezGetShader(gsKey);
    const char* fsSource = pezGetShader(fsKey);

    const char* msg = "Can't find %s shader: '%s'.\n";
    pezCheck(vsSource != 0, msg, "vertex", vsKey);
    pezCheck(gsKey == 0 || gsSource != 0, msg, "geometry", gsKey);
    pezCheck(fsKey == 0 || fsSource != 0, msg, "fragment", fsKey);

    GLint compileSuccess;
    GLchar compilerSpew[256];
    GLuint programHandle = glCreateProgram();

    GLuint vsHandle = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vsHandle, 1, &vsSource, 0);
    glCompileShader(vsHandle);
    glGetShaderiv(vsHandle, GL_COMPILE_STATUS, &compileSuccess);
    glGetShaderInfoLog(vsHandle, sizeof(compilerSpew), 0, compilerSpew);
    pezCheck(compileSuccess, "Can't compile %s:\n%s", vsKey, compilerSpew);
    glAttachShader(programHandle, vsHandle);

    GLuint gsHandle;
    if (gsKey) {
        gsHandle = glCreateShader(GL_GEOMETRY_SHADER);
        glShaderSource(gsHandle, 1, &gsSource, 0);
        glCompileShader(gsHandle);
        glGetShaderiv(gsHandle, GL_COMPILE_STATUS, &compileSuccess);
        glGetShaderInfoLog(gsHandle, sizeof(compilerSpew), 0, compilerSpew);
        pezCheck(compileSuccess, "Can't compile %s:\n%s", gsKey, compilerSpew);
        glAttachShader(programHandle, gsHandle);
    }

    GLuint fsHandle;
    if (fsKey) {
        fsHandle = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fsHandle, 1, &fsSource, 0);
        glCompileShader(fsHandle);
        glGetShaderiv(fsHandle, GL_COMPILE_STATUS, &compileSuccess);
        glGetShaderInfoLog(fsHandle, sizeof(compilerSpew), 0, compilerSpew);
        pezCheck(compileSuccess, "Can't compile %s:\n%s", fsKey, compilerSpew);
        glAttachShader(programHandle, fsHandle);
    }

    glBindAttribLocation(programHandle, SlotPosition, "Position");
    glBindAttribLocation(programHandle, SlotTexCoord, "TexCoord");
    glLinkProgram(programHandle);

    GLint linkSuccess;
    glGetProgramiv(programHandle, GL_LINK_STATUS, &linkSuccess);
    glGetProgramInfoLog(programHandle, sizeof(compilerSpew), 0, compilerSpew);

    if (!linkSuccess) {
        pezPrintString("Link error.\n");
        if (vsKey) pezPrintString("Vertex Shader: %s\n", vsKey);
        if (gsKey) pezPrintString("Geometry Shader: %s\n", gsKey);
        if (fsKey) pezPrintString("Fragment Shader: %s\n", fsKey);
        pezPrintString("%s\n", compilerSpew);
    }

    return programHandle;
}

SlabPod CreateSlab(GLsizei width, GLsizei height, GLsizei depth, int numComponents)
{
    SlabPod slab;
    slab.Ping = CreateVolume(width, height, depth, numComponents);
    slab.Pong = CreateVolume(width, height, depth, numComponents);
    return slab;
}

SurfacePod CreateSurface(GLsizei width, GLsizei height, int numComponents)
{
    GLuint fboHandle;
    glGenFramebuffers(1, &fboHandle);
    glBindFramebuffer(GL_FRAMEBUFFER, fboHandle);

    GLuint textureHandle;
    glGenTextures(1, &textureHandle);
    glBindTexture(GL_TEXTURE_2D, textureHandle);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    switch (numComponents) {
        case 1:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, 0);
            break;
        case 2:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, width, height, 0, GL_RG, GL_FLOAT, 0);
            break;
        case 3:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, width, height, 0, GL_RGB, GL_FLOAT, 0);
            break;
        case 4:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, 0);
            break;
    }

    pezCheck(GL_NO_ERROR == glGetError(), "Unable to create normals texture");

    GLuint colorbuffer;
    glGenRenderbuffers(1, &colorbuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, colorbuffer);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureHandle, 0);
    pezCheck(GL_NO_ERROR == glGetError(), "Unable to attach color buffer");

    pezCheck(GL_FRAMEBUFFER_COMPLETE == glCheckFramebufferStatus(GL_FRAMEBUFFER), "Unable to create FBO.");
    SurfacePod surface = { fboHandle, textureHandle };

    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    surface.Width = width;
    surface.Height = height;
    surface.Depth = 1;
    return surface;
}

SurfacePod CreateVolume(GLsizei width, GLsizei height, GLsizei depth, int numComponents)
{
    GLenum err = glGetError();
    std::stringstream errorBuffer;

    errorBuffer<<err;

    if(err != GL_NO_ERROR)
      pezCheck(0,errorBuffer.str().c_str());

    GLuint fboHandle;
    glGenFramebuffers(1, &fboHandle);
    glBindFramebuffer(GL_FRAMEBUFFER, fboHandle);

    GLuint textureHandle;
    glGenTextures(1, &textureHandle);
    glBindTexture(GL_TEXTURE_3D, textureHandle);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); //clamps edges of 3D volume texture (so color is restricted to the volume)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); //along the s,t and r texture coordinates
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    switch (numComponents) {
        case 1:
            glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, width, height, depth, 0, GL_RED, GL_FLOAT, 0);
            break;
        case 2:
            glTexImage3D(GL_TEXTURE_3D, 0, GL_RG32F, width, height, depth, 0, GL_RG, GL_FLOAT, 0);
            break;
        case 3:
            glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB32F, width, height, depth, 0, GL_RGB, GL_FLOAT, 0);
            break;
        case 4:
            glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA32F, width, height, depth, 0, GL_RGBA, GL_FLOAT, 0);
            break;
    }

    errorBuffer<<err;

    if(err != GL_NO_ERROR)
      pezCheck(0,errorBuffer.str().c_str());

    GLuint colorbuffer;
    glGenRenderbuffers(1, &colorbuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, colorbuffer);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, textureHandle, 0);
    pezCheck(GL_NO_ERROR == glGetError(), "Unable to attach color buffer");

    pezCheck(GL_FRAMEBUFFER_COMPLETE == glCheckFramebufferStatus(GL_FRAMEBUFFER), "Unable to create FBO.");
    SurfacePod surface = { fboHandle, textureHandle };

    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    glBindFramebuffer(GL_FRAMEBUFFER, 0); //the 0 value breaks the binding (as desired)
    surface.Width = width;
    surface.Height = height;
    surface.Depth = depth;
    return surface;
}

static void ResetState()
{
    glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_3D, 0);
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_3D, 0);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_3D, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDisable(GL_BLEND);
}

void InitSlabOps()
{
    Programs.Advect = LoadProgram("Fluid.Vertex", "Fluid.PickLayer", "Fluid.Advect");
    Programs.AdvectVelocity = LoadProgram("Fluid.Vertex", "Fluid.PickLayer", "Fluid.AdvectVelocity");
    Programs.Jacobi = LoadProgram("Fluid.Vertex", "Fluid.PickLayer", "Fluid.Jacobi");
    Programs.SubtractGradient = LoadProgram("Fluid.Vertex", "Fluid.PickLayer", "Fluid.SubtractGradient");
    Programs.ComputeDivergence = LoadProgram("Fluid.Vertex", "Fluid.PickLayer", "Fluid.ComputeDivergence");
    Programs.ApplyImpulse = LoadProgram("Fluid.Vertex", "Fluid.PickLayer", "Fluid.Splat");
    Programs.ApplyBuoyancy = LoadProgram("Fluid.Vertex", "Fluid.PickLayer", "Fluid.Buoyancy");
}

void InitSlabOpsNotStaggered(){
  Programs.Advect = LoadProgram("Fluid_not_staggered.Vertex", "Fluid_not_staggered.PickLayer", "Fluid_not_staggered.Advect");
  Programs.Jacobi = LoadProgram("Fluid_not_staggered.Vertex", "Fluid_not_staggered.PickLayer", "Fluid_not_staggered.Jacobi");
  Programs.SubtractGradient = LoadProgram("Fluid_not_staggered.Vertex", "Fluid_not_staggered.PickLayer", "Fluid_not_staggered.SubtractGradient");
  Programs.ComputeDivergence = LoadProgram("Fluid_not_staggered.Vertex", "Fluid_not_staggered.PickLayer", "Fluid_not_staggered.ComputeDivergence");
  Programs.ApplyImpulse = LoadProgram("Fluid_not_staggered.Vertex", "Fluid_not_staggered.PickLayer", "Fluid_not_staggered.Splat");
  Programs.ApplyBuoyancy = LoadProgram("Fluid_not_staggered.Vertex", "Fluid_not_staggered.PickLayer", "Fluid_not_staggered.Buoyancy");
  Programs.UpdateObstacles = LoadProgram("Fluid_not_staggered.Vertex", "Fluid_not_staggered.PickLayer", "Fluid_not_staggered.UpdateObstacles");
  Programs.UpdateObstacleSpeeds = LoadProgram("Fluid_not_staggered.Vertex", "Fluid_not_staggered.PickLayer", "Fluid_not_staggered.UpdateObstacleSpeeds");
  Programs.ApplyObstacleSpeeds = LoadProgram("Fluid_not_staggered.Vertex", "Fluid_not_staggered.PickLayer", "Fluid_not_staggered.ApplyObstacleSpeeds");
  Programs.VorticityConfinement = LoadProgram("Fluid_not_staggered.Vertex", "Fluid_not_staggered.PickLayer", "Fluid_not_staggered.VorticityConfinement");
  Programs.CopyVelocity = LoadProgram("Fluid_not_staggered.Vertex", "Fluid_not_staggered.PickLayer", "Fluid_not_staggered.CopyVelocity");
}

void SwapSurfaces(SlabPod* slab)
{
    SurfacePod temp = slab->Ping;
    slab->Ping = slab->Pong;
    slab->Pong = temp;
}

void ClearSurface(SurfacePod s, float v)
{
    glBindFramebuffer(GL_FRAMEBUFFER, s.FboHandle);
    glClearColor(v, v, v, v);
    glClear(GL_COLOR_BUFFER_BIT);
}

void AdvectVelocity(SurfacePod velocity, SurfacePod source, SurfacePod obstacles, SurfacePod dest, float dissipation, float dt)
{
    glUseProgram(Programs.AdvectVelocity);

    SetUniform("InverseSize", recipPerElem(Vector3(float(GridWidth), float(GridHeight), float(GridDepth))));
    SetUniform("TimeStep", dt);
    SetUniform("Dissipation", dissipation);
    SetUniform("SourceTexture", 1);
    SetUniform("Obstacles", 2);

    glBindFramebuffer(GL_FRAMEBUFFER, dest.FboHandle);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, velocity.ColorTexture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_3D, source.ColorTexture);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_3D, obstacles.ColorTexture);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, dest.Depth);

    ResetState();
}

void Advect(SurfacePod velocity, SurfacePod source, SurfacePod obstacles, SurfacePod obstacleSpeeds, SurfacePod dest, float dissipation, float dt)
{
    glUseProgram(Programs.Advect);

    SetUniform("InverseSize", recipPerElem(Vector3(float(GridWidth), float(GridHeight), float(GridDepth))));
    SetUniform("TimeStep", TimeStep);
    SetUniform("Dissipation", dissipation);
    SetUniform("SourceTexture", 1);
    SetUniform("Obstacles", 2);
    SetUniform("ObstacleSpeeds", 3);

    glBindFramebuffer(GL_FRAMEBUFFER, dest.FboHandle);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, velocity.ColorTexture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_3D, source.ColorTexture);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_3D, obstacles.ColorTexture);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_3D, obstacleSpeeds.ColorTexture);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, dest.Depth);

    ResetState();
}

///
/// \brief Jacobi
/// \param pressure
/// \param divergence
/// \param obstacles
/// \param dest
///
void Jacobi(SurfacePod pressure, SurfacePod divergence, SurfacePod obstacles, SurfacePod dest)
{
    glUseProgram(Programs.Jacobi);

    SetUniform("Alpha", -CellSize * CellSize);
    SetUniform("InverseBeta", inverseBeta);
    SetUniform("Divergence", 1);
    SetUniform("Obstacles", 2);

    glBindFramebuffer(GL_FRAMEBUFFER, dest.FboHandle);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, pressure.ColorTexture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_3D, divergence.ColorTexture);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_3D, obstacles.ColorTexture);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, dest.Depth);
    ResetState();
}

///
/// \brief Jacobi_no_program - debugging function for testing if glBindTexture calls are working as expected, do not use
/// \param pressure
/// \param divergence
/// \param obstacles
/// \param dest
///
void Jacobi_no_program(SurfacePod pressure, SurfacePod divergence, SurfacePod obstacles, SurfacePod dest)
{
    glBindFramebuffer(GL_FRAMEBUFFER, dest.FboHandle);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, pressure.ColorTexture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_3D, divergence.ColorTexture);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_3D, obstacles.ColorTexture);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, dest.Depth);
    ResetState();
}

///
/// \brief SubtractGradient
/// \param velocity
/// \param pressure
/// \param obstacles
/// \param obstacleSpeeds
/// \param dest
/// \param velSTD
///
void SubtractGradient(SurfacePod velocity, SurfacePod pressure, SurfacePod obstacles, SurfacePod obstacleSpeeds, SurfacePod dest, float velSTD)
{
    glUseProgram(Programs.SubtractGradient);

    SetUniform("GradientScale", GradientScale);

    SetUniform("HalfInverseCellSize", 0.5f / CellSize);
    SetUniform("Pressure", 1);
    SetUniform("Obstacles", 2);
    SetUniform("ObstacleSpeeds", 3);

    glBindFramebuffer(GL_FRAMEBUFFER, dest.FboHandle);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, velocity.ColorTexture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_3D, pressure.ColorTexture);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_3D, obstacles.ColorTexture);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_3D, obstacleSpeeds.ColorTexture);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, dest.Depth);
    ResetState();
}

///
/// \brief ComputeDivergence
/// \param velocity
/// \param obstacles
/// \param dest
///
void ComputeDivergence(SurfacePod velocity, SurfacePod obstacles, SurfacePod dest)
{
    glUseProgram(Programs.ComputeDivergence);

    SetUniform("HalfInverseCellSize", 0.5f / CellSize);
    SetUniform("Obstacles", 1);

    glBindFramebuffer(GL_FRAMEBUFFER, dest.FboHandle);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, velocity.ColorTexture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_3D, obstacles.ColorTexture);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, dest.Depth);
    ResetState();
}

///
/// \brief ComputeDivergenceCNN
/// \param velocity
/// \param obstacles
/// \param dest
///
void ComputeDivergenceCNN(SurfacePod velocity, SurfacePod obstacles, SurfacePod dest)
{
    glUseProgram(Programs.ComputeDivergence);

    SetUniform("HalfInverseCellSize", 0.5f / CellSize);
    SetUniform("Obstacles", 1);

    glBindFramebuffer(GL_FRAMEBUFFER, dest.FboHandle);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, velocity.ColorTexture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_3D, obstacles.ColorTexture);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, dest.Depth);
    ResetState();
}

///
/// \brief ApplyImpulse
/// \param dest
/// \param obstacles
/// \param position
/// \param value
/// \param jitter
/// \param time
///
void ApplyImpulse(SurfacePod dest, SurfacePod obstacles, Vector3 position, float value, float jitter, float time)
{
    glUseProgram(Programs.ApplyImpulse);

    SetUniform("Point", position);
    SetUniform("Radius", SplatRadius);
    SetUniform("FillColor", Vector3(value, value, value));
    SetUniform("Jitter", jitter); //jitter is set to zero by default outside this function
    SetUniform("Time", time);

    glBindFramebuffer(GL_FRAMEBUFFER, dest.FboHandle);
    glEnable(GL_BLEND);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, dest.Depth);
    ResetState();
}

///
/// \brief UpdateObstacles
/// \param obstacles
/// \param time
///
void UpdateObstacles(SurfacePod obstacles, float time){

    return;
    glUseProgram(Programs.UpdateObstacles);

    //just pass in the start and finish, as three floats each
    int centerX = (int)((obstacles.Width/2) * (float)(sin(obstacleSpeedMultiplier*time)) + obstacles.Width/2);
    int startX = centerX - CubeSize.x/2;
    int endX = startX + CubeSize.x;
    int startY = (obstacles.Height - CubeSize.y) / 2;
    int endY = startY + CubeSize.y;
    int startZ = (obstacles.Depth - CubeSize.z) / 2;
    int endZ = startZ + CubeSize.z;

    SetUniform("StartX", startX);
    SetUniform("EndX", endX);
    SetUniform("StartY", startY);
    SetUniform("EndY", endY);
    SetUniform("StartZ", startZ);
    SetUniform("EndZ", endZ);

    //SetUniform("Speed", obstacleSpeedMultiplier);

    glBindFramebuffer(GL_FRAMEBUFFER, obstacles.FboHandle);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, obstacles.Depth);
    ResetState();
}

///
/// \brief UpdateObstacleSpeeds
/// \param obstacleSpeeds
/// \param obstacles
/// \param time
///
void UpdateObstacleSpeeds(SurfacePod obstacleSpeeds, SurfacePod obstacles, float time){

    glUseProgram(Programs.UpdateObstacleSpeeds);

    SetUniform("CurrentTime", time);
    SetUniform("CubeSizeX", 1.0f * CubeSize.x / obstacleSpeeds.Width);
    SetUniform("CubeSizeY", 1.0f * CubeSize.y / obstacleSpeeds.Height);
    SetUniform("CubeSizeZ", 1.0f * CubeSize.z / obstacleSpeeds.Depth);
    SetUniform("Speed", obstacleSpeedMultiplier);
    SetUniform("Amplitude", obstacleSpeedMultiplier * (obstacles.Width / 2));

    glBindFramebuffer(GL_FRAMEBUFFER, obstacleSpeeds.FboHandle);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, obstacleSpeeds.Depth);
    ResetState();
}

///
/// \brief ApplyObstacleSpeeds - overwrites the velocity with the obstacle speeds, where obstacles are present
/// \param velocity
/// \param obstacles
/// \param obstacleSpeeds
/// \param dest
///
void ApplyObstacleSpeeds(SurfacePod velocity, SurfacePod obstacles, SurfacePod obstacleSpeeds, SurfacePod dest){

    glUseProgram(Programs.ApplyObstacleSpeeds);

    SetUniform("Obstacles", 1);
    SetUniform("ObstacleSpeeds", 2);

    glBindFramebuffer(GL_FRAMEBUFFER, dest.FboHandle);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, velocity.ColorTexture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_3D, obstacles.ColorTexture);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_3D, obstacleSpeeds.ColorTexture);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, dest.Depth);
    ResetState();
}

///
/// \brief ApplyBuoyancy
/// \param velocity
/// \param temperature
/// \param density
/// \param dest
///
void ApplyBuoyancy(SurfacePod velocity, SurfacePod temperature, SurfacePod density, SurfacePod dest)
{
    glUseProgram(Programs.ApplyBuoyancy);

    SetUniform("Temperature", 1);
    SetUniform("Density", 2);
    SetUniform("AmbientTemperature", AmbientTemperature);
    SetUniform("TimeStep", TimeStep);
    SetUniform("Sigma", SmokeBuoyancy);
    SetUniform("Kappa", SmokeWeight);

    glBindFramebuffer(GL_FRAMEBUFFER, dest.FboHandle);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, velocity.ColorTexture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_3D, temperature.ColorTexture);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_3D, density.ColorTexture);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, dest.Depth);
    ResetState();
}

///
/// \brief ApplyVorticityConfinement
/// \param velocity
/// \param dest
///
void ApplyVorticityConfinement(SurfacePod velocity, SurfacePod dest){

    glUseProgram(Programs.VorticityConfinement);

    SetUniform("DoubleCellSize", CellSize*2);
    SetUniform("VorticityBoost", VorticityBoost);
    SetUniform("TimeStep", TimeStep);
    SetUniform("VelocityTextureSize", dim3(velocity.Width, velocity.Height, velocity.Depth));

    glBindFramebuffer(GL_FRAMEBUFFER, dest.FboHandle);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, velocity.ColorTexture);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, dest.Depth);
    ResetState();
}

///
/// \brief CopyVelocity
/// \param velocity
/// \param dest
///
void CopyVelocity(SurfacePod velocity, SurfacePod dest){

    glUseProgram(Programs.CopyVelocity);

    glBindFramebuffer(GL_FRAMEBUFFER, dest.FboHandle);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, velocity.ColorTexture);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, dest.Depth);
    ResetState();
}

///
/// \brief CreatePointVbo
/// \param x
/// \param y
/// \param z
/// \return
///
GLuint CreatePointVbo(float x, float y, float z)
{
    float p[3] = {x, y, z};
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(p), &p[0], GL_STATIC_DRAW);
    return vbo;
}

///
/// \brief SetUniform - set OpenGL dim3 (int3) uniform value
/// \param name - uniform name
/// \param value
///
void SetUniform(const char* name, dim3 value)
{
    GLuint program;
    glGetIntegerv(GL_CURRENT_PROGRAM, (GLint*) &program);
    GLint location = glGetUniformLocation(program, name);
    glUniform3i(location, GLint(value.x), GLint(value.y), GLint(value.z));
}

///
/// \brief SetUniform - set OpenGL int uniform value
/// \param name - uniform name
/// \param value
///
void SetUniform(const char* name, int value)
{
    GLuint program;
    glGetIntegerv(GL_CURRENT_PROGRAM, (GLint*) &program);
    GLint location = glGetUniformLocation(program, name);
    glUniform1i(location, value);
}

///
/// \brief SetUniform - set OpenGL float uniform value
/// \param name - uniform name
/// \param value
///
void SetUniform(const char* name, float value)
{
    GLuint program;
    glGetIntegerv(GL_CURRENT_PROGRAM, (GLint*) &program);
    GLint location = glGetUniformLocation(program, name);
    glUniform1f(location, value);
}

///
/// \brief SetUniform - set OpenGL Matrix4 uniform value
/// \param name - uniform name
/// \param value
///
void SetUniform(const char* name, Matrix4 value)
{
    GLuint program;
    glGetIntegerv(GL_CURRENT_PROGRAM, (GLint*) &program);
    GLint location = glGetUniformLocation(program, name);
    glUniformMatrix4fv(location, 1, 0, (float*) &value);
}

///
/// \brief SetUniform - set OpenGL Matrix3 uniform value
/// \param name - uniform name
/// \param nm - matrix value
///
void SetUniform(const char* name, Matrix3 nm)
{
    GLuint program;
    glGetIntegerv(GL_CURRENT_PROGRAM, (GLint*) &program);
    GLint location = glGetUniformLocation(program, name);
    float packed[9] = {
        nm.getRow(0).getX(), nm.getRow(1).getX(), nm.getRow(2).getX(),
        nm.getRow(0).getY(), nm.getRow(1).getY(), nm.getRow(2).getY(),
        nm.getRow(0).getZ(), nm.getRow(1).getZ(), nm.getRow(2).getZ() };
    glUniformMatrix3fv(location, 1, 0, &packed[0]);
}

///
/// \brief SetUniform - set OpenGL Vector3 uniform value
/// \param name - uniform name
/// \param value
///
void SetUniform(const char* name, Vector3 value)
{
    GLuint program;
    glGetIntegerv(GL_CURRENT_PROGRAM, (GLint*) &program);
    GLint location = glGetUniformLocation(program, name);
    glUniform3f(location, value.getX(), value.getY(), value.getZ());
}

///
/// \brief SetUniform - set OpenGL float2 uniform value
/// \param name - uniform name
/// \param x
/// \param y
///
void SetUniform(const char* name, float x, float y)
{
    GLuint program;
    glGetIntegerv(GL_CURRENT_PROGRAM, (GLint*) &program);
    GLint location = glGetUniformLocation(program, name);
    glUniform2f(location, x, y);
}

///
/// \brief SetUniform - set OpenGL Vector4 uniform value
/// \param name - uniform name
/// \param value
///
void SetUniform(const char* name, Vector4 value)
{
    GLuint program;
    glGetIntegerv(GL_CURRENT_PROGRAM, (GLint*) &program);
    GLint location = glGetUniformLocation(program, name);
    glUniform4f(location, value.getX(), value.getY(), value.getZ(), value.getW());
}

///
/// \brief SetUniform - set OpenGL Point3 uniform value
/// \param name - uniform name
/// \param value
///
void SetUniform(const char* name, Point3 value)
{
    GLuint program;
    glGetIntegerv(GL_CURRENT_PROGRAM, (GLint*) &program);
    GLint location = glGetUniformLocation(program, name);
    glUniform3f(location, value.getX(), value.getY(), value.getZ());
}

///
/// \brief CreateQuadVbo
/// \return - a vertex buffer object
///
GLuint CreateQuadVbo()
{
    short positions[] = {
        -1, -1,
         1, -1,
        -1,  1,
         1,  1,
    };
    GLuint vbo;
    GLsizeiptr size = sizeof(positions);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, size, positions, GL_STATIC_DRAW);
    return vbo;
}

///
/// \brief WriteToFile - dumps 3D texture data in a SurfacePod to file
/// \param filename
/// \param surfacePod
///
void WriteToFile(const char* filename, SurfacePod surfacePod)
{
    size_t requiredBytes = surfacePod.Width * surfacePod.Height * surfacePod.Depth * 2;
    std::vector<unsigned char> cache(requiredBytes);
    glBindTexture(GL_TEXTURE_3D, surfacePod.ColorTexture);
    glGetTexImage(GL_TEXTURE_3D, 0, GL_RED, GL_HALF_FLOAT, &cache[0]);
    FILE* voxelsFile = fopen(filename, "wb");
    size_t bytesWritten = fwrite(&cache[0], 1, requiredBytes, voxelsFile);
    pezCheck(bytesWritten == requiredBytes, "Unable to dump out volume texture.");
}

///
/// \brief ReadFromFile - reads 3D texture data from file and puts it in a SurfacePod
/// \param filename - file containing raw bytes of texture
/// \param density
///
void ReadFromFile(const char* filename, SurfacePod density)
{
    size_t requiredBytes = density.Width * density.Height * density.Depth * 2;
    std::vector<unsigned char> cache(requiredBytes);
    FILE* voxelsFile = fopen(filename, "rb");
    size_t bytesRead = fread(&cache[0], 1, requiredBytes, voxelsFile);
    pezCheck(bytesRead == requiredBytes, "Unable to slurp up volume texture.");

    glBindTexture(GL_TEXTURE_3D, density.ColorTexture);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_R16F, density.Width, density.Height, density.Depth, 0, GL_RED, GL_HALF_FLOAT, &cache[0]);
}
