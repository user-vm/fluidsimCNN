#pragma once
#include <vector>
#include <string>
#include "vmath.hpp"
#include "pez.h"
#include <sstream>
//#include "gl3.h"

enum AttributeSlot {
    SlotPosition,
    SlotTexCoord,
};

struct TexturePod {
    GLuint Handle;
    GLsizei Width;
    GLsizei Height;
};

struct SurfacePod {
    GLuint FboHandle;
    GLuint ColorTexture;
    GLsizei Width;
    GLsizei Height;
    GLsizei Depth;
};

struct SlabPod {
    SurfacePod Ping;
    SurfacePod Pong;
};

extern float TimeStep;

GLuint LoadProgram(const char* vsKey, const char* gsKey, const char* fsKey);
void SetUniform(const char* name, dim3 value);
void SetUniform(const char* name, int value);
void SetUniform(const char* name, float value);
void SetUniform(const char* name, float x, float y);
void SetUniform(const char* name, vmath::Matrix4 value);
void SetUniform(const char* name, vmath::Matrix3 value);
void SetUniform(const char* name, vmath::Vector3 value);
void SetUniform(const char* name, vmath::Point3 value);
void SetUniform(const char* name, vmath::Vector4 value);
TexturePod LoadTexture(const char* path);
SurfacePod CreateSurface(int width, int height, int numComponents = 4);
SurfacePod CreateVolume(int width, int height, int depth, int numComponents = 4);
GLuint CreatePointVbo(float x, float y, float z);
GLuint CreateQuadVbo();
void CreateObstacles(SurfacePod dest, SurfacePod destSpeed);
SlabPod CreateSlab(GLsizei width, GLsizei height, GLsizei depth, int numComponents);
void InitSlabOps();
void InitSlabOpsNotStaggered();
void SwapSurfaces(SlabPod* slab);
void ClearSurface(SurfacePod s, float v);
void AdvectVelocity(SurfacePod velocity, SurfacePod source, SurfacePod obstacles, SurfacePod dest, float dissipation, float dt = TimeStep);
void Advect(SurfacePod velocity, SurfacePod source, SurfacePod obstacles, SurfacePod obstacleSpeeds, SurfacePod dest, float dissipation, float dt = TimeStep);
void Jacobi(SurfacePod pressure, SurfacePod divergence, SurfacePod obstacles, SurfacePod dest);
void Jacobi_no_program(SurfacePod pressure, SurfacePod divergence, SurfacePod obstacles, SurfacePod dest);
void SubtractGradient(SurfacePod velocity, SurfacePod pressure, SurfacePod obstacles, SurfacePod obstacleSpeeds, SurfacePod dest, float velSTD = NAN);
void ComputeDivergenceCNN(SurfacePod velocity, SurfacePod obstacles, SurfacePod dest);
void ComputeDivergence(SurfacePod velocity, SurfacePod obstacles, SurfacePod dest);
void ApplyImpulse(SurfacePod dest, SurfacePod obstacles, vmath::Vector3 position, float value, float jitter = 0, float time = 1.0);
void ApplyBuoyancy(SurfacePod velocity, SurfacePod temperature, SurfacePod density, SurfacePod dest);
void ApplyVorticityConfinement(SurfacePod velocity, SurfacePod dest);
void WriteToFile(const char* filename, SurfacePod density);
void ReadFromFile(const char* filename, SurfacePod density);
void UpdateObstacles(SurfacePod obstacles, float time);
void UpdateObstacleSpeeds(SurfacePod obstacleSpeeds, SurfacePod obstacles, float time);
void ApplyObstacleSpeeds(SurfacePod velocity, SurfacePod obstacles, SurfacePod obstacleSpeeds, SurfacePod dest);
void CopyVelocity(SurfacePod velocity, SurfacePod dest);

extern const float CellSize;
extern const int ViewportWidth;
extern const int ViewportHeight;
extern const int GridWidth;
extern const int GridHeight;
extern const int GridDepth;
extern const float SplatRadius;
extern const float AmbientTemperature;
extern const float ImpulseTemperature;
extern const float ImpulseDensity;
extern const int NumJacobiIterations;
extern const float SmokeBuoyancy;
extern const float SmokeWeight;
extern const float GradientScale;
extern const float TemperatureDissipation;
extern const float VelocityDissipation;
extern const float DensityDissipation;
extern vmath::Vector3 ImpulsePosition;
extern vmath::Vector4 XYImpulsePosition;
extern vmath::Vector4 XYImpulsePositionLast;
extern const float obstacleSpeedMultiplier;
extern const float JitterTemperature;
extern const float JitterDensity;
