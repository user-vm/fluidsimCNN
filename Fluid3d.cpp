//#define GL_GLEXT_PROTOTYPES
//#include <GL/glew.h>
#include "Utility.h"
#include <cmath>
#include <cstdio>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
//#include <helper_gl.h> //this breaks things in GLEW
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <cnn.cpp> //doesn't seem to work otherwise
#include <toycnn.h>
#include <memory>
#include <chrono>

#include "defines.h"

using namespace vmath;
using std::string;

#define OpenGLError GL_NO_ERROR == glGetError(),                        \
    "%s:%d - OpenGL Error - %s", __FILE__, __LINE__, __FUNCTION__   \

#define CheckOpenGLError(x) GL_NO_ERROR == x,                \
    "%s:%d - OpenGL Error - %s", __FILE__, __LINE__, __FUNCTION__   \

int k = 0;

int currentLoop = 0;

int useStaggered = 1;

int useJacobi = 0;

int numberOfBanks = 3;

static struct {
    SlabPod Velocity;
    SlabPod Density;
    SlabPod Pressure;
    SlabPod Temperature;
} Slabs;

static struct {
    SurfacePod Divergence;
    SurfacePod Obstacles;
    SurfacePod ObstacleSpeeds;
    SurfacePod LightCache;
    SurfacePod BlurredDensity;
} Surfaces;

static struct {
    Matrix4 Projection;
    Matrix4 Modelview;
    Matrix4 View;
    Matrix4 ModelviewProjection;
} Matrices;

static struct {
    GLuint CubeCenter;
    GLuint FullscreenQuad;
} Vaos;

extern const float CellSize;
extern const float inverseBeta;

static const Point3 EyePosition = Point3(0, 0, 1.7);//(0, 0, 2.0) gives near clipping
static GLuint RaycastProgram;
static GLuint RaycastVelProgram;
static GLuint LightProgram;
static GLuint BlurProgram;
static float FieldOfView = 0.7f;
static bool SimulateFluid = true;
static const float DefaultThetaX = 0;
static const float DefaultThetaY = 0.75f;
static float ThetaX = DefaultThetaX;
static float ThetaY = DefaultThetaY;
static int ViewSamples = GridWidth*2;
static int LightSamples = GridWidth;
static float Fips = -4;

GLuint bufferID;
struct cudaGraphicsResource *cuda_buffer_resource, *pressurePing_buffer_resource,
        *pressurePong_buffer_resource, *divergence_buffer_resource, *obstacles_buffer_resource;//*cuda_fbo_resource; //handles OpenGL-CUDA exchange

struct cudaGraphicsResource *pressurePing_texture_resource, *pressurePong_texture_resource,
        *divergence_texture_resource, *obstacles_texture_resource, *obstacleSpeeds_texture_resource, *velocityPing_texture_resource, *velocityPong_texture_resource;

float *divergenceFBOptr, *pressurePongFBOptr;
size_t divergenceFBOsize, pressurePongFBOsize;

cudaArray *pressurePingCudaArray, *pressurePongCudaArray, *divergenceCudaArray, *obstaclesCudaArray, *obstacleSpeedsCudaArray, *velocityPingCudaArray, *velocityPongCudaArray;

cudaSurfaceObject_t pressurePongSurf,pressurePingSurf;
cudaChannelFormatDesc pressurePingFormat;

cudaTextureObject_t divergenceTex, obstaclesTex;

struct cudaResourceDesc pressurePingResDesc, pressurePongResDesc, divergenceResDesc,obstaclesResDesc, obstacleSpeedsResDesc, velocityPingResDesc,velocityPongResDesc;
struct cudaTextureDesc  texDesc;

float4 *d_buffer, *pressurePing_buffer, *pressurePong_buffer, *divergence_buffer, *obstacles_buffer;

#if USE_CNN==0
std::unique_ptr<ToyCNN> toyCNN;
#else
std::unique_ptr<DefaultNet<float>> defaultNet;
#endif

PezConfig PezGetConfig()
{
    PezConfig config;
    config.Title = "Fluid3d";
    config.Width = 853*2;
    config.Height = 480*2;
    config.Multisampling = 0;
    config.VerticalSync = 0;
    return config;
}

void PezInitialize()
{

    infoMsg("nyeh\n");
    //findCUDAGLDevice();
    //int devID = gpuGetMaxGflopsDeviceId();

    //printf("devID = %d", devID);

    //checkCudaErrors(cudaGLSetGLDevice(devID));


    //THIS IS NOT NEEDED WHEN RUNNING FROM WITHIN QTCREATOR IF CUDA_VISIBLE_DEVICES IS SET TO 1 IN PROJECTS>BUILD>BUILD ENVIRONMENT ON TWO GPUS
    //will fail with cudaErrorNoDevice if run with CUDA_VISIBLE_DEVICES=1 when only one GPU is available
    //still requires CUDA_VISIBLE_DEVICES set to 1 when run from command line on two GPUs (if you don't want the memory mapping to be on both devices; other than that it works fine)
    //what is done here should still cause the memory of the Titan to still be mapped, which might mean initialization is
    //should try running with CUDA_VISIBLE_DEVICES=1
    int deviceCount;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));

    if(deviceCount > 1){
        cudaDeviceProp currentDeviceProperties;
        std::string deviceName;
        std::string deviceToFind = "GeForce GTX 1070";

        int deviceToUse = -1;

        //this will probably only work here
        for(int i=0;i<deviceCount;i++){
            checkCudaErrors(cudaGetDeviceProperties(&currentDeviceProperties,i));
            deviceName = std::string(currentDeviceProperties.name);
            if(currentDeviceProperties.major == 9999){
                deviceToUse = -1;
                printf("No CUDA-enabled GPU found.\n");
                break;
            }
            if(deviceName == deviceToFind){
                deviceToUse = i;
                break;
            }
        }

        if(deviceToUse == -1){
            printf("Device \"GeForce GTX 1070\" not found. May need to change device name (deviceToFind) in PezInitialize in Fluid3d.cpp; use cudaGetDeviceProperties for device from 0 to cudaGetDeviceCount-1 to see the names of available GPUs on your system.\n");
            exit(0);
        }

        checkCudaErrors(cudaGLSetGLDevice(deviceToUse));
        checkCudaErrors(cudaSetDevice(deviceToUse));

        printf("deviceName = %s; deviceToUse = %d\n",deviceName.c_str(),deviceToUse);

        //exit(0);
    }
    else{
        checkCudaErrors(cudaGLSetGLDevice(0));
        checkCudaErrors(cudaSetDevice(0));
        printf("Just one GPU found.\n");
    }

    //this isn't robust, should use the device names
    /*
  if(deviceCount == 1){
    checkCudaErrors(cudaGLSetGLDevice(0));
    checkCudaErrors(cudaSetDevice(0));
  else{
    checkCudaErrors(cudaGLSetGLDevice(1));
    checkCudaErrors(cudaSetDevice(1));
  }
  */

    //exit(0);

    infoMsg("meh\n");

    GLenum err = glGetError();
    std::stringstream errorBuffer;

    PezConfig cfg = PezGetConfig();

    errorBuffer<<err;

    if(err != GL_NO_ERROR)
        pezCheck(0,errorBuffer.str().c_str());

    RaycastProgram = LoadProgram("Raycast.VS", "Raycast.GS", "Raycast.FS");
    RaycastVelProgram = LoadProgram("Raycast.VS", "Raycast.GS", "Raycast.FSVel");
    LightProgram = LoadProgram("Fluid.Vertex", "Fluid.PickLayer", "Light.Cache");
    BlurProgram = LoadProgram("Fluid.Vertex", "Fluid.PickLayer", "Light.Blur");

    err  =glGetError();

    errorBuffer<<err;

    if(err != GL_NO_ERROR)
        pezCheck(0,errorBuffer.str().c_str());

    glGenVertexArrays(1, &Vaos.CubeCenter);
    glBindVertexArray(Vaos.CubeCenter);
    CreatePointVbo(0, 0, 0);
    glEnableVertexAttribArray(SlotPosition);
    glVertexAttribPointer(SlotPosition, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);

    glGenVertexArrays(1, &Vaos.FullscreenQuad);
    glBindVertexArray(Vaos.FullscreenQuad);
    CreateQuadVbo();
    glEnableVertexAttribArray(SlotPosition);
    glVertexAttribPointer(SlotPosition, 2, GL_SHORT, GL_FALSE, 2 * sizeof(short), 0);

    errorBuffer<<err;

    if(err != GL_NO_ERROR)
        pezCheck(0,errorBuffer.str().c_str());

    glGenBuffers(1,&bufferID);
    glBindBuffer(GL_ARRAY_BUFFER, bufferID);

    errorBuffer<<err;

    if(err != GL_NO_ERROR)
        pezCheck(0,errorBuffer.str().c_str());

    //if(useStaggered)
        Slabs.Velocity = CreateSlab(GridWidth + 1, GridHeight + 1, GridDepth + 1, 3); //for a staggered grid. This creates some unused velocity indices, but
    //that is not easily corrected with how textures are implemented
    //else
    //    Slabs.Velocity = CreateSlab(GridWidth, GridHeight, GridDepth, 3);

    Slabs.Density = CreateSlab(GridWidth, GridHeight, GridDepth, 1);
    Slabs.Pressure = CreateSlab(GridWidth, GridHeight, GridDepth, 1);
    Slabs.Temperature = CreateSlab(GridWidth, GridHeight, GridDepth, 1);
    Surfaces.Divergence = CreateVolume(GridWidth, GridHeight, GridDepth, 1); //this worked with 1, but not properly for staggered
    Surfaces.LightCache = CreateVolume(GridWidth, GridHeight, GridDepth, 1);
    Surfaces.BlurredDensity = CreateVolume(GridWidth, GridHeight, GridDepth, 1);

    //if(useStaggered)
    //    InitSlabOps();
    //else
        InitSlabOpsNotStaggered();
    Surfaces.Obstacles = CreateVolume(GridWidth, GridHeight, GridDepth, 1);

    //if ObstacleSpeeds is a surface, it needs to know the time from first timer call
    //need to decide how to smaple obstacle speeds
    if(useStaggered)
        Surfaces.ObstacleSpeeds = CreateVolume(GridWidth+1, GridHeight+1, GridDepth+1, 3);
    else
        Surfaces.ObstacleSpeeds = CreateVolume(GridWidth, GridHeight, GridDepth, 3);

    CreateObstacles(Surfaces.Obstacles, Surfaces.ObstacleSpeeds);
    ClearSurface(Slabs.Temperature.Ping, AmbientTemperature);

    //glDisable(GL_DEPTH_TEST);
    //glDisable(GL_CULL_FACE);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pezCheck(OpenGLError);

    //register the main OpenGL buffer with CUDA (actually how about no)
    //checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_buffer_resource, bufferID, cudaGraphicsMapFlagsNone));

    //register relevant FBOv buffers for the CUDA projection phase
    //checkCudaErrors(cudaGraphicsGLRegisterBuffer(&pressurePing_buffer_resource, Slabs.Pressure.Ping.FboHandle, cudaGraphicsMapFlagsNone));//pressurePing_buffer_resource
    //checkCudaErrors(cudaGraphicsGLRegisterBuffer(&pressurePong_buffer_resource, Slabs.Pressure.Pong.FboHandle, cudaGraphicsMapFlagsNone));
    //checkCudaErrors(cudaGraphicsGLRegisterBuffer(&divergence_buffer_resource, Surfaces.Divergence.FboHandle, cudaGraphicsMapFlagsNone));
    //checkCudaErrors(cudaGraphicsGLRegisterBuffer(&obstacles_buffer_resource, Surfaces.Obstacles.FboHandle, cudaGraphicsMapFlagsNone));

    //register all the 3D textures corresponding to the FBOs
    checkCudaErrors(cudaGraphicsGLRegisterImage(&pressurePing_texture_resource, Slabs.Pressure.Ping.ColorTexture, GL_TEXTURE_3D,cudaGraphicsRegisterFlagsNone));
    checkCudaErrors(cudaGraphicsGLRegisterImage(&pressurePong_texture_resource, Slabs.Pressure.Pong.ColorTexture, GL_TEXTURE_3D,cudaGraphicsRegisterFlagsNone));
    checkCudaErrors(cudaGraphicsGLRegisterImage(&divergence_texture_resource, Surfaces.Divergence.ColorTexture, GL_TEXTURE_3D,cudaGraphicsRegisterFlagsNone));
    checkCudaErrors(cudaGraphicsGLRegisterImage(&obstacles_texture_resource, Surfaces.Obstacles.ColorTexture, GL_TEXTURE_3D,cudaGraphicsRegisterFlagsNone));
    checkCudaErrors(cudaGraphicsGLRegisterImage(&obstacleSpeeds_texture_resource, Surfaces.ObstacleSpeeds.ColorTexture, GL_TEXTURE_3D,cudaGraphicsRegisterFlagsNone));
    checkCudaErrors(cudaGraphicsGLRegisterImage(&velocityPing_texture_resource, Slabs.Velocity.Ping.ColorTexture, GL_TEXTURE_3D,cudaGraphicsRegisterFlagsNone));
    checkCudaErrors(cudaGraphicsGLRegisterImage(&velocityPong_texture_resource, Slabs.Velocity.Pong.ColorTexture, GL_TEXTURE_3D,cudaGraphicsRegisterFlagsNone));

    //the pressurePong cudaArray needs to be associated with a surface object in order to be editable
    //might not work because cudaArray was not created with the cudaArraySurfaceLoadStore flag

    //pressurePingResDesc = cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindFloat);
    memset(&pressurePingResDesc, 0, sizeof(pressurePingResDesc));
    pressurePingResDesc.resType = cudaResourceTypeArray;

    //this is useless, why did you write it
    //pressurePongResDesc = cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindFloat);
    memset(&pressurePongResDesc, 0, sizeof(pressurePongResDesc));
    pressurePongResDesc.resType = cudaResourceTypeArray;


    //velocityPingResDesc = cudaCreateChannelDesc(32,32,32,32,cudaChannelFormatKindFloat); //this makes no sense
    memset(&velocityPingResDesc, 0, sizeof(velocityPingResDesc));
    velocityPingResDesc.resType = cudaResourceTypeArray;

    memset(&velocityPongResDesc, 0, sizeof(velocityPongResDesc));
    velocityPongResDesc.resType = cudaResourceTypeArray;

    //divergenceResDesc = cudaCreateChannelDesc(32,32,32,0,cudaChannelFormatKindFloat);
    memset(&divergenceResDesc, 0, sizeof(divergenceResDesc));
    divergenceResDesc.resType = cudaResourceTypeArray;

    //obstaclesResDesc = cudaCreateChannelDesc(32,32,32,0,cudaChannelFormatKindFloat);
    memset(&obstaclesResDesc, 0, sizeof(obstaclesResDesc));
    obstaclesResDesc.resType = cudaResourceTypeArray;

    memset(&obstacleSpeedsResDesc, 0, sizeof(obstacleSpeedsResDesc));
    obstacleSpeedsResDesc.resType = cudaResourceTypeArray;

    //this is not used for the convolution projection, just Jacobi_CUDA and Toy
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;

    //the cudnn stuff

    //initialize class instances etc.
    //use unique pointers later
    //dim3 divergenceSize = dim3(Surfaces.Divergence.Width, Surfaces.Divergence.Height, Surfaces.Divergence.Depth);
    //dim3 hiddenSize = dim3(Surfaces.Divergence.Width/2, Surfaces.Divergence.Height/2, Surfaces.Divergence.Depth/2);

#if USE_CNN==0
    toyCNN = std::unique_ptr<ToyCNN>(new ToyCNN(divergenceSize, hiddenSize));
#else
    if(numberOfBanks==1)
        defaultNet = std::unique_ptr<DefaultNet<float>>(new DefaultNet<float>("out_torch_order.bin", true, NULL, 1)); //NULL because we don't know the input size (that parameter is probably useless)
    else if(numberOfBanks==3)
        defaultNet = std::unique_ptr<DefaultNet<float>>(new DefaultNet<float>("out_torch_order_3banks.bin", true, NULL, 3));
    else{
        printf("Error: No trained model for %d banks.", numberOfBanks);
        exit(0);}
#endif
    //tensor descriptors etc.
}

void PezRender()
{
    pezCheck(OpenGLError);
    PezConfig cfg = PezGetConfig();

    static int thisLoop = 0;

    //if(thisLoop == 0)
    //glObjectLabel(GL_TEXTURE, Slabs.Density.Ping.ColorTexture, -1, "DENSITYPINGTEXTURE");
    //glObjectLabel(GL_TEXTURE, Slabs.Density.Pong.ColorTexture, -1, "DENSITYPONGTEXTURE");
    //glObjectLabel(GL_TEXTURE, Slabs.Velocity.Ping.ColorTexture, -1, "VELOCITYPINGTEXTURE");
    //glObjectLabel(GL_TEXTURE, Slabs.Velocity.Pong.ColorTexture, -1, "VELOCITYPONGTEXTURE");
    //glObjectLabel(GL_TEXTURE, Slabs.Temperature.Ping.ColorTexture, -1, "TEMPERATUREPINGTEXTURE");
    //glObjectLabel(GL_TEXTURE, Slabs.Temperature.Pong.ColorTexture, -1, "TEMPERATUREPONGTEXTURE");

    // Blur and brighten the density map:
    bool BlurAndBrighten = true;//false;//true;
    if (BlurAndBrighten) {
        glDisable(GL_BLEND);
        glBindFramebuffer(GL_FRAMEBUFFER, Surfaces.BlurredDensity.FboHandle);
        glViewport(0, 0, Slabs.Density.Ping.Width, Slabs.Density.Ping.Height);
        glBindVertexArray(Vaos.FullscreenQuad);
        glBindTexture(GL_TEXTURE_3D, Slabs.Density.Ping.ColorTexture);
        glUseProgram(BlurProgram);
        SetUniform("DensityScale", 5.0f);
        SetUniform("StepSize", sqrtf(2.0) / float(ViewSamples));
        SetUniform("InverseSize", recipPerElem(Vector3(float(GridWidth), float(GridHeight), float(GridDepth))));
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, GridDepth);
    }
    pezCheck(OpenGLError);

    // Generate the light cache:
    bool CacheLights = true;
    if (CacheLights) {
        glDisable(GL_BLEND);
        glBindFramebuffer(GL_FRAMEBUFFER, Surfaces.LightCache.FboHandle);
        glViewport(0, 0, Surfaces.LightCache.Width, Surfaces.LightCache.Height);
        glBindVertexArray(Vaos.FullscreenQuad);
        glBindTexture(GL_TEXTURE_3D, Surfaces.BlurredDensity.ColorTexture);
        glUseProgram(LightProgram);
        SetUniform("LightStep", sqrtf(2.0) / float(LightSamples));
        SetUniform("LightSamples", LightSamples);
        SetUniform("InverseSize", recipPerElem(Vector3(float(GridWidth), float(GridHeight), float(GridDepth))));
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, GridDepth);
    }

    // Perform raycasting:
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, cfg.Width, cfg.Height);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_BLEND);
    glBindVertexArray(Vaos.CubeCenter);
    glActiveTexture(GL_TEXTURE0);
    if (BlurAndBrighten)
        glBindTexture(GL_TEXTURE_3D, Surfaces.BlurredDensity.ColorTexture);
    else{
        glBindTexture(GL_TEXTURE_3D, Slabs.Density.Ping.ColorTexture);
        //if(thisLoop == 0)
        //glObjectLabel(GL_TEXTURE_3D, Slabs.Density.Ping.ColorTexture, -1, "DENSITYPINGTEXTURE");
    }

    thisLoop++;
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_3D, Surfaces.LightCache.ColorTexture);

    //remove if not trying to render obstacles

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_3D, Surfaces.Obstacles.ColorTexture);

    glUseProgram(RaycastProgram);
    SetUniform("ModelviewProjection", Matrices.ModelviewProjection);
    SetUniform("Modelview", Matrices.Modelview);
    SetUniform("ViewMatrix", Matrices.View);
    SetUniform("ProjectionMatrix", Matrices.Projection);
    SetUniform("ViewSamples", ViewSamples);
    SetUniform("EyePosition", EyePosition);
    SetUniform("Density", 0);
    SetUniform("LightCache", 1);
    SetUniform("RayOrigin", Vector4(transpose(Matrices.Modelview) * EyePosition).getXYZ());
    SetUniform("FocalLength", 1.0f / std::tan(FieldOfView / 2));
    SetUniform("WindowSize", float(cfg.Width), float(cfg.Height));
    SetUniform("StepSize", sqrtf(2.0) / float(ViewSamples));
    SetUniform("Obstacles", 2);
    SetUniform("obstacleColor", Vector3(1.0f, 0.0f, 0.0f));
    glDrawArrays(GL_POINTS, 0, 1);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_3D, 0);
    glActiveTexture(GL_TEXTURE0);
/*
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_3D, Surfaces.LightCache.ColorTexture);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_3D, Slabs.Velocity.Pong.ColorTexture);

    glUseProgram(RaycastVelProgram);
    SetUniform("ModelviewProjection", Matrices.ModelviewProjection);
    SetUniform("Modelview", Matrices.Modelview);
    SetUniform("ViewMatrix", Matrices.View);
    SetUniform("ProjectionMatrix", Matrices.Projection);
    SetUniform("ViewSamples", ViewSamples);
    SetUniform("EyePosition", EyePosition);
    SetUniform("Density", 0);
    SetUniform("LightCache", 1);
    SetUniform("RayOrigin", Vector4(transpose(Matrices.Modelview) * EyePosition).getXYZ());
    SetUniform("FocalLength", 1.0f / std::tan(FieldOfView / 2));
    SetUniform("WindowSize", float(cfg.Width), float(cfg.Height));
    SetUniform("StepSize", sqrtf(2.0) / float(ViewSamples));
    SetUniform("Velocity", 2);
    SetUniform("velocityColor", Vector3(0.0f, 1.0f, 0.0f));
    glDrawArrays(GL_POINTS, 0, 1);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_3D, 0);
    glActiveTexture(GL_TEXTURE0);*/

    pezCheck(OpenGLError);
}

//needs array dimensions (try some dim3 vars)
extern "C" cudaError_t
Jacobi_CUDA(cudaSurfaceObject_t pressurePingArray,
            cudaSurfaceObject_t pressurePongSurf,
            cudaTextureObject_t divergenceArray,
            cudaTextureObject_t obstaclesArray,
            dim3 textureDims,
            float alpha, float inverseBeta, int numLoops, int currentLoop);

/*
Jacobi_CUDA(pressurePingCudaArray, pressurePing_texture_resource, pressurePongCudaArray, pressurePong_texture_resource,
            divergenceCudaArray, divergence_texture_resource, obstaclesCudaArray, obstacles_texture_resource);
*/

size_t getSurfacePodSize(SurfacePod surfacePod){

    return surfacePod.Width * surfacePod.Height * surfacePod.Depth * sizeof(GLsizei);
}

//texture<float,1,cudaReadModeElementType> pressurePingTex, pressurePongTex, divergenceTex, obstaclesTex;

void swapCudaBuffers(float *&a, float*&b){
    float* temp;
    temp = a;
    a = b;
    b = temp;
}

//this is in cnn.cu (at least for now) (contains nothing right now)
//extern "C" cudaError_t updateObstaclesWrapper_float(dim3 obstacleExtent);

//these are in Fluid3d.cu
//velocitySize is used where it's used so we don't need two functions for staggered and non-staggered
//for a similar reason we do not fuse addGravity and addBuoyancy
extern "C" cudaError_t clampVelocityWrapper_float(float* velocity, dim3 velocitySize, float clampMin, float clampMax);
extern "C" cudaError_t addBuoyancyStaggeredWrapper_float(float* velocity, cudaTextureObject_t density, float* temperature, float* obstacles, dim3 domainSize, float3 gravity, float ambTemp, float buoyancyConstantAlpha, float buoyancyConstantBeta, float dt);
extern "C" cudaError_t addBuoyancyNotStaggeredWrapper_float(float* velocity, cudaTextureObject_t density, float* temperature, float* obstacles, dim3 domainSize, float3 gravity, float ambTemp, float buoyancyConstantAlpha, float buoyancyConstantBeta, float dt);
extern "C" cudaError_t addGravityWrapper_float(float* velocity, dim3 velocitySize, float3 gravity, float dt);
extern "C" cudaError_t advectVelocityCudaStaggeredWrapper_float(float* velocityIn, float* velocityOut, float* obstacles, float cellDim, dim3 domainSize, float dt);
extern "C" cudaError_t advectVelocityCudaNotStaggeredWrapper_float(float* velocityIn, float* velocityOut, float* obstacles, float cellDim, dim3 domainSize, float dt);
extern "C" cudaError_t advectCudaStaggeredWrapper_float(float* velocity, float* scalarFieldIn, float* scalarFieldOut, float* obstacles, dim3 domainSize, float cellDim, float dt);
extern "C" cudaError_t advectCudaNotStaggeredWrapper_float(float* velocity, float* scalarFieldIn, float* scalarFieldOut, float* obstacles, dim3 domainSize, float cellDim, float dt);
extern "C" cudaError_t advectDensityCudaStaggeredWrapper_float(float* velocityIn, cudaTextureObject_t densityPingTex, cudaSurfaceObject_t densityPongSurf, float* obstacles, dim3 domainSize, float cellDim, float dt);
extern "C" cudaError_t advectDensityCudaNotStaggeredWrapper_float(float* velocityIn, cudaTextureObject_t densityPingTex, cudaSurfaceObject_t densityPongSurf, float* obstacles, dim3 domainSize, float cellDim, float dt);
extern "C" cudaError_t vorticityConfinement_float(float* velocityIn, float* velocityOut, dim3 velocitySize);
extern "C" cudaError_t setWallBcsStaggeredWrapper_float(float* velocityIn, dim3 domainSize);
extern "C" cudaError_t setWallBcsNotStaggeredWrapper_float(float* velocityIn, dim3 domainSize);

extern "C" cudaError_t updateObstaclesWrapper_float(float* obstacles, float* velocityIn, dim3 obstacleExtent);
extern "C" cudaError_t updateObstaclesWrapper_double(double* obstacles, double* velocityIn, dim3 obstacleExtent);
extern "C" cudaError_t updateObstaclesWrapper_half(void* obstacles, void* velocityIn, dim3 obstacleExtent);

extern "C" cudaError_t subtractGradientStaggeredCuda_float(float* velocity, float* pressure, float* density, float* obstacles, dim3 velocitySize, float cellDim, float dt);
extern "C" cudaError_t subtractGradientNotStaggeredCuda_float(float* velocity, float* pressure, float* density, float* obstacles, dim3 velocitySize, float cellDim, float dt);

extern "C" cudaError_t applyInflowsStaggered_float(float* velocityIn, cudaSurfaceObject_t densityPingSurf, float* obstacles, float* temperature, float* inflowVelocity, float* inflowDensity, float* inflowTemperature, dim3 domainSize, float dt);
extern "C" cudaError_t applyInflowsNotStaggered_float(float* velocityIn, cudaSurfaceObject_t densityPingSurf, float* obstacles, float* temperature, float* inflowVelocity, float* inflowDensity, float* inflowTemperature, dim3 domainSize, float dt);

extern "C" cudaError_t applyControlledInflowsStaggered_float(float* velocityIn, cudaSurfaceObject_t densityPingSurf, float* obstacles, int posX, int posY, int posZ, int radius, float densityValue, float temperatureValue, float3 velocityValue, float* temperature, float* inflowVelocity, float* inflowDensity, float* inflowTemperature, dim3 domainSize, float dt);

//extern "C" cudaError_t initializeInflowsStaggered_float(float* inflowVelocity, float* inflowDensity, dim3 domainSize, int3 center, int radius);
//extern "C" cudaError_t initializeInflowsNotStaggered_float(float* inflowVelocity, float* inflowDensity, dim3 domainSize, int3 center, int radius);

extern "C" cudaError_t initializeInflowsStaggered_float(float* inflowVelocity, float* inflowDensity, float *inflowTemperature, dim3 domainSize, dim3 center, int radius, float ambTemp);
extern "C" cudaError_t initializeInflowsNotStaggered_float(float* inflowVelocity, float* inflowDensity, float* inflowTemperature, dim3 domainSize, dim3 center, int radius, float ambTemp);

extern "C" cudaError_t calculateDivergenceStaggeredWrapper_float(float* velocity, float* divergence, float cellDim, dim3 domainSize);
extern "C" cudaError_t calculateDivergenceNotStaggeredWrapper_float(float* velocity, float* divergence, float cellDim, dim3 domainSize);

extern "C" cudaError_t checkDivergenceWrapper_float(float* divergence, dim3 domainSize);

extern "C" cudaError_t JacobiCudaBuffers_float(float* pressure, float* divergence, float* obstacles, dim3 domainSize, float alpha, float inverseBeta, int numLoops);

extern "C" cudaError_t printCudaBuffersWrapper_float(float* buffer, dim3 size, char* name);
extern "C" cudaError_t printCudaBuffersWrapper3D_float(float* buffer, dim3 size, char* name);
extern "C" cudaError_t printCudaBuffersSideBySideWrapper3D_float(float* buffer1, float* buffer2, dim3 size, char* name1, char* name2);
extern "C" cudaError_t printTextureWrapper_float(cudaTextureObject_t aTexture, dim3 size, char* name);

extern "C" cudaError_t copyTextureCudaWrapper_float(cudaTextureObject_t src, cudaSurfaceObject_t dest, dim3 size);

extern "C" cudaError_t checkGradientFactorWrapper_float(float* pressure, float* velocity, dim3 domainSize);
//printTextureWrapper_float(densityPongTex, domainSize, "density")

extern "C" cudaError_t setBuffer3DToScalar_float(float* dest, float valX, float valY, float valZ, dim3 size);
extern "C" cudaError_t setBufferValuesWrapper_float(float* velocity, float* pressure, dim3 velocitySize, dim3 domainSize);

#if USE_CNN_CUDA_ADVECTION > 0
//BOOKMARK
cudaExtent densityExtent;
dim3 domainSize, velocitySize;
float *density, *velocityIn, *velocityOut, *pressure, *pressure2,  *obstacles, *input, *divergence, *temperatureIn, *temperatureOut, *inflowDensity, *inflowVelocity, *inflowTemperature;
struct cudaGraphicsResource *densityPing_texture_resource, *densityPong_texture_resource;
cudaArray *densityPingCudaArray, *densityPongCudaArray;
cudaTextureObject_t densityPingTex, densityPongTex;
cudaSurfaceObject_t densityPingSurf, densityPongSurf;
struct cudaResourceDesc densityPingResDesc, densityPongResDesc;
struct cudaTextureDesc densityPingTexDesc, densityPongTexDesc;
float buoyancyConstantAlpha = 0.01f, buoyancyConstantBeta = 0.01f; //NEED TO SET THESE TO A REASONABLE VALUE
float ambTemp = 0.0f; //this shouldn't matter as long as the temperature is relative to the value
float3 gravity = {.x = 0.0f, .y = -9.81f, .z = 0.0f};
//float3 gravity;
//gravity.x = 0.0f;
//gravity.y = -9.81f;
//gravity.z = 0.0f;

float* velTest;

//density does not need to be copied, can use textures and surfaces
//WHAT IS CALLED DENSITY HERE IS IN FACT SMOKE CONCENTRATION, AND HAS NOTHING TO DO WITH THE DENSITY IN THE POISSON EQUATION
void CudaSimulationLoop(int radius = 16){

    static int currentLoop = 0;
    static auto lastTime = std::chrono::steady_clock::now();
    static dim3 center;
    float3 velocityValue = make_float3(0.0,0.0,0.0);

    if(currentLoop == 0){
        checkCudaErrors(cudaGraphicsGLRegisterImage(&densityPing_texture_resource, Slabs.Density.Ping.ColorTexture, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsNone));
        checkCudaErrors(cudaGraphicsGLRegisterImage(&densityPong_texture_resource, Slabs.Density.Pong.ColorTexture, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsNone));

        memset(&densityPingResDesc, 0, sizeof(densityPingResDesc));
        densityPingResDesc.resType = cudaResourceTypeArray;
        memset(&densityPongResDesc, 0, sizeof(densityPongResDesc));
        densityPongResDesc.resType = cudaResourceTypeArray;
        memset(&densityPingTexDesc, 0, sizeof(densityPingTexDesc));
        texDesc.readMode = cudaReadModeElementType;
        memset(&densityPongTexDesc, 0, sizeof(densityPongTexDesc));
        texDesc.readMode = cudaReadModeElementType;

    }

    printf("\n\n\n################### LOOP %d #######################\n\n\n", currentLoop);

    infoMsg("mark CudaSimulationLoop");
    checkCudaErrors(cudaGraphicsMapResources(1, &densityPing_texture_resource));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&densityPingCudaArray, densityPing_texture_resource,0,0));
    checkCudaErrors(cudaGraphicsMapResources(1, &densityPong_texture_resource));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&densityPongCudaArray, densityPong_texture_resource,0,0));

    if(currentLoop == 0){

        unsigned int densityFlags;
        cudaChannelFormatDesc densityDesc;
        checkCudaErrors(cudaArrayGetInfo(&densityDesc, &densityExtent, &densityFlags, densityPingCudaArray));

        printf("densityDesc.x = %d, densityDesc.y = %d, densityDesc.z = %d, densityDesc.w = %d, densityDesc.f = %d\n",
               densityDesc.x, densityDesc.y, densityDesc.z, densityDesc.w, int(densityDesc.f));

        printf("densityExtent = %d %d %d (WHD)\n", densityExtent.width, densityExtent.height, densityExtent.depth);

        int velocitySizeTotal;
        int domainSizeTotal;

        if(useStaggered)
            velocitySizeTotal = 3 * (densityExtent.width+1)*(densityExtent.height+1)*(densityExtent.depth+1);
        else
            velocitySizeTotal = 3 * densityExtent.width * densityExtent.height * densityExtent.depth;

        checkCudaErrors(cudaMalloc(&velocityIn, velocitySizeTotal * sizeof(float)));
        checkCudaErrors(cudaMalloc(&velocityOut, velocitySizeTotal * sizeof(float)));
        checkCudaErrors(cudaMemset(velocityIn, 0, velocitySizeTotal*sizeof(float)));
        checkCudaErrors(cudaMemset(velocityOut, 0, velocitySizeTotal*sizeof(float)));


        if(1){
            dim3 velocitySize2 = dim3((densityExtent.width+1),(densityExtent.height+1),(densityExtent.depth+1));
            checkCudaErrors(cudaMalloc(&velTest, sizeof(float)*velocitySizeTotal));
            checkCudaErrors(cudaMemset(velTest, 0, sizeof(float) * velocitySizeTotal));
            checkCudaErrors(setBuffer3DToScalar_float(velTest, 0.0, 11.0, 0.0, velocitySize2));
        }

        domainSizeTotal = densityExtent.width * densityExtent.height * densityExtent.depth;

        printf("\n\n\nvelocitySizeTotal = %d, domainSizeTotal = %d\n\n\n", velocitySizeTotal, domainSizeTotal);

        //allocate pressure, velocity divergence, and obstacles IN A BLOCK, TOGETHER, in that order
        checkCudaErrors(cudaMalloc(&input, 3 * domainSizeTotal * sizeof(float)));
        checkCudaErrors(cudaMemset(input,0,3 * domainSizeTotal * sizeof(float)));
        pressure = input;
        divergence = input + domainSizeTotal;
        obstacles = input + 2 * domainSizeTotal;

        checkCudaErrors(cudaMalloc(&temperatureIn, densityExtent.width * densityExtent.height * densityExtent.depth * sizeof(float)));
        checkCudaErrors(cudaMalloc(&temperatureOut, densityExtent.width * densityExtent.height * densityExtent.depth * sizeof(float)));
        checkCudaErrors(cudaMemset(temperatureIn, 0, densityExtent.width * densityExtent.height * densityExtent.depth * sizeof(float)));
        checkCudaErrors(cudaMemset(temperatureOut, 0, densityExtent.width * densityExtent.height * densityExtent.depth * sizeof(float)));
        //should set the initial temperature to ambTemp; better force 0 ambTemp and subtract its value from any inflow temperature values the user sets; this should work as long as the temperature dependence is a linear expression of (T-Tamb)

        //allocate and initialize inflow/outflow regions
        checkCudaErrors(cudaMalloc(&inflowDensity, densityExtent.width * densityExtent.height * densityExtent.depth * sizeof(float))); //this is for both inflows and outflows (can be positive or negative)
        checkCudaErrors(cudaMalloc(&inflowTemperature, densityExtent.width * densityExtent.height * densityExtent.depth * sizeof(float)));
        checkCudaErrors(cudaMalloc(&inflowVelocity, velocitySizeTotal * sizeof(float)));

#define min(a,b,c) (((a)>(b))?((b)>(c)?(c):(b)):(((a)>(c))?(c):(a)))
        domainSize = dim3(densityExtent.width, densityExtent.height, densityExtent.depth);
        center.x = domainSize.x / 2;
        center.y = domainSize.y / 2;
        center.z = domainSize.z / 2;
        checkCudaErrors(cudaMemset(inflowVelocity, 0, velocitySizeTotal*sizeof(float)));
        checkCudaErrors(cudaMemset(inflowDensity, 0, domainSizeTotal*sizeof(float)));
        checkCudaErrors(cudaMemset(inflowTemperature, 0, domainSizeTotal*sizeof(float)));
        if(useStaggered){
            //checkCudaErrors(initializeInflowsStaggered_float(inflowVelocity, inflowDensity, domainSize, make_int3(domainSize.x/2, domainSize.y/2, domainSize.z/2), min(domainSize.x, domainSize.y, domainSize.z)/8));
            checkCudaErrors(initializeInflowsStaggered_float(inflowVelocity, inflowDensity, inflowTemperature, domainSize, center, min(domainSize.x, domainSize.y, domainSize.z)/8, ambTemp));}
        else
        {
            //checkCudaErrors(initializeInflowsNotStaggered_float(inflowVelocity, inflowDensity, domainSize, make_int3(domainSize.x/2, domainSize.y/2, domainSize.z/2), min(domainSize.x, domainSize.y, domainSize.z)/8));
            checkCudaErrors(initializeInflowsNotStaggered_float(inflowVelocity, inflowDensity, inflowTemperature, domainSize, center, min(domainSize.x, domainSize.y, domainSize.z)/8, ambTemp));
        }
#undef min
    }

    densityPingResDesc.res.array.array = densityPingCudaArray;
    densityPongResDesc.res.array.array = densityPongCudaArray;

    //if(currentLoop%2==0){
    if(0){
        checkCudaErrors(cudaCreateSurfaceObject(&densityPingSurf, &densityPingResDesc));
        checkCudaErrors(cudaCreateTextureObject(&densityPingTex, &densityPingResDesc, &densityPingTexDesc, NULL));
        checkCudaErrors(cudaCreateSurfaceObject(&densityPongSurf, &densityPongResDesc));
        checkCudaErrors(cudaCreateTextureObject(&densityPongTex, &densityPongResDesc, &densityPongTexDesc, NULL));}
    else{
        checkCudaErrors(cudaCreateSurfaceObject(&densityPongSurf, &densityPingResDesc));
        checkCudaErrors(cudaCreateTextureObject(&densityPongTex, &densityPingResDesc, &densityPingTexDesc, NULL));
        checkCudaErrors(cudaCreateSurfaceObject(&densityPingSurf, &densityPongResDesc));
        checkCudaErrors(cudaCreateTextureObject(&densityPingTex, &densityPongResDesc, &densityPongTexDesc, NULL));
    }

    //--- PROPER LOOP STARTS HERE ---

    checkCudaErrors(updateObstaclesWrapper_float(obstacles, velocityIn, domainSize)); //will also probably take the obstacles and the velocity

    //write what the order in torch is here
    //tfluids.advectScalar for all scalars (just density in torch. Here, should use the usual buoyancy force val*(T-T0)*(-gvec))
    //tfluids.advectVel (velocity advected last)
    //setConstVals (probably does nothing) (it does some weird stuff about boundary conditions, try ignoring it and see if it goes away)
    //tfluids.addBuoyancy
    //tfluids.addGravity (does not affect the velocity field in torch, I think)
    //tfluids.vorticityConfinezment
    //tfluids.setWallBcs (as part of model:forward()
    //setConstVals again
    //will probably try to do object motion here (take newly occupied cells into account for next step)
    //push density out of obstacles here
    //clamp velocity to (-1e6, 1e6)
    //calculate velocity divergence (THIS needs to be part of input)

    auto currentTime = std::chrono::steady_clock::now();
    auto dt = ((std::chrono::duration<float>)(currentTime - lastTime)).count();//((std::chrono::duration<double>)(lastTime - currentTime)).count();
    lastTime = currentTime;

    //dt = 1/60.0;
    //dt = 0.25f;//0.25f;

    if(useJacobiCuda){
        dt = 0.25f;
    }

    //this is for testing that the output for a certain input is the same in torch as it is here
    //the velocity size here is domainSize.n+1 in all dimensions, but it is domainSize.n in torch; this may create issues at some edges
    if(0){

        float* velocityCPU, *pressureCPU;
        int velocitySizeTotal;

        if(useStaggered){
            velocitySizeTotal = (domainSize.x+1) * (domainSize.y+1) * (domainSize.z+1) * 3;
            velocitySize.x = domainSize.x+1;
            velocitySize.y = domainSize.y+1;
            velocitySize.z = domainSize.z+1;}
        else{
            velocitySizeTotal = domainSize.x * domainSize.y * domainSize.z * 3;
            velocitySize.x = domainSize.x;
            velocitySize.y = domainSize.y;
            velocitySize.z = domainSize.z;}

        int domainSizeTotal = domainSize.x * domainSize.y * domainSize.z;

        printf("velocitySize.x = %d, velocitySize.y = %d, velocitySize.z = %d\n", velocitySize.x, velocitySize.y, velocitySize.z);

        checkCudaErrors(setBufferValuesWrapper_float(velocityIn, pressure, velocitySize, domainSize));
        /*
        velocityCPU = (float*)(malloc(sizeof(float)*velocitySizeTotal));
        pressureCPU = (float*)(malloc(sizeof(float)*domainSizeTotal));

        checkCudaErrors(cudaMemcpy(velocityCPU, velocityIn, sizeof(float)*velocitySizeTotal, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(pressureCPU, pressure, sizeof(float)*domainSizeTotal, cudaMemcpyDeviceToHost));

        std::ofstream outFile_initial;

        outFile_initial.open("UPred_pPred_CPP_initial_26-09-2018_05-40.txt");

        outFile_initial<<"UPred\n";

        //now write the values to file
        for(int i=0; i<velocitySizeTotal; i++) //doesn't account for different sizes of velocity tensor
            outFile_initial<<velocityCPU[i]<<"\n";

        outFile_initial<<"pPred\n";

        for(int i=0;i<domainSize.x*domainSize.y*domainSize.z;i++)
            outFile_initial<<pressureCPU[i]<<"\n";

        //exit(0);
*/


        checkCudaErrors(calculateDivergenceStaggeredWrapper_float(velocityIn, divergence, CellSize, domainSize));
        defaultNet->loadInput(input, pressure, velocityIn, domainSize, velocitySize);
        defaultNet->forward();

        checkCudaErrors(cudaDeviceSynchronize());
        /*
        float* inputCPU = (float*)(malloc(sizeof(float)*domainSizeTotal*3));
        checkCudaErrors(cudaMemcpy(inputCPU, defaultNet->inputLayer->data, 3*sizeof(float)*domainSizeTotal, cudaMemcpyDeviceToHost));

        std::ofstream outFile;

        outFile.open("input_conv_CPP.txt");

        for(int i=0;i<3*domainSizeTotal;i++)
            outFile<<inputCPU[i]<<"\n";
applyControlledInflowsStaggered_float
        outFile.close();
        exit(0);
*/

        /*
        float* divergenceCPU = (float*)(malloc(sizeof(float)*domainSizeTotal));
        checkCudaErrors(cudaMemcpy(divergenceCPU, divergence, sizeof(float)*domainSizeTotal, cudaMemcpyDeviceToHost));

        std::ofstream outFile;

        outFile.open("divergence_CPP.txt");

        for(int i=0;i<domainSizeTotal;i++)
            outFile<<divergenceCPU[i]<<"\n";

        outFile.close();
        exit(0);
        */
        //now copy the values to the CPU

        /*
        velocityCPU = (float*)(malloc(sizeof(float)*velocitySizeTotal));
        pressureCPU = (float*)(malloc(sizeof(float)*domainSizeTotal));

        checkCudaErrors(cudaMemcpy(velocityCPU, velocityIn, sizeof(float)*velocitySizeTotal, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(pressureCPU, pressure, sizeof(float)*domainSizeTotal, cudaMemcpyDeviceToHost));

        //outFile_initial.close();

        std::ofstream outFile;

        outFile.open("UPred_pPred_CPP_27-09-2018_09-40.txt");

        outFile<<"UPred\n";

        //now write the values to file
        for(int i=0; i<velocitySizeTotal; i++) //doesn't account for different sizes of velocity tensor
            outFile<<velocityCPU[i]<<"\n";

        outFile<<"pPred\n";

        for(int i=0;i<domainSize.x*domainSize.y*domainSize.z;i++)
            outFile<<pressureCPU[i]<<"\n";

        outFile.close();

        exit(0);*/
    }

    //there should be some inflows or something (at what point in the loop?)

    printf("velocityIn at beginning -> %p", velocityIn);

    //velocityDissipation is not used. Density dissipation might be used
    if(useStaggered){
        domainSize = dim3(densityExtent.width, densityExtent.height, densityExtent.depth);
        velocitySize = dim3(densityExtent.width+1, densityExtent.height+1, densityExtent.depth+1);
        checkCudaErrors(cudaDeviceSynchronize());
        //checkCudaErrors(advectCudaStaggeredWrapper_float(velocityIn, pressureIn, pressureOut, dt)); //torch version does not advect pressure; chaging that would probably break the inference

        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(advectCudaStaggeredWrapper_float(velocityIn, temperatureIn, temperatureOut, obstacles, domainSize, CellSize, dt)); //this fails too somtimes if the density advection doesn't
        checkCudaErrors(cudaDeviceSynchronize());
        swapCudaBuffers(temperatureIn, temperatureOut);
        //apply inflows now (so they don't get advected with the rest of the velocity field, but are still applied in time to advect the scalar fields)
        /*
        if(currentLoop == 1){
            checkCudaErrors(printCudaBuffersWrapper3D_float(velocityIn, velocitySize, "velocityIn"));
            printf("Exiting (%s:%d, %s)",__FILE__,__LINE__,__FUNCTION__);
            exit(0);}*/


        checkCudaErrors(advectDensityCudaStaggeredWrapper_float(velocityIn, densityPingTex, densityPongSurf, obstacles, domainSize, CellSize, dt)); //ERROR HERE (not on first loop)
        checkCudaErrors(advectDensityCudaStaggeredWrapper_float(velocityIn, densityPingTex, densityPongSurf, obstacles, domainSize, CellSize, dt)); //ERROR HERE (not on first loop)
        //checkCudaErrors(advectDensityCudaStaggeredWrapper_float(velTest, densityPingTex, densityPongSurf, obstacles, domainSize, CellSize, dt)); //ERROR HERE (not on first loop)
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(advectVelocityCudaStaggeredWrapper_float(velocityIn, velocityOut, obstacles, CellSize, domainSize, dt)); //this might need to be at the end, depending on how it is in the torch version. //->out of resources
        //checkCudaErrors(cudaErrorInvalidValue);
        checkCudaErrors(cudaDeviceSynchronize()); //ERROR HERE on first loop
        //if swapCudaBuffers is done twice, velocityIn should be continguous with pressure and temperature
        swapCudaBuffers(velocityIn, velocityOut); //this DOES work

        //if(currentLoop <10){float
        //might need to move this back to the beginning (change to densityPingSurf)
        //checkCudaErrors(applyInflowsStaggered_float(velocityIn, densityPongSurf, obstacles, temperatureIn, inflowVelocity, inflowDensity, inflowTemperature, domainSize, dt)); //takes obstacles so that density is not inflowed into occupied cells; need to ensure non-negative density values after outflows are applied
        //}
        checkCudaErrors(applyControlledInflowsStaggered_float(velocityIn, densityPongSurf, obstacles, ImpulsePosition.getX(), ImpulsePosition.getY(), ImpulsePosition.getZ(), SplatRadius, ImpulseDensity, ImpulseTemperature, velocityValue, temperatureIn, inflowVelocity, inflowDensity, inflowTemperature, domainSize, dt));

        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(addBuoyancyStaggeredWrapper_float(velocityIn, densityPongTex, temperatureIn, obstacles, domainSize, gravity, ambTemp, buoyancyConstantAlpha, buoyancyConstantBeta, dt)); //might need to modify the buoyancy force in torch and retrain if this breaks anything
        //checkCudaErrors(addGravityWrapper_float(velocityIn, velocitySize, gravity));
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(vorticityConfinement_float(velocityIn, velocityOut, velocitySize)); //for now, does nothing; since vorticity depends on the velocity neighborhood, cannot overwrite it]
        //DO NOT UNCOMMENT UNTIL VORTICITY CONFINEMENT IS IMPLEMENTED //swapCudaBuffers(velocityIn, velocityOut); //do not need for ineffective vorticityConfinement_float
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(setWallBcsStaggeredWrapper_float(velocityIn, domainSize)); //this does nothing if there are no obstacles
        //checkCudaErrors(clampVelocityWrapper_float(velocityIn, domainSize, VELOCITY_MIN, VELOCITY_MAX)); //might need VELOCITY_MIN_FLOAT etc.
        checkCudaErrors(calculateDivergenceStaggeredWrapper_float(velocityIn, divergence, CellSize, domainSize));
    }
    else{
        domainSize = dim3(densityExtent.width, densityExtent.height, densityExtent.depth);
        velocitySize = domainSize;
        //checkCudaErrors(advectCudaNotStaggeredWrapper_float(velocityIn, pressure, pressureOut, dt));
        checkCudaErrors(applyInflowsNotStaggered_float(velocityIn, densityPingSurf, obstacles, temperatureIn, inflowVelocity, inflowDensity, inflowTemperature, domainSize, dt));
        //checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(advectCudaNotStaggeredWrapper_float(velocityIn, temperatureIn, temperatureOut, obstacles, domainSize, CellSize, dt));
        swapCudaBuffers(temperatureIn, temperatureOut);
        checkCudaErrors(advectDensityCudaNotStaggeredWrapper_float(velocityIn, densityPingTex, densityPongSurf, obstacles, domainSize, CellSize, dt));
        checkCudaErrors(advectVelocityCudaNotStaggeredWrapper_float(velocityIn, velocityOut, obstacles, CellSize, domainSize, dt)); //this might need to be at the end, depending on how it is in the torch version.
        swapCudaBuffers(velocityIn, velocityOut);
        checkCudaErrors(addBuoyancyNotStaggeredWrapper_float(velocityIn, densityPongTex, temperatureIn, obstacles, domainSize, gravity, ambTemp, buoyancyConstantAlpha, buoyancyConstantBeta, dt));
        //checkCudaErrors(addGravityWrapper_float(velocityIn, velocitySize, gravity)); //don't add gravity to smoke
        checkCudaErrors(vorticityConfinement_float(velocityIn, velocityOut, domainSize)); //for now, does nothing
        //swapCudaBuffers(velocityIn, velocityOut); //do not need for ineffective vorticityConfinement_float
        checkCudaErrors(setWallBcsNotStaggeredWrapper_float(velocityIn, domainSize));
        checkCudaErrors(clampVelocityWrapper_float(velocityIn, domainSize, VELOCITY_MIN, VELOCITY_MAX));
        checkCudaErrors(calculateDivergenceNotStaggeredWrapper_float(velocityIn, divergence, CellSize, domainSize));
    }

    //checkCudaErrors(printCudaBuffersWrapper3D_float(velocityIn, velocitySize, "velocityIn"));

    printf("velocityIn after advection and stuff -> %p", velocityIn);

    checkCudaErrors(cudaDeviceSynchronize());

    //checkCudaErrors(printCudaBuffersWrapper3D_float(velocityIn, velocitySize, "velocityIn"));
    //printf("Exiting (%s:%d, %s)",__FILE__,__LINE__,__FUNCTION__);
    //exit(0);

    //printf("\n\n\n\n\n\n\n");

    //infoMsg("mark CNNProjectionCudaWrapper\n");

    //defaultNet->forward(divergenceCudaArray, pressurePongCudaArray);
    //this might actually require the divergence of the pressure (for some reason)

    //printf("pressure2 = %p\n", pressure2);
    //
    // DOES NOT WORK WITHOUT THIS!
    //
    //defaultNet->getOutput(pressure2);
    //printf("pressure2 = %p\n", pressure2);

    //checkCudaErrors(printCudaBuffersWrapper3D_float(velocityIn, velocitySize, "velocityIn"));
    //printf("Exiting (%s:%d, %s)",__FILE__,__LINE__,__FUNCTION__);
    //exit(0);

    //need to copy pressure2 to pressure instead of swapping, at least until not allocating the output in DefaultNet works
    //cudaMemcpy(pressure, pressure2, domainSize.x*domainSize.y*domainSize.z*sizeof(float), cudaMemcpyDeviceToDevice);
    //swapCudaBuffers(pressure, pressure2);

    /*
    printf("Exiting...");
    exit(1);
    */
    //toyCNN->ForwardPropagationV2(divergenceCudaArray, pressurePingSurf); //this is the version with surfaces (and maybe textures) instead of the target (and maybe source cudaArrays)

    //need to replace this with something, idk what
    //Jacobi_no_program(pressurePing, divergence, obstacles, pressurePong);
    /*
    checkCudaErrors(printTextureWrapper_float(densityPongTex, domainSize, "density"));
    printf("Exiting (%s:%d, %s)",__FILE__,__LINE__,__FUNCTION__);
    exit(0);
*/
    //copy the densityPong over densityPing to avoid texure ping-ponging confusion
    //checkCudaErrors(forceDensityOffset_float(density)); //make sure the density interop isn't faulty
    checkCudaErrors(copyTextureCudaWrapper_float(densityPongTex, densityPingSurf, domainSize));

    checkCudaErrors(cudaGraphicsUnmapResources(1, &densityPing_texture_resource, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &densityPong_texture_resource, 0));

    //k = k%2;

    //SubtractGradient(Slabs.Velocity.Ping, Slabs.Pressure.Ping, Surfaces.Obstacles, Slabs.Velocity.Pong); //<--- replace with kernel version
    //SwapSurfaces(&Slabs.Velocity);

    if(useJacobiCuda){
        //the gradient subtraction does not work yet, so do not use
        float alpha = -CellSize * CellSize;
        //printf("\ninverseBeta = %f", inverseBeta);
        int numLoops = NumJacobiIterations; //40
        //float inverseBeta = 0.1666f; //same value as in Utility.cpp
        checkCudaErrors(JacobiCudaBuffers_float(pressure, divergence, obstacles, domainSize, alpha, inverseBeta, numLoops));
    }
    else{

        //printf("velocitySize = (%d, %d, %d)\n", velocitySize.x, velocitySize.y, velocitySize.z);
        defaultNet->loadInput(input, pressure, velocityIn, domainSize, velocitySize);
        defaultNet->forward();

        checkCudaErrors(cudaDeviceSynchronize());

        //checkCudaErrors(checkGradientFactorWrapper_float(pressure, velocityIn, domainSize));
        checkCudaErrors(cudaDeviceSynchronize());

        if(useStaggered)
            checkCudaErrors(subtractGradientStaggeredCuda_float(velocityIn, pressure, density, obstacles, velocitySize, CellSize, dt)); //density buffer is not allocated and is not used
        else
            checkCudaErrors(subtractGradientNotStaggeredCuda_float(velocityIn, pressure, density, obstacles, velocitySize, CellSize, dt));
    }

    //SwapSurfaces(&Slabs.Density);
    //SimulateFluid = !SimulateFluid;

    checkCudaErrors(cudaDeviceSynchronize());

    //calculateDivergenceStaggeredWrapper_float(velocityIn, divergence, CellSize, domainSize);
    //checkDivergenceWrapper_float(divergence, domainSize);

    checkCudaErrors(cudaDeviceSynchronize());

    //exit(0);

    currentLoop++;
}

#else
#if USE_CNN > 0
float* velocityPingBuffer;

void CNNProjectionCudaWrapper(SurfacePod pressurePing, SurfacePod divergence, SurfacePod obstacles, SurfacePod pressurePong, SurfacePod velocityPing, SurfacePod velocityPong){

    static int currentLoop = 0;

    infoMsg("mark CNNProjectionCudaWrapper\n");

    if(currentLoop == 0){
    checkCudaErrors(cudaGraphicsMapResources(1, &pressurePing_texture_resource));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&pressurePingCudaArray, pressurePing_texture_resource,0,0));

    checkCudaErrors(cudaGraphicsMapResources(1, &divergence_texture_resource));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&divergenceCudaArray, divergence_texture_resource,0,0));

    checkCudaErrors(cudaGraphicsMapResources(1, &obstacles_texture_resource));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&obstaclesCudaArray, obstacles_texture_resource,0,0));

    checkCudaErrors(cudaGraphicsMapResources(1, &obstacleSpeeds_texture_resource));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&obstacleSpeedsCudaArray, obstacleSpeeds_texture_resource,0,0));

    //pressurepong needs a surface, since we will write to it (probably with a kernel)
    checkCudaErrors(cudaGraphicsMapResources(1, &pressurePong_texture_resource));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&pressurePongCudaArray, pressurePong_texture_resource,0,0));

    checkCudaErrors(cudaGraphicsMapResources(1, &velocityPing_texture_resource));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&velocityPingCudaArray, velocityPing_texture_resource,0,0));

    checkCudaErrors(cudaGraphicsMapResources(1, &velocityPong_texture_resource));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&velocityPongCudaArray, velocityPong_texture_resource,0,0));}
/*
    //printCudaBuffersWrapper3D_float(velocityPingCudaArray,);
    //if(currentLoop == 1){
        printf("first\n");
        printf("velocityPingCudaArray at iteration number %d\n", currentLoop);
        checkCudaErrors(printDataCudaArrayContents3DWrapper(velocityPingCudaArray));
        checkCudaErrors(cudaDeviceSynchronize());
        //exit(0);
        printf("\n\n");
        printf("divergenceCudaArray at iteration number %d\n", currentLoop);
        checkCudaErrors(printDataCudaArrayContentsWrapper(divergenceCudaArray));
        checkCudaErrors(cudaDeviceSynchronize());
        printf("\n\n");
        printf("pressure at iteration number %d\n", currentLoop);
        printCudaBuffersWrapper_float(defaultNet->pressure, dim3(64,64,64), "pressure");
        checkCudaErrors(cudaDeviceSynchronize());
        printf("\n\n");//}
        //*/

    //copy velocity to cpu, save it

    //need to make sure the divergence computation matches the torch version

    //addScalarToVelocity();

    //checkCudaErrors(copyCudaArrayToDeviceArrayWrapper3D_float(velocityPingCudaArray, velocityPingBuffer));

    //copy velocityPingCudaArray to a GPU buffer
    /*
    if(currentLoop == 0)
        checkCudaErrors(cudaMalloc(&velocityPingBuffer, 3*velocityPing.Width*velocityPing.Height*velocityPing.Depth*sizeof(float)));

    checkCudaErrors(copyCudaArrayToDeviceArrayWrapper3D_float(velocityPingCudaArray, velocityPingBuffer));*/

    //copy this GPU buffer to the CPU

    defaultNet->loadInput(pressurePingCudaArray, obstaclesCudaArray, divergenceCudaArray, pressurePongCudaArray, velocityPingCudaArray, velocityPingCudaArray);

    /*
    if(currentLoop %2 == 0)
        defaultNet->loadInput(pressurePingCudaArray, obstaclesCudaArray, divergenceCudaArray, pressurePongCudaArray, velocityPingCudaArray, velocityPongCudaArray);
    else
        defaultNet->loadInput(pressurePingCudaArray, obstaclesCudaArray, divergenceCudaArray, pressurePongCudaArray, velocityPongCudaArray, velocityPingCudaArray);
    */
/*
    //if(currentLoop == 0){
        printf("last\n");
        printf("velocityPingCudaArray at iteration number %d\n", currentLoop);
        checkCudaErrors(printDataCudaArrayContents3DWrapper(velocityPingCudaArray));
        checkCudaErrors(cudaDeviceSynchronize());
        printf("\n\n");
        printf("divergenceCudaArray at iteration number %d\n", currentLoop);
        checkCudaErrors(printDataCudaArrayContentsWrapper(divergenceCudaArray));
        checkCudaErrors(cudaDeviceSynchronize());
        printf("\n\n");
        printf("pressure at iteration number %d\n", currentLoop);
        //printCudaBuffersWrapper_float(defaultNet->pressure, dim3(64,64,64), "pressure");
        checkCudaErrors(cudaDeviceSynchronize());
        printf("\n\n");//}
*/
    checkCudaErrors(cudaDeviceSynchronize());

    defaultNet->forward(false, true, (float)(currentLoop));

    //printf("");

    //this has been moved to DefaultNet->forward()
    //if(useStaggered)
        //subtractGradientStaggeredCuda_float(defaultNet->velocity, defaultNet->pressure, NULL, defaultNet->divergence, dim3(defaultNet->outputLayer->tensorDims[4], defaultNet->outputLayer->tensorDims[3], defaultNet->outputLayer->tensorDims[2]), CellSize, dt); //NULL corresponds to the density, which isn't used anyway
    //else
    //    subtractGradientNotStaggeredCuda_float(defaultNet->velocity, defaultNet->pressure, NULL, defaultNet->divergence, dim3(defaultNet->outputLayer->tensorDims[4], defaultNet->outputLayer->tensorDims[3], defaultNet->outputLayer->tensorDims[2]), CellSize, dt); //needs velocity and pressure; they are already there

    //checkCudaErrors(copyCudaArrayToDeviceArrayWrapper3D_float(velocityPingCudaArray, velocityPingBuffer));

    //subtractGradientNotStaggeredCuda_float(float* velocity, float* pressure, float* density, float* obstacles, dim3 velocitySize, float cellDim){
    defaultNet->getOutput(pressurePingCudaArray);
    //need to overwrite velocityPing with the gradient-subtracted version

    checkCudaErrors(copyDeviceArrayToCudaArray3DWrapper_float(defaultNet->velocity, velocityPingCudaArray));

    //checkCudaErrors(printDataCudaArrayContents3DWrapper(velocityPingCudaArray));

    //copy the projected velocity to a CPU buffer
    //checkCudaErrors(copyCudaArrayToDeviceArrayWrapper3D_float(velocityPingCudaArray, velocityPingBuffer));
    //checkCudaErrors(cudaMemcpy(velocityPingBufferCPU2, velocityPingBuffer, 3*velocityPing.Width*velocityPing.Height*velocityPing.Depth*sizeof(float), cudaMemcpyDeviceToHost));

    //print the GPU buffers side by side
    //checkCudaErrors(printCudaBuffersSideBySideWrapper3D_float(velocityPingBuffer, defaultNet->velocity, dim3(velocityPing.Width, velocityPing.Height, velocityPing.Depth), "velocityPingBefore", "velocityPingAfter"));

    /*
    if(currentLoop%2==0)
        checkCudaErrors(copyDeviceArrayToCudaArray3DWrapper_float(defaultNet->velocity, velocityPongCudaArray));
    else
        checkCudaErrors(copyDeviceArrayToCudaArray3DWrapper_float(defaultNet->velocity, velocityPingCudaArray));
    */

    //printCudaBuffersWrapper3D_float(defaultNet->velocity, dim3(64,64,64), "velocityPing");
/*
    if(currentLoop == 1){
        printf("eh\n");
        printf("velocityPingCudaArray at iteration number %d\n", currentLoop);
        checkCudaErrors(printDataCudaArrayContents3DWrapper(velocityPingCudaArray));
        checkCudaErrors(cudaDeviceSynchronize());
        printf("\n\n");
        printf("divergenceCudaArray at iteration number d\n", currentLoop);
        checkCudaErrors(printDataCudaArrayContentsWrapper(divergenceCudaArray));
        checkCudaErrors(cudaDeviceSynchronize());
        printf("\n\n");
        printf("pressure at iteration number %d\n", currentLoop);
        printCudaBuffersWrapper_float(defaultNet->pressure, dim3(64,64,64), "pressure");
        checkCudaErrors(cudaDeviceSynchronize());
        printf("\n\n");}
*/
    currentLoop++;

    checkCudaErrors(cudaDeviceSynchronize());

    //toyCNN->ForwardPropagationV2(divergenceCudaArray, pressurePingSurf); //this is the version with surfaces (and maybe textures) instead of the target (and maybe source cudaArrays)

    //printCudaBuffersWrapper_float(defaultNet->pressure, dim3(64,64,64), "pressure");
    //printCudaBuffersWrapper3D_float(defaultNet->velocity, dim3(64,64,64), "velocity");
/*
    //if(currentLoop == 0){
    checkCudaErrors(cudaGraphicsUnmapResources(1, &pressurePing_texture_resource, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &pressurePong_texture_resource, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &divergence_texture_resource, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &obstacles_texture_resource, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &obstacleSpeeds_texture_resource, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &velocityPing_texture_resource, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &velocityPong_texture_resource, 0));//}*/

    //as far as I can tell, without the program this just loads and unloads the active textures wihtout doing anything
    //Jacobi_no_program(pressurePing, divergence, obstacles, pressurePong);
}
/*
extern "C" addObstaclesWrapper_float(cudaSurfaceObject_t densityPingSurf, cudaTextureObject_t obstaclesTex, dim3 domainSize);

void addObstaclesToDensityTexture(){

    //will try copying to densityPing first (so obstacles don't overwrite irreversibly), and copy back before we need densityPing again

    checkCudaErrors(cudaGraphicsMapResources(1, &densityPong_texture_resource));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&densityPongCudaArray, densityPong_texture_resource,0,0));

    checkCudaErrors(cudaCreateSurfaceObject(&, &))

    checkCudaErrors(addObstaclesWrapper_float(densityPongSurf, obstaclesTex, domainSize));

    checkCudaErrors(cudaGraphicsUnmapResources(1, &densityPong_texture_resources, 0));
}
*/
#else
void ToyCudaWrapper(SurfacePod pressurePing, SurfacePod divergence, SurfacePod obstacles, SurfacePod pressurePong){

    printf("mark ToyCudaWrapper");

    checkCudaErrors(cudaGraphicsMapResources(1, &pressurePing_texture_resource));
    //checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**) ));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&pressurePingCudaArray, pressurePing_texture_resource,0,0));

    //checkCudaErrors(cudaGraphicsMapResources(1, &pressurePong_texture_resource));
    //checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**) &pressurePongFBOptr, &pressurePongFBOsize, pressurePong_texture_resource));
    //checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&pressurePongCudaArray, pressurePong_texture_resource,0,0));

    checkCudaErrors(cudaGraphicsMapResources(1, &divergence_texture_resource));
    //checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**) &divergenceFBOptr, &divergenceFBOsize, divergence_texture_resource));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&divergenceCudaArray, divergence_texture_resource,0,0));

    //checkCudaErrors(cudaGraphicsMapResources(1, &obstacles_texture_resource));
    //checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&obstaclesCudaArray, obstacles_texture_resource,0,0));
    /*
  //additions start here
  pressurePingResDesc.res.array.array = pressurePingCudaArray;
  //pressurePongResDesc.res.array.array = pressurePongCudaArray;
  divergenceResDesc.res.array.array = divergenceCudaArray;
  obstaclesResDesc.res.array.array = obstaclesCudaArray;

  checkCudaErrors(cudaCreateSurfaceObject(&pressurePingSurf,&pressurePingResDesc));
  checkCudaErrors(cudaCreateTextureObject(&divergenceTex,  &divergenceResDesc, &texDesc,NULL));
  checkCudaErrors(cudaCreateTextureObject(&obstaclesTex,   &obstaclesResDesc, &texDesc,NULL));

  // Create the surface objects

  //checkCudaErrors(cudaCreateSurfaceObject(&pressurePongSurf, &pressurePongResDesc));

  //and end here
  */

    toyCNN->ForwardPropagation(divergenceCudaArray, pressurePongCudaArray);

    //toyCNN->ForwardPropagationV2(divergenceCudaArray, pressurePingSurf); //this is the version with surfaces (and maybe textures) instead of the target (and maybe source cudaArrays)

    checkCudaErrors(cudaGraphicsUnmapResources(1, &pressurePing_texture_resource, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &pressurePong_texture_resource, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &divergence_texture_resource, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &obstacles_texture_resource, 0));

    Jacobi_no_program(pressurePing, divergence, obstacles, pressurePong);

}
#endif
#endif

//this is probably not the wrapper, the extern function probably is, but whatever
void Jacobi_CUDA_wrapper(SurfacePod pressurePing, SurfacePod divergence, SurfacePod obstacles, SurfacePod pressurePong, int currentLoop){

    dim3 pressurePingDims;

    pressurePingDims.x = pressurePing.Width;
    pressurePingDims.y = pressurePing.Height;
    pressurePingDims.z = pressurePing.Depth;
    //map the 3D textures
    checkCudaErrors(cudaGraphicsMapResources(1, &pressurePing_texture_resource));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&pressurePingCudaArray, pressurePing_texture_resource,0,0));

    checkCudaErrors(cudaGraphicsMapResources(1, &pressurePong_texture_resource));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&pressurePongCudaArray, pressurePong_texture_resource,0,0));

    checkCudaErrors(cudaGraphicsMapResources(1, &divergence_texture_resource));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&divergenceCudaArray, divergence_texture_resource,0,0));

    checkCudaErrors(cudaGraphicsMapResources(1, &obstacles_texture_resource));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&obstaclesCudaArray, obstacles_texture_resource,0,0));

    pressurePingResDesc.res.array.array = pressurePingCudaArray;
    pressurePongResDesc.res.array.array = pressurePongCudaArray;
    divergenceResDesc.res.array.array = divergenceCudaArray;
    obstaclesResDesc.res.array.array = obstaclesCudaArray;

    checkCudaErrors(cudaCreateSurfaceObject(&pressurePingSurf,&pressurePingResDesc));
    checkCudaErrors(cudaCreateTextureObject(&divergenceTex,  &divergenceResDesc, &texDesc,NULL));
    checkCudaErrors(cudaCreateTextureObject(&obstaclesTex,   &obstaclesResDesc, &texDesc,NULL));

    // Create the surface objects

    checkCudaErrors(cudaCreateSurfaceObject(&pressurePongSurf, &pressurePongResDesc));

    //surface object creation and binding might only be required once (you don't need it when using cudaCreateSurfaceObject)

    //checkCudaErrors(cudaBindSurfaceToArray(&pressurePongSurf, pressurePongCudaArray, &pressurePingFormat));

    //do the CUDA implementation of the Jacobi
    //try without sending the d_buffer first
    float alpha = -CellSize * CellSize;

    Jacobi_CUDA(pressurePingSurf, pressurePongSurf, divergenceTex, obstaclesTex, pressurePingDims, alpha, inverseBeta, NumJacobiIterations, currentLoop); //they all have the same size

    //you may need to free the texture objects or something

    //checkCudaErrors(cudaDeviceSynchronize());
    //checkCudaErrors(cudaGetLastError());

    //printf("\n\n\n\n");

    checkCudaErrors(cudaGraphicsUnmapResources(1, &pressurePing_texture_resource, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &pressurePong_texture_resource, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &divergence_texture_resource, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &obstacles_texture_resource, 0));

    //still need to call this to draw the stuff and swap the framebuffer
    //the glUseProgram is commented out

    //everything except the call to ResetState() is commented out
    //Jacobi_no_program(pressurePing, divergence, obstacles, pressurePong);

    /*
 * struct SurfacePod {
    GLuint FboHandle;
    GLuint ColorTexture;
    GLsizei Width;
    GLsizei Height;
    GLsizei Depth;
};*/
}

void SwapCudaArrays(cudaArray* &swap1, cudaArray* &swap2){

    cudaArray* swap;

    swap = swap1;
    swap1 = swap2;
    swap2 = swap;
}

void JacobiProgramLoop(){

    for (int i = 0; i < NumJacobiIterations; ++i) {
        Jacobi(Slabs.Pressure.Ping, Surfaces.Divergence, Surfaces.Obstacles, Slabs.Pressure.Pong);
        SwapSurfaces(&Slabs.Pressure);
    }
}

void PezUpdate(float seconds)
{
    static int numLoop = 0;

    setbuf(stdout, NULL);

    pezCheck(OpenGLError);
    PezConfig cfg = PezGetConfig();

    //seconds = 0.0001;

    //float dt = seconds * 0.0001f;
    static auto lastTime = std::chrono::steady_clock::now();
    static auto beginning = lastTime;
    //auto currentTime = lastTime;
    static int currentLoop = 0;
    static float dt = 0.025f;
    static float ct = 0;

    Vector3 up(1, 0, 0); Point3 target(0);
    Matrices.View = Matrix4::lookAt(EyePosition, target, up);
    Matrix4 modelMatrix = Matrix4::identity();
    modelMatrix *= Matrix4::rotationX(ThetaX);
    modelMatrix *= Matrix4::rotationY(ThetaY);
    Matrices.Modelview = Matrices.View * modelMatrix;
    Matrices.Projection = Matrix4::perspective(
                FieldOfView,
                float(cfg.Width) / cfg.Height, // Aspect Ratio
                0.001f,   // Near Plane
                1.0f);  // Far Plane
    Matrices.ModelviewProjection = Matrices.Projection * Matrices.Modelview;

    //printf("\n\ndet(Matrices.View) = %f\n\n", determinant(Matrices.View));
    //printf("\n\ndet(Matrices.Projection) = %f\n\n", determinant(Matrices.Projection));

    //XYImpulse position is a Vector4 with z=0 and w=1 always, x and y are changed with LCtrl + mouse move
    //if(determinant(Matrices.ModelviewProjection) != 0.0f)
    //Matrix4 aMatrix = inverse(Matrices.ModelviewProjection);

    /*printf("\n");
  for(int i=0;i<4;i++){
      for(int j=0;j<4;j++)
          printf("Matrices.ModelviewProjection[%d,%d] = %f;  ", i,j,Matrices.ModelviewProjection.getElem(i,j));
      printf("\n");}
  printf("\n");

  for(int k=0;k<4;k++)
      printf("XYImpulsePosition[%d] = %f;  ", k,XYImpulsePosition[k]);
  */

    //printf("\n\nImpulsePosition4.getX() = %f, ImpulsePosition4.getY() = %f, ImpulsePosition4.getZ() = %f, ImpulsePosition4.getW() = %f\n\n", ImpulsePosition4.getX(), ImpulsePosition4.getY(), ImpulsePosition4.getZ(), ImpulsePosition4.getW());
    //printf("\n\nImpulsePosition4.getX() = %f, ImpulsePosition4.getY() = %f, ImpulsePosition4.getZ() = %f, ImpulsePosition4.getW() = %f\n\n", ImpulsePosition4.getX(), ImpulsePosition4.getY(), ImpulsePosition4.getZ(), ImpulsePosition4.getW());
    ImpulsePosition += (inverse(Matrices.ModelviewProjection) * (XYImpulsePosition-XYImpulsePositionLast)).getXYZ();
    XYImpulsePositionLast = XYImpulsePosition;
    //XYImpulsePosition = Vector4(0,0,0,0);

    auto currentTime = std::chrono::steady_clock::now();
    if(dt == -1.0f)
        dt = ((std::chrono::duration<float>)(currentTime - lastTime)).count();
    lastTime = currentTime;

    ct = 1.0f + ((std::chrono::duration<float>)(currentTime - beginning)).count();

    if(useJacobi){
        dt = 0.0025f;
        TimeStep = dt;}
    else{
        TimeStep = dt;
    }

    //if(numLoop == 1)
    //    TimeStep = dt*1000.0f;
    //else
    //    TimeStep = dt*4.0f;
    //TimeStep = dt*1000;

#if USE_CNN_CUDA_ADVECTION == 0

    if (SimulateFluid) {
        glBindVertexArray(Vaos.FullscreenQuad);
        glViewport(0, 0, GridWidth, GridHeight);

        //if(useStaggered)
        //    AdvectVelocity(Slabs.Velocity.Ping, Slabs.Velocity.Ping, Surfaces.Obstacles, Slabs.Velocity.Pong, VelocityDissipation);
        //else
            Advect(Slabs.Velocity.Ping, Slabs.Velocity.Ping, Surfaces.Obstacles, Surfaces.ObstacleSpeeds, Slabs.Velocity.Pong, VelocityDissipation);

        //printf("no - %d\n",int(glGetError()));

        Advect(Slabs.Velocity.Ping, Slabs.Temperature.Ping, Surfaces.Obstacles, Surfaces.ObstacleSpeeds, Slabs.Temperature.Pong, TemperatureDissipation);
        SwapSurfaces(&Slabs.Temperature);
        Advect(Slabs.Velocity.Ping, Slabs.Density.Ping, Surfaces.Obstacles, Surfaces.ObstacleSpeeds, Slabs.Density.Pong, DensityDissipation);
        SwapSurfaces(&Slabs.Density);
        SwapSurfaces(&Slabs.Velocity);
        ApplyBuoyancy(Slabs.Velocity.Ping, Slabs.Temperature.Ping, Slabs.Density.Ping, Slabs.Velocity.Pong);
        SwapSurfaces(&Slabs.Velocity);
        //if we apply this function, we won't need to flip the velocity use (Ping or Pong) in CNNProjectionCudaWrapper, because SwapSurfaces is called four times on velocity
        //might make more sense to move this after impulse application
        //if(numLoop>=1){
        ApplyVorticityConfinement(Slabs.Velocity.Ping, Slabs.Velocity.Pong);
        SwapSurfaces(&Slabs.Velocity);//}

        //velocityPing is new now
        //if(!useStaggered)
        //  ApplyImpulse(Slabs.Velocity.Ping, ImpulsePosition, ImpulseTemperature); //don't need to swap after this
        ApplyImpulse(Slabs.Temperature.Ping, Surfaces.Obstacles, ImpulsePosition, ImpulseTemperature, JitterTemperature, ct);
        ApplyImpulse(Slabs.Density.Ping, Surfaces.Obstacles, ImpulsePosition, ImpulseDensity, JitterDensity, ct);

        //update obstacles here
        //if(numLoop == 0)
            UpdateObstacles(Surfaces.Obstacles, ((std::chrono::duration<float>)(currentTime - beginning)).count()); //the speed is just a multiplier
        UpdateObstacleSpeeds(Surfaces.ObstacleSpeeds, Surfaces.Obstacles, ((std::chrono::duration<float>)(currentTime - beginning)).count());
        //obstacles will change how the divergence calculation works

        ApplyObstacleSpeeds(Slabs.Velocity.Ping, Surfaces.Obstacles, Surfaces.ObstacleSpeeds, Slabs.Velocity.Pong); //->this seems to destroy everything
        SwapSurfaces(&Slabs.Velocity);

        //CopyVelocity(Slabs.Velocity.Ping, Slabs.Velocity.Pong); //remove this and apply ping-ponging in CNNProjectionCudaWrapper if things start working
        //SwapSurfaces(&Slabs.Velocity);

        //probably need to change this
//#if USE_CNN == 0

//#endif
        //DON'T USE THIS
        //ComputeDivergenceCNN(Slabs.Velocity.Ping, Surfaces.Obstacles, Surfaces.Divergence); //this just needs halfinversewhatever to have the correct value
        //SwapSurfaces(&Slabs.Velocity);
        //ClearSurface(Slabs.Pressure.Ping, 0);

        if(useJacobi){
            ComputeDivergence(Slabs.Velocity.Ping, Surfaces.Obstacles, Surfaces.Divergence); //might be useful in the CNN version to avoid nans
            //ClearSurface(Slabs.Pressure.Ping, 0);
            for (int i = 0; i < NumJacobiIterations; ++i) {
                Jacobi(Slabs.Pressure.Ping, Surfaces.Divergence, Surfaces.Obstacles, Slabs.Pressure.Pong);
                SwapSurfaces(&Slabs.Pressure);
            }

            //if(k%2)
            //Jacobi_CUDA_wrapper(Slabs.Pressure.Pong, Surfaces.Divergence, Surfaces.Obstacles, Slabs.Pressure.Ping);
            //else
            //Jacobi_CUDA_wrapper(Slabs.Pressure.Ping, Surfaces.Divergence, Surfaces.Obstacles, Slabs.Pressure.Pong, currentLoop);
            //JacobiProgramLoop();
            //k++;

            //SwapCudaArrays(pressurePingCudaArray, pressurePongCudaArray);
            SubtractGradient(Slabs.Velocity.Ping, Slabs.Pressure.Ping, Surfaces.Obstacles, Surfaces.ObstacleSpeeds, Slabs.Velocity.Pong);
            SwapSurfaces(&Slabs.Velocity);
        }
        else{
#if USE_CNN==0
            ToyCudaWrapper(Slabs.Pressure.Ping, Surfaces.Divergence, Surfaces.Obstacles, Slabs.Pressure.Pong);
            SubtractGradient(Slabs.Velocity.Ping, Slabs.Pressure.Ping, Surfaces.Obstacles, Slabs.Velocity.Pong);
#else
            //velocityPing is still the new one here
            CNNProjectionCudaWrapper(Slabs.Pressure.Ping, Surfaces.Divergence, Surfaces.Obstacles, Slabs.Pressure.Pong, Slabs.Velocity.Ping, Slabs.Velocity.Pong);
            //addObstaclesToDensityTexture(Surfaces.Obstacles, Slabs.Density.Pong); //add obstacle voxels to density texture, which will be removed at the very beginning of the next loop
            //SwapSurfaces(&Slabs.Pressure);
            //SubtractGradient(Slabs.Velocity.Ping, Slabs.Pressure.Ping, Surfaces.Obstacles, Slabs.Velocity.Pong);//, defaultNet->velSTD);
#endif
        }

        ApplyObstacleSpeeds(Slabs.Velocity.Ping, Surfaces.Obstacles, Surfaces.ObstacleSpeeds, Slabs.Velocity.Pong);
        SwapSurfaces(&Slabs.Velocity);
        CopyVelocity(Slabs.Velocity.Ping, Slabs.Velocity.Pong); //remove this and apply ping-ponging in CNNProjectionCudaWrapper if things start working
        SwapSurfaces(&Slabs.Velocity);
        //SwapSurfaces(&Slabs.Velocity);
        //SimulateFluid = !SimulateFluid;
        currentLoop++;
    }
#else
    CudaSimulationLoop(); //needs current loop?
#endif

    //k = k%2;


    //infoMsg("%d\n",int(glGetError()));
    auto someError = glGetError();
    //printf("gluErrorString(glGetError()) = %s (%d)\n", gluErrorString(someError), someError);
    pezCheck(CheckOpenGLError(someError)); //<---ERROR HERE

    dt = -1.0f;

    //SwapSurfaces(&Slabs.Velocity);

    numLoop++;
}
/*
void PezHandleMouse(int x, int y, int action)
{
  static bool MouseDown = false;
  static int StartX, StartY;
  static const float Speed = 0.005f;
  if (action == PEZ_DOWN) {
      StartX = x;
      StartY = y;
      MouseDown = true;
    } else if (MouseDown && action == PEZ_MOVE) {
      ThetaX = DefaultThetaX + Speed * (x - StartX);
      ThetaY = DefaultThetaY + Speed * (y - StartY);
    } else if (action == PEZ_UP) {
      MouseDown = false;
    }
}
*/
void PezHandleMouse(int x, int y, int action, bool controlDown)
{
    static bool MouseDown = false;
    static int StartX, StartY, StartXCtrl = x, StartYCtrl = y;
    static const float Speed = 0.05f;
    static bool controlWasDown = controlDown;

    if(!controlWasDown && controlDown){
        XYImpulsePosition = Matrices.ModelviewProjection * Vector4(ImpulsePosition, 0);
        XYImpulsePositionLast = XYImpulsePosition;
        controlWasDown = controlDown;
        StartXCtrl = x;
        StartYCtrl = y;
        return;
    }

    if(controlDown){

        if (action == PEZ_MOVE) {
            //printf("x = %d; StartXCtrl = ; y = ; StartYCtrl = \n");
            XYImpulsePosition.setX(XYImpulsePosition.getX() + (x - StartXCtrl) * Speed);// * 1.0 / PezGetConfig().Width);//1.0 / PezGetConfig().Width * (StartXCtrl - x));
            XYImpulsePosition.setY(XYImpulsePosition.getY() - (y - StartYCtrl) * Speed);// * 1.0/ PezGetConfig().Height);//1.0 / PezGetConfig().Height * (StartYCtrl - y));
            StartXCtrl = x;
            StartYCtrl = y;

            //StartXCtrl = XYImpulsePosition.getX() * PezGetConfig().Width - x;
            //StartYCtrl = XYImpulsePosition.getY() * PezGetConfig().Height - y;
            //StartYCtrl = x;
            //StartXCtrl = y;
            //StartXCtrl = Speed * (x - StartXCtrl);
            //StartYCtrl = Speed * (y - StartYCtrl);
        }
        //convert to 3D coordinates based on camera position
    }
    else{
        if (action == PEZ_DOWN) {
            StartX = x;
            StartY = y;
            MouseDown = true;
        } else if (MouseDown && action == PEZ_MOVE) {
            ThetaX = DefaultThetaX + Speed * (x - StartX);
            ThetaY = DefaultThetaY + Speed * (y - StartY);
        } else if (action == PEZ_UP) {
            MouseDown = false;
        }
        //(Matrices.ModelviewProjection);
    }

    //StartXCtrl = x;
    //StartYCtrl = y;

    controlWasDown = controlDown;
}

void PezHandleKey(char c)
{
    if (c == ' ') {
        SimulateFluid = !SimulateFluid;
    }
}
