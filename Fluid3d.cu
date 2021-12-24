  //Modified from the Jacobi program in Fluid.glsl, in the fluidsim project by Philip Rideout

#ifndef _FLUID_3D_CU_
#define _FLUID_3D_CU_

#include <cuda_runtime_api.h>
#include <math.h>
#include <cuda_fp16.h>
#include <helper_cuda.h>
#include <cudnn.h>
#include <cublas.h>

//we'll have blocks = (16,16,16), threads = (8,8,8), leading to (128,128,128) total (even though we only need (64,64,64) with the original configuration
#define NUM_BLOCKS_X 17
#define NUM_BLOCKS_Y 17
#define NUM_BLOCKS_Z 17

#define THREADSPERBLOCK_X 8 //this should probably be changed; 32 for innermost
#define THREADSPERBLOCK_Y 8
#define THREADSPERBLOCK_Z 8

//consider the cuda_helper thing

template<typename T>
__global__ void setBufferValues(T* velocity, T* pressure, dim3 velocitySize, dim3 domainSize){

    int num = blockIdx.x*blockDim.x+threadIdx.x;

    int x = num % velocitySize.x;
    int y =(num / velocitySize.x) % velocitySize.y;
    int z = num /(velocitySize.x  * velocitySize.y);

    if(x >= velocitySize.x || y>=velocitySize.y || z>=velocitySize.z)
        return;

    T velocityValX = sqrt((x+1) * 1.1) + sqrt(2.3*(y+1)) + sqrt(1.7 * (z+1)) + sqrt(9.7);
    T velocityValY = sqrt((x+1) * 1.1) + sqrt(2.3*(y+1)) + sqrt(1.7 * (z+1)) + sqrt(9.7*2);
    T velocityValZ = sqrt((x+1) * 1.1) + sqrt(2.3*(y+1)) + sqrt(1.7 * (z+1)) + sqrt(9.7*3);

    //TODO: this squashes double values; will probably need template specialization if double is ever used. But this function will probably be useless by then.
    velocity[num*3  ] = fmodf(velocityValX,1.0);
    velocity[num*3+1] = fmodf(velocityValY,1.0);
    velocity[num*3+2] = fmodf(velocityValZ,1.0);

    if(x >= domainSize.x || y>=domainSize.y || z>=domainSize.z)
        return;

    T pressureVal = sqrt(3.1*(x+1)*(x+1) + (y+1)*(y+1) + 2.5*(z+1)*(z+1));

    pressure[x + domainSize.x*y + domainSize.x* domainSize.y*z] = fmodf(pressureVal,1.0);
}

extern "C" cudaError_t setBufferValuesWrapper_float(float* velocity, float* pressure, dim3 velocitySize, dim3 domainSize){

    int velocitySizeTotal = velocitySize.x * velocitySize.y * velocitySize.z;

    setBufferValues<float><<<velocitySizeTotal/1024+(velocitySizeTotal%1024!=0), 1024>>>(velocity, pressure, velocitySize, domainSize);
    return cudaPeekAtLastError();
}

template<typename T>
__global__ void setBuffer3DToScalar(T* dest, T valX, T valY, T valZ, dim3 size){

    int num = blockIdx.x*blockDim.x+threadIdx.x;

    int x = num % size.x;
    int y =(num / size.x) % size.y;
    int z = num /(size.x  * size.y);

    if(x >= size.x || y >= size.y || z >= size.z)
        return;

    dest[num*3] = valX;
    dest[num*3+1] = valY;
    dest[num*3+2] = valZ;

}

extern "C" cudaError_t setBuffer3DToScalar_float(float* dest, float valX, float valY, float valZ, dim3 size){

    int sizeTotal = size.x * size.y * size.z;

    setBuffer3DToScalar<float><<<sizeTotal/1024+(sizeTotal%1024!=0), 1024>>>(dest, valX, valY, valZ, size);

    return cudaPeekAtLastError();
}

template<typename T>
__global__ void copyTextureCuda(cudaTextureObject_t src, cudaSurfaceObject_t dest, dim3 size){

    int num = blockIdx.x*blockDim.x+threadIdx.x;

    int x = num % size.x;
    int y =(num / size.x) % size.y;
    int z = num /(size.x  * size.y);

    if(x >= size.x || y >= size.y || z >= size.z)
        return;

    surf3Dwrite<T>(tex3D<T>(src, x, y, z), dest, x*sizeof(T), y, z);
}

extern "C" cudaError_t copyTextureCudaWrapper_float(cudaTextureObject_t src, cudaSurfaceObject_t dest, dim3 size){

    int domainSizeTotal = size.x * size.y * size.z;
    copyTextureCuda<float><<<domainSizeTotal/1024+(domainSizeTotal%1024!=0), 1024>>>(src, dest, size);
    return cudaPeekAtLastError();
}

template<typename T>
__global__ void printBuffer(T* buffer, dim3 size, char* name){

    for(int z=0; z<size.z; z++)
        for(int y=0; y<size.y; y++)
            for(int x=0; x<size.x; x++)
                if(buffer[x + size.x * y + size.x * size.y * z] !=0)
                    printf("%s[%d,%d,%d] = %f\n", name, x, y, z, buffer[x + size.x * y + size.x * size.y * z]);
}

template<typename T>
__global__ void printBuffer3D(T* buffer, dim3 size, char* name){

    int num;

    for(int z=0; z<size.z; z++)
        for(int y=0; y<size.y; y++)
            for(int x=0; x<size.x; x++)
                if(buffer[num*3] !=0 || buffer[num*3+1] !=0 || buffer[num*3+2] !=0)
                    printf("%s[%d,%d,%d] = (%f,%f,%f)\n", name, x, y, z,
                           buffer[num*3],
                           buffer[num*3+1],
                           buffer[num*3+2]);
}

template<typename T>
__global__ void printBufferSideBySide3D(T* buffer1, T* buffer2, dim3 size, char* name1, char* name2){

    //this will also print any nan values, regardless of whether both buffer values are nan
    int num;

    for(int z=0; z<size.z; z++)
        for(int y=0; y<size.y; y++)
            for(int x=0; x<size.x; x++){
                num = x + size.x * y + size.x * size.y * z;
                if(isnan(buffer1[num*3]) || isnan(buffer1[num*3+1]) || isnan(buffer1[num*3+2]) ||
                   isnan(buffer2[num*3]) || isnan(buffer2[num*3+1]) || isnan(buffer2[num*3+2]))
                    printf("%s[%d,%d,%d] = (%f,%f,%f); %s[%d,%d,%d] = (%f,%f,%f) ---------------------------------------\n",
                           name1, x, y, z,
                           buffer1[num*3],
                           buffer1[num*3+1],
                           buffer1[num*3+2],
                           name2, x, y, z,
                           buffer2[num*3],
                           buffer2[num*3+1],
                           buffer2[num*3+2]);}
}

template<typename T>
__global__ void printTexture(cudaTextureObject_t aTexture, dim3 size, char* name){

    T val;

    for(int z=0; z<size.z; z++)
        for(int y=0; y<size.y; y++)
            for(int x=0; x<size.x; x++){
                val = tex3D<T>(aTexture, x, y, z);
                if(val !=0)
                    printf("%s[%d,%d,%d] = %f\n", name, x, y, z, val);
            }
}

extern "C" cudaError_t printCudaBuffersWrapper_float(float* buffer, dim3 size, char* name){

    return (cudaError_t)(0);
    int i;
    for(i=0;name[i]!=NULL;i++);
    char* nameCuda;
    printf("i = %d\n", i);
    checkCudaErrors(cudaMalloc(&nameCuda, i*sizeof(char)));
    checkCudaErrors(cudaMemcpy(nameCuda, name, i*sizeof(char), cudaMemcpyHostToDevice));
    printBuffer<float><<<1,1>>>(buffer, size, nameCuda);
    checkCudaErrors(cudaDeviceSynchronize());
    return cudaPeekAtLastError();
}

extern "C" cudaError_t printCudaBuffersWrapper3D_float(float* buffer, dim3 size, char* name){

    int i;
    for(i=0;name[i]!=NULL;i++);
    char* nameCuda;
    printf("i = %d\n", i);
    checkCudaErrors(cudaMalloc(&nameCuda, i*sizeof(char)));
    checkCudaErrors(cudaMemcpy(nameCuda, name, i*sizeof(char), cudaMemcpyHostToDevice));
    printBuffer3D<float><<<1,1>>>(buffer, size, nameCuda);
    checkCudaErrors(cudaDeviceSynchronize());
    return cudaPeekAtLastError();
}

extern "C" cudaError_t printCudaBuffersSideBySideWrapper3D_float(float* buffer1, float* buffer2, dim3 size, char* name1, char* name2){

    int i1;
    for(i1=0;name1[i1]!=NULL;i1++);
    int i2;
    for(i2=0;name2[i2]!=NULL;i2++);
    char *name1Cuda, *name2Cuda;
    printf("i1 = %d; i2 = %d\n", i1, i2);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMalloc(&name1Cuda, i1*sizeof(char)));
    checkCudaErrors(cudaMemcpy(name1Cuda, name1, i1*sizeof(char), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc(&name2Cuda, i2*sizeof(char)));
    checkCudaErrors(cudaMemcpy(name2Cuda, name2, i2*sizeof(char), cudaMemcpyHostToDevice));
    printBufferSideBySide3D<float><<<1,1>>>(buffer1, buffer2, size, name1Cuda, name2Cuda);
    checkCudaErrors(cudaDeviceSynchronize());
    return cudaPeekAtLastError();
}

extern "C" cudaError_t printTextureWrapper_float(cudaTextureObject_t aTexture, dim3 size, char* name){

    int i;
    for(i=0;name[i]!=NULL;i++);
    char* nameCuda;
    printf("i = %d\n", i);
    checkCudaErrors(cudaMalloc(&nameCuda, i*sizeof(char)));
    checkCudaErrors(cudaMemcpy(nameCuda, name, i*sizeof(char), cudaMemcpyHostToDevice));
    printTexture<float><<<1,1>>>(aTexture, size, nameCuda);
    checkCudaErrors(cudaDeviceSynchronize());
    return cudaPeekAtLastError();
}

#define getObstacles(dimX, dimY, dimZ) obstacles[(dimX) + domainSize.x * (dimY) + domainSize.x * domainSize.y * (dimZ)]
#define  getPressure(dimX, dimY, dimZ)  pressure[(dimX) + domainSize.x * (dimY) + domainSize.x * domainSize.y * (dimZ)]
template<typename T>
__global__ void JacobiCudaBuffers(T* pressure, T* divergence, T* obstacles, dim3 domainSize, T alpha, T inverseBeta){

    int num = blockIdx.x*blockDim.x+threadIdx.x;

    int x = num % domainSize.x;
    int y =(num / domainSize.x) % domainSize.y;
    int z = num /(domainSize.x  * domainSize.y);

    T oN, oS, oE, oW, oU, oD, pN, pS, pE, pW, pU, pD, pC;

    //assume value of obstacle texture is 0 when out of bounds in the OpenGL shader versions
    if(y<domainSize.y)
        oN = getObstacles(  x,y+1,  z);
    else
        oN = 0;

    if(y>0)
        oS = getObstacles(  x,y-1,  z);
    else
        oS = 0;

    if(x<domainSize.x)
        oE = getObstacles(x+1,  y,  z);
    else
        oE = 0;

    if(x>0)
        oW = getObstacles(x-1,  y,  z);
    else
        oW = 0;

    if(z<domainSize.z)
        oU = getObstacles(  x,  y,z+1);
    else
        oU = 0;

    if(z>0)
        oD = getObstacles(  x,  y,z-1);
    else
        oD = 0;

    pC = pressure[num];

    if (oN > 0) pN = pC; else if(y==domainSize.y-1) pN = 0; else pN = getPressure(  x,y+1,  z);
    if (oS > 0) pS = pC; else if(y==0) pS = 0; else pS = getPressure(  x,y-1,  z);
    if (oE > 0) pE = pC; else if(x==domainSize.x-1) pE = 0; else pE = getPressure(x+1,  y,  z);
    if (oW > 0) pW = pC; else if(x==0) pW = 0; else pW = getPressure(x-1,  y,  z);
    if (oU > 0) pU = pC; else if(z==domainSize.z-1) pU = 0; else pU = getPressure(  x,  y,z+1);
    if (oD > 0) pD = pC; else if(z==0) pD = 0; else pD = getPressure(  x,  y,z-1);

    float bC = divergence[num];

    pressure[num] = (pW + pE + pS + pN + pU + pD + alpha * bC) * inverseBeta;
}
#undef getObstacles
#undef getPressure

extern "C" cudaError_t
JacobiCudaBuffers_float(float* pressure, float* divergence, float* obstacles, dim3 domainSize, float alpha, float inverseBeta, int numLoops)
{

    int domainSizeTotal = domainSize.x * domainSize.y * domainSize.z;
    for(int i=0;i<numLoops;i++)
        JacobiCudaBuffers<float><<<domainSizeTotal/1024+(domainSizeTotal%1024!=0), 1024>>>(pressure, divergence, obstacles, domainSize, alpha, inverseBeta);

    return cudaPeekAtLastError();
}

__global__ void makePressure(cudaSurfaceObject_t pressurePingSurf,
                             cudaSurfaceObject_t pressurePongSurf,
                             cudaTextureObject_t divergenceTex,
                             cudaTextureObject_t obstaclesTex,
                             dim3 textureDims,
                             float alpha, float inverseBeta){

  //  Potentially useful note: Texture references are read-only. If you write to a CUDA array using a surface reference,
  //  that memory traffic goes through the L1/L2 cache hierarchy and is not coherent with the texture cache.

  int x = blockIdx.x*blockDim.x+threadIdx.x;
  int y = blockIdx.y*blockDim.y+threadIdx.y;
  int z = blockIdx.z*blockDim.z+threadIdx.z;

  if(x >= textureDims.x || y >= textureDims.y || z >= textureDims.z)
    return;

  float pC;
  surf3Dread(&pC, pressurePingSurf, x   *sizeof(float), y  , z  );

  float oN = tex3D<float>(obstaclesTex,x,y+1,z);
  float oS = tex3D<float>(obstaclesTex,x,y-1,z);
  float oE = tex3D<float>(obstaclesTex,x+1,y,z);
  float oW = tex3D<float>(obstaclesTex,x-1,y,z);
  float oU = tex3D<float>(obstaclesTex,x,y,z+1);
  float oD = tex3D<float>(obstaclesTex,x,y,z-1);

  float pN, pS, pE, pW, pU, pD;

  //since all the edges are obstacles, this should work
  if (oN > 0) pN = pC; else if(y==textureDims.y-1) pN = 0; else surf3Dread(&pN, pressurePingSurf, x   *sizeof(float), y+1, z  );
  if (oS > 0) pS = pC; else if(y==0) pS = 0; else surf3Dread(&pS, pressurePingSurf, x   *sizeof(float), y-1, z  );
  if (oE > 0) pE = pC; else if(x==textureDims.x-1) pE = 0; else surf3Dread(&pE, pressurePingSurf,(x+1)*sizeof(float), y  , z  );
  if (oW > 0) pW = pC; else if(x==0) pW = 0; else surf3Dread(&pW, pressurePingSurf,(x-1)*sizeof(float), y  , z  );
  if (oU > 0) pU = pC; else if(z==textureDims.z-1) pU = 0; else surf3Dread(&pU, pressurePingSurf, x   *sizeof(float), y  , z+1);
  if (oD > 0) pD = pC; else if(z==0) pD = 0; else surf3Dread(&pD, pressurePingSurf, x   *sizeof(float), y  , z-1);

  //bC is the divergence at this point
  float bC = tex3D<float>(divergenceTex, x,y,z);
  float oC = tex3D<float>(obstaclesTex,x,y,z);

  float temp = (pW + pE + pS + pN + pU + pD + alpha * bC) * inverseBeta;

  surf3Dwrite(temp, pressurePongSurf,x*sizeof(float),y,z);
}

//TODO: may need to change cudaTextureObject_t back to texture<float,1,cudaReadModeElementType>

extern "C" cudaError_t
Jacobi_CUDA(cudaSurfaceObject_t pressurePingSurf,
            cudaSurfaceObject_t pressurePongSurf,
            cudaTextureObject_t divergenceTex,
            cudaTextureObject_t obstaclesTex,
            dim3 textureDims,
            float alpha, float inverseBeta, int numLoops, int currentLoop){

  dim3 numBlocks = dim3(NUM_BLOCKS_X,NUM_BLOCKS_Y,NUM_BLOCKS_Z);
  dim3 threadsPerBlock = dim3(THREADSPERBLOCK_X,THREADSPERBLOCK_Y,THREADSPERBLOCK_Z);

  int i;

  cudaResourceDesc pPingResDesc, pPongResDesc;

  cudaGetSurfaceObjectResourceDesc(&pPingResDesc, pressurePingSurf);
  cudaGetSurfaceObjectResourceDesc(&pPongResDesc, pressurePongSurf);

  //TODO: numLoops is even, so we can disconsider the swap FOR NOW
  for(i=0;i<numLoops;i++){
    if(i%2==0)
      makePressure<<<numBlocks, threadsPerBlock>>>(pressurePingSurf, pressurePongSurf,
                                                   divergenceTex, obstaclesTex,
                                                   textureDims, alpha, inverseBeta);
    else
      makePressure<<<numBlocks, threadsPerBlock>>>(pressurePongSurf, pressurePingSurf,
                                                   divergenceTex, obstaclesTex,
                                                   textureDims, alpha, inverseBeta);

    //TODO: might need this:
    //cudaDeviceSynchronize();
    }

  return cudaPeekAtLastError();
}

template<typename T>
__global__ void updateObstacles(T* obstacles, T* velocityIn, dim3 obstacleExtent){

    //does nothing, can update for whatever time-dependent obstacles we need
    //this will require changes to setWallBcsStaggered
    //if Unity is used, need to get movement from it
    //not sure there's an easy way to do rapid voxelization using Unity objects
    //could try sending the global positions of all the vertices, alaos the faces and edges, and doing it in CUDA somehow
    //in the first phase, could just use a simple-geometry object that can be voxelized trivially
    //this should also update the voxel speed, wherever they're contained
}

extern "C" cudaError_t updateObstaclesWrapper_float(float* obstacles, float* velocityIn, dim3 obstacleExtent){

    int obstacleExtentTotal = obstacleExtent.x * obstacleExtent.y * obstacleExtent.z;

    updateObstacles<float><<<obstacleExtentTotal/1024+(obstacleExtentTotal%1024!=0),1024>>>(obstacles, velocityIn, obstacleExtent);
    return cudaPeekAtLastError();
}

extern "C" cudaError_t updateObstaclesWrapper_double(double* obstacles, double* velocityIn, dim3 obstacleExtent){

    int obstacleExtentTotal = obstacleExtent.x * obstacleExtent.y * obstacleExtent.z;

    updateObstacles<double><<<obstacleExtentTotal/1024+(obstacleExtentTotal%1024!=0),1024>>>(obstacles, velocityIn, obstacleExtent);
    return cudaPeekAtLastError();
}

extern "C" cudaError_t updateObstaclesWrapper_half(dim3 obstacleExtent){

    printf("Half support not yet implemented (%s:%d)\n", __FILE__, __LINE__);
    exit(1);

    return cudaPeekAtLastError();
}

#define min(a,b) ((a>b)?(b):(a))
#define max(a,b) ((a<b)?(b):(a))
template<typename T>
__global__ void clampVelocity(T* velocity, dim3 velocitySize, T clampMin, T clampMax){

    int num = blockIdx.x*blockDim.x+threadIdx.x;

    velocity[num*3] = max(min(clampMax, velocity[num*3]),clampMin);
    velocity[num*3+1] = max(min(clampMax, velocity[num*3+1]),clampMin);
    velocity[num*3+2] = max(min(clampMax, velocity[num*3+2]),clampMin);
}


extern "C" cudaError_t clampVelocityWrapper_float(float* velocity, dim3 velocitySize, float clampMin, float clampMax){

    int velocitySizeTotal = velocitySize.x*velocitySize.y*velocitySize.z;

    clampVelocity<float><<<velocitySizeTotal/1024+(velocitySizeTotal%1024!=0), 1024>>>(velocity, velocitySize, clampMin, clampMax);

    return cudaPeekAtLastError();
}

extern "C" cudaError_t advectVelocityCudaNotStaggeredWrapper_float(float* velocityIn, float* velocityOut, float* obstacles, float cellDim, dim3 domainSize, float dt);
extern "C" cudaError_t advectCudaNotStaggeredWrapper_float(float* velocityIn, float* scalarField, float* scalarFieldOut, float dt){ return cudaPeekAtLastError();}
extern "C" cudaError_t vorticityConfinement_float(float* velocityIn, float* velocityOut, dim3 velocitySize){return cudaPeekAtLastError();}
extern "C" cudaError_t setWallBcsStaggeredWrapper_float(float* velocityIn, dim3 domainSize){return cudaPeekAtLastError();}
extern "C" cudaError_t setWallBcsNotStaggeredWrapper_float(float* velocityIn, dim3 domainSize){return cudaPeekAtLastError();}

template<typename T>
__global__ void checkDivergence(T* divergence, dim3 domainSize){

    //this should be done with loop unrolling, but it's just for debugging so who cares
    float total = 0;

    for(int i=1;i<domainSize.x-1;i++)
        for(int j=1;j<domainSize.y-1;j++)
            for(int k=1;k<domainSize.z-1;k++){
                total += divergence[i + j*domainSize.y + k*domainSize.z*domainSize.y]*divergence[i + j*domainSize.y + k*domainSize.z*domainSize.y];
            }

    printf("\nSum of squared divergences = %f\n\n", total);
}

extern "C" cudaError_t checkDivergenceWrapper_float(float* divergence, dim3 domainSize){

    checkDivergence<float><<<1,1>>>(divergence, domainSize);
    return cudaPeekAtLastError();
}

//TODO: this needs addGravityStaggered and addGravityNotStaggered if using the obstacle mask when adding gravity
template<typename T>
__global__ void addGravity(T* velocity, dim3 velocitySize, T gravityX, T gravityY, T gravityZ, T dt){

    int num = blockIdx.x*blockDim.x+threadIdx.x;

    if(num >= velocitySize.x * velocitySize.y * velocitySize.z)
        return;

    //this is for addGravityStaggered, if we implement it
    /*
    int x = num %(velocitySize.y * domainSize.z);
    int y =(num % doSize.z)/ domainSize.x;
    int z = num /(domainSize.x);

    if(isObstacle())
        velocity[num*3]   += gravity[0];
    velocity[num*3+1] += gravity[1];
    velocity[num*3+2] += gravity[2];
    */

    velocity[num*3]   += gravityX * dt;
    velocity[num*3+1] += gravityY * dt;
    velocity[num*3+2] += gravityZ * dt;

}

extern "C" cudaError_t addGravityWrapper_float(float* velocity, dim3 velocitySize, float3 gravity, float dt){

    int velocitySizeTotal = velocitySize.x*velocitySize.y*velocitySize.z;
    addGravity<float><<<velocitySizeTotal/1024+(velocitySizeTotal%1024!=0), 1024>>>(velocity, velocitySize, gravity.x, gravity.y, gravity.z, dt);
    return cudaPeekAtLastError();
}

//this is the triliniear interpolation function
//unmasked textures don't need an interpolation function
#define gridGet(dimX,dimY,dimZ) grid[(dimX) + (dimY) * gridSize.x + (dimZ) * gridSize.y * gridSize.z]
template<typename T>
__device__ void trilerp(T* grid, float3 index, dim3 gridSize, T* result){

    int3 indexFloor;

    indexFloor.x = floor(index.x);
    indexFloor.y = floor(index.y);
    indexFloor.z = floor(index.z);

    float xd, yd, zd;

    xd = index.x - indexFloor.x;
    yd = index.y - indexFloor.y;
    zd = index.z - indexFloor.z;

    T c00 = gridGet(indexFloor.x,   indexFloor.y,   indexFloor.z  ) * (1.0f - xd)
          + gridGet(indexFloor.x+1, indexFloor.y,   indexFloor.z  ) * xd;

    T c01 = gridGet(indexFloor.x,   indexFloor.y+1, indexFloor.z  ) * (1.0f - xd)
          + gridGet(indexFloor.x+1, indexFloor.y+1, indexFloor.z  ) * xd;

    T c10 = gridGet(indexFloor.x,   indexFloor.y,   indexFloor.z+1) * (1.0f - xd)
          + gridGet(indexFloor.x+1, indexFloor.y,   indexFloor.z+1) * xd;

    T c11 = gridGet(indexFloor.x,   indexFloor.y+1, indexFloor.z+1) * (1.0f - xd)
          + gridGet(indexFloor.x+1, indexFloor.y+1, indexFloor.z+1) * xd;

    c00 = c00 * (1 - yd) + c10 * yd;
    c10 = c01 * (1 - yd) + c11 * yd;

    *result = c00 * (1 - zd) + c10 * zd;
}

#define isNotObstacle(dimX,dimY,dimZ) (inverseMask[(dimX)+gridSize.x*(dimY)+gridSize.x*gridSize.y*(dimZ)]==0)
//Note: the gridGet thing causes repeated multiplications that would otherwise be avoided
template<typename T>
__device__ void trilerpInverseMasked(T* grid, T* inverseMask, float3 index, dim3 gridSize, T* result){

    int3 indexFloor;

    indexFloor.x = floor(index.x);
    indexFloor.y = floor(index.y);
    indexFloor.z = floor(index.z);

    float xd, yd, zd;

    xd = index.x - indexFloor.x;
    yd = index.y - indexFloor.y;
    zd = index.z - indexFloor.z;

    //grid
    T c00 = gridGet(indexFloor.x,  indexFloor.y,  indexFloor.z  ) * isNotObstacle(indexFloor.x,  indexFloor.y,  indexFloor.z  ) * (1.0f - xd)
          + gridGet(indexFloor.x+1,indexFloor.y,  indexFloor.z  ) * isNotObstacle(indexFloor.x+1,indexFloor.y,  indexFloor.z  ) * xd;

    T c01 = gridGet(indexFloor.x,  indexFloor.y+1,indexFloor.z  ) * isNotObstacle(indexFloor.x,  indexFloor.y+1,indexFloor.z  ) * (1.0f - xd)
          + gridGet(indexFloor.x+1,indexFloor.y+1,indexFloor.z  ) * isNotObstacle(indexFloor.x+1,indexFloor.y+1,indexFloor.z  ) * xd;

    T c10 = gridGet(indexFloor.x,  indexFloor.y,  indexFloor.z+1) * isNotObstacle(indexFloor.x,  indexFloor.y,  indexFloor.z+1) * (1.0f - xd)
          + gridGet(indexFloor.x+1,indexFloor.y,  indexFloor.z+1) * isNotObstacle(indexFloor.x+1,indexFloor.y,  indexFloor.z+1) * xd;

    T c11 = gridGet(indexFloor.x,  indexFloor.y+1,indexFloor.z+1) * isNotObstacle(indexFloor.x,  indexFloor.y+1,indexFloor.z+1) * (1.0f - xd)
          + gridGet(indexFloor.x+1,indexFloor.y+1,indexFloor.z+1) * isNotObstacle(indexFloor.x+1,indexFloor.y+1,indexFloor.z+1) * xd;

    c00 = c00 * (1 - yd) + c10 * yd;
    c10 = c01 * (1 - yd) + c11 * yd;

    T c = c00 * (1 - zd) + c10 * zd;

    c00 = isNotObstacle(indexFloor.x,  indexFloor.y,  indexFloor.z  ) * (1.0f - xd)
          + isNotObstacle(indexFloor.x+1,indexFloor.y,  indexFloor.z  ) * xd;

    c01 = isNotObstacle(indexFloor.x,  indexFloor.y+1,indexFloor.z  ) * (1.0f - xd)
          + isNotObstacle(indexFloor.x,  indexFloor.y+1,indexFloor.z  ) * xd;

    c10 = isNotObstacle(indexFloor.x,  indexFloor.y,  indexFloor.z+1) * (1.0f - xd)
          + isNotObstacle(indexFloor.x,  indexFloor.y,  indexFloor.z+1) * xd;

    c10 = isNotObstacle(indexFloor.x,  indexFloor.y+1,indexFloor.z+1) * (1.0f - xd)
          + isNotObstacle(indexFloor.x,  indexFloor.y+1,indexFloor.z+1) * xd;

    c00 = c00 * (1 - yd) + c10 * yd;
    c10 = c01 * (1 - yd) + c11 * yd;

    T p = c00 * (1 - zd) + c10 * zd;

    if(p != 0)
        *result = c/p;
    else
        *result = NAN;
}
#undef isNotObstacle
#undef gridGet

extern "C" float getStandardDeviationFlatWrapper_float(float* tensor, size_t tensorSize);

#define isNotObstacle(dimX,dimY,dimZ) ((dimX>=0)&&(dimY>=0)&&(dimZ>=0)&&(dimX<domainSize.x)&&(dimY<domainSize.y)&&(dimZ<domainSize.z)&&(obstacles[(dimX)+domainSize.x*(dimY)+domainSize.x*domainSize.y*(dimZ)]==0))
template<typename T>
__global__ void addBuoyancyStaggered(T* velocity, cudaTextureObject_t density, T* temperature, T* obstacles, dim3 domainSize, float3 gravity, T ambTemp, T alpha, T beta, T dt){

    int num = blockIdx.x * blockDim.x + threadIdx.x;

    //bouyancy is added to the velocity, so need domainSize.n+1

    dim3 velocitySize;

    velocitySize.x = domainSize.x+1;
    velocitySize.y = domainSize.y+1;
    velocitySize.z = domainSize.z+1;

    int x = num % velocitySize.x;
    int y =(num / velocitySize.x) % velocitySize.y;
    int z = num /(velocitySize.x  * velocitySize.y);

    if(x >= velocitySize.x || y >= velocitySize.y || z >= velocitySize.z)// || obstacles[num]!=0) //REMOVE THE -1 FROM EACH
        return;

    //TODO (old, possibly outdated):
    //what's done here won't work unless obstacle velocity is applied to velocity grid (it is, but set to zero in torch configuration and current configuration)
    //if the edge density and temperature are not repeated for boundaries, the smoke might "curl" improperly at domain boundaries
    //clamp everything to ([0,domainSize.x-1],[0,domainSize.y-1],[0,domainSize.z-1])
    //might not need the weird indices if cudaAddressModeClamp is used in the textureDesc
    //temperature value needs to be calculated
    //density needs to be interpolated by hand so occupied cells can be ignored
    //do not take cell occupation into account until the obstacle updating mechanism is known
    //if using repeated boundary values, use (tex3D<T>(density, x-0.5*(x!=0)-0.5*(x==(domainSize.x)), y-(y==domainSize.y), z-(z==domainSize.z)) * alpha for density

    velocity[num*3]   += (tex3D<T>(density, x-0.5*(x!=0)-0.5*(x==(domainSize.x)), y-(y==domainSize.y), z-(z==domainSize.z)) * alpha //(tex3D<T>(density, x-0.5, y, z) * alpha //this will probably work, using density=0 outside the domain
                       -  ((y<domainSize.y && z<domainSize.z)?(((x!=0)?(temperature[x-1   + y*(velocitySize.x-1) + z*(velocitySize.y-1)*(velocitySize.x-1)]):ambTemp) + ((x<domainSize.x)?(temperature[x + y*(velocitySize.y-1) + z*(velocitySize.y-1)*(velocitySize.x-1)]):ambTemp)):(2*ambTemp)) * beta)
                       *  gravity.x * dt;

    //tex3D<T>(density, x-(x==domainSize.x), y-0.5*(y!=0)-0.5*(y==(domainSize.y)), z-(z==domainSize.z)) * alpha //for edge-repeating density

    velocity[num*3+1] += (tex3D<T>(density, x-(x==domainSize.x), y-0.5*(y!=0)-0.5*(y==(domainSize.y)), z-(z==domainSize.z)) * alpha
                       -  ((x<domainSize.x && z<domainSize.z)?(((y!=0)?(temperature[x + (y-1)*(velocitySize.x-1) + z*(velocitySize.y-1)*(velocitySize.x-1)]):ambTemp) + ((y<domainSize.y)?(temperature[x + y*(velocitySize.x-1) + z*(velocitySize.y-1)*(velocitySize.x-1)]):ambTemp)):(2*ambTemp)) * beta)
                       *  gravity.y * dt;

    velocity[num*3+2] += (tex3D<T>(density, x-(x==domainSize.x), y-(y==domainSize.y), z-0.5*(z!=0)-0.5*(z==(domainSize.z))) * alpha
                       -  ((x<domainSize.x && y<domainSize.y)?(((z!=0)?(temperature[x + y*(velocitySize.x-1) + (z-1)*(velocitySize.x-1)*(velocitySize.y-1)]):ambTemp) + ((z<domainSize.z)?(temperature[x + y*(velocitySize.x-1) + z*(velocitySize.y-1)*(velocitySize.x-1)]):ambTemp)):(2*ambTemp)) * beta)
                       *  gravity.z * dt;

}

//if temperature pointer is NULL, then the torch version (with a temperature-independent buoyancy force) is used
//need to NOT add buoyancy to obstacle cells
extern "C" cudaError_t addBuoyancyStaggeredWrapper_float(float* velocity, cudaTextureObject_t density, float* temperature, float* obstacles, dim3 domainSize, float3 gravity, float ambTemp, float buoyancyConstantAlpha, float buoyancyConstantBeta, float dt){

    int velocitySizeTotal = (domainSize.x+1) * (domainSize.y+1) * (domainSize.z+1);
    if(temperature != NULL)
        addBuoyancyStaggered<float><<<velocitySizeTotal/1024+(velocitySizeTotal%1024!=0),1024>>>(velocity, density, temperature, obstacles, domainSize, gravity, ambTemp, buoyancyConstantAlpha, buoyancyConstantBeta, dt);
    else
        addBuoyancyStaggered<float><<<velocitySizeTotal/1024+(velocitySizeTotal%1024!=0),1024>>>(velocity, density, temperature, obstacles, domainSize, gravity, ambTemp, buoyancyConstantAlpha, 0.0f, dt);
    //fbuoy = (alpha*s - beta*(T-Tamb)) * (gx,gy,gz), s is density, (gx,gy,gz) is the gravitational acceleration, beta is the buoyancy constant; normal gravity is (0,-9.81,0)

    return cudaPeekAtLastError();
}

template<typename T>
__global__ void addBuoyancyNotStaggered(T* velocity, cudaTextureObject_t density, T* temperature, T* obstacles, dim3 domainSize, float3 gravity, T ambTemp, T alpha, T beta){

    int num = blockIdx.x * blockDim.x + threadIdx.x;

    int x = num %(domainSize.y * domainSize.z);
    int y =(num % domainSize.z)/ domainSize.x;
    int z = num /(domainSize.x);

    if(x >= domainSize.x || y >= domainSize.y || z >= domainSize.z)
        return;

    //TODO: need to fix this somewhere else (subtracting ambTemp before multiplying by beta
    T totalBuoyancyFactor = tex3D<T>(density, x,y,z) * alpha - (temperature[x+domainSize.x*y+domainSize.x*domainSize.y*z]-ambTemp) * beta;

    velocity[num*3]   += totalBuoyancyFactor * gravity.x;
    velocity[num*3+1] += totalBuoyancyFactor * gravity.y;
    velocity[num*3+2] += totalBuoyancyFactor * gravity.z;

}

//use the SemiLagrange (or SemiLagrangeEuler maybe) from tfluids/third_party/tfluids.cu, or do it the way it's done in the Siggraph 2007 notes
extern "C" cudaError_t addBuoyancyNotStaggeredWrapper_float(float* velocity, cudaTextureObject_t density, float* temperature, float* obstacles, dim3 domainSize, float3 gravity, float ambTemp, float buoyancyConstantAlpha, float buoyancyConstantBeta, float dt){

    int velocitySizeTotal = (domainSize.x+1) * (domainSize.y+1) * (domainSize.z+1);
    if(temperature != NULL)
        addBuoyancyNotStaggered<float><<<velocitySizeTotal/1024+(velocitySizeTotal%1024!=0), 1024>>>(velocity, density, temperature, obstacles, domainSize, gravity, ambTemp, buoyancyConstantAlpha, buoyancyConstantBeta);
    else
        addBuoyancyNotStaggered<float><<<velocitySizeTotal/1024+(velocitySizeTotal%1024!=0), 1024>>>(velocity, density, temperature, obstacles, domainSize, gravity, ambTemp, buoyancyConstantAlpha, 0.0f);

    return cudaPeekAtLastError();
}

//velocity is not relative to cell dimension, so cellDim is requried as a parameter
//might not need to disregard occupied cells, so we won't use obstacles as a parameter yet
template<typename T>
__global__ void advectVelocityStaggered(T* velocityIn, T* velocityOut, T cellDim, dim3 velocitySize, T dt, int numTries=1){

    int num = blockIdx.x * blockDim.x + threadIdx.x;

    int x = num % velocitySize.x;
    int y =(num / velocitySize.x) % velocitySize.y;
    int z = num /(velocitySize.x  * velocitySize.y);

    if(x >= velocitySize.x || y >= velocitySize.y || z >= velocitySize.z)
        return;

    float3 currentPointVelocity, currentPoint;

    currentPointVelocity.x = velocityIn[num*3];
    currentPointVelocity.y = velocityIn[num*3+1];
    currentPointVelocity.z = velocityIn[num*3+2];

    //currentPointVelocity.x = 0;
    //currentPointVelocity.y = 0;
    //currentPointVelocity.z = 0;

    //should explicitly check if any component of cPV is zero to avoid rounding errors with the next procedure, both here, at every subDt step, and in the other advection functions

    currentPoint.x = x;
    currentPoint.y = y;
    currentPoint.z = z;

    float xVAdv, yVAdv, zVAdv, xVAdv2, yVAdv2, zVAdv2;
    T subDt = dt;
    float maxX, maxY, maxZ;

    do{
        numTries--;

        //the LHS variables correspond to the velocity grid
        xVAdv = currentPoint.x - currentPointVelocity.x * subDt / cellDim;
        yVAdv = currentPoint.y - currentPointVelocity.y * subDt / cellDim;
        zVAdv = currentPoint.z - currentPointVelocity.z * subDt / cellDim;

        if(xVAdv>=velocitySize.x || xVAdv<0 || yVAdv>=velocitySize.y || yVAdv<0 || zVAdv>=velocitySize.z || zVAdv<0){
            if(currentPointVelocity.x < 0) //if negative
                maxX = (velocitySize.x-1-currentPoint.x)/currentPointVelocity.x; //this will be negative, since cPV.s is negative
            else if(currentPointVelocity.x > 0) //disconsider zero values, as we can not divide by them
                maxX = -currentPoint.x/currentPointVelocity.x; //this will be negative, since cPV.x is positive
            if(currentPointVelocity.y < 0)
                maxY = max(maxX,(velocitySize.y-1-currentPoint.y)/currentPointVelocity.y); //need max(), because we want the smallest absolute value, and all compared values are negative
            else if(currentPointVelocity.y > 0)
                maxY = max(maxX,-currentPoint.y/currentPointVelocity.y);
            if(currentPointVelocity.z < 0)
                maxZ = max(maxY,(velocitySize.z-1-currentPoint.z)/currentPointVelocity.z);
            else if(currentPointVelocity.z > 0)
                maxZ = max(maxY,-currentPoint.z/currentPointVelocity.z);

            //give the values to xVAdv, yVAdv, zVAdv
            //maxZ is always negative, so this should be correct (the term added to currentPoint.n has the same sign as the in-bounds version)
            //this version should be correct according to dimensional analysis
            xVAdv2 = currentPoint.x + maxZ * currentPointVelocity.x;
            yVAdv2 = currentPoint.y + maxZ * currentPointVelocity.y;
            zVAdv2 = currentPoint.z + maxZ * currentPointVelocity.z;

            xVAdv2 = max(0,min(xVAdv2, velocitySize.x-1));
            yVAdv2 = max(0,min(yVAdv2, velocitySize.y-1));
            zVAdv2 = max(0,min(zVAdv2, velocitySize.z-1));

            subDt -= subDt * fabs((xVAdv2-currentPoint.x + yVAdv2-currentPoint.y + zVAdv2-currentPoint.z) / (xVAdv - currentPoint.x + yVAdv - currentPoint.y + zVAdv - currentPoint.z)); //the fabs should have no effect unless algorithm is incorrect
            xVAdv = xVAdv2;
            yVAdv = yVAdv2;
            zVAdv = zVAdv2;

            //TODO: Possibly needed; if we are still at the same point, which is what happens if we are at an edge and are trying to advect through that same edge, or if velocity is zero here, then use this point and exit
            //if(xVAdv==currentPoint.x && yVAdv==currentPoint.y && zVAdv==currentPoint.z)
            //    break;
        }
        else{
            subDt = 0;
        }

        xVAdv = max(0,min(xVAdv, velocitySize.x-1));
        yVAdv = max(0,min(yVAdv, velocitySize.y-1));
        zVAdv = max(0,min(zVAdv, velocitySize.z-1));

        currentPoint.x = xVAdv;
        currentPoint.y = yVAdv;
        currentPoint.z = zVAdv;

        currentPointVelocity.x = velocityIn[(int(xVAdv)  + velocitySize.x * int(yVAdv)  + velocitySize.y * velocitySize.x *   int(zVAdv))*3] * (floor(xVAdv) + 1 - xVAdv)
                               + velocityIn[(max(int(xVAdv+1), velocitySize.x-1) + velocitySize.x * int(yVAdv)  + velocitySize.y * velocitySize.x *   int(zVAdv))*3] * (xVAdv - floor(xVAdv));//(xVAdv+1, yVAdv, zVAdv).x * (xVAdv - floor(xVAdv));

        currentPointVelocity.y = velocityIn[(int(xVAdv)  + velocitySize.x * int(yVAdv)  + velocitySize.y * velocitySize.x *   int(zVAdv))*3+1] * (floor(yVAdv) + 1 - yVAdv)
                               + velocityIn[(int(xVAdv)  + velocitySize.x * max(int(yVAdv+1), velocitySize.y-1) + velocitySize.y * velocitySize.x *   int(zVAdv))*3+1] * (yVAdv - floor(yVAdv));

        currentPointVelocity.z = velocityIn[(int(xVAdv)  + velocitySize.x * int(yVAdv)  + velocitySize.y * velocitySize.x *   int(zVAdv))*3+2] * (floor(zVAdv) + 1 - zVAdv)
                               + velocityIn[(int(xVAdv)  + velocitySize.x * int(yVAdv)  + velocitySize.y * velocitySize.x * max(int(zVAdv+1), velocitySize.z-1))*3+2] * (zVAdv - floor(zVAdv));

    }while(numTries!=0 && subDt!=0);

    velocityOut[num*3]   = currentPointVelocity.x;
    velocityOut[num*3+1] = currentPointVelocity.y;
    velocityOut[num*3+2] = currentPointVelocity.z;
}

extern "C" cudaError_t advectVelocityCudaStaggeredWrapper_float(float* velocityIn, float* velocityOut, float* obstacles, float cellDim, dim3 domainSize, float dt){

    dim3 velocitySize(domainSize.x+1, domainSize.y+1, domainSize.z+1);
    printf("\n\nadvectVelocityCudaStaggeredWrapper_float; velocitySize = (%d, %d, %d)\n\n", velocitySize.x, velocitySize.y, velocitySize.z);
    int velocitySizeTotal = velocitySize.x * velocitySize.y * velocitySize.z;
#define advectVelocityNumThreads 512
    advectVelocityStaggered<float><<<velocitySizeTotal/advectVelocityNumThreads +(velocitySizeTotal%advectVelocityNumThreads!=0), advectVelocityNumThreads>>>(velocityIn, velocityOut, cellDim, velocitySize, dt);
#undef advectVelocityNumThreads
    return cudaPeekAtLastError();
}



extern "C" cudaError_t advectVelocityCudaNotStaggeredWrapper_float(float* velocityIn, float* velocityOut, float* obstacles, float cellDim, dim3 domainSize, float dt){

    return cudaPeekAtLastError();
}

template<typename T>
struct vec3D{
    T x, y, z;
};

template<typename T>
__global__ void calculateDivergenceStaggered(T* velocity, T* divergence, T cellDim, dim3 domainSize){

    int num = blockIdx.x * blockDim.x + threadIdx.x;

    int x = num % domainSize.x;
    int y =(num / domainSize.x) % domainSize.y;
    int z = num /(domainSize.x  * domainSize.y);

    if(x>=domainSize.x && y>=domainSize.y && z>=domainSize.z)
        return;

    //torch version has opposite sign for some reason
    divergence[num] =-( (velocity[(x+1 +  y * (domainSize.x+1) + z  * (domainSize.y+1) * (domainSize.x+1))*3  ] - velocity[(x + y * (domainSize.x+1) + z * (domainSize.y+1) * (domainSize.x+1))*3  ])
                      + (velocity[(x   +(y+1)*(domainSize.x+1) + z  * (domainSize.y+1) * (domainSize.x+1))*3+1] - velocity[(x + y * (domainSize.x+1) + z * (domainSize.y+1) * (domainSize.x+1))*3+1])
                      + (velocity[(x   +  y * (domainSize.x+1) +(z+1)*(domainSize.y+1) * (domainSize.x+1))*3+2] - velocity[(x + y * (domainSize.x+1) + z * (domainSize.y+1) * (domainSize.x+1))*3+2]));/// cellDim; //there is no division by cellDim in torch
}

template<typename T>
__global__ void calculateDivergenceNotStaggered(T* velocity, T* divergence, T cellDim, dim3 domainSize){

    int num = blockIdx.x * blockDim.x + threadIdx.x;

    int x = num % domainSize.x;
    int y =(num / domainSize.x) % domainSize.y;
    int z = num /(domainSize.x  * domainSize.y);

    if(x>=domainSize.x && y>=domainSize.y && z>=domainSize.z)
        return;

    //can set component to 0 if this edge version doesn't work
    divergence[num] = ( (((num>0)?(velocity[(num-1)*3  ]):0) - ((num<domainSize.x-1)?(velocity[(num+1)*3]):0))
                      + (((num>0)?(velocity[(num-1)*3+1]):0) - ((num<domainSize.y-1)?(velocity[(num+domainSize.x)*3+1]):0))
                      + (((num>0)?(velocity[(num-1)*3+2]):0) - ((num<domainSize.z-1)?(velocity[(num+domainSize.x*domainSize.y)*3+2]):0)));// / (2*cellDim);
}

extern "C" cudaError_t calculateDivergenceStaggeredWrapper_float(float* velocity, float* divergence, float cellDim, dim3 domainSize){

    int domainSizeTotal = domainSize.x * domainSize.y * domainSize.z;
    calculateDivergenceStaggered<float><<<domainSizeTotal/1024+(domainSizeTotal%1024!=0), 1024>>>(velocity, divergence, cellDim, domainSize);
    checkCudaErrors(cudaDeviceSynchronize());
    return cudaPeekAtLastError();
}

extern "C" cudaError_t calculateDivergenceNotStaggeredWrapper_float(float* velocity, float* divergence, float cellDim, dim3 domainSize){

    int domainSizeTotal = domainSize.x * domainSize.y * domainSize.z;
    calculateDivergenceNotStaggered<float><<<domainSizeTotal/1024+(domainSizeTotal%1024!=0), 1024>>>(velocity, divergence, cellDim, domainSize);
    return cudaPeekAtLastError();
}

//the velocity is advected last, so only velocityIn can be used for scalar field advection
//TODO: this uses a type of binary search, might be incorrect (look into Runge-Kutta coefficients)
template<typename T>
__global__ void advectDensityStaggered(T* velocity, cudaTextureObject_t densityPingTex, cudaSurfaceObject_t densityPongSurf, T* obstacles, dim3 domainSize, T cellDim, T dt, int numTries=1){

    int num = blockIdx.x * blockDim.x + threadIdx.x;

    int x = num % domainSize.x;
    int y =(num / domainSize.x) % domainSize.y;
    int z = num /(domainSize.x  * domainSize.y);

    int velocitySizeTotal = domainSize.x*domainSize.y*domainSize.z*3;

    if(x >= domainSize.x || y >= domainSize.y || z >= domainSize.z)
        return;

    //if cell is occupied, then skip it and set the surface to zero
    if(obstacles[num]!=0){
        return;}

    //also need to do something about out-of-bounds advection (can just clamp to the edge along the backtraced segment)

    //texture coordinates are float regardless of the texture bit depth (sizeof(T))
    float xAdv, yAdv, zAdv, xAdv2, yAdv2, zAdv2;
    T c00, c01, c10, c11, c1, c0, c, p00, p01, p10, p11, p1, p0, p, c_bis;

    bool t = false;

    c = tex3D<T>(densityPingTex, x, y, z);

    c = 0.0;

    float3 currentPoint, currentPointVelocity;

    currentPoint.x = __int2float_rd(x);
    currentPoint.y = __int2float_rd(y);
    currentPoint.z = __int2float_rd(z);

    currentPointVelocity.x = (velocity[(x + (domainSize.x+1)*y + (domainSize.x+1)*(domainSize.y+1)*z)*3]   + velocity[(x+1 + (domainSize.x+1)*y     + (domainSize.x+1)*(domainSize.y+1)* z  )*3]) / 2.0;
    currentPointVelocity.y = (velocity[(x + (domainSize.x+1)*y + (domainSize.x+1)*(domainSize.y+1)*z)*3+1] + velocity[(x   + (domainSize.x+1)*(y+1) + (domainSize.x+1)*(domainSize.y+1)* z  )*3+1]) / 2.0;
    currentPointVelocity.z = (velocity[(x + (domainSize.x+1)*y + (domainSize.x+1)*(domainSize.y+1)*z)*3+2] + velocity[(x   + (domainSize.x+1)*y     + (domainSize.x+1)*(domainSize.y+1)*(z+1))*3+2]) / 2.0;

    float subDt = dt;

    int xAdvFloor, yAdvFloor, zAdvFloor;
    float maxX, maxY, maxZ;

    float xd, yd, zd;

    bool a = true;

    bool o1, o2, o3, o4, o5, o6, o7, o8;

    do{
        numTries--;
        xAdv = currentPoint.x - currentPointVelocity.x * subDt / cellDim;
        yAdv = currentPoint.y - currentPointVelocity.y * subDt / cellDim;
        zAdv = currentPoint.z - currentPointVelocity.z * subDt / cellDim;

        //this is the boundary clamping
        //check if these are within boundaries, clamp along backtraced segment if they aren't
        if(xAdv>=domainSize.x || xAdv<0 || yAdv>=domainSize.y || yAdv<0 || zAdv>=domainSize.z || zAdv<0){
            //you are going in the OPPOSITE direction of the velocity at currentPoint when doing advection
            if(currentPointVelocity.x < 0) //if negative
                maxX = (domainSize.x-1-currentPoint.x)/currentPointVelocity.x; //this will be negative, since cPV.s is negative
            else if(currentPointVelocity.x > 0) //disconsider zero values, as we can not divide by them
                maxX = -currentPoint.x/currentPointVelocity.x; //this will be negative, since cPV.x is positive
            if(currentPointVelocity.y < 0)
                maxY = max(maxX,(domainSize.y-1-currentPoint.y)/currentPointVelocity.y); //need max(), because we want the smallest absolute value, and all compared values are negative
            else if(currentPointVelocity.y > 0)
                maxY = max(maxX,-currentPoint.y/currentPointVelocity.y);
            if(currentPointVelocity.z < 0)
                maxZ = max(maxY,(domainSize.z-1-currentPoint.z)/currentPointVelocity.z);
            else if(currentPointVelocity.z > 0)
                maxZ = max(maxY,-currentPoint.z/currentPointVelocity.z);

            xAdv2 = currentPoint.x + maxZ * currentPointVelocity.x; //this will be in the direction oppposite to the velocity vector
            yAdv2 = currentPoint.y + maxZ * currentPointVelocity.y;
            zAdv2 = currentPoint.z + maxZ * currentPointVelocity.z;

            xAdv2 = max(0,min(xAdv2, domainSize.x-1));
            yAdv2 = max(0,min(yAdv2, domainSize.y-1));
            zAdv2 = max(0,min(zAdv2, domainSize.z-1));

            //set subDt
            subDt = subDt * /*fabs*/((xAdv2-currentPoint.x)+(yAdv2-currentPoint.y)+(zAdv2-currentPoint.z)) / ((xAdv-currentPoint.x)+(yAdv-currentPoint.y)+(zAdv-currentPoint.z));
            xAdv = xAdv2;
            yAdv = yAdv2;
            zAdv = zAdv2;
        }

        //these are the crucial domain boundary guards
        xAdv = max(0,min(xAdv, domainSize.x-1));
        yAdv = max(0,min(yAdv, domainSize.y-1));
        zAdv = max(0,min(zAdv, domainSize.z-1));

        xAdvFloor = max(__float2int_rd(xAdv),0);
        yAdvFloor = max(__float2int_rd(yAdv),0);
        zAdvFloor = max(__float2int_rd(zAdv),0);

        //integral part of modff is not needed (we got it already with nAdvFloor), so give it to c00 and overwrite the value after
        xd = modff(xAdv, &c00);
        yd = modff(yAdv, &c00);
        zd = modff(zAdv, &c00);

        //copy-paste from https://en.wikipedia.org/wiki/Trilinear_interpolation
        //this can probably be optimized, but cannot be automatically interpolated with the texture-oriented hardware capabilities
        //the texture can be clamped automatically; could use this
        c00 = tex3D<T>(densityPingTex, xAdvFloor,  yAdvFloor,  zAdvFloor  ) * isNotObstacle(xAdvFloor,  yAdvFloor,  zAdvFloor  ) * (1.0f - xd)
              + tex3D<T>(densityPingTex, min(xAdvFloor+1,domainSize.x-1),yAdvFloor,  zAdvFloor  ) * isNotObstacle(xAdvFloor+1,yAdvFloor,  zAdvFloor  ) * xd;

        c01 = tex3D<T>(densityPingTex, xAdvFloor,  min(yAdvFloor+1,domainSize.y-1),zAdvFloor  ) * isNotObstacle(xAdvFloor,  yAdvFloor+1,zAdvFloor  ) * (1.0f - xd)
              + tex3D<T>(densityPingTex, min(xAdvFloor+1,domainSize.x-1),min(yAdvFloor+1,domainSize.y-1),zAdvFloor  ) * isNotObstacle(xAdvFloor+1,yAdvFloor+1,zAdvFloor  ) * xd;

        c10 = tex3D<T>(densityPingTex, xAdvFloor,  yAdvFloor,  min(zAdvFloor+1,domainSize.z-1)) * isNotObstacle(xAdvFloor,  yAdvFloor,  zAdvFloor+1) * (1.0f - xd)
              + tex3D<T>(densityPingTex, min(xAdvFloor+1,domainSize.x-1),yAdvFloor,  min(zAdvFloor+1,domainSize.z-1)) * isNotObstacle(xAdvFloor+1,yAdvFloor,  zAdvFloor+1) * xd;

        c11 = tex3D<T>(densityPingTex, xAdvFloor,  min(yAdvFloor+1,domainSize.y-1),min(zAdvFloor+1,domainSize.z-1)) * isNotObstacle(xAdvFloor,  yAdvFloor+1,zAdvFloor+1) * (1.0f - xd)
              + tex3D<T>(densityPingTex, min(xAdvFloor+1,domainSize.x-1),min(yAdvFloor+1,domainSize.y-1),min(zAdvFloor+1,domainSize.z-1)) * isNotObstacle(xAdvFloor+1,yAdvFloor+1,zAdvFloor+1) * xd;

        c0 = c00 * (1 - yd) + c10 * yd;
        c1 = c01 * (1 - yd) + c11 * yd;

        c = c0 * (1 - zd) + c1 * zd;

        p00 = isNotObstacle(xAdvFloor,  yAdvFloor,  zAdvFloor  ) * (1.0f - xd)
              + isNotObstacle(xAdvFloor+1,yAdvFloor,  zAdvFloor  ) * xd;

        p01 = isNotObstacle(xAdvFloor,  yAdvFloor+1,zAdvFloor  ) * (1.0f - xd)
              + isNotObstacle(xAdvFloor+1,yAdvFloor+1,zAdvFloor  ) * xd;

        p10 = isNotObstacle(xAdvFloor,  yAdvFloor,  zAdvFloor+1) * (1.0f - xd)
              + isNotObstacle(xAdvFloor+1,yAdvFloor,  zAdvFloor+1) * xd;

        p11 = isNotObstacle(xAdvFloor,  yAdvFloor+1,zAdvFloor+1) * (1.0f - xd)
              + isNotObstacle(xAdvFloor+1,yAdvFloor+1,zAdvFloor+1) * xd;

        p0 = p00 * (1 - yd) + p10 * yd;
        p1 = p01 * (1 - yd) + p11 * yd;

        p = p0 * (1 - zd) + p1 * zd;

        o1 = isNotObstacle(xAdvFloor,  yAdvFloor,  zAdvFloor  );
        o2 = isNotObstacle(xAdvFloor,  yAdvFloor+1,zAdvFloor  );
        o3 = isNotObstacle(xAdvFloor,  yAdvFloor,  zAdvFloor+1);
        o4 = isNotObstacle(xAdvFloor,  yAdvFloor+1,zAdvFloor+1);
        o5 = isNotObstacle(xAdvFloor+1,yAdvFloor,  zAdvFloor  );
        o6 = isNotObstacle(xAdvFloor+1,yAdvFloor+1,zAdvFloor  );
        o7 = isNotObstacle(xAdvFloor+1,yAdvFloor,  zAdvFloor+1);
        o8 = isNotObstacle(xAdvFloor+1,yAdvFloor+1,zAdvFloor+1);

        if(o1 || o2 || o3 || o4 || o5 || o6 || o7 || o8){
            //the "share" of ignored interpolation points (that are occupied by obstacles) needs to be divided between the remaining points,
            //so that the interpolation result is a a proper average

            c /= p;
            subDt = dt - subDt; //if a subsegment was projected back due to the else branch being used in some previous iteration, repeat with the rest of the time
            dt = subDt;
            //if()
            currentPoint.x = xAdv;
            currentPoint.y = yAdv;
            currentPoint.z = zAdv;

#define vel(dimX,dimY,dimZ, dimension) velocity[((dimX) + (dimY)*(domainSize.x+1) + (dimZ)*(domainSize.x+1)*(domainSize.y+1))*3+(dimension)] //the dimension component (0->x, 1->y, 2->z) of velocity cell (dimX, dimY, dimZ)
#define getIndex(dimX,dimY,dimZ, dimension) ((dimX) + (dimY)*(domainSize.x+1) + (dimZ)*(domainSize.x+1)*(domainSize.y+1))*3+(dimension)

            currentPointVelocity.x = vel(int(xAdv+0.5), yAdvFloor, zAdvFloor, 0) * ((xAdv+0.5) - float(floor(xAdv+0.5)))
                                   + vel(int(xAdv+1.5), yAdvFloor, zAdvFloor, 0) * (float(floor(xAdv+1.5)) - (xAdv+0.5));
            currentPointVelocity.y = vel(xAdvFloor, int(yAdv+0.5), zAdvFloor, 1) * ((yAdv+0.5) - float(floor(yAdv+0.5)))
                                   + vel(xAdvFloor, int(yAdv+1.5), zAdvFloor, 1) * (float(floor(yAdv+1.5)) - (yAdv+0.5));
            currentPointVelocity.z = vel(xAdvFloor, yAdvFloor, int(zAdv+0.5), 2) * ((zAdv+0.5) - float(floor(zAdv+0.5)))
                                   + vel(xAdvFloor, yAdvFloor, int(zAdv+1.5), 2) * (float(floor(zAdv+1.5)) - (zAdv+0.5));

        }
        else{
            //if all points are ignored due to occupation by obstacles
            //this halves the dt until an unoccupied cell is found, and then uses the velocity at this point to backtrace the rest of dt, just like with the original point
            //a maximum number of tries is done, if dt is not reduced to zero or dt*currentVelocity is smaller than a cell width (or something close to it) until then,
            //the last found point or the original point that was backtraced from is used
            //this is based on the idea that obstacles and any jagged features will usually be significantly larger than the cell size, which makes the segment half closer to
            //the original backtracing point more likely to be free(r)

            subDt /= 2;
            }
        }while(numTries != 0 && subDt != 0);

#undef vel
    surf3Dwrite<T>(c, densityPongSurf, x*sizeof(T), y, z);
}

//does dt need to be of type T? (check timer precision)
template<typename T>
__global__ void advectStaggered(T* velocity, T* scalarFieldIn, T* scalarFieldOut, T* obstacles, dim3 domainSize, T cellDim, float dt, int numTries=5){

    int num = blockIdx.x * blockDim.x + threadIdx.x;

    int x = num %(domainSize.y * domainSize.z);
    int y =(num % domainSize.z)/ domainSize.x;
    int z = num /(domainSize.x);

    if(x >= domainSize.x || y >= domainSize.y || z >= domainSize.z)
        return;

    //if cell is occupied, then skip it and set the surface to zero
    if(obstacles[num]!=0){
        scalarFieldIn[num] = 0;
        return;}

    //also need to do something about out-of-bounds advection (can just clamp to the edge along the backtraced segment)

    //texture coordinates are float regardless of the texture bit depth (sizeof(T))
    float xAdv, yAdv, zAdv, xAdv2, yAdv2, zAdv2;
    int xAdvFloor, yAdvFloor, zAdvFloor;
    T c00, c01, c10, c11, c1, c0, c, p00, p01, p10, p11, p1, p0, p;
    float3 currentPoint, currentPointVelocity;

    T currentValue = scalarFieldIn[num];

    currentPoint.x = float(x);
    currentPoint.y = float(y);
    currentPoint.z = float(z);

    currentPointVelocity.x = (velocity[(x + (domainSize.x+1)*y + (domainSize.x+1)*(domainSize.y+1)*z)*3]   + velocity[(x+1 + (domainSize.x+1)*y     + (domainSize.x+1)*(domainSize.y+1)* z  )*3]) / 2.0;
    currentPointVelocity.y = (velocity[(x + (domainSize.x+1)*y + (domainSize.x+1)*(domainSize.y+1)*z)*3+1] + velocity[(x+1 + (domainSize.x+1)*(y+1) + (domainSize.x+1)*(domainSize.y+1)* z  )*3]) / 2.0;
    currentPointVelocity.z = (velocity[(x + (domainSize.x+1)*y + (domainSize.x+1)*(domainSize.y+1)*z)*3+2] + velocity[(x+1 + (domainSize.x+1)*y     + (domainSize.x+1)*(domainSize.y+1)*(z+1))*3]) / 2.0;

    float subDt = dt;

    float maxX, maxY, maxZ, xd, yd, zd;

    do{
        numTries--;

        //what isa written as x,y,z and velocity need to change sometimes (when currentPoint is updated)
        //x -> currentPoint.x
        xAdv = currentPoint.x - currentPointVelocity.x * subDt / cellDim;
        yAdv = currentPoint.y - currentPointVelocity.y * subDt / cellDim;
        zAdv = currentPoint.z - currentPointVelocity.z * subDt / cellDim;


        //this is the boundary clamping

        //check if these are within boundaries, clamp along backtraced segment if they aren't
        if(xAdv>=domainSize.x || xAdv<0 || yAdv>=domainSize.y || yAdv<0 || zAdv>=domainSize.z || zAdv<0){ //(*)
            //this isn't right
            //you are going in the OPPOSITE direction of the velocity at currentPoint when doing advection
            if(currentPointVelocity.x < 0) //if negative
                maxX = (domainSize.x-1-currentPoint.x)/currentPointVelocity.x; //this will be negative, since cPV.s is negative
            else if(currentPointVelocity.x) //disconsider zero values, as we can not divide by them
                maxX = -currentPoint.x/currentPointVelocity.x; //this will be negative, since cPV.x is positive
            if(currentPointVelocity.y < 0)
                maxY = max(maxX,(domainSize.y-1-currentPoint.y)/currentPointVelocity.y); //need max(), because we want the smallest absolute value, and all compared values are negative
            else if(currentPointVelocity.y > 0)
                maxY = max(maxX,-currentPoint.y/currentPointVelocity.y);
            if(currentPointVelocity.z < 0)
                maxZ = max(maxY,(domainSize.z-1-currentPoint.z)/currentPointVelocity.z);
            else if(currentPointVelocity.z > 0)
                maxZ = max(maxY,-currentPoint.z/currentPointVelocity.z);

            xAdv2 = currentPoint.x + maxZ * currentPointVelocity.x;// * subDt / cellDim; //this will be in the direction oppposite to the velocity vector
            yAdv2 = currentPoint.y + maxZ * currentPointVelocity.y;// * subDt / cellDim;
            zAdv2 = currentPoint.z + maxZ * currentPointVelocity.z;// * subDt / cellDim;

            //this happens if we were already at an edge and the direction opposite the velocity at this point would send us past this same edge
            //will not happen if there was an obstacle here last time, since the second part of the do while would have halved the distance
            //currentValue has the proper value, since it got it from xAdv, yAdv, zAdv in the previous iteration, so do not set it here
            if(xAdv2 == xAdv && yAdv2 == yAdv && zAdv2 == zAdv){
                currentPoint.x = xAdv2;
                currentPoint.y = yAdv2;
                currentPoint.z = zAdv2;

                break;}

            //set subDt
            subDt = subDt * ((xAdv2-currentPoint.x)+(yAdv2-currentPoint.y)+(zAdv2-currentPoint.z)) / ((xAdv-currentPoint.x)+(yAdv-currentPoint.y)+(zAdv-currentPoint.z)); //this should protect against 0 components of velocity
            xAdv = xAdv2;
            yAdv = yAdv2;
            zAdv = zAdv2;
        }

        xAdv = max(0,min(xAdv, domainSize.x-1));
        yAdv = max(0,min(yAdv, domainSize.y-1));
        zAdv = max(0,min(zAdv, domainSize.z-1));

        //this might not work for doubles
        xAdvFloor = __float2int_rd(xAdv);
        yAdvFloor = __float2int_rd(yAdv);
        zAdvFloor = __float2int_rd(zAdv);

        //give xd, yd, zd their values
        xd = xAdv - xAdvFloor;
        yd = yAdv - yAdvFloor;
        zd = yAdv - zAdvFloor;

        //copy-paste from https://en.wikipedia.org/wiki/Trilinear_interpolation
        //this can probably be optimized, but cannot be automatically interpolated with the texture-oriented hardware capabilities
        //the texture can be clamped automatically; could use this

        c00 = scalarFieldIn[xAdvFloor+yAdvFloor*domainSize.x+zAdvFloor*domainSize.x*domainSize.y] * isNotObstacle(xAdvFloor,  yAdvFloor,  zAdvFloor  ) * (1.0f - xd)
              + scalarFieldIn[(xAdvFloor+1)+yAdvFloor*domainSize.x+zAdvFloor*domainSize.x*domainSize.y] * isNotObstacle(xAdvFloor+1,yAdvFloor,  zAdvFloor  ) * xd;

        c01 = scalarFieldIn[xAdvFloor+(yAdvFloor+1)*domainSize.x+zAdvFloor*domainSize.x*domainSize.y] * isNotObstacle(xAdvFloor,  yAdvFloor+1,zAdvFloor  ) * (1.0f - xd)
              + scalarFieldIn[(xAdvFloor+1)+(yAdvFloor+1)*domainSize.x+zAdvFloor*domainSize.x*domainSize.y] * isNotObstacle(xAdvFloor+1,yAdvFloor+1,zAdvFloor  ) * xd;

        c10 = scalarFieldIn[xAdvFloor+yAdvFloor*domainSize.x+(zAdvFloor+1)*domainSize.x*domainSize.y] * isNotObstacle(xAdvFloor,  yAdvFloor,  zAdvFloor+1) * (1.0f - xd)
              + scalarFieldIn[xAdvFloor+1+yAdvFloor*domainSize.x+(zAdvFloor+1)*domainSize.x*domainSize.y] * isNotObstacle(xAdvFloor+1,yAdvFloor,  zAdvFloor+1) * xd;

        c11 = scalarFieldIn[xAdvFloor+(yAdvFloor+1)*domainSize.x+(zAdvFloor+1)*domainSize.x*domainSize.y] * isNotObstacle(xAdvFloor,  yAdvFloor+1,zAdvFloor+1) * (1.0f - xd)
              + scalarFieldIn[xAdvFloor+1+(yAdvFloor+1)*domainSize.x+(zAdvFloor+1)*domainSize.x*domainSize.y] * isNotObstacle(xAdvFloor+1,yAdvFloor+1,zAdvFloor+1) * xd;

        c0 = c00 * (1 - yd) + c10 * yd;
        c1 = c01 * (1 - yd) + c11 * yd;

        c = c0 * (1 - zd) + c1 * zd;

        p00 = isNotObstacle(xAdvFloor,  yAdvFloor,  zAdvFloor  ) * (1.0f - xd)
              + isNotObstacle(xAdvFloor+1,yAdvFloor,  zAdvFloor  ) * xd;

        p01 = isNotObstacle(xAdvFloor,  yAdvFloor+1,zAdvFloor  ) * (1.0f - xd)
              + isNotObstacle(xAdvFloor,  yAdvFloor+1,zAdvFloor  ) * xd;

        p10 = isNotObstacle(xAdvFloor,  yAdvFloor,  zAdvFloor+1) * (1.0f - xd)
              + isNotObstacle(xAdvFloor,  yAdvFloor,  zAdvFloor+1) * xd;

        p10 = isNotObstacle(xAdvFloor,  yAdvFloor+1,zAdvFloor+1) * (1.0f - xd)
              + isNotObstacle(xAdvFloor,  yAdvFloor+1,zAdvFloor+1) * xd;

        p0 = p00 * (1 - yd) + p10 * yd;
        p1 = p01 * (1 - yd) + p11 * yd;

        p = p0 * (1 - zd) + p1 * zd;

        if(p!=0){ //are all points ignored?
            //the "share" of ignored interpolation points (that are occupied by obstacles) needs to be divided between the remaining points,
            //so that the interpolation result is a a proper average
            c /= p;
            subDt = dt - subDt; //if a subsegment was projected back due to the else branch being used in some previous iteration, repeat with the rest of the time
            dt = subDt;
            currentPoint.x = xAdv;
            currentPoint.y = yAdv;
            currentPoint.z = zAdv;

            currentValue = c;

            //this shouldn't be masked, since the motion of obstacles pushes the density too

            currentPointVelocity.x = velocity[(x   + y *  (domainSize.x + 1) + z * (domainSize.y + 1) * (domainSize.z + 1)) * 3  ] * (floor(xAdv)+1-xAdv)
                                   + velocity[(x+1 + y *  (domainSize.x + 1) + z * (domainSize.y + 1) * (domainSize.z + 1)) * 3  ] * (xAdv-floor(xAdv));
            currentPointVelocity.y = velocity[(x   + y *  (domainSize.x + 1) + z * (domainSize.y + 1) * (domainSize.z + 1)) * 3+1] * (floor(yAdv)+1-yAdv)
                                   + velocity[(x   +(y+1)*(domainSize.x + 1) + z * (domainSize.y + 1) * (domainSize.z + 1)) * 3+1] * (yAdv-floor(yAdv));
            currentPointVelocity.z = velocity[(x   + y *  (domainSize.x + 1) + z * (domainSize.y + 1) * (domainSize.z + 1)) * 3+2] * (floor(zAdv)+1-zAdv)
                                   + velocity[(x   + y *  (domainSize.x + 1)+(z+1)*(domainSize.y + 1) * (domainSize.z + 1)) * 3+2] * (zAdv-floor(zAdv));

            //pseudocode
            //currentPointVelocity.x = velocity(x,y,z).x * (floor(xAdv)+1-xAdv) + velocity(x+1,y,z).x * (xAdv-floor(xAdv));
            //currentPointVelocity.y = velocity(x,y,z).y * (floor(yAdv)+1-) + velocity(x,y+1,z).y * (yAdv-floor(yAdv));
            //currentPointVelocity.z = velocity(x,y,z).z * (zAdv-floor(zAdv)) + velocity(x,y,z+1).z * (zAdv-floor(zAdv));
        }
        else{
            //if all points are ignored due to occupation by obstacles
            //this halves the dt until an unoccupied cell is found, and then uses the velocity at this point to backtrace the rest of dt, just like with the original point
            //a maximum number of tries is done, if dt is not reduced to zero or dt*currentVelocity is smaller than a cell width (or something close to it) until then,
            //the last found point or the original point that was backtraced from is used
            //this is based on the idea that obstacles and any jagged features will usually be significantly larger than the cell size, which makes the segment half closer to
            //the original backtracing point more likely to be free(r)

            subDt /= 2;
            }
        }while(numTries != 0 && subDt != 0);

    scalarFieldOut[num] = currentValue;
}

#undef isNotObstacle

//__global__ void advectStaggered(T* velocity, T* scalarFieldIn, T* scalarFieldOut, T* obstacles, dim3 domainSize, T cellDim, float dt, int numTries=5){
extern "C" cudaError_t advectDensityCudaStaggeredWrapper_float(float* velocityIn, cudaTextureObject_t densityPingTex, cudaSurfaceObject_t densityPongSurf, float* obstacles, dim3 domainSize, float cellDim, float dt){

    int domainSizeTotal = domainSize.x*domainSize.y*domainSize.z;

    //if the debugging printf calls are included, this fails for 1024 threads per block (cudaErrorLaunchOutOfResources). The number of threads has been reduced to prevent this.
#define numThreadsDensity 512
    advectDensityStaggered<float><<<domainSizeTotal/numThreadsDensity+(domainSizeTotal%numThreadsDensity!=0), numThreadsDensity>>>(velocityIn, densityPingTex, densityPongSurf, obstacles, domainSize, cellDim, dt);
#undef numThreadsDensity

    return cudaPeekAtLastError();
}

//this is for the temperature advection
extern "C" cudaError_t advectCudaStaggeredWrapper_float(float* velocity, float* scalarFieldIn, float* scalarFieldOut, float* obstacles, dim3 domainSize, float cellDim, float dt){

    int domainSizeTotal = domainSize.x * domainSize.y * domainSize.z;
    advectStaggered<float><<<domainSizeTotal/1024+(domainSizeTotal%1024!=0), 1024>>>(velocity, scalarFieldIn, scalarFieldOut, obstacles, domainSize, cellDim, dt);

    return cudaPeekAtLastError();
}

template<typename T>
__global__ void advectDensityNotStaggered(T* velocity, cudaTextureObject_t densityPingTex, cudaSurfaceObject_t densityPongSurf, T* obstacles, dim3 domainSize, T cellDim, float dt, int numTries=5){

    int num = blockIdx.x * blockDim.x + threadIdx.x;

    int x = num % domainSize.x;
    int y =(num / domainSize.x) % domainSize.y;
    int z = num /(domainSize.x  * domainSize.y);

    if(x >= domainSize.x || y >= domainSize.y || z >= domainSize.z)
        return;

    surf3Dwrite(tex3D<T>(densityPingTex, x - (velocity[num*3]) * dt / 2.0,
                                         y - (velocity[num*3+1]) * dt / 2.0,
                                         z - (velocity[num*3+2]) * dt / 2.0),
                densityPongSurf, x*sizeof(T), y, z);
}

extern "C" cudaError_t advectDensityCudaNotStaggeredWrapper_float(float* velocityIn, cudaTextureObject_t densityPingTex, cudaSurfaceObject_t densityPongSurf, float* obstacles, dim3 domainSize, float cellDim, float dt){

    int domainSizeTotal = domainSize.x*domainSize.y*domainSize.z;
    advectDensityNotStaggered<float><<<domainSizeTotal/1024+(domainSizeTotal%1024!=0), 1024>>>(velocityIn, densityPingTex, densityPongSurf, obstacles, domainSize, cellDim, dt);

    return cudaPeekAtLastError();
}

//updating moving obstacles requires pushing density
//This is the Lua code from FluidNet for velocity advection, which has been adapted for use here
/*
// *****************************************************************************
// velocityUpdateForward
// *****************************************************************************

__global__ void velocityUpdateForward(
    CudaFlagGrid flags, CudaMACGrid vel, CudaRealGrid pressure,
    const int32_t bnd) {
  int32_t b, chan, k, j, i;
  if (GetKernelIndices(flags, b, chan, k, j, i)) {
    return;
  }
  if (i < bnd || i > flags.xsize() - 1 - bnd || = 29 53 9
(x,y,z) = 30 53 9
(x,y,z) = 31 53 9
(x,y,z) = 32 53 9
(
      j < bnd || j > flags.ysize() - 1 - bnd ||
      (flags.is_3d() && (k < bnd || k > flags.zsize() - 1 - bnd))) {
    // Manta doesn't touch the velocity on the boundaries (i.e.
    // it stays constant).
    return;
  }

  if (flags.isFluid(i, j, k, b)) {
    if (flags.isFluid(i - 1, j, k, b)) {
      vel(i, j, k, 0, b) -= (pressure(i, j, k, b) -
                             pressure(i - 1, j, k, b));
    }
    if (flags.isFluid(i, j - 1, k, b)) {
      vel(i, j, k, 1, b) -= (pressure(i, j, k, b) -
                             pressure(i, j - 1, ko, b));
    }
    if (flags.is_3d() && flags.isFluid(i, j, k - 1, b)) {
      vel(i, j, k, 2, b) -= (pressure(i, j, k, b) -
                             pressure(i, j, k - 1, b));
    }

    if (flags.isEmpty(i - 1, j, k, b)) {
      vel(i, j, k, 0, b) -= pressure(i, j, k, b);
    }
    if (flags.isEmpty(i, j - 1, k, b)) {
      vel(i, j, k, 1, b) -= pressure(i, j, k, b);
    }
    if (flags.is_3d() && flags.isEmpty(i, j, k - 1, b)) {
      vel(i, j, k, 2, b) -= pressure(i, j, k, b);
    }
  }
  else if (flags.isEmpty(i, j, k, b) && !flags.isOutflow(i, j, k, b)) {
    // don't change velocities in outflow cells
    if (flags.isFluid(i - 1, j, k, b)) {
      vel(i, j, k, 0, b) += pressure(i - 1, j, k, b);
    } else {
      vel(i, j, k, 0, b)  = 0.f;
    }
    if (flags.isFluid(i, j - 1, k, b)) {
      vel(i, j, k, 1, b) += pressure(i, j - 1, k, b);
    } else {
      vel(i, j, k, 1, b)  = 0.f;
    }
    if (flags.is_3d()) {
      if (flags.isFluid(i, j, k - 1, b)) {
        vel(i, j, k, 2, b) += pressure(i, j, k - 1, b);
      } else {
        vel(i, j, k, 2, b)  = 0.f;
      }
    }
}

static int tfluids_CudaMain_velocityUpdateForward(lua_State* L) {
  THCState* state = cutorch_getstate(L);

  // Get the args from the lua stack. NOTE: We do ALL arguments (size checking)
  // on the lua stack.
  THCudaTensor* tensor_u = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 1, "torch.CudaTensor"));
  THCudaTensor* tensor_flags = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 2, "torch.CudaTensor"));
  THCudaTensor* tensor_p = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 3, "torch.CudaTensor"));
  const bool is_3d = static_cast<bool>(lua_toboolean(L, 4));

  CudaFlagGrid flags = toCudaFlagGrid(state, tensor_flags, is_3d);
  CudaMACGrid vel = toCudaMACGrid(state, tensor_u, is_3d);
  CudaRealGrid pressure = toCudaRealGrid(state, tensor_p, is_3d);

  const int32_t bnd = 1;
  // LaunchKernel args: lua_State, func, domain, args...
  LaunchKernel(L, &velocityUpdateForward, flags,
               flags, vel, pressure, bnd);

  return 0;  // Recall: number of return values on the lua stack.
}

*/

//in velocityUpdateForward() in torch/tfluids/third_party/tfluids.cu, the pressure divergence is not divided by the density
//what is labelled as density is in fact smoke concentration, it has NOTHING TO DO with the density factor in the Poisson equation
//
//since the velocity field is normalized, and the exact pressure units are not relevant outside the neural network, and
//also probably the fact that the fluid is considered incompressible
template<typename T>
__global__ void subtractGradientStaggered(T* velocity, T* pressure, T* density, T* obstacles, dim3 velocitySize, T cellDim, T dt){

    int num = blockIdx.x * blockDim.x + threadIdx.x;

    int x = num % velocitySize.x;
    int y =(num / velocitySize.x) % velocitySize.y;
    int z = num /(velocitySize.x  * velocitySize.y);

    dim3 domainSize;

    domainSize.x = velocitySize.x - 1;
    domainSize.y = velocitySize.y - 1;
    domainSize.z = velocitySize.z - 1;

    if(x >= velocitySize.x || y >= velocitySize.y || z >= velocitySize.z)
        return;

    int px = 0.0f, py = 0.0f, pz = 0.0f;

    //leave boundary velocities unchanged (as in torch)
    if(x<velocitySize.x-1 && y<velocitySize.y-1 && z<velocitySize.z-1){
        if(x>0){
            //the velocity is not normalized, and the pressure is not divided by its normalization factor
            velocity[num*3]   -= (pressure[x + domainSize.x * y + domainSize.x * domainSize.y * z] - pressure[x-1 + domainSize.x * y + domainSize.x * domainSize.y * z]);
            px = (pressure[x + domainSize.x * y + domainSize.x * domainSize.y * z] - pressure[x-1 + domainSize.x * y + domainSize.x * domainSize.y * z]);
        }

        if(y>0){
            velocity[num*3+1] -= (pressure[x + domainSize.x * y + domainSize.x * domainSize.y * z] - pressure[x   + domainSize.x * (y-1) + domainSize.x*domainSize.y*z]);
            py = (pressure[x + domainSize.x * y + domainSize.x * domainSize.y * z] - pressure[x   + domainSize.x * (y-1) + domainSize.x*domainSize.y*z]);
        }

        if(z>0){
            velocity[num*3+2] -= (pressure[x + domainSize.x * y + domainSize.x * domainSize.y * z] - pressure[x   + domainSize.x * y + domainSize.x*domainSize.y*(z-1)]);
            pz = (pressure[x + domainSize.x * y + domainSize.x * domainSize.y * z] - pressure[x   + domainSize.x * y + domainSize.x*domainSize.y*(z-1)]);
        }

    }


}

template<typename T>
__global__ void checkGradientFactor(T* pressure, T* velocity, dim3 domainSize){

    int num = blockIdx.x * blockDim.x + threadIdx.x;

    int x = num % domainSize.x;
    int y =(num / domainSize.x) % domainSize.y;
    int z = num /(domainSize.x  * domainSize.y);

    if(x > domainSize.x-2 || y > domainSize.y-2 || z > domainSize.z-2 || x<1 || y<1 || z<1)
        return;

#define getPressure(dimX,dimY,dimZ) pressure[(dimX) + (dimY) * domainSize.x + (dimZ) * domainSize.x * domainSize.y]
#define getVelocity(dimX, dimY, dimZ, dimension) velocity[((dimX) + (dimY) * (domainSize.x+1) + (dimZ) * (domainSize.x+1) * (domainSize.y+1))*3 + (dimension)]

    T pVal = 6*getPressure(x,y,z) - getPressure(x+1,y,z) - getPressure(x,y+1,z) - getPressure(x,y,z+1) - getPressure(x-1,y,z) - getPressure(x,y-1,z) - getPressure(x,y,z-1);
    T vVal = getVelocity(x+1,y,z,0) - getVelocity(x,y,z,0) + getVelocity(x,y+1,z,1) - getVelocity(x,y,z,1) + getVelocity(x,y,z+1,2) - getVelocity(x,y,z,2);

#undef getPressure
#undef getVelocity
}

extern "C" cudaError_t checkGradientFactorWrapper_float(float* pressure, float* velocity, dim3 domainSize){

    int domainSizeTotal = domainSize.x * domainSize.y * domainSize.z;

    checkGradientFactor<float><<<domainSizeTotal/1024+(domainSizeTotal%1024!=0), 1024>>>(pressure, velocity, domainSize);

    return cudaPeekAtLastError();
}

extern "C" cudaError_t subtractGradientStaggeredCuda_float(float* velocity, float* pressure, float* density, float* obstacles, dim3 velocitySize, float cellDim, float dt){

    int velocitySizeTotal = velocitySize.x * velocitySize.y * velocitySize.z;

    subtractGradientStaggered<float><<<velocitySizeTotal/1024+(velocitySizeTotal%1024!=0), 1024>>>(velocity, pressure, density, obstacles, velocitySize, cellDim, dt);

    return cudaPeekAtLastError();
}

template<typename T>
__global__ void subtractGradientNotStaggered(T* velocity, T* pressure, T* density, T* obstacles, dim3 velocitySize, T cellDim, T dt){

    int num = blockIdx.x * blockDim.x + threadIdx.x;

    int x = num % velocitySize.x;
    int y =(num / velocitySize.x) % velocitySize.y;
    int z = num /(velocitySize.x  * velocitySize.y);

    if(x>= velocitySize.x || y>= velocitySize.y || z>=velocitySize.z)
        return;

    //in the torch version, the velocities at the boundaries are not changed (the outer pressure is not known, but may be considered constant if needed)
    //might have excessive otherwise

    //if there are no obstacles
    //if there are obstacles, might need setWallBcs, or maybe updatingextern "C" cudaError_t subtractGradientStaggeredCuda_float(float* velocity, float* pressure, float* density, float* obstacles, dim3 velocitySize, float cellDim, float dt){ the obstacles will set the velocities to the correct values

    //check how obstacles are handled
    //leave boundary velocities unchanged

    if(x<velocitySize.x-1 && y<velocitySize.y-1 && z<velocitySize.z-1){
        if(x>0)
            velocity[num*3]   -= (pressure[x+1+ velocitySize.x * y + velocitySize.x * velocitySize.y * z] - pressure[x-1 + velocitySize.x * y + velocitySize.x * velocitySize.y * z]);
        if(y>0)
            velocity[num*3+1] -= (pressure[x + velocitySize.x * y + velocitySize.x * velocitySize.y * z] - pressure[x   + velocitySize.x * (y-1) + velocitySize.x*velocitySize.y*z]);
        if(z>0)
            velocity[num*3+2] -= (pressure[x + velocitySize.x * y + velocitySize.x * velocitySize.y * z] - pressure[x   + velocitySize.x * y + velocitySize.x*velocitySize.y*(z-1)]);
    }
}

extern "C" cudaError_t subtractGradientNotStaggeredCuda_float(float* velocity, float* pressure, float* density, float* obstacles, dim3 velocitySize, float cellDim, float dt){

    int velocitySizeTotal = (velocitySize.x+1)*(velocitySize.y+1)*(velocitySize.z+1);

    subtractGradientNotStaggered<float><<<velocitySizeTotal/1024+(velocitySizeTotal%1024!=0), 1024>>>(velocity, pressure, density, obstacles, velocitySize, cellDim, dt);

    return cudaPeekAtLastError();
}

//this will be a sphere centered at the center of the domain, with applied velocity in the +x direction
//if you want the inflow position to be controllable, it will need to be called on every iteration (or a modified applyInflows needs to be used)
template<typename T>
__global__ void initializeInflowsStaggered(T* inflowVelocity, T* inflowDensity, T* inflowTemperature, dim3 domainSize, dim3 center, int radius, T ambTemp){

    T velValX = 1.0;
    T velValY = 0.0;
    T velValZ = 0.0;

    int num = blockIdx.x * blockDim.x + threadIdx.x;

    int x = num % domainSize.x;
    int y =(num / domainSize.x) % domainSize.y;
    int z = num /(domainSize.x  * domainSize.y);

    if(x >= domainSize.x || y >= domainSize.y || z >= domainSize.z)
        return;

    //given the structure of a MAC grid, the velocity grid used has some elements outside the domain at the faces at domainSize.n
    //these values are not modified in either branch of the following if statement
    //this formulation leads to a race condition, and is the reason why
    if(norm3df(int(center.x)-x,int(center.y)-y,int(center.z)-z)<=radius){
        inflowDensity[num] = 0.1;
        inflowTemperature[num] = ambTemp + 1.0;
        inflowVelocity[(x +   (domainSize.x+1) * y   + (domainSize.x+1) * (domainSize.y+1) * z)  *3  ] = velValX;//1.0; //(x,y,z).x
        inflowVelocity[(x +   (domainSize.x+1) * y   + (domainSize.x+1) * (domainSize.y+1) * z)  *3+1] = velValY;//1.0; //(x,y,z).y
        inflowVelocity[(x +   (domainSize.x+1) * y   + (domainSize.x+1) * (domainSize.y+1) * z)  *3+2] = velValZ;//1.0; //(x,y,z).z
        inflowVelocity[((x+1)+(domainSize.x+1) * y   + (domainSize.x+1) * (domainSize.y+1) * z)  *3  ] = velValX;//1.0; //(x+1,y,z).x
        inflowVelocity[(x +   (domainSize.x+1)*(y+1) + (domainSize.x+1) * (domainSize.y+1) * z)  *3+1] = velValY;//1.0; //(x,y+1,z).y
        inflowVelocity[(x +   (domainSize.x+1) * y   + (domainSize.x+1) * (domainSize.y+1)*(z+1))*3+2] = velValZ;//1.0; //(x,y,z+1).z
    }
}

extern "C" cudaError_t initializeInflowsStaggered_float(float* inflowVelocity, float* inflowDensity, float *inflowTemperature, dim3 domainSize, dim3 center, int radius, float ambTemp){

    int velocitySizeTotal = (domainSize.x+1) * (domainSize.y+1) * (domainSize.z+1);

    initializeInflowsStaggered<float><<<velocitySizeTotal/1024+(velocitySizeTotal%1024!=0), 1024>>>(inflowVelocity, inflowDensity, inflowTemperature, domainSize, center, radius, ambTemp);

    return cudaPeekAtLastError();
}


template<typename T>
__global__ void initializeInflowsNotStaggered(T* inflowVelocity, T* inflowDensity, T* inflowTemperature, dim3 domainSize, dim3 center, int radius, T ambTemp){

    T velVal = 10.0f;
    int num = blockIdx.x * blockDim.x + threadIdx.x;

    int x = num % domainSize.x;
    int y =(num / domainSize.x) % domainSize.y;
    int z = num /(domainSize.x  * domainSize.y);

    if(x >= domainSize.x || y >= domainSize.y || z >= domainSize.z)
        return;

    if(norm3df(int(center.x)-x,int(center.y)-y,int(center.z)-z)<=radius){ //TODO: replace center with int3 if you can to avoid casting
        inflowDensity[num] = 1.0;
        inflowTemperature[num] = ambTemp + 10.0;
        inflowVelocity[(x +   (domainSize.x+1) * y   + (domainSize.x+1) * (domainSize.y+1) * z)  *3  ] = velVal;
        inflowVelocity[(x +   (domainSize.x+1) * y   + (domainSize.x+1) * (domainSize.y+1) * z)  *3+1] = velVal;
        inflowVelocity[(x +   (domainSize.x+1) * y   + (domainSize.x+1) * (domainSize.y+1) * z)  *3+2] = velVal;
    }

}

extern "C" cudaError_t initializeInflowsNotStaggered_float(float* inflowVelocity, float* inflowDensity, float* inflowTemperature, dim3 domainSize, dim3 center, int radius, float ambTemp){

    int velocitySizeTotal = domainSize.x * domainSize.y * domainSize.z;

    initializeInflowsNotStaggered<float><<<velocitySizeTotal/1024+(velocitySizeTotal%1024!=0), 1024>>>(inflowVelocity, inflowDensity, inflowTemperature, domainSize, center, radius, ambTemp);

    return cudaPeekAtLastError();
}

template<typename T>
__global__ void applyInflowsStaggered(T* velocityIn, cudaSurfaceObject_t densityPingSurf, T* obstacles, T* temperature, T* inflowVelocity, T* inflowDensity, T* inflowTemperature, dim3 domainSize, T dt);

template<>
__global__ void applyInflowsStaggered<float>(float* velocityIn, cudaSurfaceObject_t densityPingSurf, float* obstacles, float* temperature, float* inflowVelocity, float* inflowDensity, float* inflowTemperature, dim3 domainSize, float dt){

    //this will be a central sphere
    int num = blockIdx.x * blockDim.x + threadIdx.x;

    int x = num % (domainSize.x+1);
    int y =(num / (domainSize.x+1)) % (domainSize.y+1);
    int z = num /((domainSize.x+1)  * (domainSize.y+1));

    if(x > domainSize.x || y > domainSize.y || z > domainSize.z) //if it is
        return;

    if(x < domainSize.x && y < domainSize.y && z < domainSize.z){
        surf3Dwrite(max(0,(surf3Dread<float>(densityPingSurf,x*sizeof(float),y,z)+inflowDensity[x + y * domainSize.x + z * domainSize.x * domainSize.y]*dt)),densityPingSurf,x*sizeof(float),y,z);
        temperature[x + domainSize.x * y + domainSize.x * domainSize.y * z] = inflowTemperature[x + domainSize.x * y + domainSize.x * domainSize.y * z];}

    //in this current form, we cannot have a zero-velocity speed inflow or outflow; use a tiny but nonzero velocity
    //if you don't like that, work out a condition dependant on nonzero density at this point, but mind the staggered grid
    if(inflowVelocity[num * 3] != 0 ||
       inflowVelocity[num * 3 + 1] != 0 ||
       inflowVelocity[num * 3 + 2] != 0){

        velocityIn[num*3]   = inflowVelocity[num*3];  //velocity in inflows/outflows is overwritten, not added
        velocityIn[num*3+1] = inflowVelocity[num*3+1];
        velocityIn[num*3+2] = inflowVelocity[num*3+2];}
}

extern "C" cudaError_t applyInflowsStaggered_float(float* velocityIn, cudaSurfaceObject_t densityPingSurf, float* obstacles, float* temperature, float* inflowVelocity, float* inflowDensity, float* inflowTemperature, dim3 domainSize, float dt){

    int velocitySizeTotal = (domainSize.x+1) * (domainSize.y+1) * (domainSize.z+1);

    applyInflowsStaggered<float><<<velocitySizeTotal/1024+(velocitySizeTotal%1024!=0), 1024>>>(velocityIn, densityPingSurf, obstacles, temperature, inflowVelocity, inflowDensity, inflowTemperature, domainSize, dt);

    return cudaPeekAtLastError();
}

template<typename T, typename T3>
__global__ void applyControlledInflowsStaggered(T* velocityIn, cudaSurfaceObject_t densityPingSurf, T* obstacles, int posX, int posY, int posZ, int radius, T densityValue, T temperatureValue, T3 velocityValue, T* temperature, T* inflowVelocity, T* inflowDensity, T* inflowTemperature, dim3 domainSize, T dt){

    //this will be a central sphere
    int num = blockIdx.x * blockDim.x + threadIdx.x;

    int x = num % (domainSize.x+1);
    int y =(num / (domainSize.x+1)) % (domainSize.y+1);
    int z = num /((domainSize.x+1)  * (domainSize.y+1));

    if(x > domainSize.x || y > domainSize.y || z > domainSize.z) //if it is
        return;

    if(x < domainSize.x && y < domainSize.y && z < domainSize.z){
        surf3Dwrite(max(0,(surf3Dread<T>(densityPingSurf,x*sizeof(T),y,z)+inflowDensity[x + y * domainSize.x + z * domainSize.x * domainSize.y]*dt)),densityPingSurf,x*sizeof(T),y,z);
        temperature[x + domainSize.x * y + domainSize.x * domainSize.y * z] = inflowTemperature[x + domainSize.x * y + domainSize.x * domainSize.y * z];}

    //in this current form, we cannot have a zero-velocity speed inflow or outflow; use a tiny but nonzero velocity
    //if you don't like that, work out a condition dependant on nonzero density at this point, but mind the staggered grid
    if(inflowVelocity[num * 3] != 0 ||
       inflowVelocity[num * 3 + 1] != 0 ||
       inflowVelocity[num * 3 + 2] != 0){

        velocityIn[num*3]   = inflowVelocity[num*3];  //velocity in inflows/outflows is overwritten, not added
        velocityIn[num*3+1] = inflowVelocity[num*3+1];
        velocityIn[num*3+2] = inflowVelocity[num*3+2];}

    if(abs(x-posX) < radius && abs(y-posY) < radius && abs(z-posZ) < radius){
        velocityIn[num * 3]     += velocityValue.x;
        velocityIn[num * 3 + 1] += velocityValue.y;
        velocityIn[num * 3 + 2] += velocityValue.z;
        temperature[x + domainSize.x * y + domainSize.x * domainSize.y * z] += temperatureValue;
        surf3Dwrite(max(0, surf3Dread<T>(densityPingSurf,x*sizeof(T),y,z) + densityValue), densityPingSurf, x*sizeof(T), y, z);}
}

//this is the same as applyInflowsStaggered, except that it allows a spherical user-controlled density inflow of position (posX, posY), radius radius and constant density, velocity and temperature inflow _in addition_ to the other inflows
extern "C" cudaError_t applyControlledInflowsStaggered_float(float* velocityIn, cudaSurfaceObject_t densityPingSurf, float* obstacles, int posX, int posY, int posZ, int radius, float densityValue, float temperatureValue, float3 velocityValue, float* temperature, float* inflowVelocity, float* inflowDensity, float* inflowTemperature, dim3 domainSize, float dt){

    int velocitySizeTotal = (domainSize.x+1) * (domainSize.y+1) * (domainSize.z+1);

    applyControlledInflowsStaggered<float, float3><<<velocitySizeTotal/1024+(velocitySizeTotal%1024!=0), 1024>>>(velocityIn, densityPingSurf, obstacles, posX, posY, posZ, radius, densityValue, temperatureValue, velocityValue, temperature, inflowVelocity, inflowDensity, inflowTemperature, domainSize, dt);

    return cudaPeekAtLastError();
}

template<typename T>
__global__ void applyInflowsNotStaggered(T* velocityIn, cudaSurfaceObject_t densityPingSurf, T* obstacles, T* temperature, T* inflowVelocity, T* inflowDensity, T* inflowTemperature, dim3 domainSize, T dt);

template<>
__global__ void applyInflowsNotStaggered<float>(float* velocityIn, cudaSurfaceObject_t densityPingSurf, float* obstacles, float* temperature, float* inflowVelocity, float* inflowDensity, float* inflowTemperature, dim3 domainSize, float dt){

    //this will be a central sphere
    int num = blockIdx.x * blockDim.x + threadIdx.x;

    int x = num % domainSize.x;
    int y =(num / domainSize.x) % domainSize.y;
    int z = num /(domainSize.x  * domainSize.y);

    if(x >= domainSize.x || y >= domainSize.y || z >= domainSize.z)
        return;

    //this is for inflow initialization, move it where it needs to be
    //do NOT change occupied cells
    surf3Dwrite(max(0,(surf3Dread<float>(densityPingSurf,x*sizeof(float),y,z)*(!obstacles[num])+inflowDensity[num])*dt),densityPingSurf,x*sizeof(float),y,z);
    velocityIn[num*3]   += inflowVelocity[num*3];
    velocityIn[num*3+1] += inflowVelocity[num*3+1];
    velocityIn[num*3+2] += inflowVelocity[num*3+2];

    temperature[x + domainSize.x * y + domainSize.x * domainSize.y * z] = inflowTemperature[x + domainSize.x * y + domainSize.x * domainSize.y * z];
}

extern "C" cudaError_t applyInflowsNotStaggered_float(float* velocityIn, cudaSurfaceObject_t densityPingSurf, float* obstacles, float* temperature, float* inflowVelocity, float* inflowDensity, float* inflowTemperature, dim3 domainSize, float dt){

    int domainSizeTotal = domainSize.x * domainSize.y * domainSize.z;

    applyInflowsNotStaggered<float><<<domainSizeTotal/1024+(domainSizeTotal%1024!=0), 1024>>>(velocityIn, densityPingSurf, obstacles, temperature, inflowVelocity, inflowDensity, inflowTemperature, domainSize, dt);

    return cudaPeekAtLastError();
}
#undef min
#undef max

#endif
