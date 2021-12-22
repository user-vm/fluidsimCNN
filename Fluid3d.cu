  //Modified from the Jacobi program in Fluid.glsl, in the fluidsim project by Philip Rideout

#ifndef _FLUID_3D_CU_
#define _FLUID_3D_CU_

#include <cuda_runtime_api.h>
#include <math.h>
#include <cuda_fp16.h>
#include <helper_cuda.h>
#include <cudnn.h>
#include <cublas.h>

//try this later, using "c - How to use extern cuda device variables" stackoverflow bookmark
//extern int GridWidth, GridHeight, GridDepth;

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

    //this squashes double values; will probably need template specialization if double is ever used. But this function will probably be useless by then.
    velocity[num*3  ] = fmodf(velocityValX,1.0);// - int(velocityValX);
    velocity[num*3+1] = fmodf(velocityValY,1.0);// - int(velocityValY);
    velocity[num*3+2] = fmodf(velocityValZ,1.0);// - int(velocityValZ);

    if(x >= domainSize.x || y>=domainSize.y || z>=domainSize.z)
        return;

    T pressureVal = sqrt(3.1*(x+1)*(x+1) + (y+1)*(y+1) + 2.5*(z+1)*(z+1));

    pressure[x + domainSize.x*y + domainSize.x* domainSize.y*z] = fmodf(pressureVal,1.0);// - int(pressureVal);
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
                /*
                if((buffer1[num*3] !=0 || buffer1[num*3+1] !=0 || buffer1[num*3+2] !=0 ||
                    buffer2[num*3] !=0 || buffer2[num*3+1] !=0 || buffer2[num*3+2] !=0) &&
                  (buffer1[num*3] != buffer2[num*3] || buffer1[num*3+1] != buffer2[num*3+1] || buffer1[num*3+2] != buffer2[num*3+2]))*/
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
        oN = getObstacles(  x,y+1,  z);//tex3D<float>(obstaclesTex,x,y+1,z);
    else
        oN = 0;

    if(y>0)
        oS = getObstacles(  x,y-1,  z);//tex3D<float>(obstaclesTex,x,y-1,z);
    else
        oS = 0;

    if(x<domainSize.x)
        oE = getObstacles(x+1,  y,  z);//tex3D<float>(obstaclesTex,x+1,y,z);
    else
        oE = 0;

    if(x>0)
        oW = getObstacles(x-1,  y,  z);//tex3D<float>(obstaclesTex,x-1,y,z);
    else
        oW = 0;

    if(z<domainSize.z)
        oU = getObstacles(  x,  y,z+1);//tex3D<float>(obstaclesTex,x,y,z+1);
    else
        oU = 0;

    if(z>0)
        oD = getObstacles(  x,  y,z-1);//tex3D<float>(obstaclesTex,x,y,z-1);
    else
        oD = 0;

    pC = pressure[num];

    if (oN > 0) pN = pC; else if(y==domainSize.y-1) pN = 0; else pN = getPressure(  x,y+1,  z); //surf3Dread(&pN, pressurePingSurf, x   *sizeof(float), y+1, z  );
    if (oS > 0) pS = pC; else if(y==0) pS = 0; else pS = getPressure(  x,y-1,  z); //surf3Dread(&pS, pressurePingSurf, x   *sizeof(float), y-1, z  );
    if (oE > 0) pE = pC; else if(x==domainSize.x-1) pE = 0; else pE = getPressure(x+1,  y,  z);//surf3Dread(&pE, pressurePingSurf,(x+1)*sizeof(float), y  , z  );
    if (oW > 0) pW = pC; else if(x==0) pW = 0; else pW = getPressure(x-1,  y,  z); //surf3Dread(&pW, pressurePingSurf,(x-1)*sizeof(float), y  , z  );
    if (oU > 0) pU = pC; else if(z==domainSize.z-1) pU = 0; else pU = getPressure(  x,  y,z+1); //surf3Dread(&pU, pressurePingSurf, x   *sizeof(float), y  , z+1);
    if (oD > 0) pD = pC; else if(z==0) pD = 0; else pD = getPressure(  x,  y,z-1); //surf3Dread(&pD, pressurePingSurf, x   *sizeof(float), y  , z-1);

    float bC = divergence[num];//tex3D<float>(divergenceTex, x,y,z);

    //float oC = getObstacles(x,y,z);

    pressure[num] = (pW + pE + pS + pN + pU + pD + alpha * bC) * inverseBeta;

    //surf3Dwrite(temp, pressurePongSurf,x*sizeof(float),y,z);
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
/*
extern "C" cudaError_t
subtractGradientJacobiStaggered(){


}

extern "C" cudaError_t
subtractGradientJacobiNotStaggered(){

    ivec3 T = ivec3(gl_FragCoord.xy, gLayer);

    vec3 oC = texelFetch(Obstacles, T, 0).xyz;
    if (oC.x > 0) {
        FragColor = oC.yzx;
        return;
    }

    // Find neighboring pressure:
    float pN = ;//texelFetchOffset(Pressure, T, 0, ivec3(0, 1, 0)).r;
    float pS = ;//texelFetchOffset(Pressure, T, 0, ivec3(0, -1, 0)).r;
    float pE = ;//texelFetchOffset(Pressure, T, 0, ivec3(1, 0, 0)).r;
    float pW = ;//texelFetchOffset(Pressure, T, 0, ivec3(-1, 0, 0)).r;
    float pU = ;//texelFetchOffset(Pressure, T, 0, ivec3(0, 0, 1)).r;
    float pD = ;//texelFetchOffset(Pressure, T, 0, ivec3(0, 0, -1)).r;
    float pC = ;//texelFetch(Pressure, T, 0).r;

    // Find neighboring obstacles:
    vec3 oN = texelFetchOffset(Obstacles, T, 0, ivec3(0, 1, 0)).xyz;
    vec3 oS = texelFetchOffset(Obstacles, T, 0, ivec3(0, -1, 0)).xyz;
    vec3 oE = texelFetchOffset(Obstacles, T, 0, ivec3(1, 0, 0)).xyz;
    vec3 oW = texelFetchOffset(Obstacles, T, 0, ivec3(-1, 0, 0)).xyz;
    vec3 oU = texelFetchOffset(Obstacles, T, 0, ivec3(0, 0, 1)).xyz;
    vec3 oD = texelFetchOffset(Obstacles, T, 0, ivec3(0, 0, -1)).xyz;

    // Use center pressure for solid cells:
    vec3 obstV = vec3(0);
    vec3 vMask = vec3(1);

    if (oN.x > 0) { pN = pC; obstV.y = oN.z; vMask.y = 0; }
    if (oS.x > 0) { pS = pC; obstV.y = oS.z; vMask.y = 0; }
    if (oE.x > 0) { pE = pC; obstV.x = oE.y; vMask.x = 0; }
    if (oW.x > 0) { pW = pC; obstV.x = oW.y; vMask.x = 0; }
    if (oU.x > 0) { pU = pC; obstV.z = oU.x; vMask.z = 0; }
    if (oD.x > 0) { pD = pC; obstV.z = oD.x; vMask.z = 0; }

    // Enforce the free-slip boundary condition:
    vec3 oldV = texelFetch(Velocity, T, 0).xyz;
    vec3 grad = vec3(pE - pW, pN - pS, pU - pD) * GradientScale;
    vec3 newV = oldV - grad;
    FragColor = (vMask * newV) + obstV; //newV;
}
*/

/*
Jacobi_CUDA(cudaSurfaceObject_t pressurePingSurf,
            cudaSurfaceObject_t pressurePongSurf,
            cudaTextureObject_t divergenceTex,
            cudaTextureObject_t obstaclesTex,
            dim3 textureDims,
            float alpha, float inverseBeta, int numLoops, int currentLoop){
*/

__global__ void makePressure(cudaSurfaceObject_t pressurePingSurf,
                             cudaSurfaceObject_t pressurePongSurf,
                             cudaTextureObject_t divergenceTex,
                             cudaTextureObject_t obstaclesTex,
                             dim3 textureDims,
                             float alpha, float inverseBeta){

  //SOME THINGS TO REMEMBER:
  //
  //Stackoverflow copypasta:
  //  Texture references are read-only. If you write to a CUDA array using a surface reference,
  //  that memory traffic goes through the L1/L2 cache hierarchy and is not coherent with the texture cache.

  int x = blockIdx.x*blockDim.x+threadIdx.x;
  int y = blockIdx.y*blockDim.y+threadIdx.y;
  int z = blockIdx.z*blockDim.z+threadIdx.z;

  if(x >= textureDims.x || y >= textureDims.y || z >= textureDims.z)
    return;
/*
  //just to attempt to avoid segfault
  if(x == textureDims.x-1 || y == textureDims.y-1 || z == textureDims.z-1)
    return;
*/
/*
  if(x == 0 || y == 0 || z == 0)
    return;
*/

  //half pN = tex1D(pressurePing,textureDims.x * (y + textureDims.y * z) + (x + 1));

  //if you try automatic interpolation, it probably won't work right if you use 1D textures instead of 3D textures (linear instead of trilinear)
  //these all used to be half
  //this  probably segfaults at the texture edges (the edges are obstacles, so it doesn't matter for now)
  /*
  float pC = tex3D<float>(pressurePingTex,x,y,z);
  float pN = tex3D<float>(pressurePingTex,x,y+1,z);
  float pS = tex3D<float>(pressurePingTex,x,y-1,z);
  float pE = tex3D<float>(pressurePingTex,x+1,y,z);
  float pW = tex3D<float>(pressurePingTex,x-1,y,z);
  float pU = tex3D<float>(pressurePingTex,x,y,z+1);
  float pD = tex3D<float>(pressurePingTex,x,y,z-1);
  */

  float pC;
  surf3Dread(&pC, pressurePingSurf, x   *sizeof(float), y  , z  );

  //printf("%d %d %d\n", int(x),int(y),int(z));

  //half oC = tex1D(obstaclesTex,textureDims.x * ( y    + textureDims.y *  z   ) +  x   );
  //these all used to be half
  float oN = tex3D<float>(obstaclesTex,x,y+1,z);
  float oS = tex3D<float>(obstaclesTex,x,y-1,z);
  float oE = tex3D<float>(obstaclesTex,x+1,y,z);
  float oW = tex3D<float>(obstaclesTex,x-1,y,z);
  float oU = tex3D<float>(obstaclesTex,x,y,z+1);
  float oD = tex3D<float>(obstaclesTex,x,y,z-1);

  //printf("oN = %f; oS = %f; oE = %f; oW = %f; oU = %f; oD = %f\n",oN,oS,oE,oW,oU,oD);

  float pN, pS, pE, pW, pU, pD;

  //printf("textureDims = ( %d, %d, %d )", int(textureDims.x), int(textureDims.y), int(textureDims.z));

  //since all the edges are obstacles, this should work
  if (oN > 0) pN = pC; else if(y==textureDims.y-1) pN = 0; else surf3Dread(&pN, pressurePingSurf, x   *sizeof(float), y+1, z  );
  if (oS > 0) pS = pC; else if(y==0) pS = 0; else surf3Dread(&pS, pressurePingSurf, x   *sizeof(float), y-1, z  );
  if (oE > 0) pE = pC; else if(x==textureDims.x-1) pE = 0; else surf3Dread(&pE, pressurePingSurf,(x+1)*sizeof(float), y  , z  );
  if (oW > 0) pW = pC; else if(x==0) pW = 0; else surf3Dread(&pW, pressurePingSurf,(x-1)*sizeof(float), y  , z  );
  if (oU > 0) pU = pC; else if(z==textureDims.z-1) pU = 0; else surf3Dread(&pU, pressurePingSurf, x   *sizeof(float), y  , z+1);
  if (oD > 0) pD = pC; else if(z==0) pD = 0; else surf3Dread(&pD, pressurePingSurf, x   *sizeof(float), y  , z-1);

  //pN = pC;
  //pE = pC;
  //pU = pC;

  //bC is the divergence at this point
  float bC = tex3D<float>(divergenceTex, x,y,z);
/*
  if(bC > 0)
    printf("divergence[%d][%d][%d] = %f;  ",x,y,z,bC);
*/
/*
  if(pC > 0)
    printf("pressure[%d][%d][%d] = %f;  ",x,y,z,pC);
*/

  //does this have a point?
  float oC = tex3D<float>(obstaclesTex,x,y,z);


  /*
  if(oC > 0 && x>0 && x<63 && y>0 && y<63 && z>0 && z<63 && currentJacobiIteration==0 && currentLoop==0)
    //printf("obstacles[%d][%d][%d] = %f;  ",x,y,z,oC);
    printf("%d %d %d\n",x,y,z);
*/
  //int a = 8;//20;
/*
  if(x%a == 0 && y%a == 0 && z%a == 0){
    printf("pressurePingTex[%d][%d][%d] = %f;  ",x,y,z,pC);
    printf("divergenceTex[%d][%d][%d] = %f\n\n",x,y,z,bC);
    //printf("sizeof(divergenceTex) = %d;  ",int(sizeof(divergenceTex)));
    //printf("sizeof(obstaclesTex) = %d  \n",int(sizeof(obstaclesTex)));
  }
*/
  float temp = (pW + pE + pS + pN + pU + pD + alpha * bC) * inverseBeta;

  surf3Dwrite(temp, pressurePongSurf,x*sizeof(float),y,z);
/*
  if(x%a == 0 && y%a == 0 && z%a == 0){
    printf("pressurePongSurf[%d][%d][%d] = %f;  \n\n",x,y,z,temp);//pressurePongSurf);
    //printf("divergenceTex[%d][%d][%d] = %f;  \n\n",x,y,z,divergenceTex);
  }*/
  //printf("pressurePingTex[%d][%d][%d] = %f",int(x),int(y),int(z),tex3D<float>(pressurePingTex,x,y,z));
/*
  if(tex3D<float>(obstaclesTex,x,y,z))
    printf("obstaclesTex[%d][%d][%d] = %f",int(x),int(y),int(z),tex3D<float>(obstaclesTex,x,y,z));
*/
}

//may need to change cudaTextureObject_t back to texture<float,1,cudaReadModeElementType>

extern "C" cudaError_t
Jacobi_CUDA(cudaSurfaceObject_t pressurePingSurf,
            cudaSurfaceObject_t pressurePongSurf,
            cudaTextureObject_t divergenceTex,
            cudaTextureObject_t obstaclesTex,
            dim3 textureDims,
            float alpha, float inverseBeta, int numLoops, int currentLoop){

  //uint x = textureDims.x;
  //uint y = textureDims.y;
  //uint z = textureDims.z;

  dim3 numBlocks = dim3(NUM_BLOCKS_X,NUM_BLOCKS_Y,NUM_BLOCKS_Z);
  dim3 threadsPerBlock = dim3(THREADSPERBLOCK_X,THREADSPERBLOCK_Y,THREADSPERBLOCK_Z);

  //try cycling the chevron parameters and see if anything speeds up
  int i;

  cudaResourceDesc pPingResDesc, pPongResDesc;

  cudaGetSurfaceObjectResourceDesc(&pPingResDesc, pressurePingSurf);
  cudaGetSurfaceObjectResourceDesc(&pPongResDesc, pressurePongSurf);

  //printf("height = %d, width = %d, pitch = %d\n", pPingResDesc.res.pitch2D.height);

  /*
   *
   *
cudaArray_t array
struct cudaChannelFormatDesc desc
void * devPtr
size_t  height
cudaMipmappedArray_t mipmap
size_t  pitchInBytes
enumcudaResourceType resType
size_t  sizeInBytes
size_t  width
*/

  //numLoops is even, so we can disconsider the swap FOR NOW
  for(i=0;i<numLoops;i++){
    if(i%2==0)
      makePressure<<<numBlocks, threadsPerBlock>>>(pressurePingSurf, pressurePongSurf,
                                                   divergenceTex, obstaclesTex,
                                                   textureDims, alpha, inverseBeta);//, currentLoop, i);
    else
      makePressure<<<numBlocks, threadsPerBlock>>>(pressurePongSurf, pressurePingSurf,
                                                   divergenceTex, obstaclesTex,
                                                   textureDims, alpha, inverseBeta);//, currentLoop, i);

    //MIGHT NEED TO UNCOMMENT THIS
    //cudaDeviceSynchronize();
    }

  /*
  makePressure<<<WIDTH.HEIGHT,DEPTH>>>(pressurePingArray,pressurePingDims,
                                       pressurePongArray,pressurePongDims,
                                       divergenceArray,  divergenceDims,
                                       obstaclesArray,   obstaclesDims,
                                       alpha, inverseBeta);

  */

  return cudaPeekAtLastError();
}

/* The original Jacobi program from Fluid.glsl
-- Jacobi

out vec4 FragColor;

uniform sampler3D Pressure;
uniform sampler3D Divergence;
uniform sampler3D Obstacles;

uniform float Alpha;
uniform float InverseBeta;

in float gLayer;

void main()
{
    ivec3 T = ivec3(gl_FragCoord.xy, gLayer);

    // Find neighboring pressure:
    vec4 pN = texelFetchOffset(Pressure, T, 0, ivec3(0, 1, 0));
    vec4 pS = texelFetchOffset(Pressure, T, 0, ivec3(0, -1, 0));
    vec4 pE = texelFetchOffset(Pressure, T, 0, ivec3(1, 0, 0));
    vec4 pW = texelFetchOffset(Pressure, T, 0, ivec3(-1, 0, 0));
    vec4 pU = texelFetchOffset(Pressure, T, 0, ivec3(0, 0, 1));
    vec4 pD = texelFetchOffset(Pressure, T, 0, ivec3(0, 0, -1));
    vec4 pC = texelFetch(Pressure, T, 0);

    // Find neighboring obstacles:
    vec3 oN = texelFetchOffset(Obstacles, T, 0, ivec3(0, 1, 0)).xyz;
    vec3 oS = texelFetchOffset(Obstacles, T, 0, ivec3(0, -1, 0)).xyz;
    vec3 oE = texelFetchOffset(Obstacles, T, 0, ivec3(1, 0, 0)).xyz;
    vec3 oW = texelFetchOffset(Obstacles, T, 0, ivec3(-1, 0, 0)).xyz;
    vec3 oU = texelFetchOffset(Obstacles, T, 0, ivec3(0, 0, 1)).xyz;
    vec3 oD = texelFetchOffset(Obstacles, T, 0, ivec3(0, 0, -1)).xyz;

    // Use center pressure for solid cells:
    if (oN.x > 0) pN = pC;
    if (oS.x > 0) pS = pC;
    if (oE.x > 0) pE = pC;
    if (oW.x > 0) pW = pC;
    if (oU.x > 0) pU = pC;
    if (oD.x > 0) pD = pC;

    vec4 bC = texelFetch(Divergence, T, 0);
    FragColor = (pW + pE + pS + pN + pU + pD + Alpha * bC) * InverseBeta;
}
*/

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
    //int D = obstacleExtent.z;
    //int H = obstacleExtent.y;
    //int W = obstacleExtent.x;

    //updateObstacles<float><<<D*H*W/1024,1024>>>();
    return cudaPeekAtLastError();
}

#define min(a,b) ((a>b)?(b):(a))
#define max(a,b) ((a<b)?(b):(a))
template<typename T>
__global__ void clampVelocity(T* velocity, dim3 velocitySize, T clampMin, T clampMax){

    int num = blockIdx.x*blockDim.x+threadIdx.x;
/*
    int x = num %(velocitySize.y * velocitySize.z);
    int y =(num % velocitySize.z)/ velocitySize.x;
    int z = num /(velocitySize.x);
*/

    //int x = num % velocitySize.x;
    //int y =(num / velocitySize.x) % velocitySize.y;
    //int z = num /(velocitySize.x  * velocitySize.y);

    velocity[num*3] = max(min(clampMax, velocity[num*3]),clampMin);
    velocity[num*3+1] = max(min(clampMax, velocity[num*3+1]),clampMin);
    velocity[num*3+2] = max(min(clampMax, velocity[num*3+2]),clampMin);
}


extern "C" cudaError_t clampVelocityWrapper_float(float* velocity, dim3 velocitySize, float clampMin, float clampMax){

    int velocitySizeTotal = velocitySize.x*velocitySize.y*velocitySize.z;

    clampVelocity<float><<<velocitySizeTotal/1024+(velocitySizeTotal%1024!=0), 1024>>>(velocity, velocitySize, clampMin, clampMax);

    return cudaPeekAtLastError();
}

//extern "C" cudaError_t addBuoyancyStaggeredWrapper_float(float* velocity, cudaSurfaceObject_t densityPongSurf, float* temperature, dim3 domainSize, float buoyancyConstant);
//extern "C" cudaError_t addBuoyancyNotStaggeredWrapper_float(float* velocity, float* density, float* temperature, dim3 domainSize, float buoyancyConstant);
//extern "C" cudaError_t advectVelocityCudaStaggeredWrapper_float(float* velocityIn, float* velocityOut, float* obstacles, dim3 domainSize);
extern "C" cudaError_t advectVelocityCudaNotStaggeredWrapper_float(float* velocityIn, float* velocityOut, float* obstacles, float cellDim, dim3 domainSize, float dt);
//extern "C" cudaError_t advectCudaStaggeredWrapper_float(float* velocityIn, float* scalarFieldIn, float* scalarFieldOut, float dt);
extern "C" cudaError_t advectCudaNotStaggeredWrapper_float(float* velocityIn, float* scalarField, float* scalarFieldOut, float dt){ return cudaPeekAtLastError();}
//extern "C" cudaError_t advectDensityCudaStaggeredWrapper_float(float* velocityIn, cudaTextureObject_t densityPingTex, cudaSurfaceObject_t densityPongSurf, float *obstacles, float cellDim, float dt);
//extern "C" cudaError_t advectDensityCudaNotStaggeredWrapper_float(float* velocityIn, cudaTextureObject_t densityPingTex, cudaSurfaceObject_t densityPongSurf, float dt);
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
                //if(divergence[i + j*domainSize.y + k*domainSize.z*domainSize.y] !=0 )
                //    printf("divergence[%d,%d,%d] = %f\n", i, j, k, divergence[i + j*domainSize.y + k*domainSize.z*domainSize.y]);
            }

    printf("\nSum of squared divergences = %f\n\n", total);
}

extern "C" cudaError_t checkDivergenceWrapper_float(float* divergence, dim3 domainSize){

    checkDivergence<float><<<1,1>>>(divergence, domainSize);
    //printf("Exiting... (Fluid3d.cu, chekcDivergenceWrapper_float)");
    //exit(0);
    return cudaPeekAtLastError();
}

//this needs addGravityStaggered and addGravityNotStaggered if using the obstacle mask when adding gravity
//don't do that for now
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

//LEAVE THIS ALONE UNTIL YOU'RE SURE YOU NEED IT
/*
template<typename T>
__global__ void addGravityStaggered(T* velocity, T* obstacles, dim3 velocitySize, float gravity[3]){

    int num = blockIdx.x*blockDim.x+threadIdx.x;

    if(obstacles[num]==0 || obstacles[num+1]==0)
        velocity = ;
}

extern "C" cudaError_t addGravityStaggeredWrapper_float(float* velocity, dim3 velocitySize, float3 gravity){

    float g[3];
    int velocitySizeTotal = velocitySize.x*velocitySize.y*velocitySize.z;
    addGravityStaggered<float><<<velocitySizeTotal/1024+(velocitySizeTotal%1024!=0), 1024>>>(velocity, velocitySize, g);
    return cudaPeekAtLastError();
}
*/

//this is the triliniear interpolation function
//unmasked textures don't need an interpolation function
//does it need __forceinline__?
//does it need to check for out-of-bounds?
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
//the gridGet thing causes repeated multiplications that would otherwise be avoided
template<typename T>
__device__ void trilerpInverseMasked(T* grid, T* inverseMask, float3 index, dim3 gridSize, T* result){

    //T* grid, float3 index, dim3 gridSize, T* result)

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

/*
//this only interpolates between values with inverseMask==0 (basically for ignoring occupied cells)
//what about out-of-bounds?
template<typename T>
__device__ void trilerpTexInverseMasked(cudaTextureObject_t grid, T* inverseMask, float3 index, T* result){
extern "C" cudaError_t addGravityWrapper_float(float* velocity, dim3 velocitySize, float3 gravity){

    float g[3];
    int velocitySizeTotal = velocitySize.x*velocitySize.y*velocitySize.z;
    addGravity<float><<<velocitySizeTotal/1024+(velocitySizeTotal%1024!=0), 1024>>>(velocity, velocitySize, g);
    return cudaPeekAtLastError();
}
    //floor is double, indices are float for textures (interoplation factor or whatever it's called is 9-bit)
    float3 indexFloor(float(floor(index.x)), float(floor(index.y)), float(floor(index.z)));

    T c00 = tex3D<T>(grid, indexFloor.x,  indexFloor.y,  indexFloor.z  ) * isNotObstacle(indexFloor.x,  indexFloor.y,  indexFloor.z  ) * (1.0f - xd)
          + tex3D<T>(grid, indexFloor.x+1,indexFloor.y,  indexFloor.z  ) * isNotObstacle(indexFloor.x+1,indexFloor.y,  indexFloor.z  ) * xd;

    T c01 = tex3D<T>(grid, indexFloor.x,  indexFloor.y+1,indexFloor.z  ) * isNotObstacle(indexFloor.x,  indexFloor.y+1,indexFloor.z  ) * (1.0f - xd)
          + tex3D<T>(grid, indexFloor.x+1,indexFloor.y+1,indexFloor.z  ) * isNotObstacle(indexFloor.x+1,indexFloor.y+1,indexFloor.z  ) * xd;

    T c10 = tex3D<T>(grid, indexFloor.x,  indexFloor.y,  indexFloor.z+1) * isNotObstacle(indexFloor.x,  indexFloor.y,  indexFloor.z+1) * (1.0f - xd)
          + tex3D<T>(grid, indexFloor.x+1,indexFloor.y,  indexFloor.z+1) * isNotObstacle(indexFloor.x+1,indexFloor.y,  indexFloor.z+1) * xd;

    T c11 = tex3D<T>(grid, indexFloor.x,  indexFloor.y+1,indexFloor.z+1) * isNotObstacle(indexFloor.x,  indexFloor.y+1,indexFloor.z+1) * (1.0f - xd)
          + tex3D<T>(grid, indexFloor.x+1,indexFloor.y+1,indexFloor.z+1) * isNotObstacle(indexFloor.x+1,indexFloor.y+1,indexFloor.z+1) * xd;

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
*/

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

    //need to fix this somewhere else (subtracting ambTemp before multiplying by beta
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

    //int numTriesInitial = numTries;
/*
    if(x == 32 && y == 32 && z == 32)
        printf("cellDim = %f\n", cellDim);
    return;
*/
    //needs to do the out-of-bounds thing, but not any obstacle things
    //still uses numTries, although that might be silly here
    do{
        numTries--;

        //the LHS variables correspond to the velocity grid
        xVAdv = currentPoint.x - currentPointVelocity.x * subDt / cellDim;
        yVAdv = currentPoint.y - currentPointVelocity.y * subDt / cellDim;
        zVAdv = currentPoint.z - currentPointVelocity.z * subDt / cellDim;
/*
        if(x == 16 && y == 16 && z == 16)
            printf("(%d, %d, %d) %f %f %f\n", x, y, z, xVAdv, yVAdv, zVAdv);
*/
/*
        if(x == 32 && y == 32 && z == 32)
            printf("currentPoint = (%f,%f,%f); currentPointVelocity = (%f,%f,%f); velocity(%d,%d,%d) = %f; cellDim = %f\n", currentPoint.x, currentPoint.y, currentPoint.z, currentPointVelocity.x, currentPointVelocity.y, currentPointVelocity.z, x, y, z, velocityIn, cellDim);
*/
        if(xVAdv>=velocitySize.x || xVAdv<0 || yVAdv>=velocitySize.y || yVAdv<0 || zVAdv>=velocitySize.z || zVAdv<0){
/*
            if(numTries == numTriesInitial-1 && x == 32 && y == 32 && z == 32)
                printf("ADVVEL, (%d, %d, %d)\n", x, y, z);
*/
            //THIS
            //also need to update subDt
            //do not need to handle zero velocities (if all three are zero, cannot have out-of-bounds condition, if one is nonzero, it will determine maxZ, and the other three will stay the same because their currentPointVelocity.n is zero)
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
            xVAdv2 = currentPoint.x + maxZ * currentPointVelocity.x; //* subDt / cellDim;
            yVAdv2 = currentPoint.y + maxZ * currentPointVelocity.y; //* subDt / cellDim;
            zVAdv2 = currentPoint.z + maxZ * currentPointVelocity.z; //* subDt / cellDim;

            xVAdv2 = max(0,min(xVAdv2, velocitySize.x-1));
            yVAdv2 = max(0,min(yVAdv2, velocitySize.y-1));
            zVAdv2 = max(0,min(zVAdv2, velocitySize.z-1));

            subDt -= subDt * fabs((xVAdv2-currentPoint.x + yVAdv2-currentPoint.y + zVAdv2-currentPoint.z) / (xVAdv - currentPoint.x + yVAdv - currentPoint.y + zVAdv - currentPoint.z)); //the fabs should have no effect unless algorithm is incorrect
            xVAdv = xVAdv2;
            yVAdv = yVAdv2;
            zVAdv = zVAdv2;

            //if we are still at the same point, which is what happens if we are at an edge and are trying to advect through that same edge, or if velocity is zero here, then use this point and exit
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
/*
        if(float(int(xVAdv))!=floorf(xVAdv))
            printf("INT CASTING FAILED");
*/
        currentPointVelocity.x = velocityIn[(int(xVAdv)  + velocitySize.x * int(yVAdv)  + velocitySize.y * velocitySize.x *   int(zVAdv))*3] * (floor(xVAdv) + 1 - xVAdv)
                               + velocityIn[(max(int(xVAdv+1), velocitySize.x-1) + velocitySize.x * int(yVAdv)  + velocitySize.y * velocitySize.x *   int(zVAdv))*3] * (xVAdv - floor(xVAdv));//(xVAdv+1, yVAdv, zVAdv).x * (xVAdv - floor(xVAdv));

        currentPointVelocity.y = velocityIn[(int(xVAdv)  + velocitySize.x * int(yVAdv)  + velocitySize.y * velocitySize.x *   int(zVAdv))*3+1] * (floor(yVAdv) + 1 - yVAdv)
                               + velocityIn[(int(xVAdv)  + velocitySize.x * max(int(yVAdv+1), velocitySize.y-1) + velocitySize.y * velocitySize.x *   int(zVAdv))*3+1] * (yVAdv - floor(yVAdv));

        currentPointVelocity.z = velocityIn[(int(xVAdv)  + velocitySize.x * int(yVAdv)  + velocitySize.y * velocitySize.x *   int(zVAdv))*3+2] * (floor(zVAdv) + 1 - zVAdv)
                               + velocityIn[(int(xVAdv)  + velocitySize.x * int(yVAdv)  + velocitySize.y * velocitySize.x * max(int(zVAdv+1), velocitySize.z-1))*3+2] * (zVAdv - floor(zVAdv));
/*
        if(x%10 == 0 && y%10 == 0 && z%10==0 && (velocityIn[(x + velocitySize.x * y + velocitySize.x * velocitySize.y * z) *3]!=0 && velocityIn[(x + velocitySize.x * y + velocitySize.x * velocitySize.y * z) *3 + 1] && velocityIn[(x + velocitySize.x * y + velocitySize.x * velocitySize.y * z) *3 + 2]))
            printf("x = %d; y = %d; z = %d; velocity(x,y,z) = (%f,%f,%f); xVAdv = %f; yVAdv = %f; zVAdv = %f;\n floor(xVAdv) + 1 - xVAdv = %f; xVAdv - floor(xVAdv) = %f; floor(yVAdv) + 1 - yVAdv = %f; yVAdv - floor(yVAdv) = %f; floor(zVAdv) + 1 - zVAdv = %f; zVAdv - floor(zVAdv) = %f; dt = %f\n\n",
                   x, y, z, velocityIn[(x + velocitySize.x * y + velocitySize.x * velocitySize.y * z) *3], velocityIn[(x + velocitySize.x * y + velocitySize.x * velocitySize.y * z) *3 + 1], velocityIn[(x + velocitySize.x * y + velocitySize.x * velocitySize.y * z) *3 + 2],
                   xVAdv, yVAdv, zVAdv, floor(xVAdv) + 1 - xVAdv, xVAdv - floor(xVAdv), floor(yVAdv) + 1 - yVAdv, yVAdv - floor(yVAdv), floor(zVAdv) + 1 - zVAdv, zVAdv - floor(zVAdv), dt);
*/
        //printf("currentPointVelocity = ()", currentPointVelocity.x, currentPointVelocity.y, currentPointVelocity.z);

        //printf();

        //pseudocode
        //currentPointVelocity.x = velocity(xVAdv,yVAdv,zVAdv).x * (floor(xVAdv) + 1 - xVAdv) + velocity(xVAdv+1, yVAdv, zVAdv).x * (xVAdv - floor(xVAdv));
        //currentPointVelocity.y = velocity(xVAdv,yVAdv,zVAdv).y * (floor(yVAdv) + 1 - yVAdv) + velocity(xVAdv, yVAdv+1, zVAdv).y * (yVAdv - floor(yVAdv));
        //currentPointVelocity.z = velocity(xVAdv,yVAdv,zVAdv).z * (floor(zVAdv) + 1 - zVAdv) + velocity(xVAdv, yVAdv, zVAdv+1).z * (zVAdv - floor(zVAdv));

        //it can't be both of these, also currentPoint needs to be updated
        //currentPointVelocity.x = (velocityIn[num*3] + velocityIn[num*3+1]) / 2.0;
        //currentPointVelocity.y = (velocityIn[num*3+1] + velocityIn[num*3+1+velocitySize.x]) / 2.0;
        //currentPointVelocity.z = (velocityIn[num*3+2] + velocityIn[num*3+2+velocitySize.x*velocitySize.y]) / 2.0;

        //currentPointVelocity.x = velocityIn[num*3];
        //currentPointVelocity.y = velocityIn[num*3+1];
        //currentPointVelocity.z = velocityIn[num*3+2];


    }while(numTries!=0 && subDt!=0);

    //int xNewFloor = int(currentPoint.x), yNewFloor = int(currentPoint.y), zNewFloor = int(currentPoint.z);

    //pseudocode
    //velOut(x,y,z).x = (velIn(xNewFloor,yNewFloor,zNewFloor).x*(xNewFloor+1-currentPoint.x) + velIn(xNewFloor+1,yNewFloor,zNewFloor).x*(currentPoint.x-xNewFloor)); //velIn(int(currentPoint.x), int(currentPoint.y), int(currentPoint.z));
    //velOut(x,y,z).y = (velIn(xNewFloor,yNewFloor,zNewFloor).y*(yNewFloor+1-currentPoint.y) + velIn(xNewFloor,yNewFloor+1,zNewFloor).y*(currentPoint.y-yNewFloor));
    //velOut(x,y,z).z = (velIn(xNewFloor,yNewFloor,zNewFloor).z*(zNewFloor+1-currentPoint.z) + velIn(xNewFloor,yNewFloor,zNewFloor+1).z*(currentPoint.z-zNewFloor));

    //there should be no out-of-bounds at this point

/*
    if(currentPoint.x > velocitySize.x-1 || currentPoint.y > velocitySize.y-1 || currentPoint.z > velocitySize.z-1){
        printf("currentPoint.x = %f; currentPoint.y = %f; currentPoint.z = %f\n", currentPoint.x, currentPoint.y, currentPoint.z);
        return;}
*/
    //if(subDt == 0.0f){
        velocityOut[num*3]   = currentPointVelocity.x;
        velocityOut[num*3+1] = currentPointVelocity.y;
        velocityOut[num*3+2] = currentPointVelocity.z;//}
    /*else{
        velocityOut[num*3]   = 0;
        velocityOut[num*3+1] = 0;
        velocityOut[num*3+2] = 0;
    }*/
/*
    if(x == 32 && y == 32 && z == 32)
        printf("currentPoint = (%f,%f,%f); currentPointVelocity = (%f,%f,%f); velocity(%d,%d,%d) = (%f, %f, %f)\n",
               currentPoint.x, currentPoint.y, currentPoint.z, currentPointVelocity.x, currentPointVelocity.y, currentPointVelocity.z, x, y, z, velocityIn[num*3], velocityIn[num*3+1], velocityIn[num*3+2]);*/
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

//__global__ void advectDensityStaggeredClassic()

//the velocity is advected last, so only velocityIn can be used for scalar field advection <-------
//does this need to be before or after the obstacles move?
//this uses a type of binary search, might be incorrect (look into Runge-Kutta coefficients)
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
        //this is unlikely to be correct (first value must be a pointer, not a value)
        //surf3Dwrite(0, densityPongSurf, x*sizeof(T),y,z);
        return;}

    //if cell is an inflow, don't do advection on it (?)
    //if()

    //also need to do something about out-of-bounds advection (can just clamp to the edge along the backtraced segment)

    //texture coordinates are float regardless of the texture bit depth (sizeof(T))
    float xAdv, yAdv, zAdv, xAdv2, yAdv2, zAdv2;
    T c00, c01, c10, c11, c1, c0, c, p00, p01, p10, p11, p1, p0, p, c_bis;

    bool t = false;

    c = tex3D<T>(densityPingTex, x, y, z);
    /*
    if(y == 0 && c !=0){
        c_bis = c;
        t = true;
        printf("x = %d, y = %d, z = %d, c = %f\n", x, y, z, c);
    }*/

    c = 0.0;

    //printf("%f\n", c);
    //return;

    float3 currentPoint, currentPointVelocity;

    currentPoint.x = __int2float_rd(x);
    currentPoint.y = __int2float_rd(y);
    currentPoint.z = __int2float_rd(z);

    currentPointVelocity.x = (velocity[(x + (domainSize.x+1)*y + (domainSize.x+1)*(domainSize.y+1)*z)*3]   + velocity[(x+1 + (domainSize.x+1)*y     + (domainSize.x+1)*(domainSize.y+1)* z  )*3]) / 2.0;
    currentPointVelocity.y = (velocity[(x + (domainSize.x+1)*y + (domainSize.x+1)*(domainSize.y+1)*z)*3+1] + velocity[(x   + (domainSize.x+1)*(y+1) + (domainSize.x+1)*(domainSize.y+1)* z  )*3+1]) / 2.0;
    currentPointVelocity.z = (velocity[(x + (domainSize.x+1)*y + (domainSize.x+1)*(domainSize.y+1)*z)*3+2] + velocity[(x   + (domainSize.x+1)*y     + (domainSize.x+1)*(domainSize.y+1)*(z+1))*3+2]) / 2.0;

    float subDt = dt;

    //printf("%f\n", subDt);

    int xAdvFloor, yAdvFloor, zAdvFloor;
    float maxX, maxY, maxZ;

    float xd, yd, zd;

    //since a small enough subDt will bring us back to the original cell we backtraced from (which , we only need the
    //while(numTries!=0 && subDt<)

    //printf("numTries = %d\n", numTries);

    bool a = true;

    bool o1, o2, o3, o4, o5, o6, o7, o8;

    do{
        numTries--;

        //what isa written as x,y,z and velocity need to change sometimes (when currentPoint is updated)
        //x -> currentPoint.x
        xAdv = currentPoint.x - currentPointVelocity.x * subDt / cellDim;
        yAdv = currentPoint.y - currentPointVelocity.y * subDt / cellDim;
        zAdv = currentPoint.z - currentPointVelocity.z * subDt / cellDim;
/*
        if(a){
            printf("x = %d; y = %d; z = %d; xAdv = %f; yAdv = %f; zAdv = %f; currentPointVelocity.x = %f, currentPointVelocity.y = %f, currentPointVelocity.z = %f, cellDim = %f\n", x, y, z, xAdv, yAdv, zAdv, currentPointVelocity.x, currentPointVelocity.y, currentPointVelocity.z, cellDim);
            a = false;
        return;}*/
/*
        printf("cellDim = %f;xAdv = %f, yAdv = %f, zAdv = %f\n", cellDim, xAdv, yAdv, zAdv);
        return;
*/

        //this is the boundary clamping
        //check if these are within boundaries, clamp along backtraced segment if they aren't
        if(xAdv>=domainSize.x || xAdv<0 || yAdv>=domainSize.y || yAdv<0 || zAdv>=domainSize.z || zAdv<0){
            //this isn't right
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

            //probably due to insufficient precision, currentPoint might still reside very slightly outside the domain, so the calculation of currentPointVelocity will do boundary checks
/*
            //if(maxZ == NAN){
                printf("maxZ=%f (x,y,z) = (%d, %d, %d)\n", maxZ, x, y, z);
                return;
             //   return;}*/



            xAdv2 = currentPoint.x + maxZ * currentPointVelocity.x; //this will be in the direction oppposite to the velocity vector
            yAdv2 = currentPoint.y + maxZ * currentPointVelocity.y;
            zAdv2 = currentPoint.z + maxZ * currentPointVelocity.z;
/*
            if(xAdv2 >= domainSize.x || yAdv2 >= domainSize.y || zAdv2 >= domainSize.z || xAdv < 0 || yAdv < 0 || zAdv < 0){
                printf("(xAdv2, yAdv2, zAdv2) = (%f, %f, %f); (xAdv, yAdv, zAdv) = (%f, %f, %f); (x,y,z) = (%d, %d, %d); currentPointVelocity = (%f, %f, %f)\n",
                       xAdv2, yAdv2, zAdv2, xAdv, yAdv, zAdv, x, y, z,
                       currentPointVelocity.x, currentPointVelocity.y, currentPointVelocity.z);
                return;}*/

            xAdv2 = max(0,min(xAdv2, domainSize.x-1));
            yAdv2 = max(0,min(yAdv2, domainSize.y-1));
            zAdv2 = max(0,min(zAdv2, domainSize.z-1));

            //might just break here
            /*
            if(xAdv2 == xAdv && yAdv2 == yAdv && zAdv2 == zAdv){ //this happens if we were already at an edge and the direction opposite the velocity at this point would send us past this same edge
                currentPoint.x = xAdv2;
                currentPoint.y = yAdv2;
                currentPoint.z = zAdv2;
                break;}*/

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
/*
        if(xAdvFloor > domainSize.x-1 || yAdvFloor > domainSize.y-1 || zAdvFloor > domainSize.z-1 || xAdvFloor < 0 || yAdvFloor < 0 || zAdvFloor < 0){
            printf("(xAdv, yAdv, zAdv) = (%f, %f, %f); (xAdvFloor, yAdvFloor, zAdvFloor) = (%d, %d, %d); (x,y,z) = (%d, %d, %d); currentPointVelocity = (%f, %f, %f)\n",
                   xAdv, yAdv, zAdv, xAdvFloor, yAdvFloor, zAdvFloor, x, y, z,
                   currentPointVelocity.x, currentPointVelocity.y, currentPointVelocity.z);
            return;
        }*/

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



/*
        if(c!=0.0f &&numTries==0){
            printf("x = %d, y = %d, z = %d, xAdv = %f, yAdv = %f, zAdv = %f, xAdvFloor = %d, yAdvFloor = %d, zAdvFloor = %d, xd = %f, yd = %f, zd = %f, c00 = %f, c01 = %f, c10 = %f, c11 = %f, c0 = %f, c1 = %f, c = %f\n",\
                   x, y, z, xAdv, yAdv, zAdv, xAdvFloor, yAdvFloor, zAdvFloor, xd, yd, zd, c00, c01, c10, c11, c0, c1, c);
            }//return;}
*/
        //return;

/*
        if(a){
            printf("xd = %f, yd = %f, zd = %f, xAdv = %f, yAdv = %f, zAdv = %f, xAdvFloor = %d, yAdvFloor = %d, zAdvFloor = %d\n", xd, yd, zd, xAdv, yAdv, zAdv, xAdvFloor, yAdvFloor, zAdvFloor);
            return;}
            */
/*
        if(a && c!=0){
            printf("c = %f\n",c);
            return;
            a = false;}
*/

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
/*
        if(y == 0 && c !=0)
            printf("x = %d, y = %d, z = %d, xAdv = %f, yAdv = %f, zAdv = %f, xAdvFloor = %d, yAdvFloor = %d, zAdvFloor = %d, xd = %f, yd = %f, zd = %f,\nc00 = %f, c01 = %f, c10 = %f, c11 = %f, c0 = %f, c1 = %f, c = %f, p00 = %f, p01 = %f, p10 = %f, p11 = %f, p0 = %f, p1 = %f, p = %f\n\n",\
                    x, y, z, xAdv, yAdv, zAdv, xAdvFloor, yAdvFloor, zAdvFloor, xd, yd, zd, c00, c01, c10, c11, c0, c1, c, p00, p01, p10, p11, p0, p1, p);
*/

        if(o1 || o2 || o3 || o4 || o5 || o6 || o7 || o8){ //are all points ignored?
            //the "share" of ignored interpolation points (that are occupied by obstacles) needs to be divided between the remaining points,
            //so that the interpolation result is a a proper average

            c /= p;
/*
            if(c > c00+0.00001f && c > c01+0.00001f && c > c10 +0.00001f && c > c11+0.00001f)
                printf("x = %d, y = %d, z = %d, xAdv = %f, yAdv = %f, zAdv = %f, xAdvFloor = %d, yAdvFloor = %d, zAdvFloor = %d, xd = %f, yd = %f, zd = %f,\nc00 = %f, c01 = %f, c10 = %f, c11 = %f, c0 = %f, c1 = %f, c = %f, p00 = %f, p01 = %f, p10 = %f, p11 = %f, p0 = %f, p1 = %f, p = %f\n\n",\
                        x, y, z, xAdv, yAdv, zAdv, xAdvFloor, yAdvFloor, zAdvFloor, xd, yd, zd, c00, c01, c10, c11, c0, c1, c, p00, p01, p10, p11, p0, p1, p);
*/
            subDt = dt - subDt; //if a subsegment was projected back due to the else branch being used in some previous iteration, repeat with the rest of the time
            dt = subDt;
            //if()
            currentPoint.x = xAdv;
            currentPoint.y = yAdv;
            currentPoint.z = zAdv;

            //if(xAdv >= )

            //this shouldn't be masked, since the motion of obstacles pushes the density too
//#define vel(dimX,dimY,dimZ, dimension) velocity[(max(0,min(dimX,domainSize.x)) + max(0,min(dimY,domainSize.y)*(domainSize.x+1)) + max(0,min(domainSize.z,dimZ))*(domainSize.x+1)*(domainSize.y+1))*3+dimension] //the dimension component (0->x, 1->y, 2->z) of velocity cell (dimX, dimY, dimZ)
//#define getIndex(dimX,dimY,dimZ, dimension) (max(0,min(dimX,domainSize.x)) + max(0,min(dimY,domainSize.y))*(domainSize.x+1) + max(0,min(domainSize.z,dimZ))*(domainSize.x+1)*(domainSize.y+1))*3+dimension
#define vel(dimX,dimY,dimZ, dimension) velocity[((dimX) + (dimY)*(domainSize.x+1) + (dimZ)*(domainSize.x+1)*(domainSize.y+1))*3+(dimension)] //the dimension component (0->x, 1->y, 2->z) of velocity cell (dimX, dimY, dimZ)
#define getIndex(dimX,dimY,dimZ, dimension) ((dimX) + (dimY)*(domainSize.x+1) + (dimZ)*(domainSize.x+1)*(domainSize.y+1))*3+(dimension)
            //change this in other place (weights)
/*
            if(getIndex(int(xAdv+0.5), yAdvFloor, zAdvFloor, 0)>=velocitySizeTotal || getIndex(int(xAdv+1.5), yAdvFloor, zAdvFloor, 0)>=velocitySizeTotal
            || getIndex(xAdvFloor, int(yAdv+0.5), zAdvFloor, 1)>=velocitySizeTotal || getIndex(xAdvFloor, int(yAdv+1.5), zAdvFloor, 1)>=velocitySizeTotal
            || getIndex(xAdvFloor, yAdvFloor, int(zAdv+0.5), 2)>=velocitySizeTotal || getIndex(xAdvFloor, yAdvFloor, int(zAdv+1.5), 2)>=velocitySizeTotal
            || getIndex(int(xAdv+0.5), yAdvFloor, zAdvFloor, 0)<=0 || getIndex(int(xAdv+1.5), yAdvFloor, zAdvFloor, 0)<=0
            || getIndex(xAdvFloor, int(yAdv+0.5), zAdvFloor, 1)<=0 || getIndex(xAdvFloor, int(yAdv+1.5), zAdvFloor, 1)<=0
            || getIndex(xAdvFloor, yAdvFloor, int(zAdv+0.5), 2)<=0 || getIndex(xAdvFloor, yAdvFloor, int(zAdv+1.5), 2)<=0){
                printf("(xAdv, yAdv, zAdv) = (%f, %f, %f); (x,y,z) = (%d, %d, %d); currentPointVelocity = (%f, %f, %f)\n",
                       xAdv, yAdv, zAdv, x, y, z,
                       currentPointVelocity.x, currentPointVelocity.y, currentPointVelocity.z);
                return;}
*/
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

    //if(currentPoint.x != x || currentPoint.y != y || currentPoint.z != z)
    /*
    if(currentPointVelocity.x !=0 || currentPointVelocity.y!=0 || currentPointVelocity.z != 0)
        printf("density[%d, %d, %d] = %f; velocity[x,y,z] = (%f, %f, %f); currentPoint = (%f, %f, %f); currentPointVelocity = (%f, %f, %f)\n",
               x, y, z, tex3D<T>(densityPingTex, currentPoint.x, currentPoint.y,currentPoint.z),
               vel(x,y,z,0), vel(x,y,z,1), vel(x,y,z,2),
               currentPoint.x, currentPoint.y, currentPoint.z,
               currentPointVelocity.x, currentPointVelocity.y, currentPointVelocity.z);*/
#undef vel
    //if(currentLoop!=2)
    //printf("numTries = %d", numTries);
/*
    if(tex3D<T>(densityPingTex, currentPoint.x, currentPoint.y, currentPoint.z) != tex3D<T>(densityPingTex, x, y, z))
        printf("densityPingTex[%f, %f, %f] = %f (NEW); densityPingTex[%d, %d, %d] = %f (OLD)\n",
               currentPoint.x, currentPoint.y, currentPoint.z, tex3D<T>(densityPingTex, currentPoint.x, currentPoint.y, currentPoint.z),
               x, y, z, tex3D<T>(densityPingTex, x, y, z));*/
    /*if(yAdvFloor == 0 && c !=0 && t){
        surf3Dwrite<T>(c, densityPongSurf, x*sizeof(T), y, z); //<---MISALIGNED ADDRESS ERROR HERE (once upon a time)
        //printf("x = %d, y = %d, z = %d, c = %f\n", x, y, z, c);
        printf("x = %d, y = %d, z = %d, xAdv = %f, yAdv = %f, zAdv = %f, xAdvFloor = %d, yAdvFloor = %d, zAdvFloor = %d, xd = %f, yd = %f, zd = %f,\nc00 = %f, c01 = %f, c10 = %f, c11 = %f, c0 = %f, c1 = %f, c = %f, p00 = %f, p01 = %f, p10 = %f, p11 = %f, p0 = %f, p1 = %f, p = %f, C_BIS = %f, t = %d\n\n",\
                x, y, z, xAdv, yAdv, zAdv, xAdvFloor, yAdvFloor, zAdvFloor, xd, yd, zd, c00, c01, c10, c11, c0, c1, c, p00, p01, p10, p11, p0, p1, p, c_bis, t);}
    else{*/
        surf3Dwrite<T>(c, densityPongSurf, x*sizeof(T), y, z);
    //}
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

    //since a small enough subDt will bring us back to the original cell we backtraced from (which , we only need the
    //while(numTries!=0 && subDt<)
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

        //it should be impossitex3D<T>(densityPingTex, x, y, z)ble for nAdvFloor values to cause out-of-bounds in the follwing, due to the previous check (*)
        //may still need to verify that nAdvFloor<domainSize.n

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

            //WTF IS THIS?
            /*
#define vel(dimX,dimY,dimZ) velocity[dimX + dimY*(domainSize.x+1) + dimZ*(domainSize.x+1)*(domainSize.y+1)] //turns single-indexed into triple-indexed

            currentPointVelocity.x = vel(int(xAdv+0.5), yAdvFloor, zAdvFloor) * ((xAdv+0.5) - float(floor(xAdv+0.5)))
                                   + vel(int(xAdv+1.5), yAdvFloor, zAdvFloor) * (float(floor(xAdv+1.5)) - (xAdv+0.5));
            currentPointVelocity.y = vel(xAdvFloor, int(yAdv+0.5), zAdvFloor) * ((xAdv+0.5) - float(floor(xAdv+0.5)))
                                   + vel(xAdvFloor, int(yAdv+1.5), zAdvFloor) * (float(floor(xAdv+1.5)) - (xAdv+0.5));
            currentPointVelocity.z = vel(xAdvFloor, yAdvFloor, int(zAdv+0.5)) * ((xAdv+0.5) - float(floor(xAdv+0.5)))
                                   + vel(xAdvFloor, yAdvFloor, int(zAdv+1.5)) * (float(floor(xAdv+1.5)) - (xAdv+0.5));
#undef vel
*/
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

    //this should make it similar to the advection scheme in the original Rideout thing when uncommented AND numTries is set to 1
    //actually it's kinda dubious to do this for temperature
    /*
    if(subDt!=0)
        scalarFieldOut[num] = 0;
    */

    /*
    if(numTries != 0)
    //if(currentPoint.x!=x && currentPoint.y!=y && currentPoint.z!=z)
        printf("currentPoint.x = %f, currentPoint.y = %f, currentPoint.z = %f, x = %d, y = %d, z = %d\n", currentPoint.x, currentPoint.y, currentPoint.z, x, y, z);
    */
    scalarFieldOut[num] = currentValue;
}

#undef isNotObstacle

//__global__ void advectStaggered(T* velocity, T* scalarFieldIn, T* scalarFieldOut, T* obstacles, dim3 domainSize, T cellDim, float dt, int numTries=5){
extern "C" cudaError_t advectDensityCudaStaggeredWrapper_float(float* velocityIn, cudaTextureObject_t densityPingTex, cudaSurfaceObject_t densityPongSurf, float* obstacles, dim3 domainSize, float cellDim, float dt){

    int domainSizeTotal = domainSize.x*domainSize.y*domainSize.z;
    //densityPingTex does not seem to have been initialized properly
    //printf("domainSizeTotal = %d\n", domainSizeTotal);
    //printf("domainSize = (%d, %d, %d)\n", domainSize.x, domainSize.y, domainSize.z);
    //printf("kernel parameters <<<%d, %d>>>", domainSizeTotal/1024+(domainSizeTotal%1024!=0), 1024);

    //printf("cellDim = %f", cellDim);
    //printf("Exiting (%s:%d, %s)",__FILE__,__LINE__,__FUNCTION__);
    //exit(0);


    //trying to pass this as a parameter causes cellDim to be zero for some reason
    //static int numLoop = 0;
    //int numLoopCopy = numLoop;

    //if the debugging printf calls are included, this fails for 1024 threads per block (cudaErrorLaunchOutOfResources). The number of threads has been reduced to prevent this.
#define numThreadsDensity 512
    advectDensityStaggered<float><<<domainSizeTotal/numThreadsDensity+(domainSizeTotal%numThreadsDensity!=0), numThreadsDensity>>>(velocityIn, densityPingTex, densityPongSurf, obstacles, domainSize, cellDim, dt);
#undef numThreadsDensity

    //numLoop +=1;

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
//also probably the fact that the fluid is considered incompressible, the
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

    //THIS
    /*
    if(x!=0 && y!=0 && z!=0 && x%16==0 && y%16==0 && z%16==0)
        printf("velocity[%d,%d,%d].x = %f, velocity[x,y,z].y = %f, velocity[x,y,z].z = %f; pressure[x,y,z] = %f, pressure[x-1,y,z] = %f, pressure[x,y-1,z] = %f, pressure[x,y,z-1] = %f\n",
               x, y, z, velocity[num],
               pressure[x   + domainSize.x * y     + domainSize.x * domainSize.y *  z   ],
               pressure[x-1 + domainSize.x * y     + domainSize.x * domainSize.y *  z   ],
               pressure[x   + domainSize.x * (y-1) + domainSize.x * domainSize.y *  z   ],
               pressure[x   + domainSize.x * y     + domainSize.x * domainSize.y * (z-1)]);
    */

    //printf("(x,y,z) = %d %d %d\n",x,y,z);

    //in the torch version, the velocities at the boundaries are not changed (the outer pressure is not known, but may be considered constant if needed)
    //might have excessive otherwise

    //if there are no obstacles
    //if there are obstacles, might need setWallBcs, or maybe updating the obstacles will set the velocities to the correct values

    //T xx = pressure[64*64*64-1];
    //T xx2 = velocity[3*65*65*65-1];
    //T xx3 = density[64*64*64-1];

    if(x >= velocitySize.x || y >= velocitySize.y || z >= velocitySize.z)
        return;

    //THIS ADDS a constant value to velocity, because nothing moves otherwise
    //velocity[num*3]= 13.0f;
    //velocity[num*3+1]= 13.0f;
    //velocity[num*3+2]= 13.0f;
    //return;

    int px = 0.0f, py = 0.0f, pz = 0.0f;

    //leave boundary velocities unchanged (as in torch) <--------------
    if(x<velocitySize.x-1 && y<velocitySize.y-1 && z<velocitySize.z-1){
        if(x>0){
            //the velocity is not normalized, and the pressure is not divided by its normalization factor
            velocity[num*3]   -= (pressure[x + domainSize.x * y + domainSize.x * domainSize.y * z] - pressure[x-1 + domainSize.x * y + domainSize.x * domainSize.y * z])
                    // / ( density[x + domainSize.x * y + domainSize.x * domainSize.y * z] +  density[x-1 + domainSize.x * y + domainSize.x * domainSize.y * z])
                    // / cellDim// * dt; //there is no division
                    ;
            px = (pressure[x + domainSize.x * y + domainSize.x * domainSize.y * z] - pressure[x-1 + domainSize.x * y + domainSize.x * domainSize.y * z]);
            //not sure if cellDim is needed, but that would make sense
            //

            /*
            if(x + domainSize.x * y + domainSize.x * domainSize.y * z >= domainSize.x * domainSize.y * domainSize.z)
                printf("x = %d, y = %d, z = %d, x>0, case 1\n", x,y,z);
            if(x-1 + domainSize.x * y + domainSize.x * domainSize.y * z >= domainSize.x * domainSize.y * domainSize.z)
                printf("x = %d, y = %d, z = %d, x>0, case 2\n", x,y,z);*/

        }

        if(y>0){
            velocity[num*3+1] -= (pressure[x + domainSize.x * y + domainSize.x * domainSize.y * z] - pressure[x   + domainSize.x * (y-1) + domainSize.x*domainSize.y*z])
                    // / ( density[x + domainSize.x * y + domainSize.x * domainSize.y * z] +  density[x   + domainSize.x * (y-1) + domainSize.x*domainSize.y*z])
                    // / cellDim// * dt;
                    ;
            py = (pressure[x + domainSize.x * y + domainSize.x * domainSize.y * z] - pressure[x   + domainSize.x * (y-1) + domainSize.x*domainSize.y*z]);
            /*
            if(x + domainSize.x * y + domainSize.x * domainSize.y * z >= domainSize.x * domainSize.y * domainSize.z)
                printf("x = %d, y = %d, z = %d, y>0, case 1\n", x,y,z);
            if(x   + domainSize.x * (y-1) + domainSize.x*domainSize.y*z >= domainSize.x * domainSize.y * domainSize.z)
                printf("x = %d, y = %d, z = %d, y>0, case 2\n", x,y,z);
            */
        }

        if(z>0){
            velocity[num*3+2] -= (pressure[x + domainSize.x * y + domainSize.x * domainSize.y * z] - pressure[x   + domainSize.x * y + domainSize.x*domainSize.y*(z-1)])
                    // / ( density[x + domainSize.x * y + domainSize.x * domainSize.y * z] +  density[x   + domainSize.x * y + domainSize.x*domainSize.y*(z-1)])
                    // / cellDim// * dt;
                    ;
            /*
            if(x + domainSize.x * y + domainSize.x * domainSize.y * z >= domainSize.x * domainSize.y * domainSize.z)
                printf("x = %d, y = %d, z = %d, y>0, case 1", x,y,z);
            if(x   + domainSize.x * y + domainSize.x*domainSize.y*(z-1) >= domainSize.x * domainSize.y * domainSize.z)
                printf("x = %d, y = %d, z = %d, y>0, case 2", x,y,z);
            */
            pz = (pressure[x + domainSize.x * y + domainSize.x * domainSize.y * z] - pressure[x   + domainSize.x * y + domainSize.x*domainSize.y*(z-1)]);
        }
/*
        if(x>0 && y>0 && z>0 && (px!=0 || py!=0 || pz!=0))
            printf("pressureDiv[ %d, %d, %d] = ( %f, %f, %f)\n", x, y, z,
                                         pressure[x + domainSize.x * y + domainSize.x * domainSize.y * z] - pressure[x-1 + domainSize.x * y + domainSize.x * domainSize.y * z],
                                         pressure[x + domainSize.x * y + domainSize.x * domainSize.y * z] - pressure[x   + domainSize.x * (y-1) + domainSize.x*domainSize.y*z],
                                         pressure[x + domainSize.x * y + domainSize.x * domainSize.y * z] - pressure[x   + domainSize.x * y + domainSize.x*domainSize.y*(z-1)]);
*/
/*
        if(x>=20 && x<=44 && y>=20 && y<=44 && z>20 && z<=44 && pressure[x + domainSize.x * y + domainSize.x * domainSize.y * z] != 0){
            //printf("pressure[%d, %d, %d] = %f\n", x, y, z, pressure[x + domainSize.x * y + domainSize.x * domainSize.y * z]);
            //printf("pressure[%d, %d, %d] = %f\nvelocity[%d, %d, %d] = %f; velocity[x, y, z]; velocity[x, y, z]\n", x, y, z, pressure[x + domainSize.x * y + domainSize.x * domainSize.y * z], x, y, z, velocity[num*3], velocity[num*3+1], velocity[num*3+2]);
        }*/

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

    //T cVal = pVal * 2;
/*
    if(vVal!=0)
        printf("pVal/vVal = %f\n", pVal/vVal);*/
    /*
    if(pVal!=0)
        printf("pVal*6 = %f", pVal);
*/
#undef getPressure
#undef getVelocity
}

extern "C" cudaError_t checkGradientFactorWrapper_float(float* pressure, float* velocity, dim3 domainSize){

    int domainSizeTotal = domainSize.x * domainSize.y * domainSize.z;

    checkGradientFactor<float><<<domainSizeTotal/1024+(domainSizeTotal%1024!=0), 1024>>>(pressure, velocity, domainSize);

    return cudaPeekAtLastError();
}

extern "C" cudaError_t subtractGradientStaggeredCuda_float(float* velocity, float* pressure, float* density, float* obstacles, dim3 velocitySize, float cellDim, float dt){

    //int domainSizeTotal = domainSize.x * domainSize.y * domainSize.z;
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
            velocity[num*3]   -= (pressure[x+1+ velocitySize.x * y + velocitySize.x * velocitySize.y * z] - pressure[x-1 + velocitySize.x * y + velocitySize.x * velocitySize.y * z])
                    // /   density[x  + velocitySize.x * y + velocitySize.x * velocitySize.y * z]
                    // / (2*cellDim); //num is not the same for velocity and scalar fields
                    ;
        if(y>0)
            velocity[num*3+1] -= (pressure[x + velocitySize.x * y + velocitySize.x * velocitySize.y * z] - pressure[x   + velocitySize.x * (y-1) + velocitySize.x*velocitySize.y*z])
                    // /   density[x + velocitySize.x * y + velocitySize.x * velocitySize.y * z]
                    // / (2*cellDim);
                    ;
        if(z>0)
            velocity[num*3+2] -= (pressure[x + velocitySize.x * y + velocitySize.x * velocitySize.y * z] - pressure[x   + velocitySize.x * y + velocitySize.x*velocitySize.y*(z-1)])
                    // /   density[x + velocitySize.x * y + velocitySize.x * velocitySize.y * z]
                    // / (2*cellDim);
                    ;
    }
}

extern "C" cudaError_t subtractGradientNotStaggeredCuda_float(float* velocity, float* pressure, float* density, float* obstacles, dim3 velocitySize, float cellDim, float dt){

    //int domainSizeTotal = domainSize.x * domainSize.y * domainSize.z;
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
/*
    if(inflowDensity[num] != 0)
        printf("inflowDensity(%d,%d,%d) = %f; inflowVelocity(x,y,z).x = %f; inflowVelocity(x,y,z).y = %f; inflowVelocity(x,y,z).z = %f;\
 inflowVelocity(x+1,y,z).x = %f; inflowVelocity(x,y+1,z).y = %f; inflowVelocity(x,y,z+1).z = %f\n",\
                x, y, z, inflowDensity[num], inflowTemperature[num],
                inflowVelocity[(x +   (domainSize.x+1) * y   + (domainSize.x+1) * (domainSize.y+1) * z)  *3],
                inflowVelocity[(x +   (domainSize.x+1) * y   + (domainSize.x+1) * (domainSize.y+1) * z)  *3+1],
                inflowVelocity[(x +   (domainSize.x+1) * y   + (domainSize.x+1) * (domainSize.y+1) * z)  *3+2],
                inflowVelocity[((x+1)+(domainSize.x+1) * y   + (domainSize.x+1) * (domainSize.y+1) * z)  *3  ],
                inflowVelocity[(x +   (domainSize.x+1)*(y+1) + (domainSize.x+1) * (domainSize.y+1) * z)  *3+1],
                inflowVelocity[(x +   (domainSize.x+1) * y   + (domainSize.x+1) * (domainSize.y+1)*(z+1))*3+2]);
*/
}

extern "C" cudaError_t initializeInflowsStaggered_float(float* inflowVelocity, float* inflowDensity, float *inflowTemperature, dim3 domainSize, dim3 center, int radius, float ambTemp){

    int velocitySizeTotal = (domainSize.x+1) * (domainSize.y+1) * (domainSize.z+1);

    //printf("center = (%d,%d,%d), radius = %d\n", center.x, center.y, center.z, radius);

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

    if(norm3df(int(center.x)-x,int(center.y)-y,int(center.z)-z)<=radius){ //replace center with int3 if you can to avoid casting
        inflowDensity[num] = 1.0;
        inflowTemperature[num] = ambTemp + 10.0;
        inflowVelocity[(x +   (domainSize.x+1) * y   + (domainSize.x+1) * (domainSize.y+1) * z)  *3  ] = velVal;//1.0; //(x,y,z).x
        inflowVelocity[(x +   (domainSize.x+1) * y   + (domainSize.x+1) * (domainSize.y+1) * z)  *3+1] = velVal;//1.0; //(x,y,z).y
        inflowVelocity[(x +   (domainSize.x+1) * y   + (domainSize.x+1) * (domainSize.y+1) * z)  *3+2] = velVal;//1.0; //(x,y,z).z
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

    //bool isInScalarDomain = false;
/*
    if(x == 32 && y==32 && z==32 && x < domainSize.x && y < domainSize.y && z < domainSize.z && surf3Dread<float>(densityPingSurf,x*sizeof(float),y,z) > 0)
        printf("densityPingSurf(%d,%d,%d) = %f, velocityIn(x,y,z).x = %f, velocityIn(x,y,z).y = %f, velocityIn(x,y,z).z = %f\n",
                x, y, z, surf3Dread<float>(densityPingSurf,x*sizeof(float),y,z), velocityIn[num*3], velocityIn[num*3+1], velocityIn[num*3+2]);
*/
    if(x < domainSize.x && y < domainSize.y && z < domainSize.z){
        surf3Dwrite(max(0,(surf3Dread<float>(densityPingSurf,x*sizeof(float),y,z)+inflowDensity[x + y * domainSize.x + z * domainSize.x * domainSize.y]*dt)//*
                    /*(!(obstacles[x + y * domainSize.x + z * domainSize.x*domainSize.y]))*/),densityPingSurf,x*sizeof(float),y,z);
        temperature[x + domainSize.x * y + domainSize.x * domainSize.y * z] = inflowTemperature[x + domainSize.x * y + domainSize.x * domainSize.y * z];}
        //isInScalarDomain = true;}
/*
    if(x == 32 && y==32 && z==32 && x < domainSize.x && y < domainSize.y && z < domainSize.z && surf3Dread<float>(densityPingSurf,x*sizeof(float),y,z) > 0)
        printf("densityPingSurf(%d,%d,%d) = %f, velocityIn(x,y,z).x = %f, velocityIn(x,y,z).y = %f, velocityIn(x,y,z).z = %f\n",
                x, y, z, surf3Dread<float>(densityPingSurf,x*sizeof(float),y,z), velocityIn[num*3], velocityIn[num*3+1], velocityIn[num*3+2]);*/

    //in this current form, we cannot have a zero-velocity speed inflow or outflow; use a tiny but nonzero velocity
    //if you don't like that, work out a condition dependant on nonzero density at this point, but mind the staggered grid
    if(inflowVelocity[num * 3] != 0 ||
       inflowVelocity[num * 3 + 1] != 0 ||
       inflowVelocity[num * 3 + 2] != 0){

        velocityIn[num*3]   = inflowVelocity[num*3];  //velocity in inflows/outflows is overwritten, not added
        velocityIn[num*3+1] = inflowVelocity[num*3+1];
        velocityIn[num*3+2] = inflowVelocity[num*3+2];}
/*
    if(inflowVelocity[num*3]!=0 || inflowVelocity[num*3+1]!=0 || inflowVelocity[num*3+2]!=0)
        printf("velocityIn(%d,%d,%d).x = %f, velocityIn(x,y,z).y = %f, velocityIn(x,y,z).z = %f\n", x, y, z, velocityIn[num*3], velocityIn[num*3+1], velocityIn[num*3+2]);
*/

/*
    if(x < domainSize.x && y < domainSize.y && z < domainSize.z && surf3Dread<float>(densityPingSurf,x*sizeof(float),y,z) > 0)
        printf("densityPingSurf(%d,%d,%d) = %f, velocityIn(x,y,z).x = %f, velocityIn(x,y,z).y = %f, velocityIn(x,y,z).z = %f\n",
                x, y, z, surf3Dread<float>(densityPingSurf,x*sizeof(float),y,z), velocityIn[num*3], velocityIn[num*3+1], velocityIn[num*3+2]);
*/
/*
    if(velocityIn[num*3] > 0){buffer
        if(x < domainSize.x && y < domainSize.y && z < domainSize.z)
            printf("densityPingSurf(%d,%d,%d) = %f, velocityIn(x,y,z).x = %f, velocityIn(x,y,z).y = %f, velocityIn(x,y,z).z = %f\n",
                    x, y, z, surf3Dread<float>(densityPingSurf,x*sizeof(float),y,z), velocityIn[num*3], velocityIn[num*3+1], velocityIn[num*3+2]);
        else
            printf("velocityIn(%d,%d,%d).x = %f, velocityIn(x,y,z).y = %f, velocityIn(x,y,z).z = %f\n", x, y, z, velocityIn[num*3], velocityIn[num*3+1], velocityIn[num*3+2]);
    }*/
}

extern "C" cudaError_t applyInflowsStaggered_float(float* velocityIn, cudaSurfaceObject_t densityPingSurf, float* obstacles, float* temperature, float* inflowVelocity, float* inflowDensity, float* inflowTemperature, dim3 domainSize, float dt){

    //printf("dt = %f\n", dt);

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

    //bool isInScalarDomain = false;
/*
    if(x == 32 && y==32 && z==32 && x < domainSize.x && y < domainSize.y && z < domainSize.z && surf3Dread<T>(densityPingSurf,x*sizeof(T),y,z) > 0)
        printf("densityPingSurf(%d,%d,%d) = %f, velocityIn(x,y,z).x = %f, velocityIn(x,y,z).y = %f, velocityIn(x,y,z).z = %f\n",
                x, y, z, surf3Dread<T>(densityPingSurf,x*sizeof(T),y,z), velocityIn[num*3], velocityIn[num*3+1], velocityIn[num*3+2]);
*/
    if(x < domainSize.x && y < domainSize.y && z < domainSize.z){
        surf3Dwrite(max(0,(surf3Dread<T>(densityPingSurf,x*sizeof(T),y,z)+inflowDensity[x + y * domainSize.x + z * domainSize.x * domainSize.y]*dt)//*
                    /*(!(obstacles[x + y * domainSize.x + z * domainSize.x*domainSize.y]))*/),densityPingSurf,x*sizeof(T),y,z);
        temperature[x + domainSize.x * y + domainSize.x * domainSize.y * z] = inflowTemperature[x + domainSize.x * y + domainSize.x * domainSize.y * z];}
        //isInScalarDomain = true;}
/*
    if(x == 32 && y==32 && z==32 && x < domainSize.x && y < domainSize.y && z < domainSize.z && surf3Dread<T>(densityPingSurf,x*sizeof(T),y,z) > 0)
        printf("densityPingSurf(%d,%d,%d) = %f, velocityIn(x,y,z).x = %f, velocityIn(x,y,z).y = %f, velocityIn(x,y,z).z = %f\n",
                x, y, z, surf3Dread<T>(densityPingSurf,x*sizeof(T),y,z), velocityIn[num*3], velocityIn[num*3+1], velocityIn[num*3+2]);*/

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
/*
    if(inflowVelocity[num*3]!=0 || inflowVelocity[num*3+1]!=0 || inflowVelocity[num*3+2]!=0)
        printf("velocityIn(%d,%d,%d).x = %f, velocityIn(x,y,z).y = %f, velocityIn(x,y,z).z = %f\n", x, y, z, velocityIn[num*3], velocityIn[num*3+1], velocityIn[num*3+2]);
*/

/*
    if(x < domainSize.x && y < domainSize.y && z < domainSize.z && surf3Dread<T>(densityPingSurf,x*sizeof(T),y,z) > 0)
        printf("densityPingSurf(%d,%d,%d) = %f, velocityIn(x,y,z).x = %f, velocityIn(x,y,z).y = %f, velocityIn(x,y,z).z = %f\n",
                x, y, z, surf3Dread<T>(densityPingSurf,x*sizeof(T),y,z), velocityIn[num*3], velocityIn[num*3+1], velocityIn[num*3+2]);
*/
/*
    if(velocityIn[num*3] > 0){buffer
        if(x < domainSize.x && y < domainSize.y && z < domainSize.z)
            printf("densityPingSurf(%d,%d,%d) = %f, velocityIn(x,y,z).x = %f, velocityIn(x,y,z).y = %f, velocityIn(x,y,z).z = %f\n",
                    x, y, z, surf3Dread<T>(densityPingSurf,x*sizeof(T),y,z), velocityIn[num*3], velocityIn[num*3+1], velocityIn[num*3+2]);
        else
            printf("velocityIn(%d,%d,%d).x = %f, velocityIn(x,y,z).y = %f, velocityIn(x,y,z).z = %f\n", x, y, z, velocityIn[num*3], velocityIn[num*3+1], velocityIn[num*3+2]);
    }*/
}

//this is the same as applyInflowsStaggered, except that it allows a spherical user-controlled density inflow of position (posX, posY), radius radius and constant density, velocity and temperature inflow _in addition_ to the other inflows
extern "C" cudaError_t applyControlledInflowsStaggered_float(float* velocityIn, cudaSurfaceObject_t densityPingSurf, float* obstacles, int posX, int posY, int posZ, int radius, float densityValue, float temperatureValue, float3 velocityValue, float* temperature, float* inflowVelocity, float* inflowDensity, float* inflowTemperature, dim3 domainSize, float dt){

    //printf("dt = %f\n", dt);

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
    //float val = surf3Dread<float>(densityPingSurf,x*sizeof(float),y,z)*(!obstacles[num])+inflowDensity[num];
    //float* valP = &val;
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
