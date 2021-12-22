#ifndef _CNN_CU_
#define _CNN_CU_

#include <cuda_runtime_api.h>
#include <helper_cuda.h>
#include "defines.h"
/*
template<typename T>
__global__ void addObstacles(cudaSurfaceObject_t densityPongSurf, cudaSurfaceOb){


}
*/

template<typename T>
__global__ void getStandardDeviationFlat(T* tensor, size_t tensorSize, T* sum){

    for(int i=0;i<tensorSize;i++)
        *sum += tensor[tensorSize];
}

extern "C" float getStandardDeviationFlatWrapper_float(float* tensor, size_t tensorSize){

    float *sum, sumCPU;
    checkCudaErrors(cudaMalloc(&sum, sizeof(float)));
    checkCudaErrors(cudaMemset(sum, 0, sizeof(float)));
    getStandardDeviationFlat<float><<<1,1>>>(tensor, tensorSize, sum);

    checkCudaErrors(cudaMemcpy(&sumCPU, sum, sizeof(float), cudaMemcpyDeviceToHost));

    return sumCPU;
}

template<typename T>
__global__ void setEdges(T* buffer, dim3 domainExtent, T value){

    int num = threadIdx.x + (blockIdx.x * blockDim.x);

    int x = num % domainExtent.x;
    int y =(num / domainExtent.x) % domainExtent.y;
    int z = num /(domainExtent.x  * domainExtent.y);

    if(x==0 || y==0 || z==0 || x==domainExtent.x-1 || y==domainExtent.y-1 || z==domainExtent.z-1)//{
        buffer[num] = value;
        //printf("buffer[%d,%d,%d] = %f\n",x,y,z,buffer[num]);}
}

extern "C" cudaError_t setEdgesWrapper_float(float* buffer, dim3 domainExtent, float value){

    int domainExtentTotal = domainExtent.x*domainExtent.y*domainExtent.z;
    setEdges<float><<<domainExtentTotal/1024 + (domainExtentTotal%1024!=0), 1024>>>(buffer, domainExtent, value);
}

#define inRange(array,index,length) (length>=index)?(array[index]):(-1)
//should probably always run with <<<1,1>>>
__global__ void printSingleElement_float(float* theDeviceArray, size_t numSizes, int* sizes, int* elementToGet){

  int offset = elementToGet[0];

  for(int i=1;i<numSizes;i++)
    offset = offset * sizes[i-1] + elementToGet[i];

  printf("theDeviceArray[%d][%d][%d][%d][%d] = %f\n",
         inRange(sizes,0,numSizes),
         inRange(sizes,1,numSizes),
         inRange(sizes,2,numSizes),
         inRange(sizes,3,numSizes),
         inRange(sizes,4,numSizes),
         5);//theDeviceArray[offset]);
}
#undef inRange

extern "C" cudaError_t printSingleElementWrapper_float(float* theDeviceArray, size_t numSizes, int* sizes, int* elementToGet){

  printSingleElement_float<<<1,1>>>(theDeviceArray, numSizes, sizes, elementToGet);
  //checkCudaErrors(cudaDeviceSynchronize());

  return cudaPeekAtLastError();
}

__global__ void setDeviceArrayConstant_float(float* theDeviceArray, unsigned int totalDeviceArraySize, float value){

    //printf("index = %d; totalDeviceArraySize = %d; value = %f\n", threadIdx.x + (blockIdx.x * blockDim.x), totalDeviceArraySize, value);

  if(threadIdx.x + (blockIdx.x * blockDim.x) >= totalDeviceArraySize)
    return;

  theDeviceArray[threadIdx.x + (blockIdx.x * blockDim.x)] = value;
  //printf("index = %d; totalDeviceArraySize = %d; value = %f\n", threadIdx.x + (blockIdx.x * blockDim.x), totalDeviceArraySize, value);

  //__syncthreads();
}

extern "C" cudaError_t setDeviceArrayConstantWrapper_float(float* theDeviceArray, dim3 dimensions, float value){

  printf("value = %f\n", value);
  setDeviceArrayConstant_float<<<dimensions.x*dimensions.y*dimensions.z/1024+1, 1024>>>(theDeviceArray, (unsigned int)(dimensions.x * dimensions.y * dimensions.z), value);

  printf("kernel %d\n", (dimensions.x*dimensions.y*dimensions.z/1024+1) * 1024);

  return cudaPeekAtLastError();

  //checkCudaErrors(cudaDeviceSynchronize());
}

//can only upsample in d, h and w, upsample of n or c is not useful here (or anywhere, probably)
//this does not handle bad parameters; output dimensions must be input dimensions dot upscale dimensions
__global__ void upscaleLayer3DKernel_float(float* input, float* output, int outputDimsN, int outputDimsC, int outputDimsD, int outputDimsH, int outputDimsW,
                                           int upscaleD, int upscaleH, int upscaleW){

    int ncdhw = threadIdx.x + (blockIdx.x * blockDim.x);

    int N = outputDimsN;
    int C = outputDimsC;
    int D = outputDimsD;
    int H = outputDimsH;
    int W = outputDimsW;

    //THIS WORKS, IDK WHY
    int x = N * C * D * H * W;

    int n = ncdhw / (C*D*H*W);
    int c = (ncdhw / (D*H*W)) % C;
    int d = (ncdhw / (H * W)) % D;
    int h = (ncdhw / W) % H;
    int w = ncdhw % W;

    //printf("n = %d, c = %d, d = %d, h = %d, w = %d\n", n, c, d, h, w);

    if(n >= N || c >= C || d >= D || h >= H || w >= W)
        return;

    int Cin = C;
    int Din = D / upscaleD;
    int Hin = H / upscaleH;
    int Win = W / upscaleW;

    output[(((n * C + c) * D + d) * H + h) * W + w] = input[(((n * Cin + c) * Din + d/upscaleD) * Hin + h/upscaleH) * Win + w/upscaleW];
}

__global__ void upscaleLayer3DKernel_double(double* input, double* output, int outputDimsN, int outputDimsC, int outputDimsD, int outputDimsH, int outputDimsW,
                                            int upscaleD, int upscaleH, int upscaleW){

    int ncdhw = threadIdx.x + (blockIdx.x * blockDim.x);

    int N = outputDimsN;
    int C = outputDimsC;
    int D = outputDimsD;
    int H = outputDimsH;
    int W = outputDimsW;

    //THIS WORKS, IDK WHY
    int x = N * C * D * H * W;

    int n = ncdhw / (C*D*H*W);
    int c = (ncdhw / (D*H*W)) % C;
    int d = (ncdhw / (H * W)) % D;
    int h = (ncdhw / W) % H;
    int w = ncdhw % W;

    //printf("n = %d, c = %d, d = %d, h = %d, w = %d\n", n, c, d, h, w);

    if(n >= N || c >= C || d >= D || h >= H || w >= W)
        return;

    int Cin = C;
    int Din = D / upscaleD;
    int Hin = H / upscaleH;
    int Win = W / upscaleW;

    output[(((n * C + c) * D + d) * H + h) * W + w] = input[(((n * Cin + c) * Din + d/upscaleD) * Hin + h/upscaleH) * Win + w/upscaleW];
}

/*
__global__ void upscaleLayer3DKernel_half(void* input, void* output, int* upscale){


}
*/

//this thing was trilinear upsampling (might have incorrect offsetting, and does not handle edges); model uses nearest upsampling (VolumetricUpSamplinbgNearest)

/*
//in this kernel, n, c, d, h, and w correspond to output
//for now, the kernel will ignore the scaling parameters for the n and c dimensions, since these are not normally scaled
//this reduces the number of elements to be interpolated from 32 to 8

//this interpolates int that output(x,y,z
__global__ void upscaleLayer3DKernel_float(float* input, float* output, int* outputDims, int* upscale){

//#define N outputDims[0]
//#define C outputDims[1]
#define D outputDims[2]
#define H outputDims[3]
#define W outputDims[4]
#define output3(d,h,w) output[(d * H + h) * W + w] //allows 3-dimension retrieval of input value
#define  input3(d,h,w)  input[(d * H + h) * W + w] //allows 3-dimension retrieval of output value

  //this'll probably need more thought for optimization
  //int ncd = threadIdx.x + (blockIdx.x * blockDim.x); //replace defines
  int d = threadIdx.x + (blockIdx.x * blockDim.x);
  int h = threadIdx.y + (blockIdx.y * blockDim.y);
  int w = threadIdx.z + (blockIdx.z * blockDim.z);

  //int d = ncd % D;
  //int c = (ncd / D) % C;
  //int n = ncd / (C * D);


  if(h > H || d > D || w > W)
    return;

  int d0 = d/D;
  int h0 = h/H;
  int w0 = w/W;

  float dd = (d % D) / D;
  float hd = (h % H) / H;
  float wd = (w % W) / W;

  float c00 = input3(  d0,  h0,  w0) * (1 - dd) + input3(d0+1,  h0,  w0);
  float c01 = input3(  d0,  h0,w0+1) * (1 - dd) + input3(d0+1,  h0,w0+1);
  float c10 = input3(  d0,  h0,w0+1) * (1 - dd) + input3(d0+1,  h0,w0+1);
  float c11 = input3(  d0,h0+1,w0+1) * (1 - dd) + input3(d0+1,h0+1,w0+1);

  float c0 =  c00 * (1 - hd) + c10 * hd;
  float c1 =  c01 * (1 - hd) + c11 * hd;

  output3(d,h,w) = c0 * (1 - wd) + c1 * wd;

  //output3(d,h,w) = input3(d/D, h/H, w/W) * (x % upscale) + input3(d,h,w);
}
*/

/*
__global__ void volumetricUpSamplingNearestForward(
    const int ratio, THCDeviceTensor<float, 5> in,
    THCDeviceTensor<float, 5> out) {
  const int pnt_id = threadIdx.x + blockIdx.x * blockDim.x;
  const int chan = blockIdx.y;
  const int batch = blockIdx.z;
  if (pnt_id >= (out.getSize(2) * out.getSize(3) * out.getSize(4))) {
    return;
  }
  const int x = pnt_id % out.getSize(4);
  const int y = (pnt_id / out.getSize(4)) % out.getSize(3);
  const int z = pnt_id / (out.getSize(3) * out.getSize(4));

  const int xin = x / ratio;
  const int yin = y / ratio;
  const int zin = z / ratio;
  const float inVal = in[batch][chan][zin][yin][xin];
  out[batch][chan][z][y][x] = inVal;
}
*/

extern "C" cudaError_t upscaleLayer3DKernelWrapper_float(float* input, float* output, int* outputDims, int* upscale){

/*
    printf("\n\nNot kernel:\n");
    for(int i=0;i<5;i++)
        printf("outputDims[%d] = %d\n",i,outputDims[i]);
    for(int i=0;i<3;i++)
        printf("upscale[%d] = %d\n",i,upscale[i]);
*/
    upscaleLayer3DKernel_float<<<outputDims[0]*outputDims[1]*outputDims[2]*outputDims[3]*outputDims[4]/1024 + 1, 1024>>>(input, output, outputDims[0], outputDims[1], outputDims[2], outputDims[3], outputDims[4],
                                                                                                                         upscale[0], upscale[1], upscale[2]);
    //printf("\n\n");
    //checkCudaErrors(cudaDeviceSynchronize());
    return cudaPeekAtLastError();
}

extern "C" cudaError_t upscaleLayer3DKernelWrapper_half(void* input, void* output, int* outputDims, int* upscale){
  printf("Half support not yet implemented (%s:%d)\n", __FILE__, __LINE__);
  exit(1);

  return cudaPeekAtLastError();
}

extern "C" cudaError_t upscaleLayer3DKernelWrapper_double(double* input, double* output, int* outputDims, int* upscale){

  upscaleLayer3DKernel_double<<<outputDims[0]*outputDims[1]*outputDims[2]*outputDims[3]*outputDims[4]/1024+1, 1024>>>(input, output, outputDims[0], outputDims[1], outputDims[2], outputDims[3], outputDims[4],
                                                                                                                      upscale[0], upscale[1], upscale[2]);

  return cudaPeekAtLastError();
}

__global__ void multiplyTensorByScalar_float(float *tensor, float *tensorOut, size_t tensorSize, float scalar){

  int i = threadIdx.x+(blockIdx.x*BLOCK_SIZE_DIVISION);

  if(i < tensorSize){
    tensorOut[i] = tensor[i] * scalar;
    //if(i % (128*128))
    //  printf("tensorOut[%d] = %f\n", i, tensorOut[i]);
    }

  //__syncthreads();
}

__global__ void multiplyTensorByScalar_double(double *tensor, double *tensorOut, size_t tensorSize, double scalar){

  int i = threadIdx.x+(blockIdx.x*BLOCK_SIZE_DIVISION);

  if(i < tensorSize){
    tensorOut[i] = tensor[i] * scalar;
    //if(i % (128*128))
    //  printf("tensorOut[%d] = %f\n", i, tensorOut[i]);
    }

  //__syncthreads();
}

//TODO: Also this
/*
__global__ void multiplyTensorByScalar_half(void  int i = threadIdx.x+(blockIdx.x*BLOCK_SIZE_DIVISION);

  if(i < tensorSize){
    tensorOut[i] = tensor[i] * scalar;
    //if(i % (128*128))
    //  printf("tensorOut[%d] = %f\n", i, tensorOut[i]);
    }

  __syncthreads(); *tensor, void *tensorOut, size_t tensorSize, void *scalar){

  int i = threadIdx.x+(blockIdx.x*BLOCK_SIZE_DIVISION);

  if(i < tensorSize){
    tensorOut[i] = tensor[i] * scalar;
    //if(i % (128*128))
    //  printf("tensorOut[%d] = %f\n", i, tensorOut[i]);
    }

  __syncthreads();
}
*/

extern "C" cudaError_t multiplyTensorByScalarWrapper_float(float *tensor, float *tensorOut, size_t tensorSize, float scalar){
  multiplyTensorByScalar_float<<<tensorSize/BLOCK_SIZE_DIVISION + bool(tensorSize%BLOCK_SIZE_DIVISION), BLOCK_SIZE_DIVISION>>>(tensor, tensorOut, tensorSize, scalar);

  return cudaPeekAtLastError();
}

extern "C" cudaError_t multiplyTensorByScalarWrapper_double(double *tensor, double *tensorOut, size_t tensorSize, double scalar){
  multiplyTensorByScalar_double<<<tensorSize/BLOCK_SIZE_DIVISION + bool(tensorSize%BLOCK_SIZE_DIVISION), BLOCK_SIZE_DIVISION>>>(tensor, tensorOut, tensorSize, scalar);
  return cudaPeekAtLastError();
}


extern "C" cudaError_t multiplyTensorByScalarWrapper_half(void *tensor, void *tensorOut, size_t tensorSize, void* scalar){

  errorMsg("Half support not yet implemented (%s:%d)\n", __FILE__, __LINE__);
  exit(1);
  //multiplyTensorByScalar_half<<<tensorSize/BLOCK_SIZE_DIVISION + bool(tensorSize%BLOCK_SIZE_DIVISION), BLOCK_SIZE_DIVISION>>>(tensor, tensorOut, tensorSize, scalar);
  return cudaPeekAtLastError();
}


//overloading doesn't work in C

//might need forceinline
//if changing this, the others need to be changed in the same way (..._double) or an analogous way (..._half)
__global__ void reduceSummation_float(float * input, float * output, size_t len) {
    //@@ Load a segment of the input vector into shared memory
    __shared__ float partialSum[2 * BLOCK_SIZE];
    unsigned int t = threadIdx.x, start = 2 * blockIdx.x * BLOCK_SIZE;
    if (start + t < len){
       partialSum[t] = input[start + t];}
       //if(partialSum[t] == NAN)
       //  infoMsg("ahoSum\n");
    else
       partialSum[t] = 0;
    if (start + BLOCK_SIZE + t < len)
       partialSum[BLOCK_SIZE + t] = input[start + BLOCK_SIZE + t];
    else
       partialSum[BLOCK_SIZE + t] = 0;
    //@@ Traverse the reduction tree
    for (unsigned int stride = BLOCK_SIZE; stride >= 1; stride >>= 1) {
       __syncthreads();
       if (t < stride)
          partialSum[t] += partialSum[t+stride];
    }

    __syncthreads();
    //@@ Write the computed sum of the block to the output vector at the
    //@@ correct index
    if (t == 0)
       output[blockIdx.x] = partialSum[0];

    //recursive kernel functions only work for __device__ kernels, and only on Fermi architecture, so don't try it
}

__global__ void reduceSummation_double(double * input, double * output, size_t len) {
    //@@ Load a segment of the input vector into shared memory
    __shared__ double partialSum[2 * BLOCK_SIZE];
    unsigned int t = threadIdx.x, start = 2 * blockIdx.x * BLOCK_SIZE;
    if (start + t < len){
       partialSum[t] = input[start + t];}
       //if(partialSum[t] == NAN)
       //  infoMsg("ahoSum\n");}
    else
       partialSum[t] = 0;
    if (start + BLOCK_SIZE + t < len)
       partialSum[BLOCK_SIZE + t] = input[start + BLOCK_SIZE + t];
    else
       partialSum[BLOCK_SIZE + t] = 0;
    //@@ Traverse the reduction tree
    for (unsigned int stride = BLOCK_SIZE; stride >= 1; stride >>= 1) {
       __syncthreads();
       if (t < stride)
          partialSum[t] += partialSum[t+stride];
    }

    __syncthreads();
    //@@ Write the computed sum of the block to the output vector at the
    //@@ correct index
    if (t == 0)
       output[blockIdx.x] = partialSum[0];

    //recursive kernel functions only work for __device__ kernels, and only on Fermi architecture, so don't try it
}


//TODO
/*
//using half2 instead of half is faster according to https://devblogs.nvidia.com/mixed-precision-programming-cuda-8/
//idk whether cudnn (when using CUDNN_DATA_HALF) wants half or half2, so here's both versions; just swap the kernel call in reduceSummationWrapper_half
__global__ void reduceSummation_half(void * input, void * output, size_t len) {
    //@@ Load a segment of the input vector into shared memory
    __shared__ double partialSum[2 * BLOCK_SIZE];
    unsigned int t = threadIdx.x, start = 2 * blockIdx.x * BLOCK_SIZE;
    if (start + t < len){
       partialSum[t] = input[start + t];}
       //if(partialSum[t] == NAN)
       //  infoMsg("ahoSum\n");}
    else
       partialSum[t] = 0;
    if (start + BLOCK_SIZE + t < len)
       partialSum[BLOCK_SIZE + t] = input[start + BLOCK_SIZE + t];
    else
       partialSum[BLOCK_SIZE + t] = 0;
    //@@ Traverse the reduction tree
    for (unsigned int stride = BLOCK_SIZE; stride >= 1; stride >>= 1) {
       __syncthreads();
       if (t < stride)
          partialSum[t] += partialSum[t+stride];
    }

    __syncthreads();
    //@@ Write the computed sum of the block to the output vector at the
    //@@ correct index
    if (t == 0)
       output[blockIdx.x] = partialSum[0];

    //recursive kernel functions only work for __device__ kernels, and only on Fermi architecture, so don't try it
}
*/

#define numBlocks tensorOutSize
#define threadsPerBlock BLOCK_SIZE
extern "C" cudaError_t reduceSummationWrapper_float(float *tensor, float *tensorOut, size_t tensorSize, size_t tensorOutSize){

  reduceSummation_float<<<numBlocks, threadsPerBlock>>>(tensor, tensorOut, tensorSize);
  return cudaPeekAtLastError();

  //checkCudaErrors(cudaDeviceSynchronize());
}

extern "C" cudaError_t reduceSummationWrapper_double(double *tensor, double *tensorOut, size_t tensorSize, size_t tensorOutSize){

  reduceSummation_double<<<numBlocks, threadsPerBlock>>>(tensor, tensorOut, tensorSize);
  return cudaPeekAtLastError();
}

//half can't exist in C (according to /usr/local/cuda-9.0/include/cuda_fp16.hpp:1791), so I guess we'll just use void pointers or something
extern "C" cudaError_t reduceSummationWrapper_half(void *tensor, void *tensorOut, size_t tensorSize, size_t tensorOutSize){

  errorMsg("Half support not yet implemented (%s:%d)\n", __FILE__, __LINE__);
  exit(1);
  //reduceSummationWrapper_half<<<numBlocks, threadsPerBlock>>>(tensor, tensorOut, tensorSize);
  return cudaPeekAtLastError();
}

__global__ void reduceSummationVariance_float(float * input, float * output, size_t len, float average){//}, size_t *maxIndex) {
    //@@ Load a segment of the input vector into shared memory
    __shared__ float partialSum[2 * BLOCK_SIZE];
    unsigned int t = threadIdx.x, start = 2 * blockIdx.x * BLOCK_SIZE;
    if (start + t < len){
       partialSum[t] = (input[start + t] - average) * (input[start + t] - average);// / len; //powf((input[start + BLOCK_SIZE + t] - average), 2); //for 1st step of variance calculation
       //printf("reduceSummationVariance_float, threadIdx = %d, start = %d\n, input", int(threadIdx.x), int(start));
       //*maxIndex = start + BLOCK_SIZE + t;
      }
    else
       partialSum[t] = 0;
    if (start + BLOCK_SIZE + t < len){
       partialSum[BLOCK_SIZE + t] = (input[start + BLOCK_SIZE + t] - average) * (input[start + BLOCK_SIZE + t] - average);// / len; //powf(input[start + BLOCK_SIZE + t] - average, 2); //for 1st step of variance calculation
       //*maxIndex = start + BLOCK_SIZE + t;
    }
    else
       partialSum[BLOCK_SIZE + t] = 0;
    //@@ Traverse the reduction tree
    for (unsigned int stride = BLOCK_SIZE; stride >= 1; stride >>= 1) {
       __syncthreads();
       if (t < stride)
          partialSum[t] += partialSum[t+stride];
    }
    //@@ Write the computed sum of the block to the output vector at the
    //@@ correct index
    __syncthreads();
    if (t == 0)
       output[blockIdx.x] = partialSum[0];

    //recursive kernel functions only work for __device__ kernels, and only on Fermi architecture, so don't try it
}

__global__ void reduceSummationVariance_double(double * input, double * output, size_t len, double average){//}, size_t *maxIndex) {
    //@@ Load a segment of the input vector into shared memory
    __shared__ double partialSum[2 * BLOCK_SIZE];
    unsigned int t = threadIdx.x, start = 2 * blockIdx.x * BLOCK_SIZE;
    if (start + t < len){
       partialSum[t] = (input[start + BLOCK_SIZE + t] - average) * (input[start + BLOCK_SIZE + t] - average); //powf((input[start + BLOCK_SIZE + t] - average), 2); //for 1st step of variance calculation
       //*maxIndex = start + BLOCK_SIZE + t;
      }
    else
       partialSum[t] = 0;
    if (start + BLOCK_SIZE + t < len){
       partialSum[BLOCK_SIZE + t] = (input[start + BLOCK_SIZE + t] - average) * (input[start + BLOCK_SIZE + t] - average); //powf(input[start + BLOCK_SIZE + t] - average, 2); //for 1st step of variance calculation
       //*maxIndex = start + BLOCK_SIZE + t;
    }
    else
       partialSum[BLOCK_SIZE + t] = 0;
    //@@ Traverse the reduction tree
    for (unsigned int stride = BLOCK_SIZE; stride >= 1; stride >>= 1) {
       __syncthreads();
       if (t < stride)
          partialSum[t] += partialSum[t+stride];
    }
    //@@ Write the computed sum of the block to the output vector at the
    //@@ correct index
    //__syncthreads();
    if (t == 0)
       output[blockIdx.x] = partialSum[0];

    //recursive kernel functions only work for __device__ kernels, and only on Fermi architecture, so don't try it
}

//TODO eventually
/*
__global__ void reduceSummationVariance_double(double * input, double * output, size_t len, double average){//}, size_t *maxIndex) {
    //@@ Load a segment of the input vector into shared memory
    __shared__ double partialSum[2 * BLOCK_SIZE];
    unsigned int t = threadIdx.x, start = 2 * blockIdx.x * BLOCK_SIZE;
    if (start + t < len){
       partialSum[t] = (input[start + BLOCK_SIZE + t] - average) * (input[start + BLOCK_SIZE + t] - average); //powf((input[start + BLOCK_SIZE + t] - average), 2); //for 1st step of variance calculation
       //*maxIndex = start + BLOCK_SIZE + t;
      }
    else
       partialSum[t] = 0;
    if (start + BLOCK_SIZE + t < len){
       partialSum[BLOCK_SIZE + t] = (input[start + BLOCK_SIZE + t] - average) * (input[start + BLOCK_SIZE + t] - average); //powf(input[start + BLOCK_SIZE + t] - average, 2); //for 1st step of variance calculation
       //*maxIndex = start + BLOCK_SIZE + t;
    }
    else
       partialSum[BLOCK_SIZE + t] = 0;
    //@@ Traverse the reduction tree
    for (unsigned int stride = BLOCK_SIZE; stride >= 1; stride >>= 1) {
       __syncthreads();
       if (t < stride)
          partialSum[t] += partialSum[t+stride];
    }
    //@@ Write the computed sum of the block to the output vector at the
    //@@ correct index
    //__syncthreads();
    if (t == 0)
       output[blockIdx.x] = partialSum[0];

    //recursive kernel functions only work for __device__ kernels, and only on Fermi architecture, so don't try it
}
*/

extern "C" cudaError_t reduceSummationVarianceWrapper_float(float *tensor, float *tensorOut, size_t tensorSize, size_t tensorOutSize, float average){

  reduceSummationVariance_float<<<numBlocks, threadsPerBlock>>>(tensor, tensorOut, tensorSize, average);
  return cudaPeekAtLastError();
}

extern "C" cudaError_t reduceSummationVarianceWrapper_double(double *tensor, double *tensorOut, size_t tensorSize, size_t tensorOutSize, double average){

  reduceSummationVariance_double<<<numBlocks, threadsPerBlock>>>(tensor, tensorOut, tensorSize, average);
  return cudaPeekAtLastError();
}

extern "C" cudaError_t reduceSummationVarianceWrapper_half(void *tensor, void *tensorOut, size_t tensorSize, void* average){
//                                                                                            ^--- ?

  errorMsg("Half support not yet implemented (%s:%d)\n", __FILE__, __LINE__);
  exit(1);
  //reduceSummationVariance_half<<<numBlocks, threadsPerBlock>>>(tensor, tensorOut, tensorSize, average);
  return cudaPeekAtLastError();
}

#undef numBlocks
#undef threadsPerBlock

//this might need variants for each of float, half and double
//all mallocs, callocs and cudaMallocs need to be in the

/*
#define FLOAT float
#define numBlocks tensorOutSize
#define threadsPerBlock BLOCK_SIZE
//#define multiplyByVarianceDevice
extern "C" FLOAT multiplyByVarianceDevice(FLOAT* tensor, FLOAT* tensorOut, size_t tensorSize){

  FLOAT* tensorOutInitial = tensorOut;
  size_t tensorSizeInitial = tensorSize;
  //dim3 numBlocks = dim3(NUM_BLOCKS_X, NUM_BLOCKS_Y, NUM_BLOCKS_Z);
  //dim3 threadsPerBlock = dim3(THREADSPERBLOCK_X, THREADSPERBLOCK_Y, THREADSPERBLOCK_Z);

  //this should make the output buffering stop. doesn't work on printfs in kernels
  //setvbuf(stdout, NULL, _IONBF, 0);

  size_t tensorOutSize = tensorSize / (BLOCK_SIZE << 1) + bool(tensorSize % (BLOCK_SIZE<<1));

  //
  //starting here, we calculate the average
  //



  //CUDA_CALL(cudaDeviceSynchronize());

  FLOAT *sum;
  CUDA_CALL(cudaMalloc((void**)&sum, sizeof(FLOAT)));

  //debugSummation<<<1,1>>>(tensor, tensorSize);

  reduceSummation<<<numBlocks, threadsPerBlock>>>(tensor, tensorOut, tensorSize);
  tensorSize = tensorOutSize;
  tensorOutSize = tensorSize / (BLOCK_SIZE << 1) + bool(tensorSize % (BLOCK_SIZE<<1));
  printf("tensorOutSize %d\n", tensorOutSize);

  //this gives a slightly different result for some reason; see how it acts after multiple sum reductions
  //debugSummation<<<1,1>>>(tensorOut, tensorSize);

  //in this configuration, the initial tensor is destroyed
  while(tensorSize > 1){
      reduceSummation<<<numBlocks, threadsPerBlock>>>(tensorOut, tensorOut + tensorSize, tensorSize);
      tensorOut += tensorSize;
      tensorSize = tensorOutSize;
      tensorOutSize = tensorSize / (BLOCK_SIZE << 1) + bool(tensorSize % (BLOCK_SIZE<<1));

      //debugSummation<<<1,1>>>(tensorOut, tensorSize);

      //printf("average %d\n", tensorOutSize);
    }

  //printf("\n");

  //tensorOut[0] now contains the sum of the elements

  FLOAT average;
  CUDA_CALL(cudaMemcpy(&average, tensorOut, sizeof(FLOAT), cudaMemcpyDeviceToHost)); //cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind

  //printf();

  //need to multiply by INITIAL tensorSize
  average /= tensorSizeInitial;

  //printf("average device %f\n", average);

  //
  //now we can calculate the variance
  //with reduceSummationVariance, the partial sum gets the values (x^2-average) in the first summation reduction
  //

  tensorSize = tensorSizeInitial;
  tensorOutSize = tensorSize / (BLOCK_SIZE << 1) + bool(tensorSize % (BLOCK_SIZE<<1));

  //reset the position of the output pointer
  tensorOut = tensorOutInitial;

  reduceSummationVariance<<<numBlocks, threadsPerBlock>>>(tensor, tensorOut, tensorSize, average);
  tensorSize = tensorOutSize;
  tensorOutSize = tensorSize / (BLOCK_SIZE << 1) + bool(tensorSize % (BLOCK_SIZE<<1));

  //debugSummation<<<1,1>>>(tensorOut, tensorSize);

  printf("tensorOutSize %d", tensorOutSize);

  //printArray<<<1,1>>>(tensorOut, tensorSize);

  //DON'T flip the pointers if using offsets
  //temp = tensor;
  //tensor = tensorOut;
  //tensorOut = temp;

  while(tensorSize > 1){
      reduceSummation<<<numBlocks, threadsPerBlock>>>(tensorOut, tensorOut + tensorSize, tensorSize);
      tensorOut += tensorSize;
      tensorSize = tensorOutSize;
      tensorOutSize = tensorSize / (BLOCK_SIZE << 1) + bool(tensorSize % (BLOCK_SIZE<<1));

      //debugSummation<<<1,1>>>(tensorOut, tensorSize);

      //printf("variance %d\n", tensorOutSize);
    }

  //printf("\n");

  //
  //now we multiply by the standard deviation
  //

  FLOAT standardDeviation;
  CUDA_CALL(cudaMemcpy(&standardDeviation, tensorOut, sizeof(FLOAT), cudaMemcpyDeviceToHost));

  standardDeviation /= tensorSizeInitial;

  printf("square standardDeviation device %f\n", standardDeviation);

  standardDeviation = sqrt(standardDeviation);

  printf("standardDeviation device %f\n", standardDeviation);

  tensorSize = tensorSizeInitial;

  tensorOut = tensorOutInitial;

  multiplyByVariance<<<tensorSize/BLOCK_SIZE_DIVISION + bool(tensorSize%BLOCK_SIZE_DIVISION), BLOCK_SIZE_DIVISION>>>(tensor, tensorOut, tensorSizeInitial, standardDeviation);

  printf("tensorOut multiplyByVarianceDevice -> %p\n", tensorOut);
  printf("tensor multiplyByVarianceDevice -> %p\n", tensor);

  printArray<<<1,1>>>(tensor, 16);

  //finalTensorOutput = tensor;

  //printf("standardDeviation device %f\n", standardDeviation);
  return standardDeviation;
}
*/

__global__ void printfDebug(){

  printf("eh");
}

__global__ void printElement(float* normalArray, cudaExtent dataExtent){

  int x = blockIdx.x*blockDim.x+threadIdx.x;
  int y = blockIdx.y*blockDim.y+threadIdx.y;
  int z = blockIdx.z*blockDim.z+threadIdx.z;

  int xyz = x + y * dataExtent.width + z * dataExtent.width * dataExtent.height;

  printf("normalArray[%d] = %f", xyz, normalArray[xyz]);
}

__global__ void printTextureContentsFloat4(cudaTextureObject_t dataTexture, cudaExtent dataExtent){

  int x = blockIdx.x*blockDim.x+threadIdx.x;
  int y = blockIdx.y*blockDim.y+threadIdx.y;
  int z = blockIdx.z*blockDim.z+threadIdx.z;

  //printf("%d %d %d", x, y, z);
/*
  if(x > 10 || y > 10 || z > 10)
    return;
*/
  if(x >= dataExtent.width || y >= dataExtent.height || z >= dataExtent.depth)
    return;

  float4 here = tex3D<float4>(dataTexture,x,y,z);

  //printf("blah");

  if(here.x!=0 || here.y!=0 || here.z!=0)
  //if(isnan(here.x) || isnan(here.y) || isnan(here.z))
    printf("dataTexture[%d][%d][%d] = ( %f, %f, %f)\n", x,y,z,here.x, here.y, here.z);

  __syncthreads();
}

__global__ void printTextureContents(cudaTextureObject_t dataTexture, cudaExtent dataExtent){

  int x = blockIdx.x*blockDim.x+threadIdx.x;
  int y = blockIdx.y*blockDim.y+threadIdx.y;
  int z = blockIdx.z*blockDim.z+threadIdx.z;

  //printf("%d %d %d", x, y, z);
/*
  if(x > 10 || y > 10 || z > 10)
    return;
*/
  if(x >= dataExtent.width || y >= dataExtent.height || z >= dataExtent.depth)
    return;

  float here = tex3D<float>(dataTexture,x,y,z);

  //printf("blah");

  if(here!=0)
    printf("dataTexture[%d][%d][%d] = %f\n", x,y,z,here);

  __syncthreads();
}

extern "C" cudaError_t
printDataCudaArrayContents3DWrapper(cudaArray* data){

  cudaTextureObject_t dataTexture;
  cudaResourceDesc dataResourceDesc;
  cudaTextureDesc dataTextureDesc;

  cudaChannelFormatDesc dataChannelFormatDesc;
  cudaExtent dataExtent;
  unsigned int dataFlags;

  //we are only interested in dataExtent
  cudaArrayGetInfo(&dataChannelFormatDesc, &dataExtent, &dataFlags, data);

  memset(&dataResourceDesc, 0, sizeof(dataResourceDesc));
  dataResourceDesc.resType = cudaResourceTypeArray;

  dataResourceDesc.res.array.array = data;

  memset(&dataTextureDesc, 0, sizeof(dataTextureDesc));
  dataTextureDesc.readMode = cudaReadModeElementType;

  checkCudaErrors(cudaCreateTextureObject(&dataTexture, &dataResourceDesc, &dataTextureDesc, NULL));

  printf("\nPrinting cudaArray contents...\n");

  //these might need to be the other way around
  //also they may need to be constant, or use templates

  printf("Running kernel with dim3( %d, %d, %d), dim3( %d, %d, %d)", BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKSIZE_Z, THREADSPERBLOCK_X, THREADSPERBLOCK_Y, THREADSPERBLOCK_Z);

  printTextureContentsFloat4<<<dim3(BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKSIZE_Z), dim3(THREADSPERBLOCK_X, THREADSPERBLOCK_Y, THREADSPERBLOCK_Z)>>>(dataTexture, dataExtent);

  //checkCudaErrors(cudaDeviceSynchronize());

  //printTextureContents<<<dim3(MAXTEXTURESIZE_X/64+1, MAXTEXTURESIZE_Y/64+1, MAXTEXTURESIZE_Z/64+1), dim3(64,64,64)>>>(dataTexture, dataExtent);

  return cudaPeekAtLastError();
}

__global__ void copyCudaArrayToDeviceArray_float(cudaTextureObject_t dataTexture, float* dst, cudaExtent dataExtent, int msg){

  int x = blockIdx.x*blockDim.x+threadIdx.x;
  int y = blockIdx.y*blockDim.y+threadIdx.y;
  int z = blockIdx.z*blockDim.z+threadIdx.z;

  //printf("%d %d %d", x, y, z);

  if(x >= dataExtent.width || y >= dataExtent.height || z >= dataExtent.depth)
    return;

  float here = tex3D<float>(dataTexture,x,y,z);
  if(isnan(here))
      here = 0.0f;

  int xyz = x + y * dataExtent.width + z * dataExtent.width * dataExtent.height;

  if(msg == 10 && here!=0.0f && x%4==0 && y==32 && z==32)
      printf("obstacles[%d,%d,%d] = %f\n", x, y, z, here);

  //if(here != 0.0f && msg >=0)
  //if(here != 0.0f && msg >= 0)
  //  printf("%d %d %d = %f, %p %d\n", x, y, z, here, (void*)(dst), msg);

  dst[xyz] = here;
/*
  if(x == 55 && y == 0 && z == 0){
      printf("\n%f\n",here);
      printf("\n%f %d\n",dst[xyz],xyz);}
*/
  //__syncthreads();
}

__global__ void copyCudaArrayToDeviceArray_double(cudaTextureObject_t dataTexture, double* dst, cudaExtent dataExtent){

  int x = blockIdx.x*blockDim.x+threadIdx.x;
  int y = blockIdx.y*blockDim.y+threadIdx.y;
  int z = blockIdx.z*blockDim.z+threadIdx.z;

  //printf("%d %d %d", x, y, z);

  if(x >= dataExtent.width || y >= dataExtent.height || z >= dataExtent.depth)
    return;

  //tex3D<> has no double variant, so use the non-template variant I guess
  //double here = tex3D<double>(dataTexture,x,y,z);

  //see https://stackoverflow.com/questions/35137213/texture-objects-for-doubles
  uint2 hereUint2;
  tex3D(&hereUint2, dataTexture, x, y, z);

  double here = __hiloint2double(hereUint2.y, hereUint2.x);

  int xyz = x + y * dataExtent.width + z * dataExtent.width * dataExtent.height;

  //printf("%d %d %d = %f\n", x, y, z, here);

  dst[xyz] = here;
/*
  if(x == 55 && y == 0 && z == 0){
      printf("\n%f\n",here);
      printf("\n%f %d\n",dst[xyz],xyz);}
*/
  //__syncthreads();
}

//TODO: this as well
/*
__global__ void copyCudaArrayToDeviceArray_half(cudaTextureObject_t dataTexture, half* dst, cudaExtent dataExtent){

  int x = blockIdx.x*blockDim.x+threadIdx.x;
  int y = blockIdx.y*blockDim.y+threadIdx.y;
  int z = blockIdx.z*blockDim.z+threadIdx.z;

  //printf("%d %d %d", x, y, z);

  if(x >= dataExtent.width || y >= dataExtent.height || z >= dataExtent.depth)
    return;

  half here = tex3D<half>(dataTexture,x,y,z);

  int xyz = x + y * dataExtent.width + z * dataExtent.width * dataExtent.height;

  //printf("%d %d %d = %f\n", x, y, z, here);

  dst[xyz] = here;

  //if(x == 55 && y == 0 && z == 0){
  //    printf("\n%f\n",here);
  //    printf("\n%f %d\n",dst[xyz],xyz);}

  __syncthreads();
}
*/

//call with float,float4 or double,double4
template<typename T, typename T4>
__global__ void copyCudaArrayToDeviceArray3D(cudaTextureObject_t srcTexture, T* dst, dim3 size){

    int num = blockIdx.x * blockDim.x + threadIdx.x;

    int x = num % size.x;
    int y =(num / size.x) % size.y;
    int z = num /(size.x  * size.y);

    if(x>=size.x || y>=size.y || z>=size.z)
        return;

    float4 dstVal = tex3D<T4>(srcTexture, x, y, z);

    //if(dstVal.x !=0 || dstVal.y !=0 || dstVal.z !=0)
    //    printf("srcTexture[ %d, %d, %d] = (%f, %f, %f)\n", x, y, z, dstVal.x, dstVal.y, dstVal.z);

    dst[num*3] = isnan(dstVal.x) ? 0 : dstVal.x;
    dst[num*3+1] = isnan(dstVal.y) ? 0 : dstVal.y;
    dst[num*3+2] = isnan(dstVal.z) ? 0 : dstVal.z;
}

extern "C" cudaError_t
copyCudaArrayToDeviceArrayWrapper3D_float(cudaArray* src, float* dst){

    cudaTextureObject_t srcTexture;
    cudaResourceDesc srcResourceDesc;
    cudaTextureDesc srcTextureDesc;

    cudaChannelFormatDesc srcChannelFormatDesc;
    cudaExtent srcExtent;
    unsigned int srcFlags;

    //we are only interested in srcExtent
    checkCudaErrors(cudaArrayGetInfo(&srcChannelFormatDesc, &srcExtent, &srcFlags, src));

    memset(&srcResourceDesc, 0, sizeof(srcResourceDesc));
    srcResourceDesc.resType = cudaResourceTypeArray;

    srcResourceDesc.res.array.array = src;

    memset(&srcTextureDesc, 0, sizeof(srcTextureDesc));
    srcTextureDesc.readMode = cudaReadModeElementType;

    checkCudaErrors(cudaCreateTextureObject(&srcTexture, &srcResourceDesc, &srcTextureDesc, NULL));

    dim3 size = dim3(srcExtent.width, srcExtent.height, srcExtent.depth);
    int sizeTotal = size.x * size.y * size.z;

    printf("sizeTotal = %d", sizeTotal);

#define numThreadsCudaArrayToDevice 1024
    copyCudaArrayToDeviceArray3D<float, float4><<<sizeTotal / numThreadsCudaArrayToDevice + bool(sizeTotal%numThreadsCudaArrayToDevice!=0), numThreadsCudaArrayToDevice>>>(srcTexture, dst, size);
#undef numThreadsCudaArrayToDevice

    return cudaPeekAtLastError();
}

//this associates a texture with a cudaArray, to copy it to a device array
//this is for float contents, might make it a template (?)
//might be faster to make objects only once and send them through (memory increase is probably negligible)
extern "C" cudaError_t
copyCudaArrayToDeviceArrayWrapper_float(cudaArray* src, float* dst, int msg){

    cudaTextureObject_t srcTexture;
    cudaResourceDesc srcResourceDesc;
    cudaTextureDesc srcTextureDesc;

    cudaChannelFormatDesc srcChannelFormatDesc;
    cudaExtent srcExtent;
    unsigned int srcFlags;

    //we are only interested in srcExtent
    checkCudaErrors(cudaArrayGetInfo(&srcChannelFormatDesc, &srcExtent, &srcFlags, src));

    memset(&srcResourceDesc, 0, sizeof(srcResourceDesc));
    srcResourceDesc.resType = cudaResourceTypeArray;

    srcResourceDesc.res.array.array = src;

    memset(&srcTextureDesc, 0, sizeof(srcTextureDesc));
    srcTextureDesc.readMode = cudaReadModeElementType;

    checkCudaErrors(cudaCreateTextureObject(&srcTexture, &srcResourceDesc, &srcTextureDesc, NULL));

    //I don't think this happens
    //printf("\nPrinting cudaArray contents...\n");

    //these might need to be the other way around
    //also they may need to be constant, or use templates

    infoMsg("Running kernel with dim3( %d, %d, %d), dim3( %d, %d, %d)", BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKSIZE_Z, THREADSPERBLOCK_X, THREADSPERBLOCK_Y, THREADSPERBLOCK_Z);

    copyCudaArrayToDeviceArray_float<<<dim3(BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKSIZE_Z), dim3(THREADSPERBLOCK_X, THREADSPERBLOCK_Y, THREADSPERBLOCK_Z)>>>(srcTexture, dst, srcExtent, msg);

    //checkCudaErrors(cudaDeviceSynchronize());
    return cudaPeekAtLastError();
}

extern "C" cudaError_t
copyCudaArrayToDeviceArrayWrapper_double(cudaArray* src, double* dst){

    cudaTextureObject_t srcTexture;
    cudaResourceDesc srcResourceDesc;
    cudaTextureDesc srcTextureDesc;

    cudaChannelFormatDesc srcChannelFormatDesc;
    cudaExtent srcExtent;
    unsigned int srcFlags;

    //we are only interested in srcExtent
    checkCudaErrors(cudaArrayGetInfo(&srcChannelFormatDesc, &srcExtent, &srcFlags, src));

    memset(&srcResourceDesc, 0, sizeof(srcResourceDesc));
    srcResourceDesc.resType = cudaResourceTypeArray;

    srcResourceDesc.res.array.array = src;

    memset(&srcTextureDesc, 0, sizeof(srcTextureDesc));
    srcTextureDesc.readMode = cudaReadModeElementType;

    checkCudaErrors(cudaCreateTextureObject(&srcTexture, &srcResourceDesc, &srcTextureDesc, NULL));

    //printf("\nPrinting cudaArray contents...\n");

    //these might need to be the other way around
    //also they may need to be constant, or use templates

    infoMsg("Running kernel with dim3( %d, %d, %d), dim3( %d, %d, %d)", BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKSIZE_Z, THREADSPERBLOCK_X, THREADSPERBLOCK_Y, THREADSPERBLOCK_Z);

    copyCudaArrayToDeviceArray_double<<<dim3(BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKSIZE_Z), dim3(THREADSPERBLOCK_X, THREADSPERBLOCK_Y, THREADSPERBLOCK_Z)>>>(srcTexture, dst, srcExtent);

    //checkCudaErrors(cudaDeviceSynchronize());
    return cudaPeekAtLastError();
}

//TODO again
extern "C" cudaError_t
copyCudaArrayToDeviceArrayWrapper_half(cudaArray* src, void* dst){

  errorMsg("Half support not yet implemented (%s:%d)\n", __FILE__, __LINE__);
  exit(1);

  return cudaPeekAtLastError();
}

//BOOKMARK
__global__ void copyDeviceArrayToCudaArray_float(cudaSurfaceObject_t dstSurface, cudaExtent dataExtent, float* src){

  int x = blockIdx.x*blockDim.x+threadIdx.x;
  int y = blockIdx.y*blockDim.y+threadIdx.y;
  int z = blockIdx.z*blockDim.z+threadIdx.z;

  //x=0;
  //y=0;
  //z=0;

  if(x >= dataExtent.width || y >= dataExtent.height || z >= dataExtent.depth)
    return;
/*
  if(x%10==0 && y%10==0 && z%10==0)
    infoMsg("src[%d][%d][%d] = %f", z,y,x,src[x + y * dataExtent.width + z * dataExtent.width * dataExtent.height]);
    //%f should work for doubles and floats, due to the float being promoted to double (at least in C)
*/
  surf3Dwrite(src[x + y * dataExtent.width + z * dataExtent.width * dataExtent.height],
              dstSurface,
              x*sizeof(float),y,z); //also has boundaryMode as optional parameter

}

__global__ void copyDeviceArrayToCudaArray_double(cudaSurfaceObject_t dstSurface, cudaExtent dataExtent, double* src){

  int x = blockIdx.x*blockDim.x+threadIdx.x;
  int y = blockIdx.y*blockDim.y+threadIdx.y;
  int z = blockIdx.z*blockDim.z+threadIdx.z;

  if(x >= dataExtent.width || y >= dataExtent.height || z >= dataExtent.depth)
    return;

  if(x%10==0 && y%10==0 && z%10==0)
    infoMsg("src[%d][%d][%d] = %f", z,y,x,src[x + y * dataExtent.width + z * dataExtent.width * dataExtent.height]);
    //%f should work for doubles and floats, due to the float being promoted to double (at least in C)

  //surf3Dwrite can't handle doubles directly
  //how about long long int?
  long long int srcHereReinterpreted = __double_as_longlong(src[x + y * dataExtent.width + z * dataExtent.width * dataExtent.height]);

  //uint2 hereUint2;
  //hereUint2.= __double2

  //double here = __hiloint2double(hereUint2.y, hereUint2.x);

  //doesn't work for double
  surf3Dwrite(srcHereReinterpreted, //src[x + y * dataExtent.width + z * dataExtent.width * dataExtent.height],
              dstSurface,
              x,y,z); //also has boundaryMode as optional parameter
}

//TODO
/*
__global__ void copyDeviceArrayToCudaArray_half(){

}
*/

template<typename T, typename T4>
__device__ __forceinline__ T4 make_T4(T x, T y, T z, T w);

template<>
__device__ __forceinline__ float4 make_T4<float>(float x, float y, float z, float w){

    return make_float4(x, y, z, w);
}

template<>
__device__ __forceinline__ double4 make_T4<double>(double x, double y, double z, double w){

    return make_double4(x, y, z, w);
}

template<typename T, typename T4>
__global__ void copyDeviceArrayToCudaArray3D(cudaSurfaceObject_t dstSurface, cudaExtent destExtent, T* src){

    int x = blockIdx.x*blockDim.x+threadIdx.x;
    int y = blockIdx.y*blockDim.y+threadIdx.y;
    int z = blockIdx.z*blockDim.z+threadIdx.z;

    //x=0;
    //y=0;
    //z=0;

    if(x >= destExtent.width || y >= destExtent.height || z >= destExtent.depth)
      return;

    int num = x + y * destExtent.width + z * destExtent.width * destExtent.height;
  /*
    if(x%10==0 && y%10==0 && z%10==0)
      infoMsg("src[%d][%d][%d] = %f", z,y,x,src[x + y * dataExtent.width + z * dataExtent.width * dataExtent.height]);
      //%f should work for doubles and floats, due to the float being promoted to double (at least in C)
  */
    //won't work with doubles if this is here
    T4 srcVal = make_T4<T, T4>(src[num*3], src[num*3+1], src[num*3+2], 0);

    surf3Dwrite(srcVal,
                dstSurface,
                x*sizeof(T4),y,z); //also has boundaryMode as optional parameter
}

extern "C" cudaError_t
copyDeviceArrayToCudaArray3DWrapper_float(float* src, cudaArray* dest){

    cudaSurfaceObject_t dstSurface;

    //create cudaTextureResourceDesc for cudaArray
    cudaResourceDesc dstDesc;
    memset(&dstDesc, 0, sizeof(cudaResourceDesc));
    dstDesc.resType = cudaResourceTypeArray;
    dstDesc.res.array.array = dest;

    cudaExtent destExtent;
    cudaChannelFormatDesc destChannelFormatDesc; //useless, but needed by cudaArrayGetInfo
    unsigned int flags; //also useless

    cudaArrayGetInfo(&destChannelFormatDesc, &destExtent, &flags, dest);

    //printf("velocity destExtent (depth,height,width) = %d %d %d \n", destExtent.depth, destExtent.height, destExtent.width);
    //printf("destChannelFormatDesc.x = %d;\ndestChannelFormatDesc.y = %d;\ndestChannelFormatDesc.z = %d;\ndestChannelFormatDesc.w = %d\n\n",
    //       destChannelFormatDesc.x, destChannelFormatDesc.y, destChannelFormatDesc.z, destChannelFormatDesc.w);

    //checkCudaErrors(cudaBindSurfaceToArray(dstSurface,)); //no, not this for surface objects

    checkCudaErrors(cudaCreateSurfaceObject(&dstSurface, &dstDesc));

    //printf("Running copyDeviceArrayToCudaArray3DWrapper_float with <<<dim3( %d, %d, %d),dim3( %d, %d, %d)>>>\n", );

    copyDeviceArrayToCudaArray3D<float, float4><<<dim3(destExtent.width /THREADSPERBLOCK_X+(destExtent.width %THREADSPERBLOCK_X!=0),
                                                       destExtent.height/THREADSPERBLOCK_Y+(destExtent.height%THREADSPERBLOCK_Y!=0),
                                                       destExtent.depth /THREADSPERBLOCK_Z+(destExtent.depth %THREADSPERBLOCK_Z!=0)),
                                                  dim3(THREADSPERBLOCK_X, THREADSPERBLOCK_Y, THREADSPERBLOCK_Z)>>>(dstSurface, destExtent, src);

    checkCudaErrors(cudaDeviceSynchronize());
    return cudaPeekAtLastError();
}

extern "C" cudaError_t
copyDeviceArrayToCudaArrayWrapper_float(float* src, cudaArray* dest){

  cudaSurfaceObject_t dstSurface;

  //create cudaTextureResourceDesc for cudaArray
  cudaResourceDesc dstDesc;
  memset(&dstDesc, 0, sizeof(cudaResourceDesc));
  dstDesc.resType = cudaResourceTypeArray;
  dstDesc.res.array.array = dest;

  cudaExtent destExtent;
  cudaChannelFormatDesc destChannelFormatDesc; //useless, but needed by cudaArrayGetInfo
  unsigned int flags; //also useless

  cudaArrayGetInfo(&destChannelFormatDesc, &destExtent, &flags, dest);

  //printf("destExtent (depth,height,width) = %d %d %d \n", destExtent.depth, destExtent.height, destExtent.width);

  //checkCudaErrors(cudaBindSurfaceToArray(dstSurface,)); //no, not this for surface objects

  checkCudaErrors(cudaCreateSurfaceObject(&dstSurface, &dstDesc));

  copyDeviceArrayToCudaArray_float<<<dim3(BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKSIZE_Z), dim3(THREADSPERBLOCK_X, THREADSPERBLOCK_Y, THREADSPERBLOCK_Z)>>>(dstSurface, destExtent, src);

  //checkCudaErrors(cudaDeviceSynchronize());
  /*cudaError_t cudaErr = cudaDeviceSynchronize();

  if(cudaErr!=cudaSuccess){
    printf("FAIL copyCudaArrayToDeviceArrayWrapper -%s \n",cudaGetErrorString(cudaErr));
    return;}*/

  return cudaPeekAtLastError();
}

extern "C" cudaError_t
copyDeviceArrayToCudaArrayWrapper_double(double* src, cudaArray* dest){

  cudaSurfaceObject_t dstSurface;

  //create cudaTextureResourceDesc for cudaArray
  cudaResourceDesc dstDesc;
  memset(&dstDesc, 0, sizeof(cudaResourceDesc));
  dstDesc.resType = cudaResourceTypeArray;
  dstDesc.res.array.array = dest;

  cudaExtent destExtent;
  cudaChannelFormatDesc destChannelFormatDesc; //useless, but needed by cudaArrayGetInfo
  unsigned int flags; //also useless

  cudaArrayGetInfo(&destChannelFormatDesc, &destExtent, &flags, dest);

  //checkCudaErrors(cudaBindSurfaceToArray(dstSurface,)); //no, not this for surface objects

  checkCudaErrors(cudaCreateSurfaceObject(&dstSurface, &dstDesc));

  copyDeviceArrayToCudaArray_double<<<dim3(BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKSIZE_Z), dim3(THREADSPERBLOCK_X, THREADSPERBLOCK_Y, THREADSPERBLOCK_Z)>>>(dstSurface,destExtent,src);

  //checkCudaErrors(cudaDeviceSynchronize());
  /*cudaError_t cudaErr = cudaDeviceSynchronize();

  if(cudaErr!=cudaSuccess){
    printf("FAIL copyCudaArrayToDeviceArrayWrapper -%s \n",cudaGetErrorString(cudaErr));
    return;}*/

  return cudaPeekAtLastError();
}

extern "C" cudaError_t
copyDeviceArrayToCudaArrayWrapper_half(void* src, cudaArray* dest){

  errorMsg("Half support not yet implemented (%s:%d)\n", __FILE__, __LINE__);
  exit(1);

  return cudaPeekAtLastError();
}

//this associates a texture with a cudaArray, to print its contents
extern "C" cudaError_t
printDataCudaArrayContentsWrapper(cudaArray* data){

  cudaTextureObject_t dataTexture;
  cudaResourceDesc dataResourceDesc;
  cudaTextureDesc dataTextureDesc;

  cudaChannelFormatDesc dataChannelFormatDesc;
  cudaExtent dataExtent;
  unsigned int dataFlags;

  //we are only interested in dataExtent
  cudaArrayGetInfo(&dataChannelFormatDesc, &dataExtent, &dataFlags, data);

  memset(&dataResourceDesc, 0, sizeof(dataResourceDesc));
  dataResourceDesc.resType = cudaResourceTypeArray;

  dataResourceDesc.res.array.array = data;

  memset(&dataTextureDesc, 0, sizeof(dataTextureDesc));
  dataTextureDesc.readMode = cudaReadModeElementType;

  checkCudaErrors(cudaCreateTextureObject(&dataTexture, &dataResourceDesc, &dataTextureDesc, NULL));

  printf("\nPrinting cudaArray contents...\n");

  //these might need to be the other way around
  //also they may need to be constant, or use templates

  printf("Running kernel with dim3( %d, %d, %d), dim3( %d, %d, %d)", BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKSIZE_Z, THREADSPERBLOCK_X, THREADSPERBLOCK_Y, THREADSPERBLOCK_Z);

  printTextureContents<<<dim3(BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKSIZE_Z), dim3(THREADSPERBLOCK_X, THREADSPERBLOCK_Y, THREADSPERBLOCK_Z)>>>(dataTexture, dataExtent);

  //checkCudaErrors(cudaDeviceSynchronize());

  //printTextureContents<<<dim3(MAXTEXTURESIZE_X/64+1, MAXTEXTURESIZE_Y/64+1, MAXTEXTURESIZE_Z/64+1), dim3(64,64,64)>>>(dataTexture, dataExtent);

  return cudaPeekAtLastError();
}

extern "C" cudaError_t
printElementWrapper(float* normalArray, cudaExtent extent){

  printElement<<<1,1>>>(normalArray, extent);
  return cudaPeekAtLastError();
}

extern "C" cudaError_t
printfDebugWrapper(){

  printfDebug<<<1,10>>>();
  return cudaPeekAtLastError();
}

extern "C" cudaError_t
doPrintfWrapper(cudaTextureObject_t someTex, int i, int j, int k){

  //printf("bek");

  //doPrintf<<<10,10>>>(someTex, i, j, k);

  printfDebug<<<10,10>>>();
/*
  if(cudaDeviceSynchronize()!=cudaSuccess)
    printf("FAIL");*/
  return cudaPeekAtLastError();
}

__global__ void printfArray(float* theArray, dim3 theArraySizes, int someFactor, dim3 printStride, bool printZeros){

  //return;
  int x = blockIdx.x*blockDim.x+threadIdx.x;
  int y = blockIdx.y*blockDim.y+threadIdx.y;
  int z = blockIdx.z*blockDim.z+threadIdx.z;

  if(x >= theArraySizes.x || y >= theArraySizes.y || z >= theArraySizes.z * someFactor)
    return;

  int numEl = x + y * theArraySizes.x + z * theArraySizes.x * theArraySizes.y;

  //printf("%d ", numEl);

  //if(z % printStride.z || y % printStride.y || x % printStride.x)
  //  return;

  if(theArray[numEl] == 0 && !printZeros)
    return;

  printf("%d %d %d = %f\n", x, y, z, theArray[numEl]);//numEl/int(theArraySizes.y*theArraySizes.z), (numEl/int(theArraySizes.z))%(int(theArraySizes.y)), numEl%int(theArraySizes.z), theArray[numEl]);

  //__syncthreads();
}

///<<<NumOfBlocks, NumOfThreadsPerBlock>>>

#define min(a,b) (a<b)?(a):(b)
#define max(a,b) (a>b)?(a):(b)

//someFactor is a lazy way to allow for printing the input, which has three channels instead of one
//printStride means only the elements with z % printStride.z == 0, y % printStride.y == 0, and x % printStride.x == 0 are printed
extern "C" cudaError_t printfArrayWrapper(float* theArray, dim3 theArraySize, int someFactor, dim3 printStride, bool printZeros){

    return (cudaError_t)(0);

  infoMsg("\n printfArrayWrapper run with device array sizes %d %d %d \n", theArraySize.x, theArraySize.y, theArraySize.z);

  printfArray<<<dim3(BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKSIZE_Z * someFactor), dim3(THREADSPERBLOCK_X, THREADSPERBLOCK_Y, THREADSPERBLOCK_Z)>>>(theArray, theArraySize, someFactor, printStride, printZeros);

  //checkCudaErrors(cudaDeviceSynchronize());
  return cudaPeekAtLastError();
}

//this is modified from tfluids/third_party/tfluids.cc/.cu
//since we are breaking free of the OpenGL requirements for the velocity, could turn it into a float4 array or a triple float array
//make it a triple float array
//vel(i,j,k,l) would correspond to
//what is b? b is the batch, which is not interesting for single batch 1D memory
//mantaflow documentation might make this make sense
// *****************************************************************************
// setWallBcsForward
// *****************************************************************************
//chan

//this function should just overwrite the velocity in occupied cells with the obstacle velocities
#define vel(d,h,w,chan) velocity[((d*(obstacleExtent.y+1) + h)*(obstacleExtent.x+1) + w)*3 + chan]
#define obs(d,h,w) obstacles[(d*obstacleExtent.y + h)*obstacleExtent.x + w]
#define isObstacle(d,h,w) obs(d,h,w)==1
#define isFluid(d,h,w) obs(d,h,w)==0
template<typename T>
__global__ void setWallBcsStaggered(T* obstacles, T* velocity, dim3 obstacleExtent, bool sticky) {
  int32_t chan, k, j, i;

  const bool cur_fluid = isFluid(i, j, k);
  const bool cur_obs = isObstacle(i, j, k);
  if (!cur_fluid && !cur_obs) {
    return;
  }

  // we use i > 0 instead of bnd=1 to check outer wall
  if (i > 0 && isObstacle(i - 1, j, k)) {
    // TODO(tompson): Set to (potentially) non-zero obstacle velocity.
    vel(i, j, k, 0) = 0;
  }
  if (i > 0 && cur_obs && isFluid(i - 1, j, k)) {
    vel(i, j, k, 0) = 0;
  }
  if (j > 0 && isObstacle(i, j - 1, k)) {
    vel(i, j, k, 1) = 0;
  }
  if (j > 0 && cur_obs && isFluid(i, j - 1, k)) {
    vel(i, j, k, 1) = 0;
  }

  if (k > 0 && isObstacle(i, j, k - 1)) {
    vel(i, j, k, 2) = 0;
  }
  if (k > 0 && cur_obs && isFluid(i, j, k - 1)) {
    vel(i, j, k, 2) = 0;
  }

  //instead if the isStick flag, we use a global flag in sticky (might change it to a grid flag eventually)
/*
  if (cur_fluid) {
    if ((i > 0 && isStick(i - 1, j, k)) ||
        (i < xsize() - 1 && isStick(i + 1, j, k))) {
      vel(i, j, k, 1) = 0;
      if (vel.is_3d()) {
        vel(i, j, k, 2) = 0;
      }
    }
    if ((j > 0 && isStick(i, j - 1, k)) ||
        (j < ysize() - 1 && isStick(i, j + 1, k))) {
      vel(i, j, k, 0) = 0;
      if (vel.is_3d()) {
        vel(i, j, k, 2) = 0;
      }
    }
    if (vel.is_3d() &&
        ((k > 0 && isStick(i, j, k - 1)) ||
         (k < zsize() - 1 && isStick(i, j, k + 1)))) {
      vel(i, j, k, 0) = 0;
      vel(i, j, k, 1) = 0;
    }
  }
*/
}
#undef obs
#undef vel

//VEL() AND OBS() CHANGE HERE
template<typename T>
__global__ void setWallBcsNotStaggered(T* obstacles, T* velocity, dim3 obstacleExtent){

    int D = obstacleExtent.z; //dim3 contains ints
    int H = obstacleExtent.y;
    int W = obstacleExtent.x;

    int dhw = threadIdx.x + blockDim.x * blockIdx.x;
    int d = dhw / (H*W);
    int h = (dhw / W) % D;
    int w = dhw % W;

    if(d >= D || h>= H || w>=W)
        return;

    if(obstacles[dhw] != 0)
        velocity[dhw] = 0; //THIS DOES NOT ACCOUNT FOR MOVING OBSTACLES
}

//this makes all edges of the domain obstacles; can set the thickness
//copy from tfluids or wherever in torch
//run once at beginning
//the templating will probably have to be removed or made useless if half support is ever added, since half probably needs a specialized kernel.
template<typename T>
__global__ void setWalls(T* obstacles, dim3 domainExtent, size_t thickness){

    //1D kernels
    int dhw = threadIdx.x + blockIdx.x * blockDim.x;

    int D = domainExtent.z;
    int H = domainExtent.y;
    int W = domainExtent.x;

    int d = dhw / (H*W);
    int h = (dhw / W) % D;
    int w = dhw % W;

    if(d >= D || h>= H || w>=W)
        return;

    if(d < thickness || h < thickness || w < thickness)
        obstacles[dhw] = 1.0f;

    if(D - d <= thickness || H - h <= thickness || W - w <= thickness)
        obstacles[dhw] = 1.0f;
}

extern "C" cudaError_t setWallsWrapper_float(float* obstacles, dim3 domainExtent, size_t thickness){

    int totalThreads = domainExtent.x * domainExtent.y * domainExtent.z;

    setWalls<float><<<totalThreads/BLOCK_SIZE, BLOCK_SIZE>>>(obstacles, domainExtent, thickness);
    return cudaPeekAtLastError();
}

extern "C" cudaError_t setWallsWrapper_double(double* obstacles, dim3 domainExtent, size_t thickness){

    int totalThreads = domainExtent.x * domainExtent.y * domainExtent.z;

    setWalls<double><<<totalThreads/BLOCK_SIZE, BLOCK_SIZE>>>(obstacles, domainExtent, thickness);
    return cudaPeekAtLastError();
}

extern "C" cudaError_t setWallsWrapper_half(void* obstacles, dim3 domainExtent, size_t thickness){

    printf("Half support not yet implemented (%s:%d)\n", __FILE__, __LINE__);
    exit(1);
    /*
    int totalThreads = domainExtent.x * domainExtent.y * domainExtent.z;

    setWalls_half<<<totalThreads/BLOCK_SIZE, BLOCK_SIZE>>>(obstacles, domainExtent, thickness);
    */
    return cudaPeekAtLastError();
}

template<typename T>
__global__ void updateObstacles(dim3 obstacleExtent){//T* obstacles, T* velocityIn, dim3 obstacleExtent){

    //does nothing, can update for whatever time-dependent obstacles we need
    //this will require changes to setWallBcsStaggered
    //if Unity is used, need to get movement from it
    //not sure there's an easy way to do rapid voxelization using Unity objects
    //could try sending the global positions of all the vertices, alaos the faces and edges, and doing it in CUDA somehow
    //in the first phase, could just use a simple-geometry object that can be voxelized trivially
    //this should also update the voxel speed, wherever they're contained
}

extern "C" cudaError_t updateObstaclesDNWrapper_float(dim3 obstacleExtent){//float* obstacles, float* velocityIn, dim3 obstacleExtent){

    int D = obstacleExtent.z;
    int H = obstacleExtent.y;
    int W = obstacleExtent.x;

    updateObstacles<float><<<D*H*W/BLOCK_SIZE,BLOCK_SIZE>>>(obstacleExtent);//obstacles, velocityIn, obstacleExtent);
    return cudaPeekAtLastError();
}

extern "C" cudaError_t updateObstaclesDNWrapper_double(dim3 obstacleExtent){//double* obstacles, double* velocityIn, dim3 obstacleExtent){

    int D = obstacleExtent.z;
    int H = obstacleExtent.y;
    int W = obstacleExtent.x;

    updateObstacles<double><<<D*H*W/BLOCK_SIZE,BLOCK_SIZE>>>(obstacleExtent);//obstacles, velocityIn, obstacleExtent);
    return cudaPeekAtLastError();
}

extern "C" cudaError_t updateObstaclesDNWrapper_half(dim3 obstacleExtent){

    printf("Half support not yet implemented (%s:%d)\n", __FILE__, __LINE__);
    exit(1);
    //int D = obstacleExtent.z;
    //int H = obstacleExtent.y;
    //int W = obstacleExtent.x;

    //updateObstacles<float><<<D*H*W/BLOCK_SIZE,BLOCK_SIZE>>>();
    return cudaPeekAtLastError();
}

#undef min
#undef max

#endif
