#ifndef _CNN_CU_
#define _CNN_CU_

#include <cuda_runtime_api.h>
#include <helper_cuda.h>
#include "defines.h"

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

///
/// \brief printSingleElement_float - print float element at indices (run with <<<1,1>>>)
/// \param theDeviceArray
/// \param numSizes
/// \param sizes
/// \param elementToGet
///
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
  return cudaPeekAtLastError();
}

///
/// \brief setDeviceArrayConstant_float - set elements of 1D array to float constant "value"(run as 1D kernel)
/// \param theDeviceArray
/// \param totalDeviceArraySize
/// \param value
///
__global__ void setDeviceArrayConstant_float(float* theDeviceArray, unsigned int totalDeviceArraySize, float value){

  if(threadIdx.x + (blockIdx.x * blockDim.x) >= totalDeviceArraySize)
    return;

  theDeviceArray[threadIdx.x + (blockIdx.x * blockDim.x)] = value;
}

extern "C" cudaError_t setDeviceArrayConstantWrapper_float(float* theDeviceArray, dim3 dimensions, float value){

  //TODO: doesn't really make sense to run this as a 2D kernel given contents of setDeviceArrayConstant_float
  setDeviceArrayConstant_float<<<dimensions.x*dimensions.y*dimensions.z/1024+1, 1024>>>(theDeviceArray, (unsigned int)(dimensions.x * dimensions.y * dimensions.z), value);

  return cudaPeekAtLastError();

  //checkCudaErrors(cudaDeviceSynchronize());
}

//can only upsample in d, h and w, upsample of n or c is not useful here (or anywhere, probably)
//this does not handle bad parameters; output dimensions must be input dimensions dot upscale dimensions
///
/// \brief upscaleLayer3DKernel_float - Upscales input and puts result in output
/// \param input
/// \param output
/// \param outputDimsN
/// \param outputDimsC
/// \param outputDimsD
/// \param outputDimsH
/// \param outputDimsW
/// \param upscaleD
/// \param upscaleH
/// \param upscaleW
///
__global__ void upscaleLayer3DKernel_float(float* input, float* output, int outputDimsN, int outputDimsC, int outputDimsD, int outputDimsH, int outputDimsW,
                                           int upscaleD, int upscaleH, int upscaleW){

    int ncdhw = threadIdx.x + (blockIdx.x * blockDim.x);

    int N = outputDimsN;
    int C = outputDimsC;
    int D = outputDimsD;
    int H = outputDimsH;
    int W = outputDimsW;

    int x = N * C * D * H * W;

    int n = ncdhw / (C*D*H*W);
    int c = (ncdhw / (D*H*W)) % C;
    int d = (ncdhw / (H * W)) % D;
    int h = (ncdhw / W) % H;
    int w = ncdhw % W;

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

    int x = N * C * D * H * W;

    int n = ncdhw / (C*D*H*W);
    int c = (ncdhw / (D*H*W)) % C;
    int d = (ncdhw / (H * W)) % D;
    int h = (ncdhw / W) % H;
    int w = ncdhw % W;;

    if(n >= N || c >= C || d >= D || h >= H || w >= W)
        return;

    int Cin = C;
    int Din = D / upscaleD;
    int Hin = H / upscaleH;
    int Win = W / upscaleW;

    output[(((n * C + c) * D + d) * H + h) * W + w] = input[(((n * Cin + c) * Din + d/upscaleD) * Hin + h/upscaleH) * Win + w/upscaleW];
}

extern "C" cudaError_t upscaleLayer3DKernelWrapper_float(float* input, float* output, int* outputDims, int* upscale){

    upscaleLayer3DKernel_float<<<outputDims[0]*outputDims[1]*outputDims[2]*outputDims[3]*outputDims[4]/1024 + 1, 1024>>>(input, output, outputDims[0], outputDims[1], outputDims[2], outputDims[3], outputDims[4],
                                                                                                                         upscale[0], upscale[1], upscale[2]);
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
    }
}

__global__ void multiplyTensorByScalar_double(double *tensor, double *tensorOut, size_t tensorSize, double scalar){

  int i = threadIdx.x+(blockIdx.x*BLOCK_SIZE_DIVISION);

  if(i < tensorSize){
    tensorOut[i] = tensor[i] * scalar;
    }
}

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

//TODO: might need forceinline
//TODO: find code this is based on
//Note: if changing this, the others need to be changed in the same way (..._double) or an analogous way (..._half)
///
/// \brief reduceSummation_float - do one step of parallel sum reduction, for 1D float arrays
/// \param input - current input for this step
/// \param output - current output for this step
/// \param len - length of input
///
__global__ void reduceSummation_float(float * input, float * output, size_t len) {
    // Load a segment of the input vector into shared memory
    __shared__ float partialSum[2 * BLOCK_SIZE];
    unsigned int t = threadIdx.x, start = 2 * blockIdx.x * BLOCK_SIZE;
    if (start + t < len){
       partialSum[t] = input[start + t];}
    else
       partialSum[t] = 0;
    if (start + BLOCK_SIZE + t < len)
       partialSum[BLOCK_SIZE + t] = input[start + BLOCK_SIZE + t];
    else
       partialSum[BLOCK_SIZE + t] = 0;
    // Traverse the reduction tree
    for (unsigned int stride = BLOCK_SIZE; stride >= 1; stride >>= 1) {
       __syncthreads();
       if (t < stride)
          partialSum[t] += partialSum[t+stride];
    }

    __syncthreads();
    // Write the computed sum of the block to the output vector at the
    // correct index
    if (t == 0)
       output[blockIdx.x] = partialSum[0];

    //recursive kernel functions only work for __device__ kernels, and only on Fermi architecture, so don't try it
}

///
/// \brief reduceSummation_double - do one step of parallel sum reduction, for 1D double arrays
/// \param input - current input for this step
/// \param output - current output for this step
/// \param len - length of input
///
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

#define numBlocks tensorOutSize
#define threadsPerBlock BLOCK_SIZE
extern "C" cudaError_t reduceSummationWrapper_float(float *tensor, float *tensorOut, size_t tensorSize, size_t tensorOutSize){

  reduceSummation_float<<<numBlocks, threadsPerBlock>>>(tensor, tensorOut, tensorSize);
  return cudaPeekAtLastError();
}

extern "C" cudaError_t reduceSummationWrapper_double(double *tensor, double *tensorOut, size_t tensorSize, size_t tensorOutSize){

  reduceSummation_double<<<numBlocks, threadsPerBlock>>>(tensor, tensorOut, tensorSize);
  return cudaPeekAtLastError();
}

//half can't exist in C (according to /usr/local/cuda-9.0/include/cuda_fp16.hpp:1791), so I guess we'll just use void pointers or something
extern "C" cudaError_t reduceSummationWrapper_half(void *tensor, void *tensorOut, size_t tensorSize, size_t tensorOutSize){

  errorMsg("Half support not yet implemented (%s:%d)\n", __FILE__, __LINE__);
  exit(1);
  return cudaPeekAtLastError();
}

///
/// \brief reduceSummationVariance_float - modified parallel reduction step for variance calculation. Average must have been calculated already through reduceSummation_float
/// \param input - current input for this step
/// \param output - current output for this step
/// \param len - length of input
/// \param average - average of values in original input (at start of reduction loop)
///
__global__ void reduceSummationVariance_float(float * input, float * output, size_t len, float average){
    //@@ Load a segment of the input vector into shared memory
    __shared__ float partialSum[2 * BLOCK_SIZE];
    unsigned int t = threadIdx.x, start = 2 * blockIdx.x * BLOCK_SIZE;
    if (start + t < len){
       partialSum[t] = (input[start + t] - average) * (input[start + t] - average);//for 1st step of variance calculation
      }
    else
       partialSum[t] = 0;
    if (start + BLOCK_SIZE + t < len){
       partialSum[BLOCK_SIZE + t] = (input[start + BLOCK_SIZE + t] - average) * (input[start + BLOCK_SIZE + t] - average); //for 1st step of variance calculation
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

///
/// \brief reduceSummationVariance_double - modified parallel reduction step for variance calculation. Average must have been calculated already through reduceSummation_double
/// \param input - current input for this step
/// \param output - current output for this step
/// \param len - length of input
/// \param average - average of values in original input (at start of reduction loop)
///
__global__ void reduceSummationVariance_double(double * input, double * output, size_t len, double average){
    //@@ Load a segment of the input vector into shared memory
    __shared__ double partialSum[2 * BLOCK_SIZE];
    unsigned int t = threadIdx.x, start = 2 * blockIdx.x * BLOCK_SIZE;
    if (start + t < len){
       partialSum[t] = (input[start + BLOCK_SIZE + t] - average) * (input[start + BLOCK_SIZE + t] - average); //for 1st step of variance calculation
      }
    else
       partialSum[t] = 0;
    if (start + BLOCK_SIZE + t < len){
       partialSum[BLOCK_SIZE + t] = (input[start + BLOCK_SIZE + t] - average) * (input[start + BLOCK_SIZE + t] - average); //for 1st step of variance calculation
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

extern "C" cudaError_t reduceSummationVarianceWrapper_float(float *tensor, float *tensorOut, size_t tensorSize, size_t tensorOutSize, float average){

  reduceSummationVariance_float<<<numBlocks, threadsPerBlock>>>(tensor, tensorOut, tensorSize, average);
  return cudaPeekAtLastError();
}

extern "C" cudaError_t reduceSummationVarianceWrapper_double(double *tensor, double *tensorOut, size_t tensorSize, size_t tensorOutSize, double average){

  reduceSummationVariance_double<<<numBlocks, threadsPerBlock>>>(tensor, tensorOut, tensorSize, average);
  return cudaPeekAtLastError();
}

extern "C" cudaError_t reduceSummationVarianceWrapper_half(void *tensor, void *tensorOut, size_t tensorSize, void* average){

  errorMsg("Half support not yet implemented (%s:%d)\n", __FILE__, __LINE__);
  exit(1);
  //reduceSummationVariance_half<<<numBlocks, threadsPerBlock>>>(tensor, tensorOut, tensorSize, average);
  return cudaPeekAtLastError();
}

#undef numBlocks
#undef threadsPerBlock

///
/// \brief printfDebug - print string "DEBUG printfDebug" to check if output is being printed to screen when printf is used from a CUDA kernel
///
__global__ void printfDebug(){

  printf("DEBUG printfDebug");
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

  if(x >= dataExtent.width || y >= dataExtent.height || z >= dataExtent.depth)
    return;

  float4 here = tex3D<float4>(dataTexture,x,y,z);

  if(here.x!=0 || here.y!=0 || here.z!=0)
    printf("dataTexture[%d][%d][%d] = ( %f, %f, %f)\n", x,y,z,here.x, here.y, here.z);

  __syncthreads();
}

///
/// \brief printTextureContents - print all NONZERO elements in a CUDA texture. Note that they are not printed in any order.
/// \param dataTexture - the texture
/// \param dataExtent - dimension limits of texture
///
__global__ void printTextureContents(cudaTextureObject_t dataTexture, cudaExtent dataExtent){

  int x = blockIdx.x*blockDim.x+threadIdx.x;
  int y = blockIdx.y*blockDim.y+threadIdx.y;
  int z = blockIdx.z*blockDim.z+threadIdx.z;

  if(x >= dataExtent.width || y >= dataExtent.height || z >= dataExtent.depth)
    return;

  float here = tex3D<float>(dataTexture,x,y,z);

  if(here!=0) //remove to print all zeros too
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

  return cudaPeekAtLastError();
}

__global__ void copyCudaArrayToDeviceArray_float(cudaTextureObject_t dataTexture, float* dst, cudaExtent dataExtent, int msg){

  int x = blockIdx.x*blockDim.x+threadIdx.x;
  int y = blockIdx.y*blockDim.y+threadIdx.y;
  int z = blockIdx.z*blockDim.z+threadIdx.z;

  if(x >= dataExtent.width || y >= dataExtent.height || z >= dataExtent.depth)
    return;

  float here = tex3D<float>(dataTexture,x,y,z);
  if(isnan(here))
      here = 0.0f;

  int xyz = x + y * dataExtent.width + z * dataExtent.width * dataExtent.height;

  if(msg == 10 && here!=0.0f && x%4==0 && y==32 && z==32)
      printf("obstacles[%d,%d,%d] = %f\n", x, y, z, here);

  dst[xyz] = here;
}

///
/// \brief copyCudaArrayToDeviceArray_double - copy contents of texture to double CPU array
/// \param dataTexture - CUDA texture
/// \param dst - CPU array
/// \param dataExtent
///
__global__ void copyCudaArrayToDeviceArray_double(cudaTextureObject_t dataTexture, double* dst, cudaExtent dataExtent){

  int x = blockIdx.x*blockDim.x+threadIdx.x;
  int y = blockIdx.y*blockDim.y+threadIdx.y;
  int z = blockIdx.z*blockDim.z+threadIdx.z;

  if(x >= dataExtent.width || y >= dataExtent.height || z >= dataExtent.depth)
    return;

  //see https://stackoverflow.com/questions/35137213/texture-objects-for-doubles
  uint2 hereUint2;
  tex3D(&hereUint2, dataTexture, x, y, z);

  double here = __hiloint2double(hereUint2.y, hereUint2.x);

  int xyz = x + y * dataExtent.width + z * dataExtent.width * dataExtent.height;

  dst[xyz] = here;
}

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

    infoMsg("Running kernel with dim3( %d, %d, %d), dim3( %d, %d, %d)", BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKSIZE_Z, THREADSPERBLOCK_X, THREADSPERBLOCK_Y, THREADSPERBLOCK_Z);

    copyCudaArrayToDeviceArray_float<<<dim3(BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKSIZE_Z), dim3(THREADSPERBLOCK_X, THREADSPERBLOCK_Y, THREADSPERBLOCK_Z)>>>(srcTexture, dst, srcExtent, msg);

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

    infoMsg("Running kernel with dim3( %d, %d, %d), dim3( %d, %d, %d)", BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKSIZE_Z, THREADSPERBLOCK_X, THREADSPERBLOCK_Y, THREADSPERBLOCK_Z);

    copyCudaArrayToDeviceArray_double<<<dim3(BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKSIZE_Z), dim3(THREADSPERBLOCK_X, THREADSPERBLOCK_Y, THREADSPERBLOCK_Z)>>>(srcTexture, dst, srcExtent);

    return cudaPeekAtLastError();
}

//TODO
extern "C" cudaError_t
copyCudaArrayToDeviceArrayWrapper_half(cudaArray* src, void* dst){

  errorMsg("Half support not yet implemented (%s:%d)\n", __FILE__, __LINE__);
  exit(1);

  return cudaPeekAtLastError();
}

__global__ void copyDeviceArrayToCudaArray_float(cudaSurfaceObject_t dstSurface, cudaExtent dataExtent, float* src){

  int x = blockIdx.x*blockDim.x+threadIdx.x;
  int y = blockIdx.y*blockDim.y+threadIdx.y;
  int z = blockIdx.z*blockDim.z+threadIdx.z;

  if(x >= dataExtent.width || y >= dataExtent.height || z >= dataExtent.depth)
    return;

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

  //surf3Dwrite can't handle doubles directly, so use long long
  long long int srcHereReinterpreted = __double_as_longlong(src[x + y * dataExtent.width + z * dataExtent.width * dataExtent.height]);

  //doesn't work for double
  surf3Dwrite(srcHereReinterpreted,
              dstSurface,
              x,y,z); //Note: surf3Dwrite also has boundaryMode as optional parameter
}

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

    if(x >= destExtent.width || y >= destExtent.height || z >= destExtent.depth)
      return;

    int num = x + y * destExtent.width + z * destExtent.width * destExtent.height;

    //TODO: this won't work with doubles if this is here
    T4 srcVal = make_T4<T, T4>(src[num*3], src[num*3+1], src[num*3+2], 0);

    surf3Dwrite(srcVal,
                dstSurface,
                x*sizeof(T4),y,z); //Note: surf3Dwrite also has boundaryMode as optional parameter
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
    unsigned int flags; //also useless, but needed by cudaArrayGetInfo

    cudaArrayGetInfo(&destChannelFormatDesc, &destExtent, &flags, dest);

    checkCudaErrors(cudaCreateSurfaceObject(&dstSurface, &dstDesc));

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

  checkCudaErrors(cudaCreateSurfaceObject(&dstSurface, &dstDesc));

  copyDeviceArrayToCudaArray_float<<<dim3(BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKSIZE_Z), dim3(THREADSPERBLOCK_X, THREADSPERBLOCK_Y, THREADSPERBLOCK_Z)>>>(dstSurface, destExtent, src);

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

  checkCudaErrors(cudaCreateSurfaceObject(&dstSurface, &dstDesc));

  copyDeviceArrayToCudaArray_double<<<dim3(BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKSIZE_Z), dim3(THREADSPERBLOCK_X, THREADSPERBLOCK_Y, THREADSPERBLOCK_Z)>>>(dstSurface,destExtent,src);

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
  printf("Running kernel with dim3( %d, %d, %d), dim3( %d, %d, %d)", BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKSIZE_Z, THREADSPERBLOCK_X, THREADSPERBLOCK_Y, THREADSPERBLOCK_Z);
  printTextureContents<<<dim3(BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKSIZE_Z), dim3(THREADSPERBLOCK_X, THREADSPERBLOCK_Y, THREADSPERBLOCK_Z)>>>(dataTexture, dataExtent);

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

  printfDebug<<<10,10>>>();
  return cudaPeekAtLastError();
}

__global__ void printfArray(float* theArray, dim3 theArraySizes, int someFactor, dim3 printStride, bool printZeros){

  int x = blockIdx.x*blockDim.x+threadIdx.x;
  int y = blockIdx.y*blockDim.y+threadIdx.y;
  int z = blockIdx.z*blockDim.z+threadIdx.z;

  if(x >= theArraySizes.x || y >= theArraySizes.y || z >= theArraySizes.z * someFactor)
    return;

  int numEl = x + y * theArraySizes.x + z * theArraySizes.x * theArraySizes.y;

  if(theArray[numEl] == 0 && !printZeros)
    return;

  printf("%d %d %d = %f\n", x, y, z, theArray[numEl]);
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

//Modified from tfluids/third_party/tfluids.cc/.cu

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

  //instead of the isStick flag, we use a global flag in sticky (might change it to a grid flag eventually)
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

    return cudaPeekAtLastError();
}

template<typename T>
__global__ void updateObstacles(dim3 obstacleExtent){

    //TODO: this should be updated depending on how obstacles are implemented
}

extern "C" cudaError_t updateObstaclesDNWrapper_float(dim3 obstacleExtent){

    int D = obstacleExtent.z;
    int H = obstacleExtent.y;
    int W = obstacleExtent.x;

    updateObstacles<float><<<D*H*W/BLOCK_SIZE,BLOCK_SIZE>>>(obstacleExtent);
    return cudaPeekAtLastError();
}

extern "C" cudaError_t updateObstaclesDNWrapper_double(dim3 obstacleExtent){

    int D = obstacleExtent.z;
    int H = obstacleExtent.y;
    int W = obstacleExtent.x;

    updateObstacles<double><<<D*H*W/BLOCK_SIZE,BLOCK_SIZE>>>(obstacleExtent);
    return cudaPeekAtLastError();
}

extern "C" cudaError_t updateObstaclesDNWrapper_half(dim3 obstacleExtent){

    printf("Half support not yet implemented (%s:%d)\n", __FILE__, __LINE__);
    exit(1);

    return cudaPeekAtLastError();
}

#undef min
#undef max

#endif
