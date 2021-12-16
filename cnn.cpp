//Modified from a CUDNN LeNet example at https://github.com/tbennun/cudnn-training

#ifndef CNN_CPP
#define CNN_CPP

#include "cnn.h"
#include <sstream>
//#include <pez.h>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <chrono>
#include <thread> //#include <fstream>
#include <algorithm>
#include <sstream>
#include <iterator>
#include <cctype>
//#include <thpp/Tensor.h>

#include "defines.h"

//might want to add

extern "C" cudaError_t
doPrintfWrapper(cudaTextureObject_t someTex, int i, int j, int k);

extern "C" cudaError_t
printfDebugWrapper();

//someFactor is a lazy way to allow for printing the input, which has three channels instead of one
//printStride means only the elements with z % printStride.z == 0, y % printStride.y == 0, and x % printStride.x == 0 are printed
extern "C" cudaError_t
printfArrayWrapper(float* theArray, dim3 theArraySize, int someFactor, dim3 printStride, bool printZeros);

extern "C" cudaError_t
printElementWrapper(float* normalArray, cudaExtent extent);

extern "C" cudaError_t
printDataCudaArrayContentsWrapper(cudaArray* data);

extern "C" cudaError_t
printDataCudaArrayContents3DWrapper(cudaArray* data);

extern "C" cudaError_t
copyCudaArrayToDeviceArrayWrapper_float(cudaArray* src, float* dst, int msg);

extern "C" cudaError_t
copyCudaArrayToDeviceArrayWrapper_double(cudaArray* src, double* dst);

extern "C" cudaError_t
copyCudaArrayToDeviceArrayWrapper_half(cudaArray* src, void* dst);
/*
namespace std
{
  template<>
  struct default_delete<InputLayer3D> {
    void operator()(InputLayer3D* ptr) {}
  };
}
*/

template<typename T>
void Layer<T>::setStride(){

    tensorStrides.resize(tensorDims.size());

    tensorStrides[tensorStrides.size()-1] = 1;

    for(int i=tensorStrides.size()-2;i>=0;i--)
        tensorStrides[i] = tensorStrides[i+1] * tensorDims[i+1];
}

template<typename T>
void Layer<T>::concatenateTensorWith(Layer<T>* otherLayer){

    //this way, there will hopefully be a single std::vector containing the layers to concatenate, and all the layers in it will have concatWithPointer
    //pointing to this std vector

    Layer<T>* thisLayer = this;

    if(this->concatWith.size() == 0){
        this->concatWith.push_back(thisLayer);
        this->concatWithPointer = &(this->concatWith);}

    otherLayer->concatWithPointer = this->concatWithPointer;
    (*concatWithPointer).push_back(otherLayer);
}

template<typename T>
std::unordered_map<std::string, cudnnActivationMode_t> ConvBiasLayer3D<T>::activationModeMap = {{"sigmoid",     CUDNN_ACTIVATION_SIGMOID     },
                                                                                                {"relu",        CUDNN_ACTIVATION_RELU        },
                                                                                                {"tanh",        CUDNN_ACTIVATION_TANH        },
                                                                                                {"clippedrelu", CUDNN_ACTIVATION_CLIPPED_RELU},
                                                                                                {"elu",         CUDNN_ACTIVATION_ELU         },
                                                                                                {"identity",    CUDNN_ACTIVATION_IDENTITY    }};

template<typename T>
PoolingLayer3D<T>::PoolingLayer3D(int _dims[3]){

    for(int i=0;i<3;i++)
        dims[i] = _dims[i];
}

template<typename T>
PoolingLayer3D<T>::PoolingLayer3D(){}

template<typename T>
MergeContinguousTensorsLayer3D<T>::MergeContinguousTensorsLayer3D(){}

template<typename T>
MergeContinguousTensorsLayer3D<T>::~MergeContinguousTensorsLayer3D(){}

template<typename T>
void MergeContinguousTensorsLayer3D<T>::forward(){} //don't need to do anything, output is the same as input, and input is already set

template<typename T>
AvgPoolLayer3D<T>::AvgPoolLayer3D(int _dims[3]){

    for(int i=0;i<3;i++)
        this->dims[i] = _dims[i];

    //TODO: these two should be parameters
    int padding[] = {0,0,0};
    int stride[] = {this->dims[0],this->dims[1],this->dims[2]};

    checkCUDNN(cudnnCreatePoolingDescriptor(&this->poolingDesc));
    checkCUDNN(cudnnSetPoolingNdDescriptor(this->poolingDesc,
                                           CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
                                           CUDNN_PROPAGATE_NAN,
                                           3,
                                           this->dims,
                                           padding,
                                           stride));
}

/*
template<typename T>
Layer<T>::Layer(Layer<T>* inputLayer, std::vector<Layer<T>*> outputLayers, std::string _name){
  connectedInputLayer  = inputLayer;
  connectedOutputLayers = outputLayers;
  name = _name;
}
*/

template<typename T>
void Layer<T>::allocateTensor(T* tensorData){

    //first need to set tensor descriptor; it can only be set when the input is known. It must be known at this point.
    calculateTensorDims();

    checkCUDNN(cudnnCreateTensorDescriptor(&tensorDesc));
    checkCUDNN(cudnnSetTensorNdDescriptor(tensorDesc, getDataType(), tensorDims.size(), tensorDims.data(), tensorStrides.data()));

    checkCudaErrors(cudaDeviceSynchronize());

    if(tensorData!=NULL){
        data = tensorData;
        this->getTotalTensorSize();
        printf("GETTOTALTENSORSIZE %d", this->getTotalTensorSize());}
    else if(this->concatWithPointer == NULL || this->concatWithPointer->size() <= 1){
        infoMsg("getTotalTensorSize = %d\n", getTotalTensorSize());
        checkCudaErrors(cudaMalloc((void**)&data, sizeof(T)*getTotalTensorSize()));
        infoMsg("Allocated for %s sizeof(T) * %d bytes\n", this->name.c_str(), getTotalTensorSize());}
    else
        this->getTotalTensorSize(); //probably useless, unless the merging layer looks at the number of channels before calling this

    //printf("Allocated for %s sizeof(T) * %d bytes\n", this->name.c_str(), getTotalTensorSize());

    if(getTotalTensorSize() == 0)
        warnMsg("WARNING: Tensor has no elements; allocated memory is zero.");

    //this is poorly implemented, only exists for ConvLayer3D and ConvBiasLayer3D to determine the forward convolution algorithm
    //if variable input size is implemented, this will go into ConvBiasLayer3D<T>::forward()
    postPrep();

}

template<typename T>
cudnnActivationMode_t ConvBiasLayer3D<T>::getActivationMode(){
    return activationMode;
}

template<typename T>
void ConvBiasLayer3D<T>::postPrep(){

    //need cudnnGetConvolutionFowrwardAlgorithm here

    //algorithm for cudnnConvolutionFwdAlgo_t algo can be suggested by CUDA by using cudnnGetConvolutionForwardAlgorithm
    //use CUDNN_CONVOLUTION_FWD_PREFER_FASTEST for now with the cudnnConvolutionFwdPreference_t enum
    //CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT is for overhead, does not contain
    //see "Choosing Convolution Algo in cuDNN v2" in favorites

    //could get expected performance results witgh cudnnGetConvolutionForwardAlgorithm_v7

    size_t sizeInBytes;
    /*
  cudnnGetConvolutionForwardWorkspaceSize(
                                  *(this->getLayerCudnnHandle()),
                                  this->connectedInputLayer->tensorDesc,
                                  this->filterDesc,
                                  this->convDesc,
                                  this->tensorDesc,
                                  CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                                  &sizeInBytes );
*/
    //printf("\n\nsizeInBytes for %s is %d\n\n", this->name.c_str(), sizeInBytes);

    /*
  if(getActivationMode() == CUDNN_ACTIVATION_RELU)
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(*(this->getLayerCudnnHandle()),
                                                   this->connectedInputLayer->tensorDesc,
                                                   this->filterDesc,
                                                   this->convDesc,
                                                   this->tensorDesc,
                                                   CUDNN_CONVOLUTION_FWD_NO_WORKSPACE,//CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                   0, //this is the memory limit, which is ignored with CUDNN_CONVOLUTION_FWD_PREFER_FASTEST
                                                   &algorithm));
  else if(getActivationMode() == CUDNN_ACTIVATION_IDENTITY)*/
    algorithm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    /*  else
    errorMsg("Cannot do convolution with activation other than relu or identity");*/

    //CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM

    checkCUDNN(cudnnCreateTensorDescriptor(&uselessDesc));
    checkCUDNN(cudnnSetTensorNdDescriptor(uselessDesc, this->getDataType(), this->tensorDims.size(), this->tensorDims.data(), this->tensorStrides.data())); //since it is multiplied by zero, it doesn't matter what we put in it
    checkCudaErrors(cudaMalloc((void**)&uselessData, sizeof(T)*this->getTotalTensorSize()));
}

template<typename T>
void UpscaleLayer3D<T>::calculateTensorDims(){

    //since the dimensions are ncdhw, the last three dimensions need to be multiplied
    this->tensorDims.resize(this->connectedInputLayer->tensorDims.size());

    int offset = 5 - upscaleFactor.size();

    for(int i=0;i<this->tensorDims.size();i++)
        this->tensorDims[i] = this->connectedInputLayer->tensorDims[i];

    for(int i=2;i<this->tensorDims.size();i++)
        this->tensorDims[i] = this->connectedInputLayer->tensorDims[i] * upscaleFactor[i-2];

    infoMsg("\nUpscaleLayer\n");

    //printf("\nUpscaleLayer %s, tensorDims = %d %d %d %d %d", this->name.c_str(), this->tensorDims[0], this->tensorDims[1], this->tensorDims[2], this->tensorDims[3], this->tensorDims[4]);
    /*
  for(i=0;i<tensorDims.size();i++)
      printf("tensorDims[%d] = ");
*/
    this->getTotalTensorSize();

    this->setStride();
}

template<typename T>
void ConvBiasLayer3D<T>::calculateTensorDims(){

    this->tensorDims.resize(this->connectedInputLayer->tensorDims.size());

    int theDims[5];
    //cudnnSetCallback(CUDNN_SEV_INFO_EN,NULL,NULL);

    //printf("calculateTensorDims %s\n",this->name.c_str());

    //tensorDims.data() is 0,0,0,0,0 for some reason
    //the second parameter of this should be the descriptor of the input tensor
    checkCUDNN(cudnnGetConvolutionNdForwardOutputDim(convDesc, this->connectedInputLayer->tensorDesc, filterDesc, 5, this->tensorDims.data())); //3 or 5? 5, since it is the dimension of the output tensor
    /*
  for(int i=0;i<this->tensorDims.size();i++)
    this->tensorDims[i] = theDims[i];
*/
    this->getTotalTensorSize();

    this->setStride();

}

template<typename T>
void PoolingLayer3D<T>::calculateTensorDims(){

    //this should resize it to 5
    this->tensorDims.resize(this->connectedInputLayer->tensorDims.size());

    checkCUDNN(cudnnGetPoolingNdForwardOutputDim(poolingDesc, this->connectedInputLayer->tensorDesc, 5, this->tensorDims.data()));//idk if it's supposed to be 3 or 5, try it

    this->getTotalTensorSize(); //is this necessary?

    this->setStride();
}

template<typename T>
void InputLayer3D<T>::calculateTensorDims(){

    //do nothing, as loadInput sets the dimensions by itself for the input layer
}

template<typename T>
void AddLayer3D<T>::calculateTensorDims(){
    this->tensorDims = this->connectedInputLayer->tensorDims; //this layer does not change the dimension
    this->setStride();
}

template<typename T>
void ScalingLayer3D<T>::calculateTensorDims(){
    this->tensorDims = this->connectedInputLayer->tensorDims; //this layer does not change the dimension
    this->setStride();
}

template<typename T>
void MergeContinguousTensorsLayer3D<T>::calculateTensorDims(){

    this->tensorDims.resize(this->connectedInputLayer->tensorDims.size());

    int totalChannels = 0;

    this->tensorDims = (*(this->connectedInputLayer->concatWithPointer))[0]->tensorDims;

    //checking that spatial dimensions are identical is done when is concatenateWith is called
    for(int i=0;i<this->connectedInputLayer->concatWithPointer->size();i++){
        //(*(this->connectedInputLayer->concatWithPointer))[i]->calculateTensorDims();
        totalChannels += (*(this->connectedInputLayer->concatWithPointer))[i]->tensorDims[1]; //the C in NCDHW
    }

    this->tensorDims[1] = totalChannels;

    this->getTotalTensorSize();
    this->setStride();

    //also allocate data and set pointers for all layers in concatWith
    //this needs to be postprep
}

template<typename T>
void MergeContinguousTensorsLayer3D<T>::postPrep(){

    int currentOffset = 0;

    //set pointers for all concatWith
    for(int i=0;i<this->connectedInputLayer->concatWithPointer->size();i++){
        (*(this->connectedInputLayer->concatWithPointer))[i]->data = this->data + currentOffset;
        currentOffset = currentOffset + (*(this->connectedInputLayer->concatWithPointer))[i]->getTotalTensorSize();
    }

    //this might cause a segmentation fault at the end, if the dat is deallocated twice (probably not)
}

/*
template<typename T>
void Input
*/

template<typename T>
size_t Layer<T>::getTotalTensorSize(){

    if(tensorDims.size() == 0){
        totalTensorSize = 0;
        return 0;
    }

    if(totalTensorSize !=0)
        return totalTensorSize;

    totalTensorSize = 1;

    for(int i=0;i<tensorDims.size();i++)
        totalTensorSize *= tensorDims[i];

    return totalTensorSize;
}

template<>
void Layer<float>::setDataType(){

    dataType = CUDNN_DATA_FLOAT;
}

template<>
void Layer<double>::setDataType(){

    dataType = CUDNN_DATA_DOUBLE;
}

template<>
void Layer<half>::setDataType(){

    dataType = CUDNN_DATA_HALF;
}

template<typename T>
Layer<T>::Layer(){

    connectedOutputLayersRaw = std::vector<Layer<T>*>();
    connectedOutputLayers = std::vector<std::unique_ptr<Layer<T>>>();

    setDataType();
}

//in the absence of RAII, we'll just do this and deal with custom memory management in the Net later

template<typename T>
Layer<T>::~Layer(){}

template<typename T>
InputLayer3D<T>::~InputLayer3D(){}

//can check if layerToConnectTo is in Net by checking if the cudnnHandle matches
template<typename T>
Layer<T>* Net<T>::addLayer(std::unique_ptr<Layer<T>> layerToConnect, Layer<T>* layerToConnectTo){

    if(layerToConnectTo ==NULL){
        inputLayer = std::move(layerToConnect);
        inputLayer->connectedOutputLayers.resize(0);
        inputLayer->connectedInputLayer = NULL;
        outputLayer = inputLayer.get();
    }
    else{  //if (GetKernelIndices(flags, chan, k, j, i)) { //???
        // return;
        //}
        //layerToConnect->connectedInputLayer = layerToConnectTo;
        layerToConnectTo->connectedOutputLayers.push_back(std::move(layerToConnect)); //this switches the ownership of the object to connectedOutputLayers.back()
        layerToConnectTo->connectedOutputLayers.back()->connectedInputLayer = layerToConnectTo;
        outputLayer = layerToConnectTo->connectedOutputLayers.back().get();
        //layerToConnectTo->connectedOutputLayers.back().get();
        layerToConnectTo->connectedOutputLayers.back()->layerCudnnHandle = &cudnnHandle;}
    //layerToConnect->connectedInputLayer = layerToConnectTo;}


    //layerToConnectTo->connectedOutputLayers.push_back(std::move(layerToConnect)); //this switches the ownership of the object to connectedOutputLayers.back()

    //layerToConnect->connectedInputLayer = layerToConnectTo;

    //outputLayer = layerToConnect.get();
    //layerToConnect->layerCudnnHandle = &cudnnHandle;

    //layerToConnect->getTensorSize(); //for InputLayer3D, error if loadData or whatever it was has not been called yet; for everything except pooling, upscaling and convolution, should return the same size as the tensor data in layerToConnectTo

    return outputLayer; //this probably nulls the outputLayer unique pointer, so you should probably return outputLayer.get()
}

template<typename T>
Layer<T>* Net<T>::addLayer(std::unique_ptr<Layer<T>> layerToConnect){
    return addLayer(std::move(layerToConnect), outputLayer);
}

template<typename T>
cudnnDataType_t Net<T>::getDataType(){
    return dataType;
}

template<>
void Net<float>::setDataType(){

    dataType = CUDNN_DATA_FLOAT;
}

template<>
void Net<double>::setDataType(){

    dataType = CUDNN_DATA_DOUBLE;
}

template<>
void Net<half>::setDataType(){

    dataType = CUDNN_DATA_HALF;
}

template<typename T>
Net<T>::Net(){

    //this doesn't seem to do anything
    checkCUDNN(cudnnSetCallback(CUDNN_SEV_INFO_EN,NULL,NULL));

    checkCUDNN(cudnnCreate(&cudnnHandle));

    inputLayer = NULL;
    outputLayer = NULL;

    setDataType();
}

extern "C" cudaError_t setWallsWrapper_float(float* obstacles, dim3 domainExtent, size_t thickness);
extern "C" cudaError_t setWallsWrapper_half(void* obstacles, dim3 domainExtent, size_t thickness);
extern "C" cudaError_t setWallsWrapper_double(double* obstacles, dim3 domainExtent, size_t thickness);

extern "C" cudaError_t setEdgesWrapper_float(float* buffer, dim3 domainExtent, float value);

template<>
cudaError_t DefaultNet<float>::setWallsWrapper(float* obstacles, dim3 domainExtent, size_t thickness){
    return printCudaErrors(setWallsWrapper_float(obstacles, domainExtent, thickness));
}

template<>
cudaError_t DefaultNet<half>::setWallsWrapper(half* obstacles, dim3 domainExtent, size_t thickness){
    return printCudaErrors(setWallsWrapper_half((void*)obstacles, domainExtent, thickness));
}

template<>
cudaError_t DefaultNet<double>::setWallsWrapper(double* obstacles, dim3 domainExtent, size_t thickness){
    return printCudaErrors(setWallsWrapper_double(obstacles, domainExtent, thickness));
}

//move these to Fluid3d.cpp
extern "C" cudaError_t updateObstaclesDNWrapper_float(dim3 obstacleExtent);
extern "C" cudaError_t updateObstaclesDNWrapper_double(dim3 obstacleExtent);
extern "C" cudaError_t updateObstaclesDNWrapper_half(dim3 obstacleExtent);

template<>
cudaError_t DefaultNet<float>::updateObstaclesDNWrapper(dim3 obstacleExtent){

    return printCudaErrors(updateObstaclesDNWrapper_float(obstacleExtent));
}

template<>
cudaError_t DefaultNet<double>::updateObstaclesDNWrapper(dim3 obstacleExtent){

    return printCudaErrors(updateObstaclesDNWrapper_double(obstacleExtent));
}

template<>
cudaError_t DefaultNet<half>::updateObstaclesDNWrapper(dim3 obstacleExtent){

    return printCudaErrors(updateObstaclesDNWrapper_half(obstacleExtent));
}

extern "C" cudaError_t printCudaBuffersWrapper_float(float* buffer, dim3 size, char* name);

template<>
cudaError_t DefaultNet<float>::copyCudaArrayToDeviceArrayZeroObstacles(cudaArray* lastPressureArray, cudaArray* velDivergenceArray, cudaArray* obstacleArray, size_t inputLayerObstacleOffset, size_t inputLayerVelDivergenceOffset){

    //copyCudaArrayToDeviceArrayWrapper_float(lastPressureArray, (float*)(this->inputLayer->data), (this->numIteration==10)?0:(-1));
    copyCudaArrayToDeviceArrayWrapper_float(velDivergenceArray,
                                            (float*)this->inputLayer->data + inputLayerVelDivergenceOffset, (this->numIteration==10)?1:(-1));

    //this won't work after you move advection etc. to CUDA
    cudaChannelFormatDesc desc;
    cudaExtent obstacleExtent;
    unsigned int flags;

    printCudaErrors(cudaArrayGetInfo(&desc,&obstacleExtent,&flags,obstacleArray));

    printCudaErrors(updateObstaclesDNWrapper(dim3(obstacleExtent.width, obstacleExtent.height, obstacleExtent.depth)));

    //DO NOT NEED to memset the obstacles here; just when the DefaultNet data is allocated
    //need to call

    //can use cudaMemset for making sure the initialized memory is set to zero
    //cudaMalloc does NOT clear the memory

    //cudaMemset((float*)(this->inputLayer->data) + inputLayerVelDivergenceOffset + inputLayerObstacleOffset, 0, this->inputLayer->tensorDims);
    checkCudaErrors(copyCudaArrayToDeviceArrayWrapper_float(obstacleArray,
                                            (float*)(this->inputLayer->data) + inputLayerVelDivergenceOffset + inputLayerObstacleOffset, 10));//(this->numIteration==10)?2:(-1));

    checkCudaErrors(cudaDeviceSynchronize());
    printf("obstacles buffer values are \n");
    checkCudaErrors(printCudaBuffersWrapper_float((float*)(this->inputLayer->data) + inputLayerVelDivergenceOffset + inputLayerObstacleOffset,
                                  dim3(obstacleExtent.width, obstacleExtent.height, obstacleExtent.depth),
                                  "obstacles"));
    checkCudaErrors(cudaDeviceSynchronize());

    //checkCudaErrors(cudaDeviceSynchronize());
    //exit(0);

    return cudaGetLastError();
}

extern "C" cudaError_t
copyCudaArrayToDeviceArrayWrapper3D_float(cudaArray* src, float* dst);

template<>
cudaError_t DefaultNet<float>::copyCudaArrayToDeviceArray(cudaArray* lastPressureArray, cudaArray* velDivergenceArray, cudaArray* obstacleArray, size_t inputLayerObstacleOffset, size_t inputLayerVelDivergenceOffset){

    //what order?
    //first is pDivScaled for some reason, need pressure divergence?
    //what you need is the norm of the velocity

    //
    printCudaErrors(copyCudaArrayToDeviceArrayWrapper_float(lastPressureArray, (float*)(this->inputLayer->data), (this->numIteration==10)?0:(-1)));
    printCudaErrors(copyCudaArrayToDeviceArrayWrapper_float(velDivergenceArray,
                                                            (float*)this->inputLayer->data + inputLayerVelDivergenceOffset, (this->numIteration==10)?1:(-1)));

    //copyCudaArrayToDeviceArrayWrapper_float(obstacleArray,
    //incrementVoid(this->inputLayer->data, inputLayerObstacleOffset, sizeof(T)));

    printCudaErrors(copyCudaArrayToDeviceArrayWrapper_float(obstacleArray,
                                                            (float*)(this->inputLayer->data) + inputLayerVelDivergenceOffset + inputLayerObstacleOffset, 2));//(this->numIteration==10)?2:(-1));


    return cudaGetLastError();
}

template<>
cudaError_t DefaultNet<double>::copyCudaArrayToDeviceArray(cudaArray* lastPressureArray, cudaArray* velDivergenceArray, cudaArray* obstacleArray, size_t inputLayerObstacleOffset, size_t inputLayerVelDivergenceOffset){

    printCudaErrors(copyCudaArrayToDeviceArrayWrapper_double(lastPressureArray, (double*)(this->inputLayer->data)));
    printCudaErrors(copyCudaArrayToDeviceArrayWrapper_double(velDivergenceArray,
                                                             (double*)this->inputLayer->data + inputLayerVelDivergenceOffset));

    //copyCudaArrayToDeviceArrayWrapper_float(obstacleArray,
    //incrementVoid(this->inputLayer->data, inputLayerObstacleOffset, sizeof(T)));

    printCudaErrors(copyCudaArrayToDeviceArrayWrapper_double(obstacleArray,
                                                             (double*)(this->inputLayer->data) + inputLayerVelDivergenceOffset + inputLayerObstacleOffset));

    return cudaGetLastError();
}

template<>
cudaError_t DefaultNet<half>::copyCudaArrayToDeviceArray(cudaArray* lastPressureArray, cudaArray* velDivergenceArray, cudaArray* obstacleArray, size_t inputLayerObstacleOffset, size_t inputLayerVelDivergenceOffset){

    printCudaErrors(copyCudaArrayToDeviceArrayWrapper_half(lastPressureArray, (half*)(this->inputLayer->data)));
    printCudaErrors(copyCudaArrayToDeviceArrayWrapper_half(velDivergenceArray,
                                                           (half*)this->inputLayer->data + inputLayerVelDivergenceOffset));

    //copyCudaArrayToDeviceArrayWrapper_float(obstacleArray,
    //incrementVoid(this->inputLayer->data, inputLayerObstacleOffset, sizeof(T)));

    printCudaErrors(copyCudaArrayToDeviceArrayWrapper_half(obstacleArray,
                                                           (half*)(this->inputLayer->data) + inputLayerVelDivergenceOffset + inputLayerObstacleOffset));

    return cudaGetLastError();
}

template<>
cudaError_t DefaultNet<float>::copyCudaArrayToDeviceArray(cudaArray* array, float* buffer){

    return printCudaErrors(copyCudaArrayToDeviceArrayWrapper_float(array, buffer, -1));
}

extern "C" cudaError_t setDeviceArrayConstantWrapper_float(float* theDeviceArray, dim3 dimensions, float value);

//this can be used to set the values of the network, in order to check that the output is as expected
template<>
cudaError_t DefaultNet<float>::copyCudaArrayToDeviceArrayDummy(cudaArray* lastPressureArray, cudaArray* velDivergenceArray, cudaArray* obstacleArray, size_t inputLayerObstacleOffset, size_t inputLayerVelDivergenceOffset){

    cudaChannelFormatDesc desc;
    cudaExtent ext;
    unsigned int flags;

    printCudaErrors(cudaArrayGetInfo(&desc, &ext, &flags, lastPressureArray));
    setDeviceArrayConstantWrapper_float((float*)(this->inputLayer->data), dim3(ext.width, ext.height, ext.depth), 1.0f);

    printCudaErrors(cudaDeviceSynchronize());

    infoMsg("lastPressureArray extent = ( %d, %d, %d)\n", ext.width, ext.height, ext.depth);

    printCudaErrors(cudaArrayGetInfo(&desc, &ext, &flags, velDivergenceArray));
    setDeviceArrayConstantWrapper_float((float*)this->inputLayer->data + inputLayerVelDivergenceOffset, dim3(ext.width, ext.height, ext.depth), 1.0f);

    infoMsg("velDivergenceArray extent = ( %d, %d, %d)\n", ext.width, ext.height, ext.depth);

    printCudaErrors(cudaArrayGetInfo(&desc, &ext, &flags, obstacleArray));
    setDeviceArrayConstantWrapper_float((float*)(this->inputLayer->data) + inputLayerVelDivergenceOffset + inputLayerObstacleOffset,
                                        dim3(ext.width, ext.height, ext.depth), 0.0f);

    infoMsg("obstacleArray extent = ( %d, %d, %d)\n", ext.width, ext.height, ext.depth);

    infoMsg("inputLayerVelDivergenceOffset = %d, inputLayerObstacleOffset = %d", int(inputLayerVelDivergenceOffset), int(inputLayerObstacleOffset));
}

extern "C" cudaError_t copyDeviceArrayToCudaArray3DWrapper_float(float* src, cudaArray* dest);

extern "C" cudaError_t copyDeviceArrayToCudaArrayWrapper_float(float* src, cudaArray* dest);

extern "C" cudaError_t copyDeviceArrayToCudaArrayWrapper_double(double* src, cudaArray* dest);

extern "C" cudaError_t copyDeviceArrayToCudaArrayWrapper_half(void* src, cudaArray* dest);

//these are for copying the output to the cudaArray it needs to be in to interop back to OpenGL
template<>
cudaError_t DefaultNet<float>::copyDeviceArrayToCudaArray(cudaArray* pressureArray){

    return printCudaErrors(copyDeviceArrayToCudaArrayWrapper_float(this->outputLayer->data, pressureArray));

}

template<>
cudaError_t DefaultNet<double>::copyDeviceArrayToCudaArray(cudaArray* pressureArray){

    return printCudaErrors(copyDeviceArrayToCudaArrayWrapper_double(this->outputLayer->data, pressureArray));

}

template<>
cudaError_t DefaultNet<half>::copyDeviceArrayToCudaArray(cudaArray* pressureArray){

    return printCudaErrors(copyDeviceArrayToCudaArrayWrapper_half(this->outputLayer->data, pressureArray));

}

template<typename T>
void Net<T>::allocateTensors(T* input, T* output){ //either or both are NULL if allocation is desired for input and/or output

    std::vector<Layer<T>*> currentLayers;
    std::vector<Layer<T>*> nextLayers;
    std::vector<Layer<T>*> mergePoints;
    currentLayers.push_back(this->inputLayer.get());

    bool okMergeFlag = false;

    //this needs to be a graph parse, not a linear while
    while(currentLayers.size() != 0 || mergePoints.size() != 0){
        for(int i=0;i<currentLayers.size();i++){
            infoMsg("allocateTensors %s\n", currentLayers[i]->type().c_str());
            if(currentLayers[i]->type() == "MergeContinguousLayer3D" && !okMergeFlag)
                mergePoints.push_back(currentLayers[i]);
            else{
                if(input!=NULL && currentLayers[i]->type() =="InputLayer3D"){
                    currentLayers[i]->allocateTensor(input);
                    //printf("\ninputlayer not null\n\n");
                }
                else if(output!=NULL && currentLayers[i] == outputLayer){
                    currentLayers[i]->allocateTensor(output); //do not allocate device memory
                    //printf("\noutputlayer not null, %s\n\n", currentLayers[i]->name.c_str());
                }
                else
                    currentLayers[i]->allocateTensor();
                for(int j=0;j<currentLayers[i]->connectedOutputLayers.size();j++)
                    nextLayers.push_back(currentLayers[i]->connectedOutputLayers[j].get());
            }
        }

        okMergeFlag = false;

        if(nextLayers.size()==0 && mergePoints.size()!=0){
            currentLayers.clear();
            currentLayers.push_back(mergePoints[0]);
            mergePoints.erase(mergePoints.begin());
            okMergeFlag = true;
        }
        else{
            currentLayers.swap(nextLayers);
            nextLayers.clear();
        }
    }
}

extern "C" cudaError_t reduceSummationWrapper_float(float *tensor, float *tensorOut, size_t tensorSize, size_t tensorOutSize);
extern "C" cudaError_t reduceSummationWrapper_double(double *tensor, double *tensorOut, size_t tensorSize, size_t tensorOutSize);
extern "C" cudaError_t reduceSummationWrapper_half(void *tensor, void *tensorOut, size_t tensorSize, size_t tensorOutSize);

extern "C" cudaError_t reduceSummationVarianceWrapper_float(float *tensor, float *tensorOut, size_t tensorSize, size_t tensorOutSize, float average);
extern "C" cudaError_t reduceSummationVarianceWrapper_double(double *tensor, double *tensorOut, size_t tensorSize, size_t tensorOutSize, double average);
extern "C" cudaError_t reduceSummationVarianceWrapper_half(void *tensor, void *tensorOut, size_t tensorSize, size_t tensorOutSize, void* average);

//need a second intermediate tensor to do the calculation
//BLOCK_SIZE was 1024 at the time of writing
//this returns a pointer to a single float in GPU memory, containing the standard deviation.
//If you don't like that, uncomment the float out and cudaMemcpy at the end, change function to return float, and
//change return tensorOut to return out
float getStandardDeviation(float* tensor, float* tensorOut, size_t tensorSize){

    static float* tensorCPU = NULL;
    if(tensorCPU==NULL)
        checkCudaErrors(cudaMallocHost(&tensorCPU, tensorSize*sizeof(float)));
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(tensorCPU, tensor, tensorSize*sizeof(float), cudaMemcpyHostToDevice));

    //printf tensorCPU
    //printf()

    //return 1.0f; //placeholder
    //getNormKernel_float(this->connectedInputLayers[0]->data, this->data);

    //extern "C" FLOAT multiplyByVarianceDevice(FLOAT* tensor, FLOAT* tensorOut, size_t tensorSize){

    //when setting tensorDims, also change dataSize maybe so you don't need this

    //if not using the offset method, need to copy the initial tensor
    static float* tensor2 = NULL;

    if(tensor2 == NULL)
        checkCudaErrors(cudaMalloc(&tensor2, tensorSize*sizeof(float)));
    checkCudaErrors(cudaMemcpy(tensor2,tensor,tensorSize*sizeof(float),cudaMemcpyDeviceToDevice));

    //printf("Initial tensor is");



    //float* tensorOutInitial = tensorOut;
    size_t tensorSizeInitial = tensorSize;

    //this should make the output buffering stop. doesn't work on printfs in kernels
    //setvbuf(stdout, NULL, _IONBF, 0);

    size_t tensorOutSize = tensorSize / (BLOCK_SIZE << 1) + bool(tensorSize % (BLOCK_SIZE<<1));

    //
    //starting here, we calculate the average
    //

    //checkCudaErrors(cudaDeviceSynchronize());

    static float *sum = NULL;

    if(sum==NULL)
        checkCudaErrors(cudaMalloc((void**)&sum, sizeof(float)));
    //size_t tensorOutSizeInitial = tensorSize / (BLOCK_SIZE << 1) + bool(tensorSize % (BLOCK_SIZE<<1));

    //checkCudaErrors(cudaMalloc((void**)&tensorOut, sizeof(float) * tensorOutSizeInitial));

    //debugSummation<<<1,1>>>(tensor, tensorSize);

    infoMsg("tensorSize initial = %d\n", int(tensorSize));

    reduceSummationWrapper_float(tensor, tensorOut, tensorSize, tensorOutSize);
    float* tempTensor = tensor;
    tensor = tensorOut;
    tensorOut = tempTensor;
    tensorSize = tensorOutSize;
    tensorOutSize = tensorSize / (BLOCK_SIZE << 1) + bool(tensorSize % (BLOCK_SIZE<<1));

    infoMsg("tensorSize after first summation reduction%d\n", tensorSize);

    //this gives a slightly different result for some reason; see how it acts after multiple sum reductions
    //debugSummation<<<1,1>>>(tensorOut, tensorSize);



    //in this configuration, the initial tensor is destroyed
    while(tensorSize > 1){
        //reduceSummation<<<numBlocks, threadsPerBlock>>>(tensorOut, tensorOut + tensorSize, tensorSize);
        reduceSummationWrapper_float(tensor, tensorOut, tensorSize, tensorOutSize);
        tempTensor = tensor;
        tensor = tensorOut;
        tensorOut = tempTensor;
        tensorSize = tensorOutSize;
        tensorOutSize = tensorSize / (BLOCK_SIZE << 1) + bool(tensorSize % (BLOCK_SIZE<<1));

        //debugSummation<<<1,1>>>(tensorOut, tensorSize);

        //printf("average %d\n", tensorOutSize);
    }

    //printf("\n");

    //tensorOut[0] now contains the sum of the elements

    //checkCudaErrors(cudaDeviceSynchronize());

    float average;
    checkCudaErrors(cudaMemcpy(&average, tensor, sizeof(float), cudaMemcpyDeviceToHost)); //cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind


    infoMsg("Sum is %f\n", average);

    //need to divide by INITIAL tensorSize
    average /= tensorSizeInitial;

    infoMsg("Average is %f\n", average);

    //printf("average device %f\n", average);

    //
    //now we can calculate the variance
    //with reduceSummationVariance, the partial sum gets the values (x^2-average) in the first summation reduction
    //

    tensor = tensor2;

    tensorSize = tensorSizeInitial;
    tensorOutSize = tensorSize / (BLOCK_SIZE << 1) + bool(tensorSize % (BLOCK_SIZE<<1));

    //reduceSummationVariance<<<numBlocks, threadsPerBlock>>>(tensor, tensorOut, tensorSize, average);
    reduceSummationVarianceWrapper_float(tensor, tensorOut, tensorSize, tensorOutSize, average);
    tempTensor = tensor;
    tensor = tensorOut;
    tensorOut = tempTensor;
    tensorSize = tensorOutSize;
    tensorOutSize = tensorSize / (BLOCK_SIZE << 1) + bool(tensorSize % (BLOCK_SIZE<<1));

    //debugSummation<<<1,1>>>(tensorOut, tensorSize);

    //printf("tensorOutSize %d", tensorOutSize);

    //printArray<<<1,1>>>(tensorOut, tensorSize);

    while(tensorSize > 1){
        //reduceSummation<<<numBlocks, threadsPerBlock>>>(tensorOut, tensorOut + tensorSize, tensorSize);
        reduceSummationWrapper_float(tensor, tensorOut, tensorSize, tensorOutSize);
        tempTensor = tensor;
        tensor = tensorOut;
        tensorOut = tempTensor;
        tensorSize = tensorOutSize;
        tensorOutSize = tensorSize / (BLOCK_SIZE << 1) + bool(tensorSize % (BLOCK_SIZE<<1));

        //debugSummation<<<1,1>>>(tensorOut, tensorSize);

        //printf("variance %d\n", tensorOutSize);
    }

    //printf("\n");


    float out;
    checkCudaErrors(cudaMemcpy(&out, tensor, sizeof(float), cudaMemcpyDeviceToHost));

    infoMsg("sum of square differences is %f\n",out);

    return float(sqrt(out/(tensorSizeInitial-1)));
    //return tensorOut;

}

extern "C" cudaError_t multiplyTensorByScalarWrapper_float(float *tensor, float *tensorOut, size_t tensorSize, float scalar);

extern "C" cudaError_t multiplyTensorByScalarWrapper_double(double *tensor, double *tensorOut, size_t tensorSize, double scalar);

extern "C" cudaError_t multiplyTensorByScalarWrapper_half(void *tensor, void *tensorOut, size_t tensorSize, void* scalar);


//#define getExtents true
/*
template<typename T>
void DefaultNet<T>::loadInput(cudaArray* densityArray){

    if(firstIteration){

        cudaChannelFormatDesc densityDesc;
        cudaExtent densityExtent;
        unsigned int densityFlags;
        checkCudaErrors(cudaArrayGetInfo(&densityDesc, &densityExtent, &densityFlags, densityArray));

        printf("densityDesc.x = %d, densityDesc.y = %d, densityDesc.z = %d, densityDesc.w = %d, densityDesc.f = %d\n",
               densityDesc.x, densityDesc.y, densityDesc.z, densityDesc.w, int(densityDesc.f));


    }
}
*/

//input has size 1*3*domainSize.z*domainSize.y*domainSize.x*sizeof(T)
//it contains lastPressure, velDivergence and obstacles, continguous in memory
//THE ADVECTION ETC. ARE DONE IN FLUIDNET, BUT WITH CUDA AND WITH KERNELS IN FLUIDNET.CU
template<typename T>
void DefaultNet<T>::loadInput(T* input, T* pressureBuffer, T* velocityBuf, dim3 domainSize, dim3 velocitySize){

    //need to turn pressureArray into pressureDivergenceArray

    //the following needs to be done when input layer is created
    //checkCudaErrors(cudaMalloc(&dataToy,  sizeof(float) * dataExtent.width * dataExtent.height * dataExtent.depth));

    //since the extent of the layers cannot change mid-simulation (at least currently),

    //numIteration = 0;

    printf("velocityBuf = %p\n", velocityBuf);

    this->pressure = pressureBuffer;
    this->velocity = velocityBuf;

    if(firstIteration){ //first time loadInput is called for this DefaultNet object


        infoMsg("domainSize.x = %d, domainSize.y = %d, domainSize.z = %d\n", domainSize.x, domainSize.y, domainSize.z);
        infoMsg("velocitySize.x = %d, velocitySize.y = %d, velocitySize.z = %d\n", velocitySize.x, velocitySize.y, velocitySize.z);

        velocityBufferSize = 3*velocitySize.x * velocitySize.y * velocitySize.z;

        inputLayerVelDivergenceOffset = domainSize.x * domainSize.y * domainSize.z;
        inputLayerObstacleOffset = inputLayerVelDivergenceOffset;
        inputLayerObstacleBufferSize = inputLayerVelDivergenceOffset;

        this->inputLayer->tensorDims.resize(5);

        this->inputLayer->tensorDims[0] = 1;
        this->inputLayer->tensorDims[1] = 3;
        this->inputLayer->tensorDims[2] = domainSize.z;
        this->inputLayer->tensorDims[3] = domainSize.y;
        this->inputLayer->tensorDims[4] = domainSize.x;

        this->inputLayer->setStride();

        //need to allocate velocityBuffer and velocityBuffer2
        checkCudaErrors(cudaMalloc(&velocityBuffer, velocityBufferSize * sizeof(T)));
        checkCudaErrors(cudaMalloc(&velocityBuffer2, velocityBufferSize * sizeof(T)));

        //due to this, there should be no effort to retrieve the updated obstacle texture from the OpenGL side

        //this version should not require allocating the input again for copying

        this->allocateTensors(input, pressureBuffer);
        //this->setWallsWrapper(this->inputLayer->data + inputLayerVelDivergenceOffset + inputLayerObstacleOffset, domainSize, 1); //thickness of 1
        firstIteration = false;

        //set divergence edges to zero
        //setEdgesWrapper_float(this->inputLayer->data + inputLayerVelDivergenceOffset, domainSize, 0.0f);
    }

    //BOOKMARK
    infoMsg("velocityBufferSize = %d",int(velocityBufferSize));

    //this sucks and there has to be a better way
    //did you allocate the velocityOut?
    infoMsg("sizeof(T) = %d", sizeof(T));
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy((void*)velocityBuffer, (void*)velocityBuf, velocityBufferSize*sizeof(T), cudaMemcpyDeviceToDevice));
    velSTD = getStandardDeviation(velocityBuffer, velocityBuffer2, velocityBufferSize);
    //velSTD2 = getStandardDeviation()

    infoMsg("velSTD = %f", velSTD);
    printf("velSTD = %f\n", velSTD);
    //exit(0);

    //DISABLED FOR NETWORK TESTING
    //this needs a wrapper so that non-float buffers work

    //version that was supposed to just multiply the flags to avoid other divisions
    /*
    multiplyTensorByScalarWrapper_float(this->inputLayer->data + inputLayerVelDivergenceOffset + inputLayerObstacleOffset,
                                        this->inputLayer->data + inputLayerVelDivergenceOffset + inputLayerObstacleOffset, //twice so it's inplace
                                        inputLayerObstacleBufferSize,
                                        velSTD);*/

    //version that replicates what the torch version does
    //THIS MAY NEED TO BE REMOVED (or the flags sent to DefaultNet->forward changed) if you go back to the CUDA advection method
    if(velSTD!=0 && !isnan(velSTD))
    multiplyTensorByScalarWrapper_float(this->inputLayer->data,
                                        this->inputLayer->data, //twice so it's inplace
                                        inputLayerObstacleBufferSize*2,
                                        1.0f/velSTD);

    this->divergence = this->inputLayer->data + inputLayerObstacleBufferSize;

    numIteration++;
}

#define checkExtents true

//given that the first convolution tensor has N=3, it appears to take the divergence
//dataType has the default value CUDNN_DATA_FLOAT
//we need velocityArray to calculate the normalization
template<typename T>
void DefaultNet<T>::loadInput(cudaArray* lastPressureArray, cudaArray* obstacleArray, cudaArray* velDivergenceArray, cudaArray* pressureArray, cudaArray* velocityArray, cudaArray* velocityPongArray){//, cudnnDataType_t dataType){

    //need to turn pressureArray into pressureDivergenceArray

    //the following needs to be done when input layer is created
    //checkCudaErrors(cudaMalloc(&dataToy,  sizeof(float) * dataExtent.width * dataExtent.height * dataExtent.depth));

    //since the extent of the layers cannot change mid-simulation (at least currently),

    //numIteration = 0;

    this->obstacleArray = obstacleArray;

    if(firstIteration){ //first time loadInput is called for this DefaultNet object

        cudaChannelFormatDesc lastPressureDesc;
        cudaExtent lastPressureExtent;
        unsigned int lastPressureFlags;
        checkCudaErrors(cudaArrayGetInfo(&lastPressureDesc, &lastPressureExtent, &lastPressureFlags, lastPressureArray));

        cudaChannelFormatDesc obstacleDesc;
        cudaExtent obstacleExtent;
        unsigned int obstacleFlags;
        checkCudaErrors(cudaArrayGetInfo(&obstacleDesc, &obstacleExtent, &obstacleFlags, obstacleArray));

        cudaChannelFormatDesc velDivergenceDesc;
        cudaExtent velDivergenceExtent;
        unsigned int velDivergenceFlags;
        checkCudaErrors(cudaArrayGetInfo(&velDivergenceDesc, &velDivergenceExtent, &velDivergenceFlags, velDivergenceArray));

        cudaChannelFormatDesc velocityDesc;
        cudaExtent velocityExtent;
        unsigned int velocityFlags;
        checkCudaErrors(cudaArrayGetInfo(&velocityDesc, &velocityExtent, &velocityFlags, velocityArray));

        cudaChannelFormatDesc velocityPongDesc;
        cudaExtent velocityPongExtent;
        unsigned int velocityPongFlags;
        checkCudaErrors(cudaArrayGetInfo(&velocityPongDesc, &velocityPongExtent, &velocityPongFlags, velocityPongArray));

        /*
      //Channel format kind
      enum __device_builtin__ cudaChannelFormatKind
      {
          cudaChannelFormatKindSigned           =   0,      //< Signed channel format
          cudaChannelFormatKindUnsigned         =   1,      //< Unsigned channel format
          cudaChannelFormatKindFloat            =   2,      //< Float channel format
          cudaChannelFormatKindNone             =   3       //< No channel format
      };

      //CUDA Channel format descriptor
      struct __device_builtin__ cudaChannelFormatDesc
      {
          int                        x; //< x
          int                        y; //< y
          int                        z; //< z
          int                        w; //< w
          enum cudaChannelFormatKind f; //< Channel format kind
      };*/

        printf("velocityDesc.x = %d, velocityDesc.y = %d, velocityDesc.z = %d, velocityDesc.w = %d, velocityDesc.f = %d\n",
               velocityDesc.x, velocityDesc.y, velocityDesc.z, velocityDesc.w, int(velocityDesc.f));

        printf("velocityExtent.width = %d; velocityExtent.height = %d; velocityExtent.depth = %d", velocityExtent.width, velocityExtent.height, velocityExtent.depth);

        infoMsg("velDivergenceDesc.x = %d, velDivergenceDesc.y = %d, velDivergenceDesc.z = %d, velDivergenceDesc.w = %d, velDivergenceDesc.f = %d\n",
                velDivergenceDesc.x, velDivergenceDesc.y, velDivergenceDesc.z, velDivergenceDesc.w, int(velDivergenceDesc.f));

        infoMsg("lastPressureDesc.x = %d, lastPressureDesc.y = %d, lastPressureDesc.z = %d, lastPressureDesc.w = %d, lastPressureDesc.f = %d\n",
                lastPressureDesc.x, lastPressureDesc.y, lastPressureDesc.z, lastPressureDesc.w, int(lastPressureDesc.f));

        infoMsg("obstacleDesc.x = %d, obstacleDesc.y = %d, obstacleDesc.z = %d, obstacleDesc.w = %d, obstacleDesc.f = %d\n",
                obstacleDesc.x, obstacleDesc.y, obstacleDesc.z, obstacleDesc.w, int(obstacleDesc.f));

        //exit(0);

        velocityBufferSize = 3 * velocityExtent.depth * velocityExtent.height * velocityExtent.width;

        if(checkExtents){

            size_t velDivergenceOffset = lastPressureExtent.depth * lastPressureExtent.width * lastPressureExtent.height;
            size_t obstacleOffset = velDivergenceExtent.depth * velDivergenceExtent.width * velDivergenceExtent.height;

            //obstacleExtent.depth * obstacleExtent.width * obstacleExtent.height;

            if(velDivergenceOffset!=inputLayerVelDivergenceOffset){
                warnMsg("WARNING: For velDivergence: cudaExtent of lastPressureDesc cudaArray does not match inputLayerVelDivergenceOffset\n");
                inputLayerVelDivergenceOffset = velDivergenceOffset;}
            if(obstacleOffset!=inputLayerObstacleOffset){
                warnMsg("WARNING: For velDivergence: cudaExtents of lastPressureDesc and/or velDivergenceDesc cudaArrays does not match inputLayerObstacleOffset\n");
                inputLayerObstacleOffset = obstacleOffset;}
        }

        else{

            inputLayerVelDivergenceOffset = lastPressureExtent.depth * lastPressureExtent.width * lastPressureExtent.height;
            inputLayerObstacleOffset = velDivergenceExtent.depth * velDivergenceExtent.width * velDivergenceExtent.height;

        }

        inputLayerObstacleBufferSize = obstacleExtent.depth * obstacleExtent.width * obstacleExtent.height;

        this->inputLayer->tensorDims.resize(5);

        this->inputLayer->tensorDims[0] = 1;
        this->inputLayer->tensorDims[1] = 3;
        this->inputLayer->tensorDims[2] = velDivergenceExtent.depth;
        this->inputLayer->tensorDims[3] = velDivergenceExtent.width;
        this->inputLayer->tensorDims[4] = velDivergenceExtent.height;// 1, 3, velDivergenceExtent.depth, velDivergenceExtent.width, velDivergenceExtent.height;

        this->inputLayer->setStride();

        //need to allocate velocityBuffer and velocityBuffer2
        checkCudaErrors(cudaMalloc(&velocityBuffer, velocityBufferSize * sizeof(T)));
        checkCudaErrors(cudaMalloc(&velocityBuffer2, velocityBufferSize * sizeof(T)));

        //due to this, there should be no effort to retrieve the updated obstacle texture from the OpenGL side

        this->allocateTensors();
        //this->setWallsWrapper(this->inputLayer->data + 2*velDivergenceExtent.depth*velDivergenceExtent.width*velDivergenceExtent.height, dim3(obstacleExtent.width, obstacleExtent.height, obstacleExtent.depth), 1); //thickness of 1
        firstIteration = false;
    }

#undef checkExtents
#undef getExtents

    //INPUTLAYER NEEDS TO HAVE ITS DATA ALLOCATED BY THIS POINT
    //copyCudaArrayToDeviceArrayDummy(lastPressureArray, velDivergenceArray, obstacleArray, inputLayerObstacleOffset, inputLayerVelDivergenceOffset);
    copyCudaArrayToDeviceArrayZeroObstacles(lastPressureArray, velDivergenceArray, obstacleArray, inputLayerObstacleOffset, inputLayerVelDivergenceOffset); //->this deletes obstacles (not any more)

    pressure = this->outputLayer->data;
    divergence = this->inputLayer->data + inputLayerObstacleBufferSize;
    obstacles = this->inputLayer->data + inputLayerObstacleBufferSize*2;

    //printInputAndOutput();

    //BOOKMARK
    infoMsg("velocityBufferSize = %d",int(velocityBufferSize));

    checkCudaErrors(copyCudaArrayToDeviceArrayWrapper3D_float(velocityArray, velocityBuffer));
    checkCudaErrors(cudaDeviceSynchronize());
    velSTD = getStandardDeviation(velocityBuffer, velocityBuffer2, velocityBufferSize); //need a cudaArrayGetInfo for velocityArraySize

    //will not need this if we create a version of subtractGradient that works on cudaArrays
    checkCudaErrors(copyCudaArrayToDeviceArrayWrapper3D_float(velocityArray, velocityBuffer));

    printf("velocityBuffer loadInput\n\n");
    checkCudaErrors(printfArrayWrapper(velocityBuffer,dim3(64,64,64), 1, dim3(1,1,1),false));
    this->velocity = velocityBuffer;

    infoMsg("velSTD = %f", velSTD);
    printf("velSTD = %f\n", velSTD);

    numIteration++;
}

template<typename T>
cudnnHandle_t* Layer<T>::getLayerCudnnHandle(){
    return layerCudnnHandle;
}

extern "C" cudaError_t upscaleLayer3DKernelWrapper_float(float* input, float* output, int* outputDims, int* upscale);
extern "C" cudaError_t upscaleLayer3DKernelWrapper_half(void* input, void* output, int* outputDims, int* upscale);
extern "C" cudaError_t upscaleLayer3DKernelWrapper_double(double* input, double* output, int* outputDims, int* upscale);

//inputDims and upscale are of length 5
template<>
cudaError_t UpscaleLayer3D<float>::upscaleLayer3DKernelWrapper(float* input, float* output, int* outputDims, int* upscale){

    upscaleLayer3DKernelWrapper_float(input, output, outputDims, upscale);
}

template<>
cudaError_t UpscaleLayer3D<half>::upscaleLayer3DKernelWrapper(half* input, half* output, int* outputDims, int* upscale){

    return printCudaErrors(upscaleLayer3DKernelWrapper_half(input, output, outputDims, upscale));
}

template<>
cudaError_t UpscaleLayer3D<double>::upscaleLayer3DKernelWrapper(double* input, double* output, int* outputDims, int* upscale){

    return printCudaErrors(upscaleLayer3DKernelWrapper_double(input, output, outputDims, upscale));
}

template<typename T>
void UpscaleLayer3D<T>::forward(){

    //upscaleFactor needs to have 5 elements (or more) (it will if the constructor is correct), or else it will segfault
    checkCudaErrors(upscaleLayer3DKernelWrapper(this->connectedInputLayer->data, this->data, this->tensorDims.data(), this->upscaleFactor.data()));
}

//if upscaleFactor has more than 5 elements, all elements after the fifth are ignored
//if it has less than 5 elements, the last len(upscaleFactor) dimensions are upscaled (the order of dimensions is NCDHW)
//a 3-element upscaleFactor is thus interpreted as zyx
template<typename T>
UpscaleLayer3D<T>::UpscaleLayer3D(std::vector<int> _upscaleFactor){

#define max(x,y) ((x)>(y))?(x):(y)
    int i = max(0, 5 - _upscaleFactor.size());
#undef max
    int j;

    upscaleFactor = _upscaleFactor;
    /*
  for(j=0;i<5;i++,j++)
    upscaleFactor[i] = _upscaleFactor[j];
*/
}

template<typename T>
UpscaleLayer3D<T>::~UpscaleLayer3D(){

    //is empty in the implementation with kernels

    //checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
    //checkCUDNN(cudnnDestroyTensorDescriptor(convTensorDesc));

    //checkCudaErrors(cudaFree(convData));
}

template<typename T>
cudnnDataType_t Layer<T>::getDataType(){

    return dataType;
}

//this contains no operation, so don't need to do anything
template<typename T>
void InputLayer3D<T>::forward(){ }

//as with the others, dims = {N,C,D,H,W} (generally, N = 1)
//default: N=1,C=3(obstacle grid, velocity divergence, pressure divergence, _probably_ in that order), 128 (depth of input matrix), 128 (width), 128 (height)
//128x128x128 matrix corresponds to a 64x64x64 cell staggered MAC grid in the paper //this is probably BS
template<typename T>
InputLayer3D<T>::InputLayer3D(int dims[]){

    if(dims == NULL){
        this->getTotalTensorSize();
        return;
    }


    this->tensorDims[0] = dims[0];
    this->tensorStrides[0] = 1;

    for(int i=1; i<5; i++){
        this->tensorDims[i] = dims[i];
        this->tensorStrides[i] = dims[i-1] * this->tensorStrides[i-1];
    }

    checkCUDNN(cudnnCreateTensorDescriptor(&(this->tensorDesc)));
    checkCUDNN(cudnnSetTensorNdDescriptor(this->tensorDesc, this->getDataType(), 5, dims, this->tensorStrides.data()));
}

template<typename T>
void AddLayer3D<T>::forward(){

    //crashes with SIGSEGV
    checkCUDNN(cudnnAddTensor(*(this->getLayerCudnnHandle()),
                              &alpha, //probably should be allowed to not be 1
                              tensorDesc,
                              this->data,
                              &beta,
                              otherInputLayer->tensorDesc,
                              otherInputLayer->data));
}

template<typename T>
AddLayer3D<T>::AddLayer3D(Layer<T> *otherInputLayer){

    //alpha and beta are on the cpu
    alpha = 1.0f;
    beta = 1.0f;

    otherInputLayer = std::move(otherInputLayer);

}

template<typename T>
AddLayer3D<T>::~AddLayer3D(){

    //there are no tensors to destroy in this case, so just cudaFree all the stuff
    //will probably change if you switch to ping=-ponging, mind the segfaults
}

template<typename T>
ScalingLayer3D<T>::ScalingLayer3D(T _scaleFactor){

    scaleFactor = _scaleFactor;

    if(typeid(T) == typeid(float))
        (*multiplyTensorByScalarWrapper)() = multiplyTensorByScalarWrapper_float;
    else if(typeid(T) == typeid(double))
        (*multiplyTensorByScalarWrapper)() = multiplyTensorByScalarWrapper_double;
    else if(typeid(T) == typeid(half))
        (*multiplyTensorByScalarWrapper)() = multiplyTensorByScalarWrapper_half;
}

//the cudnnConvolutionForward function doesn't allow bias, it allows blending with the output it overwrites
//can use cudnnConvolutionBiasActivationForward for convolution, bias and activation. If a new non-cuDNN activation is ever added, can use CUDNN_ACTIVATION_IDENTITY plus a kernel function on the output.
//try to make z in cudnnConvolutionBiasActivationForward the same as the bias, with alpha2 set to zero.
template<typename T>
ConvBiasLayer3D<T>::ConvBiasLayer3D(std::vector<int> filterSize, std::vector<T> filterData, std::vector<int> biasSize, std::vector<T> biasDataCPU, std::string activationMode, double _coef){

    std::transform(activationMode.begin(), activationMode.end(), activationMode.begin(), ::tolower);

    if(activationModeMap.find(activationMode) == activationModeMap.end()){
        warnMsg("WARNING: Activation mode not recognized, using relu.\n");
        activationMode = "relu";}


    alpha2 = 0.0f;

    //biasSize will get copied to biasDims
    //filterSize will get copied to filterDims

    int ones[] = {1,1,1,1,1};

    filterDims = std::vector<int>(ones, ones+int(filterSize.size()));
    filterStrides = std::vector<int>(ones, ones+int(filterSize.size()));

    biasDims = std::vector<int>(ones, ones+int(filterSize.size()));
    biasStrides = std::vector<int>(ones, ones+int(filterSize.size()));

    for(int i=0;i<5;i++)
        std::cout<<filterSize[i];

    std::cout<<std::endl;

    coef = _coef;

    this->convData = (void*)(filterData.data());

    //int convDims[] = {1,8,3,3,3}; //NCDHW //these are the dimensions of the input, with the network described in the paper

    //filterStrides[0] = 1; //-> this is already so by default (in cnn.h)
    //biasStrides[0] = 1; //-> same here

    //expected stride, also calculate total number of elements
    int totalSize = filterSize[0];
    int totalBiasSize = biasSize[0];

    filterDims[0] = filterSize[0];
    filterStrides[0] = 1;

    //the bias needs to have the first element 1, and the second element equal to the number of feature maps
    biasDims[1] = biasSize[0];
    biasStrides[0] = 1;
    //filter = ;

    for(int i=1;i<filterSize.size();i++){
        filterDims[i] = filterSize[i];
        totalSize *= filterSize[i];
        filterStrides[i] = filterStrides[i-1] * filterSize[i];}

    for(int i=1;i<biasSize.size();i++){
        //biasDims[i] = biasSize[i];
        totalBiasSize *= biasSize[i];
        biasStrides[i] = biasStrides[i-1] * biasSize[i];
    }

    //for convDims = {1,1,1,1,1}, filterStrides should be {1,1,1,1,1}

    //alpha and beta are on the cpu
    //alpha = 1.0f;
    checkCUDNN(cudnnCreateTensorDescriptor(&convTensorDesc));

    for(int i=0;i<filterDims.size();i++)
        infoMsg("filterDims.data()[%d] = %d\n", i, filterDims.data()[i]);
    infoMsg("\n");

    for(int i=0;i<filterStrides.size();i++)
        infoMsg("filterDims.data()[%d] = %d\n", i, filterDims.data()[i]);
    infoMsg("\n");

    //the bias needs to have the first element 1, and the second element equal to the number of feature maps
    for(int i=0;i<biasDims.size();i++)
        infoMsg("biasDims.data()[%d] = %d\n", i, biasDims.data()[i]);
    infoMsg("\n");

    for(int i=0;i<biasStrides.size();i++)
        infoMsg("biasStrides.data()[%d] = %d\n", i, biasStrides.data()[i]);
    infoMsg("\n");

    //ALSO CUDNN_STATUS_BAD_PARAM
    checkCUDNN(cudnnSetTensorNdDescriptor(convTensorDesc,this->getDataType(),int(filterDims.size()),filterDims.data(),filterStrides.data())); //->filterDims should be 5 elements long, I think, not 3

    //this seems to set the values of the the device pointer y to the same SINGLE value specified by the valuePtr
    //T valuePtr[] = {1};

    //allocate GPU memory and copy the filterData to it
    checkCudaErrors(cudaMalloc(&convData, sizeof(T) * totalSize));
    checkCudaErrors(cudaMemcpy(convData, filterData.data(), sizeof(T) * totalSize, cudaMemcpyHostToDevice));

    //allocate GPU memory for the bias
    infoMsg("biasData size: %d\n", sizeof(T) * totalBiasSize);
    checkCudaErrors(cudaMalloc(&biasData, sizeof(T) * totalBiasSize));
    checkCudaErrors(cudaMemcpy(biasData, biasDataCPU.data(), sizeof(T) * totalBiasSize, cudaMemcpyHostToDevice)); //SIGSEGV here

    //bias needs a tensor descriptor
    checkCUDNN(cudnnCreateTensorDescriptor(&biasTensorDesc));

    //printf("getDataType %d\n", int(this->getDataType()));
    //CUDNN_STATUS_BAD_PARAM
    //int _biasSize[] = {1,1,1,1,1};

    //CUDNN_STATUS_BAD_PARAM
    //need to resize it to the number of dimensions of the convolution tensor
    checkCUDNN(cudnnSetTensorNdDescriptor(biasTensorDesc,this->getDataType(), int(biasDims.size()), biasDims.data(), biasStrides.data())); //Heisenbug?

    //Throws CUDNN_STATUS_BAD_PARAM if "The second dimension of biasDesc and the first dimension of filterDesc are not equal."
    //this cannot be called until addLayer, since the input tensor size must be known
    /*checkCUDNN(cudnnConvolutionBiasActivationForward(this->getLayerCudnnHandle(),
                                                   &alpha,
                                                   ));*/

    //checkCUDNN(cudnnSetTensor(*(this->getLayerCudnnHandle()), convTensorDesc, convData, valuePtr)); -> the hell is this for?

    //trying with 5 (padding only on the spatial dimensions (DHW))
    int padding[] = {0,0,0,0,0};

    //if padding needs to be 3x3x3, then we start with 0, otherwise we start with 2 (same as filterSize on RHS)
    padding[2] = filterSize[2] / 2; //this should give 1x1x1 padding for 3x3x3 and 0x0x0 padding for 1x1x1, as required; rest of the padding (on N and C) should be zero in both places
    padding[3] = filterSize[3] / 2;
    padding[4] = filterSize[4] / 2;

    int filterStride[5];

    filterStride[0] = 1;
    for(int i=1;i<5;i++)
        filterStride[i] = filterSize[i-1]* filterStride[i-1];

    int upscaleFactor[] = {1,1,1,1,1};

    checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
    //CUDNN_STATUS_BAD_PARAM
    //it is still unclear to me what the difference between CUDNN_CONVOLUTION and CUDNN_CROSS_CORRELATION is, if any

    //int filterStride_v2[] = {1,3,9}; //-> wrong way?
    int filterStride_v2[] = {1,1,1};

    //CUDNN_CROSS_CORRELATION matches the output of Torch7 and TensorFlow
    //A cross-correlation is equivalent to a convolution with its filter rotated by 180 degrees.
    checkCUDNN(cudnnSetConvolutionNdDescriptor(convDesc, 3, padding+2, filterStride_v2, upscaleFactor, CUDNN_CROSS_CORRELATION,//CUDNN_CONVOLUTION,
                                               this->getDataType()));

    //filterDesc seems to be wrong; it should be output dimensions first, input dimensions second (so 8,3,3,3,3)
    //IF CORRECT, THIS NEEDS TO BE FIXED ELSEWHERE

    //this might mean that the data is in the wrong order (?)
    checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
    checkCUDNN(cudnnSetFilterNdDescriptor(filterDesc,this->getDataType(),CUDNN_TENSOR_NCHW,5,filterDims.data()));

    //add activation based on activationMode
    checkCUDNN(cudnnCreateActivationDescriptor(&activationDescriptor));
    checkCUDNN(cudnnSetActivationDescriptor(activationDescriptor, activationModeMap[activationMode], CUDNN_NOT_PROPAGATE_NAN, coef));

    infoMsg("conv5 activation is %d\n",int(activationModeMap[activationMode]));

    this->activationMode = activationModeMap[activationMode]; //activationMode is a string, this->activationMode is a cudnnActivationMode_t
}

//THIS ENTIRE FUNCTION NEEDS TO BE CHANGED TO USE cudnnConvolutionBiasActivationForward
//Since you would otherwise need an intermediate tensor between the convolution addition and the bias addition,
//the z tensor memory waste in cudnnConvolutionBiasActivationForward would happen anyway
template<typename T>
void ConvBiasLayer3D<T>::forward(){

    infoMsg(this->name.c_str());
    infoMsg("\n");

    if(getActivationMode() == CUDNN_ACTIVATION_RELU)
        //this only seems to work for ReLU, even though it should work for Identity as well with CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
        checkCUDNN(cudnnConvolutionBiasActivationForward(*(this->getLayerCudnnHandle()),
                                                         &alpha,
                                                         this->connectedInputLayer->tensorDesc,
                                                         this->connectedInputLayer->data,
                                                         filterDesc,
                                                         convData,
                                                         convDesc,
                                                         algorithm,
                                                         workspace,
                                                         workspaceSizeInBytes,
                                                         &alpha2,
                                                         uselessDesc,
                                                         uselessData,
                                                         biasTensorDesc,
                                                         biasData,
                                                         activationDescriptor,
                                                         this->tensorDesc,
                                                         this->data));
    else if(getActivationMode() != CUDNN_ACTIVATION_IDENTITY){ //identity doesn't work with cudnnactivationforward (as it probably shouldn't, it would just be a cudaMemcpy)
        //will probably need to set the convoluton algorithm, workspace and other stuff in the constructor
        //CUDNN_EXECUTION_FAILED when CUDNN_CONVOLUTION_FWD_NO_WORKSPACE is used
        //documentation doesn't help, but the workspace size is probably too small for fastest algorithm; try changing the preference for getting the forward algorithm

        checkCUDNN(cudnnConvolutionForward(*(this->getLayerCudnnHandle()),
                                           &alpha,
                                           this->connectedInputLayer->tensorDesc, //for base class members, need either this or Base<T>:: (or using something something) (https://isocpp.org/wiki/faq/templates#nondependent-name-lookup-members)
                                           this->connectedInputLayer->data,
                                           filterDesc,
                                           convData,
                                           convDesc,
                                           algorithm,
                                           workspace,
                                           workspaceSizeInBytes,
                                           &beta,
                                           this->uselessDesc,
                                           this->uselessData)); //uselessData is an intermediary buffer, output for convolution forward and input for activation forward

        //add the bias
        //this is inplace, from uselessData to uselessData
        //alpha should be 1, beta should be zero
        checkCUDNN(cudnnAddTensor(*(this->getLayerCudnnHandle()),
                                  &alpha,
                                  biasTensorDesc,//this->connectedInputLayer->tensorDesc,
                                  biasData,//this->connectedInputLayer->data,
                                  &beta,
                                  this->uselessDesc, //this is NOT the same as this, since the bias layer has lower dimensionality
                                  this->uselessData));

        T activationAlpha = 0.0;
        T activationBeta = 1.0; //these always have these values

        checkCUDNN(cudnnActivationForward(*(this->getLayerCudnnHandle()),
                                          this->activationDescriptor,
                                          &activationAlpha,
                                          this->uselessDesc,
                                          this->uselessData,
                                          &activationBeta,
                                          this->tensorDesc,
                                          this->data));
    }
    else{
        checkCUDNN(cudnnConvolutionForward(*(this->getLayerCudnnHandle()),
                                           &alpha,
                                           this->connectedInputLayer->tensorDesc, //for base class members, need either this or Base<T>:: (or using something something) (https://isocpp.org/wiki/faq/templates#nondependent-name-lookup-members)
                                           this->connectedInputLayer->data,
                                           filterDesc,
                                           convData,
                                           convDesc,
                                           algorithm,
                                           workspace,
                                           workspaceSizeInBytes,
                                           &beta,
                                           this->tensorDesc,
                                           this->data));

        checkCUDNN(cudnnAddTensor(*(this->getLayerCudnnHandle()),
                                  &alpha,
                                  biasTensorDesc,//this->connectedInputLayer->tensorDesc,
                                  biasData,//this->connectedInputLayer->data,
                                  &alpha,
                                  this->tensorDesc, //this is NOT the same as this, since the bias layer has lower dimensionality
                                  this->data));
    }
}

//alternatively
//need to allocate uselessData and initialize uselessDesc
//this does not support anything except RELU or Identity, for the othersw do it the other way (disabled for now)

//these are overloaded because C doesn't support template functions

template<typename T>
ScalingLayer3D<T>::ScalingLayer3D(){

}

template<typename T>
NormScalingLayer3D<T>::NormScalingLayer3D(){

    normPointer = NULL;
}

template<>
inline void NormScalingLayer3D<float>::reduceSummationWrapper(float *tensor, float *tensorOut, size_t tensorSize, size_t tensorOutSize){

    reduceSummationWrapper_float(tensor, tensorOut, tensorSize, tensorOutSize);
}

template<>
inline void NormScalingLayer3D<double>::reduceSummationWrapper(double *tensor, double *tensorOut, size_t tensorSize, size_t tensorOutSize){

    reduceSummationWrapper_double(tensor, tensorOut, tensorSize, tensorOutSize);
}

template<>
inline void NormScalingLayer3D<half>::reduceSummationWrapper(half *tensor, half *tensorOut, size_t tensorSize, size_t tensorOutSize){

    reduceSummationWrapper_half((void*)(tensor), (void*)(tensorOut), tensorSize, tensorOutSize);
}

template<>
inline void NormScalingLayer3D<float>::reduceSummationVarianceWrapper(float *tensor, float *tensorOut, size_t tensorSize, size_t tensorOutSize, float average){

    reduceSummationVarianceWrapper_float(tensor, tensorOut, tensorSize, tensorOutSize, average);
}

template<>
inline void NormScalingLayer3D<double>::reduceSummationVarianceWrapper(double *tensor, double *tensorOut, size_t tensorSize, size_t tensorOutSize, double average){

    reduceSummationVarianceWrapper_double(tensor, tensorOut, tensorSize, tensorOutSize, average);
}

template<>
inline void NormScalingLayer3D<half>::reduceSummationVarianceWrapper(half *tensor, half *tensorOut, size_t tensorSize, size_t tensorOutSize, half average){

    reduceSummationVarianceWrapper_half((void*)(tensor), (void*)(tensorOut), tensorSize, tensorOutSize, (void*)(&average));
}

//DO NOT USE THIS, THE OTHER ONE IS CALLED (getStandardDeviation)
template<typename T>
T NormScalingLayer3D<T>::getNorm(){
    return (T)1;
}

template<typename T>
void ScaleByInverseNormLayer3D<T>::forward(){

    //get norm
    if(this->normPointer != NULL)
        this->scaleFactor = 1.0f / this->getNorm();
    else{
        //norm = &normPointer; //can probably remove norm member altogether, since its value is stored in ScalingLayer3D<T>->scaleFactor
        this->scaleFactor = 1.0f / *(this->normPointer);}

    ScalingLayer3D<T>::forward();
}

template<typename T>
void ScaleByNormLayer3D<T>::forward(){

    //get norm
    if(this->normPointer != NULL)
        this->scaleFactor = this->getNorm();
    else{
        //norm = &normPointer; //can probably remove norm member altogether, since its value is stored in ScalingLayer3D<T>->scaleFactor
        this->scaleFactor = 1.0f / *(this->normPointer);}

    ScalingLayer3D<T>::forward(); //<------ THIS IS POTENTIALLY PROBLEMATIC (or not) (what do you think "potentially" means?)
}

template<typename T>
void PoolingLayer3D<T>::forward(){

    checkCUDNN(cudnnPoolingForward(*(this->getLayerCudnnHandle()),
                                   poolingDesc,
                                   &alpha,
                                   this->connectedInputLayer->tensorDesc,
                                   this->connectedInputLayer->data,
                                   &beta,
                                   this->tensorDesc,
                                   this->data));

    infoMsg("poolingForward %s\n", this->name.c_str());
}

/*
template<typename T>
void AvgPoolLayer3D<T>::forward(){
  PoolingLayer3D<T>::forward();

}
*/

//might make all of the scalar parameters pointers for simplicity when calling the multiplyTensorByScalar function pointer
extern "C" cudaError_t multiplyTensorByScalarWrapper_float(float *tensor, float *tensorOut, size_t tensorSize, float scalar);
extern "C" cudaError_t multiplyTensorByScalarWrapper_double(double *tensor, double *tensorOut, size_t tensorSize, double scalar);
extern "C" cudaError_t multiplyTensorByScalarWrapper_half(void *tensor, void *tensorOut, size_t tensorSize, void* scalar);

template<>
inline cudaError_t ScalingLayer3D<float>::multiplyTensorByScalarWrapper(float *tensor, float *tensorOut, size_t tensorSize, float scalar){

    return printCudaErrors(multiplyTensorByScalarWrapper_float(tensor, tensorOut, tensorSize, scalar));
}

template<>
inline cudaError_t ScalingLayer3D<double>::multiplyTensorByScalarWrapper(double *tensor, double *tensorOut, size_t tensorSize, double scalar){

    return printCudaErrors(multiplyTensorByScalarWrapper_double(tensor, tensorOut, tensorSize, scalar));
}

template<>
inline cudaError_t ScalingLayer3D<half>::multiplyTensorByScalarWrapper(half *tensor, half *tensorOut, size_t tensorSize, half scalar){

    return printCudaErrors(multiplyTensorByScalarWrapper_half((void*)(tensor), (void*)(tensorOut), tensorSize, (void*)(&scalar)));
}


template<typename T>
void ScalingLayer3D<T>::forward(){

    multiplyTensorByScalarWrapper(this->connectedInputLayer->data, this->data, this->connectedInputLayer->getTotalTensorSize(), this->scaleFactor); //multiplys tensor by scaleFactor
}

/*
void Layer3D::forward(Net net){
  forward(net.cudnnHandle);
}
*/

template<typename T>
int DefaultNet<T>::powInt(int x, int power){

    int result = 1;

    for(int i=0;i<power;i++)
        result *= x;

    return result;
}

template<typename T>
void DefaultNet<T>::makeNet(int inputDims[], int numBanks, int bankLength){

    //bankLength is the number of convolutions in each bank (2 by default (3->8, 8->8), might make this a parameter)

    ///
    ///    conv1   conv2    conv3                          ,----,         conv4    conv5
    /// IN ----> A -----> B -----> C --------------------->|add2|-----> F -----> G -----> OUT
    ///          |                                         '----'
    ///     pool1|                                           ^
    ///          v conv2a   conv3a    upx2         ,----,    |
    ///         A1 ----->B1 ----->C1 ------>D1 --->|add1|--> E
    ///          |                                 '----'
    ///     pool2|                                   ^
    ///          v conv2b   conv3b    upx4           |
    ///         A2 ----->B2 ----->C2 ------>D2 ------'
    ///
    /// conv2, conv2a and conv2b have the same filters, but the cudnn objects have different strides
    /// A,A1,A2 could perhaps be packed together (same for B,B1,B2 as a result)
    /// upx2 and upx4 don't use
    /// THE TENSOR OUTPUT BY EACH LAYER (convolution, pooling, etc.) IS CONTAINED IN THE LAYER OBJECT CONTAINING THAT LAYER
    ///

    firstIteration = True;

    std::unique_ptr<InputLayer3D<T>> in(new InputLayer3D<T>(inputDims));
    in->name = "input";
    Layer<T>* currentPoolingOut = this->addLayer(std::move(in)); //IN

    //if you ever want to change the out-of-network scaling to a layer, use something like this, but not quite (since you need the norm of the velocity, not its divergence)
    /*
  std::unique_ptr<ScaleByInverseNormLayer3D<T>> scale1(new ScaleByInverseNormLayer3D<T>()); //with no arguments, ScalingLayer3D will
  this->addLayer(std::move(scale1));
  */

    //need to make sure the layers are added in the same order that Torch outputs them in the bin file
    size_t currentFilter = 0;

    std::unique_ptr<ConvBiasLayer3D<T>> conv;
    std::unique_ptr<UpscaleLayer3D<T>> up;
    std::unique_ptr<AvgPoolLayer3D<T>> pool;
    Layer<T> *convOut, *upOut, *firstConvOut;
    int poolDims[] = {2,2,2};
    int thePower;
    std::vector<int> upscaleDims;
    //char* nameBuffer;

    for(int i = 0;i<numBanks; i++){

        if(i != 0){
            pool = std::unique_ptr<AvgPoolLayer3D<T>>(new AvgPoolLayer3D<T>(poolDims));
            pool->name = "pool" + std::to_string(i);
            currentPoolingOut = this->addLayer(std::move(pool), currentPoolingOut); //=> Ai
        }

        convOut = currentPoolingOut;

        for(int j = 0;j<bankLength;j++){
            conv = std::unique_ptr<ConvBiasLayer3D<T>>(new ConvBiasLayer3D<T>(convTensorDims[currentFilter], filtersData[currentFilter], biasTensorDims[currentFilter], biasesData[currentFilter++], "ReLU"));

            //both the bank number and the number of layers in the bank start with 0
            conv->name = "conv_" + std::to_string(j) + "_bank_" + std::to_string(i);
            convOut = this->addLayer(std::move(conv), convOut);
        }

        if(i == 0)
            firstConvOut = convOut; //this is the layer we will concatenate all upscale outputs width

        //need to upscale by 2^i
        if(i != 0){
            thePower = powInt(2,i);
            upscaleDims = std::vector<int>({thePower, thePower, thePower});
            up = std::unique_ptr<UpscaleLayer3D<T>>(new UpscaleLayer3D<T>(upscaleDims));
            up->name = "upx"+std::to_string(upscaleDims[0]);
            upOut = this->addLayer(std::move(up), convOut);
            firstConvOut->concatenateTensorWith(upOut);
        }
    }

    if(numBanks > 1){
        std::unique_ptr<MergeContinguousTensorsLayer3D<T>> merge(new MergeContinguousTensorsLayer3D<T>()); //give it one of the layers in the continguous group
        merge->name = "merge";
        convOut = this->addLayer(std::move(merge),firstConvOut);
    }

    //remember to not prematurely delete/overwrite the output tensor or input tensors of this merge layer
    //if the data is freed, it should be only the mergeOut output tensor (as this will also free all the inputs)

    std::string activationString;

    //three more convolution layers, the last one having identity activation
    for(int i=0;i<3;i++){
        activationString = (i==2)?"Identity":"ReLU";
        conv = std::unique_ptr<ConvBiasLayer3D<T>>(new ConvBiasLayer3D<T>(convTensorDims[currentFilter], filtersData[currentFilter], biasTensorDims[currentFilter], biasesData[currentFilter++], activationString));
        conv->name = "conv_" + std::to_string(i) + "_postbanks";
        convOut = this->addLayer(std::move(conv), convOut);
    }

}

#define floor_1(x) ((x>=1)?(x):(1))

template<typename T>
//isBinary is just here to allow the constructor to be overloaded, but could also be used to separate between binary files and floats as strings in file
//numBanks is the number of resolution branches (1 + number of times we pool down)
DefaultNet<T>::DefaultNet(std::string fileName, bool isBinary, int inputSize[], int numBanks, int bankLength){// sizeofIntSource = sizeof(int), sizeofFloatSource = sizeof(float)){ //if

    infoMsg("DefaultNet constructor");

    //checkCudaErrors(cudaDeviceSynchronize());

    std::vector<std::vector<int>> allTensorDims(0); //this will have dimensions in

    std::vector<std::vector<float>> tensors(0);

    std::ifstream fin;

    fin.open(fileName, std::ios::binary);

    int currentTensorTotalElements, numDims;

    while(fin.read(reinterpret_cast<char*>(&numDims), sizeof(int))){

        allTensorDims.resize(allTensorDims.size() + 1);
        allTensorDims.back().resize(numDims);

        currentTensorTotalElements = 1;

        for(int i=0;i<numDims;i++){
            if(fin.read(reinterpret_cast<char*>(&allTensorDims.back()[i]), sizeof(int))){
                currentTensorTotalElements *= allTensorDims.back()[i];}
            else{
                errorMsg("Insufficient tensor dimensions for tensor number %d; EOF reached.",int(tensors.size()+1));
                exit(1);}
        }

        tensors.resize(tensors.size() + 1);
        tensors.back().resize(currentTensorTotalElements);

        for(int i=0;i<currentTensorTotalElements;i++){
            if(!fin.read(reinterpret_cast<char*>(&tensors.back()[i]), sizeof(float))){
                errorMsg("Insufficient tensor elements for tensor number %d; EOF reached",int(tensors.size()));
            }
        }
    }

    for(int i=0;i<10;i+=2)
        if(allTensorDims[i][0] != allTensorDims[i+1][0])
            warnMsg("WARNING: Convolution tensor conv%d does not match its bias tensor", i/2 + 1);

    convTensorDims.resize(allTensorDims.size() / 2);
    biasTensorDims.resize(convTensorDims.size());
    filtersData.resize(convTensorDims.size());
    biasesData.resize(convTensorDims.size());

    for(int i=0;i<allTensorDims.size();i+=2){
        convTensorDims[i/2].swap(allTensorDims[i]);
        biasTensorDims[i/2].swap(allTensorDims[i+1]);
        filtersData[i/2].swap(tensors[i]);
        biasesData[i/2].swap(tensors[i+1]);
    }

    makeNet(inputSize, numBanks, bankLength);

    //exit(0);
}

extern "C" cudaError_t printSingleElementWrapper_float(float* deviceArray, size_t numSizes, int* sizes, int* elementToGet);

template<>
cudaError_t DefaultNet<float>::printSingleElementWrapper(float* deviceArray, size_t numSizes, int* sizes, int* elementToGet){

    return(printCudaErrors(printSingleElementWrapper_float(deviceArray, numSizes, sizes, elementToGet)));
}

template<typename T>
T Layer<T>::getDataAtIndex(int n, int c, int d, int h, int w){

    int indexes[5];
    int index = n;
    T valueCPU;

    indexes[0] = n;
    indexes[1] = c;
    indexes[2] = d;
    indexes[3] = h;
    indexes[4] = w;

    for(int i=1;i<5;i++)
        index = index * this->tensorDims[i-1] + indexes[i];

    cudaMemcpy(&valueCPU, this->data + index, sizeof(T), cudaMemcpyDeviceToHost);

    return valueCPU;
}

template<typename T>
T Layer<T>::getDataAtIndex(int indices[5]){
    return getDataAtIndex(indices[0], indices[1], indices[2], indices[3], indices[4]);
}

template<typename T>
void Layer<T>::print(){

    infoMsg("Layer \"%s\":\n", this->name.c_str());

    infoMsg("Layer sizes -> %d %d %d %d %d\n",this->tensorDims[0],
            this->tensorDims[1],
            this->tensorDims[2],
            this->tensorDims[3],
            this->tensorDims[4]);

    float* outputTensorCPU;

    checkCudaErrors(cudaMallocHost(&outputTensorCPU, this->getTotalTensorSize() * sizeof(float)));
    checkCudaErrors(cudaMemcpy(outputTensorCPU, this->data, this->getTotalTensorSize()*sizeof(float), cudaMemcpyDeviceToHost));

    int ia[5];

    for(ia[0] = 0;ia[0]<this->tensorDims[0];ia[0]++)
        for(ia[1] = 0;ia[1]<this->tensorDims[1];ia[1]++)
            for(ia[2] = 0;ia[2]<this->tensorDims[2];ia[2]++)
                for(ia[3] = 0;ia[3]<this->tensorDims[3];ia[3]++)
                    for(ia[4] = 0;ia[4]<this->tensorDims[4];ia[4]++){
                        int index = ia[0];
                        for(int i=1;i<5;i++)
                            index = index * this->tensorDims[i] + ia[i];
                        //if(inputTensorCPU[index] != 0.0f)
                        //    printf("input[%d][%d][%d][%d][%d] = %f, index = %d\n", ia[0], ia[1], ia[2], ia[3], ia[4], double(inputTensorCPU[index]), index); //%f is used for both float and double, half needs to be converted
                        if(outputTensorCPU[index] != 0.0f)
                            infoMsg("output[%d][%d][%d][%d][%d] = %f, index = %d\n", ia[0], ia[1], ia[2], ia[3], ia[4], double(outputTensorCPU[index]), index); //%f is used for both float and double, half needs to be converted
                    }

    checkCudaErrors(cudaFreeHost(outputTensorCPU));
}

//just for debugging
//might only print nonzero elements
template<typename T>
cudaError_t DefaultNet<T>::printInputAndOutput(){

    float* outputTensorCPU;
    float* inputTensorCPU;

    printCudaErrors(cudaMallocHost(&outputTensorCPU, this->outputLayer->getTotalTensorSize() * sizeof(float)));
    printCudaErrors(cudaMemcpy(outputTensorCPU, this->outputLayer->data, this->outputLayer->getTotalTensorSize()*sizeof(float), cudaMemcpyDeviceToHost));

    printCudaErrors(cudaMallocHost(&inputTensorCPU, this->inputLayer->getTotalTensorSize() * sizeof(float)));
    printCudaErrors(cudaMemcpy(inputTensorCPU, this->inputLayer->data, this->inputLayer->getTotalTensorSize()*sizeof(float), cudaMemcpyDeviceToHost));

    infoMsg("%d\n", this->inputLayer->getTotalTensorSize());

    int ia[5];

    printCudaErrors(cudaFreeHost(inputTensorCPU));
    printCudaErrors(cudaFreeHost(outputTensorCPU));

    exit(0);

    return cudaGetLastError();
}

extern "C" cudaError_t calculateDivergenceStaggeredWrapper_float(float* velocity, float* divergence, float cellDim, dim3 domainSize);

extern "C" cudaError_t printCudaBuffersWrapper_float(float* buffer, dim3 size, char* name);
extern "C" cudaError_t printCudaBuffersWrapper3D_float(float* buffer, dim3 size, char* name);

template<typename T>
void DefaultNet<T>::forward(bool doSubtractGradient, bool doComputeGradient, T dt){

    std::vector<Layer<T>*> currentLayers;
    std::vector<Layer<T>*> nextLayers;
    std::vector<Layer<T>*> mergePoints;
    currentLayers.push_back(this->inputLayer.get());

    bool okMergeFlag = false;

    printf("velSTD = %f\n\n", velSTD);

    if(doComputeGradient){
        //this will not have correct bounds, and neither will gradioent subtraction
        checkCudaErrors(calculateDivergenceStaggeredWrapper_float(this->velocity, this->divergence, 0.0,
                                                                  dim3(this->outputLayer->tensorDims[4], this->outputLayer->tensorDims[3], this->outputLayer->tensorDims[2])));
    }

    if(velSTD!=0 && !isnan(velSTD)){
        multiplyTensorByScalarWrapper_float(this->inputLayer->data,
                                            this->inputLayer->data, //twice so it's inplace
                                            inputLayerObstacleBufferSize*2,
                                            1.0f/velSTD);

        multiplyTensorByScalarWrapper_float(this->velocity,
                                            this->velocity,
                                            velocityBufferSize,
                                            1.0f/velSTD
                                            );}

    //this needs to be a graph parse, not a linear while
    while(currentLayers.size() != 0 || mergePoints.size() != 0){
        for(int i=0;i<currentLayers.size();i++){
            if(currentLayers[i]->type() == "MergeContinguousLayer3D" && !okMergeFlag){
                mergePoints.push_back(currentLayers[i]);
                infoMsg("ecksdee\n");}
            else{
                currentLayers[i]->forward();
                checkCudaErrors(cudaDeviceSynchronize());
                infoMsg("%s\n",currentLayers[i]->name.c_str());

                for(int j=0;j<currentLayers[i]->connectedOutputLayers.size();j++)
                    nextLayers.push_back(currentLayers[i]->connectedOutputLayers[j].get());
            }
        }

        okMergeFlag = false;

        if(nextLayers.size()==0 && mergePoints.size()!=0){
            currentLayers.clear();
            currentLayers.push_back(mergePoints[0]);
            mergePoints.erase(mergePoints.begin());
            okMergeFlag = true; //can parse the next merge layer, that has just been put in currentLayers
        }
        else{
            currentLayers.swap(nextLayers);
            nextLayers.clear();
        }
    }

    if(doSubtractGradient){
    subtractGradientStaggeredCuda_float(this->velocity, this->pressure, NULL, this->divergence,
                                        dim3(this->outputLayer->tensorDims[4]+1, this->outputLayer->tensorDims[3]+1, this->outputLayer->tensorDims[2]+1),
            CellSize, 0.0); //NULL corresponds to the density, which isn't used anyway
    }
    //this should only be done if we divided by velSTD in loadinput
    //this shouldn't be here, it should be after subtractDivergence, should more both this and the initial multiplyTensorByScalaraWrapper_float to Fluid3d.cpp
    if(velSTD!=0 && !isnan(velSTD)){
        multiplyTensorByScalarWrapper_float(this->outputLayer->data,
                                            this->outputLayer->data, //twice so it's inplace
                                            inputLayerObstacleBufferSize,
                                            velSTD);

        multiplyTensorByScalarWrapper_float(this->velocity,
                                            this->velocity,
                                            velocityBufferSize,
                                            velSTD);}

}

template<typename T>
void DefaultNet<T>::getOutput(cudaArray* pressureOut){

    copyDeviceArrayToCudaArray(pressureOut);
}

//if we want the output to not overwrite the input
template<typename T>
void DefaultNet<T>::getOutput(T*& pressureOut){

    pressureOut = this->outputLayer->data;
}

#endif //CNN_CPP
