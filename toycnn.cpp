#include <toycnn.h>

///
/// \brief ToyCNN::ToyCNN - simple test CNN for debugging
/// \details Constructor. hiddenLayerSize should be half the size of divergenceSize, in all dimensions.
/// \param divergenceSize
/// \param hiddenLayerSize
///
ToyCNN::ToyCNN(dim3 inputSize_, dim3 hiddenLayerSize_){

  inputSize = inputSize_;
  hiddenLayerSize = hiddenLayerSize_;

  checkCUDNN(cudnnCreate(&cudnnHandle));

  std::vector<int> sizeVect = {2,2,2};
  std::vector<int> equalVect = {1,1,1};

  poolDown = ToyMaxPoolLayer3D(sizeVect, sizeVect);
  poolUp = ToyMaxPoolLayer3D(sizeVect, sizeVect);
  poolEqual = ToyMaxPoolLayer3D(equalVect, equalVect);

  int zeroVect[] = {0, 0, 0};

  //maybe these should be stored somewhere
  int hiddenTensorSize[] = {1,1,hiddenLayerSize.x,hiddenLayerSize.y,hiddenLayerSize.z};
  int hiddenTensorStride[] = {1,1,1,hiddenLayerSize.x,hiddenLayerSize.x*hiddenLayerSize.y};

  int dataTensorSize[] = {1,1,inputSize.x,inputSize.y,inputSize.z};
  int dataTensorStride[] = {1,1,1,inputSize.x,inputSize.y*inputSize.x};

  int targetTensorSize[] = {1,1,inputSize.x,inputSize.y,inputSize.z};
  int targetTensorStride[] = {1,1,1,inputSize.x,inputSize.y*inputSize.x};

  int nothingDataTensorSize[] = {1,1,inputSize.x,inputSize.y,inputSize.z};
  int nothingDataTensorStride[] = {1,1,1,inputSize.x,inputSize.y*inputSize.x};

  int nothingPoolingTensorSize[] = {1,1,hiddenLayerSize.x,hiddenLayerSize.y,hiddenLayerSize.z};
  int nothingPoolingTensorStride[] = {1,1,1,hiddenLayerSize.x,hiddenLayerSize.y*hiddenLayerSize.x};

  checkCUDNN(cudnnCreateTensorDescriptor(&dataTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&hiddenTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&targetTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&nothingDataTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&nothingPoolingTensor));

  checkCUDNN(cudnnCreatePoolingDescriptor(&poolDownDesc));
  checkCUDNN(cudnnCreatePoolingDescriptor(&poolUpDesc));
  checkCUDNN(cudnnCreatePoolingDescriptor(&poolEqualDesc));

  int* poolDownSize = &(poolDown.size)[0];
  int* poolDownStride = &(poolDown.stride)[0];

  int* poolEqualSize = &(poolEqual.size)[0];
  int* poolEqualStride = &(poolEqual.size)[0];

  checkCUDNN(cudnnSetTensorNdDescriptor(dataTensor,
                                        CUDNN_DATA_FLOAT,
                                        5,
                                        dataTensorSize,
                                        dataTensorStride));

  checkCUDNN(cudnnSetTensorNdDescriptor(hiddenTensor,
                                        CUDNN_DATA_FLOAT,
                                        5,
                                        hiddenTensorSize,
                                        hiddenTensorStride));

  checkCUDNN(cudnnSetTensorNdDescriptor(targetTensor,
                                        CUDNN_DATA_FLOAT,
                                        5,
                                        targetTensorSize,
                                        targetTensorStride));

  //this is used for the derivative associated with the data tensor, which is useless since the backwards pooling is used for upscaling
  checkCUDNN(cudnnSetTensorNdDescriptor(nothingDataTensor,
                                        CUDNN_DATA_FLOAT,
                                        5,
                                        nothingDataTensorSize,
                                        nothingDataTensorStride));

  checkCUDNN(cudnnSetTensorNdDescriptor(nothingPoolingTensor,
                                        CUDNN_DATA_FLOAT,
                                        5,
                                        nothingPoolingTensorSize,
                                        nothingPoolingTensorStride));

  checkCUDNN(cudnnSetPoolingNdDescriptor(poolDownDesc,
                                         CUDNN_POOLING_MAX,
                                         CUDNN_PROPAGATE_NAN,
                                         3, //dimension
                                         &(poolDown.size)[0],
                                         zeroVect,
                                         &(poolDown.stride)[0])); //the last 3 are int[] of length 3

  checkCUDNN(cudnnSetPoolingNdDescriptor(poolUpDesc,
                                         CUDNN_POOLING_MAX,
                                         CUDNN_PROPAGATE_NAN,
                                         3, //dimension
                                         &(poolUp.size)[0],
                                         zeroVect,
                                         &(poolUp.stride)[0])); //the last 3 are int[] of length 3

  //some testing thing that pools one to one
  checkCUDNN(cudnnSetPoolingNdDescriptor(poolEqualDesc,
                                         CUDNN_POOLING_MAX,
                                         CUDNN_PROPAGATE_NAN,
                                         3, //dimension
                                         &(poolEqual.size)[0],
                                         zeroVect,
                                         &(poolEqual.stride)[0])); //the last 3 are int[] of length 3

  dataToyHost = std::vector<float>(64*64*64);
  hiddenToyHost = std::vector<float>(32*32*32);
  targetToyHost = std::vector<float>(64*64*64);

  for(int i=0;i<64*64*64;i++)
    dataToyHost[i] = i;
}

extern "C" void
copyCudaArrayToDeviceArrayWrapper_float(cudaArray* src, float* dst);

///
/// \brief ToyCNN::ForwardPropagation
/// \details Runs through the CNN (the non-toy version of this will replace the Jacobi method)
/// \details
/// \details      poolDown         poolUp
/// \details data -------> hidden -------> target
/// \details
/// \param data
/// \param hidden
/// \param target
///
void ToyCNN::ForwardPropagation(cudaArray* data, cudaArray* target){

  if(cudaDeviceSynchronize()!=cudaSuccess){
    printf("FAIL texture printing  \n");
    return;}

  cudaChannelFormatDesc dataDesc;
  cudaExtent dataExtent;
  unsigned int dataFlags;

  cudaArrayGetInfo(&dataDesc, &dataExtent, &dataFlags, data);

  printf("data cudaArray: dataExtent.depth = %d, dataExtent.height = %d, dataExtent.width = %d\n", dataExtent.depth,dataExtent.height,dataExtent.width);

  float alpha = 1.0f, beta = 0.0f;

  float* dataToy;
  float* hiddenToy;
  float* targetToy;
  float* nothingDataToy;
  float* nothingPoolingToy;

  printf("sizeof(float) = %d", int(sizeof(float)));

  int totalInputSize = inputSize.x * inputSize.y * inputSize.z;
  int totalHiddenSize = hiddenLayerSize.x * hiddenLayerSize.y * hiddenLayerSize.z;

  checkCudaErrors(cudaMalloc(&dataToy,  sizeof(float) * dataExtent.width * dataExtent.height * dataExtent.depth));
  checkCudaErrors(cudaMalloc(&hiddenToy,sizeof(float) * hiddenToyHost.size()));
  checkCudaErrors(cudaMalloc(&targetToy,sizeof(float) * targetToyHost.size()));
  checkCudaErrors(cudaMalloc(&nothingDataToy,sizeof(float) * dataToyHost.size()));
  checkCudaErrors(cudaMalloc(&nothingPoolingToy,sizeof(float) * hiddenToyHost.size()));

  infoMsg("totalInputSize %d", totalInputSize);

#define W 10
#define H 10

  copyCudaArrayToDeviceArrayWrapper_float(data, dataToy);

  if(cudaDeviceSynchronize()!=cudaSuccess){
    printf("FAIL copyCudaArrayToDeviceArrayWrapper_float \n");
    return;}

  checkCudaErrors(cudaMemcpy(hiddenToy, &hiddenToyHost[0], hiddenToyHost.size() * sizeof(float), cudaMemcpyHostToDevice));

  if(cudaDeviceSynchronize()!=cudaSuccess){
    printf("FAIL cudaMemcpy \n");
    return;}

  //toy testing attempt
  checkCUDNN(cudnnPoolingForward(cudnnHandle, poolDownDesc, &alpha, dataTensor, dataToy, &beta, hiddenTensor, hiddenToy));

  cudaError_t cudaErr = cudaDeviceSynchronize();

  if(cudaErr!=cudaSuccess){
    printf("FAIL -%s \n",cudaGetErrorString(cudaErr));
    return;}

  if(cudaDeviceSynchronize()!=cudaSuccess){
    printf("FAIL\n");
    return;}

  printf("hiddenToy:\n\n");

  cudaDeviceSynchronize();
}

ToyCNN::~ToyCNN(){

  checkCUDNN(cudnnDestroy(cudnnHandle));

  checkCUDNN(cudnnDestroyTensorDescriptor(dataTensor));
  checkCUDNN(cudnnDestroyTensorDescriptor(hiddenTensor));
  checkCUDNN(cudnnDestroyTensorDescriptor(targetTensor));

  checkCUDNN(cudnnDestroyPoolingDescriptor(poolDownDesc));
  checkCUDNN(cudnnDestroyPoolingDescriptor(poolUpDesc));
  checkCUDNN(cudnnDestroyPoolingDescriptor(poolEqualDesc));

  //delete the hidden layer
  checkCudaErrors(cudaFree(hidden));

}

MaxPoolLayer::MaxPoolLayer(int size_, int stride_) : size(size_), stride(stride_) {}


ToyMaxPoolLayer3D::ToyMaxPoolLayer3D(){

  size = std::vector<int>(3);
  stride = std::vector<int>(3);

}

ToyMaxPoolLayer3D::ToyMaxPoolLayer3D(std::vector<int> size_, std::vector<int> stride_){

  size = size_;
  stride = stride_;

}
