//Modified from a CUDNN LeNet example at https://github.com/tbennun/cudnn-training

#ifndef CNN_H
#define CNN_H

#include <cudnn.h>
#include <cublas_v2.h>
#include <vector>
#include <memory>
#include <unordered_map>
#include <typeinfo>

struct nop
{
    template <typename T>
    void operator() (T const &) const noexcept { }
};

template <typename T>
using nop_unique_ptr = std::unique_ptr<T, nop>;

template<typename T>
class Net;

template<typename T>
class Layer;

template<typename T>
class MergeContinguousTensorsLayer3D;

template<typename T>
class Layer //this will be the base class for all tensors in the network
{
public:
  std::vector<std::unique_ptr<Layer<T>>> connectedOutputLayers;
  std::vector<Layer<T>*> connectedOutputLayersRaw;
  Layer<T>* connectedInputLayer; //there should be only one input layer

  std::string name;

  std::vector<int> tensorDims;
  std::vector<int> tensorStrides;

  Layer();
  ~Layer();

  virtual void forward(){printf("Virtual forward method called.\n");}

  cudnnTensorDescriptor_t tensorDesc;
  //void *data;
  T* data = NULL;

  friend class Net<T>;

  size_t getTotalTensorSize();

  cudnnDataType_t getDataType();

  cudnnHandle_t* getLayerCudnnHandle();

  void allocateTensor(T *tensorData = NULL);

  ///
  /// \brief Gets the output tensor dimensions, using the input tensor dimensions and the operation (may use cudnn get outputdims functions)
  ///
  virtual void calculateTensorDims(){}

  ///
  /// \brief Some layers (for now just the convlayer and convbiaslayer) need to initialize stuff after the output tensor size is calculated
  ///
  virtual void postPrep(){}

  void setStride();

  void setDataType();

  void concatenateTensorWith(Layer<T> *otherLayer);

  virtual std::string type(){return this->layerType;}

  std::vector<Layer<T>*>* concatWithPointer = NULL;

  friend class MergeContinguousTensorsLayer3D<T>;

  T getDataAtIndex(int n, int c, int d, int h, int w);
  T getDataAtIndex(int indexes[5]);

  void print();

private:
  std::string layerType = "Layer";
  cudnnHandle_t* layerCudnnHandle;
  cudnnDataType_t dataType;
  size_t totalTensorSize = 0;
  std::vector<Layer<T>*> concatWith; //this will have size 0 for all but one vector
};

template<typename T>
class InputLayer3D : public Layer<T>{
public:
  void forward();
  void calculateTensorDims();

  void destroy();

  InputLayer3D(int dims[5]);
  ~InputLayer3D();

  virtual std::string type(){return this->layerType;}

private:
  std::string layerType = "InputLayer3D";
};

template<typename T>
class MergeContinguousTensorsLayer3D : public Layer<T>{
public:
    void forward(); //does nothing
    void calculateTensorDims(); //adds the channel dimensions of all the input layers (could just do 8*numBanks if we know each convlayer has 8 feature maps)
    void postPrep();

    MergeContinguousTensorsLayer3D(); //gets its input group when addLayer is called
    ~MergeContinguousTensorsLayer3D();

    virtual std::string type(){return this->layerType;}

private:
    std::string layerType = "MergeContinguousLayer3D";
};

///
/// \brief The class containing the neural network
///
template<typename T>
class Net
{
public:
  Layer<T>* outputLayer; //last layer, that gives the end result
  std::unique_ptr<Layer<T>> inputLayer; //first layer, that takes the initial data

  Net();

  Layer<T>* addLayer(std::unique_ptr<Layer<T>> layerToConnect);
  Layer<T>* addLayer(std::unique_ptr<Layer<T>> layerToConnect, Layer<T> *layerToConnectTo);

  //this is mainly for the convbiaslayer
  Layer<T>* addLayer(Net<T>* netToConnect, Layer<T> *layerToConnectTo); //-> no longer useful

  cudnnDataType_t getDataType();
  cudnnHandle_t getCudnnHandle();

  std::string name;

  void setDataType();

  void allocateTensors(T* allocateInput = NULL, T* allocateOutput = NULL);

private:
  cudnnHandle_t cudnnHandle;
  cudnnDataType_t dataType; //needs to match T
};

template<typename T>
class DefaultNet: public Net<T>{
public:
  DefaultNet(std::string fileName, bool isBinary, int inputSize[] = NULL, int numBanks = 1, int bankLength = 2);

  void makeNet(int inputDims[] = NULL, int numBanks = 1, int bankLength = 2);

  void loadInput(cudaArray* lastPressureArray, cudaArray* obstacleArray, cudaArray* divergenceArray, cudaArray* pressureArray, cudaArray* velocityPingArray, cudaArray *velocityPongArray);//, cudnnDataType_t dataType = CUDNN_DATA_FLOAT);
  void loadInput(T *data, T *target);
  void loadInput(cudaArray* densityArray);
  void loadInput(T* input, T* pressureBuffer, T* velocityBuf, dim3 domainSize, dim3 velocitySize);

  cudaArray_t obstacleArray;

  T* velocityBuffer;
  T* velocityBuffer2;

  size_t velocityBufferSize;

  T velSTD;

  std::vector<std::vector<int>> convTensorDims;
  std::vector<std::vector<int>> biasTensorDims;
  std::vector<std::vector<float>> filtersData;
  std::vector<std::vector<float>> biasesData;

  void getTensorValuesFromFile(std::string fileName);

  bool firstIteration; //set to False in constructor

  int numIteration = 0;

  size_t inputLayerVelDivergenceOffset;
  size_t inputLayerObstacleOffset;
  size_t inputLayerObstacleBufferSize;

  //forward pass of entire network
  void forward(bool doSubtractGradient = false, bool doComputeGradient = false, T dt = 0.25f);

  void getOutput(cudaArray* pressureOut);
  void getOutput(T *&pressureOut);

  int powInt(int x, int power);

  cudaError_t copyCudaArrayToDeviceArrayDummy(cudaArray* lastPressureArray,
                                       cudaArray* velDivergenceArray,
                                       cudaArray* obstacleArray,
                                       size_t inputLayerObstacleOffset,
                                       size_t inputLayerVelDivergenceOffset);

  T *obstacles, *velocity, *divergence, *pressure;

  cudaError_t updateObstaclesDNWrapper(dim3 obstacleExtent);
  cudaError_t setWallsWrapper(T *obstacles, dim3 domainExtent, size_t thickness);
  cudaError_t copyCudaArrayToDeviceArrayZeroObstacles(cudaArray *lastPressureArray, cudaArray *velDivergenceArray, cudaArray *obstacleArray, size_t inputLayerObstacleOffset, size_t inputLayerVelDivergenceOffset);
  cudaError_t copyCudaArrayToDeviceArray(cudaArray *lastPressureArray, cudaArray *velDivergenceArray, cudaArray *obstacleArray, size_t inputLayerObstacleOffset, size_t inputLayerVelDivergenceOffset);
  cudaError_t copyCudaArrayToDeviceArray(cudaArray *array, T *buffer);
  cudaError_t copyDeviceArrayToCudaArray(cudaArray *pressureArray);
  cudaError_t printSingleElementWrapper(float *deviceArray, size_t numSizes, int *sizes, int *elementToGet);
  cudaError_t printInputAndOutput();
};

template<typename T>
class PoolingLayer3D : public Layer<T>{
public:
  int dims[3];

  T alpha = 1.0f;
  T beta = 0.0f;

  cudnnPoolingDescriptor_t poolingDesc;

  PoolingLayer3D(int _dims[3]);

  void calculateTensorDims();

  void forward();

  virtual std::string type(){return this->layerType;}

private:
  std::string layerType = "PoolingLayer3D";

protected:
  PoolingLayer3D();
};

template<typename T>
class AvgPoolLayer3D : public PoolingLayer3D<T>{
public:

  AvgPoolLayer3D(int _dims[3]);

  virtual std::string type(){return this->layerType;}

private:
  std::string layerType = "AvgPoolingLayer3D";
};

///
/// \brief Layer for adding two 3D tensors or the same size
///
template<typename T>
class AddLayer3D : public Layer<T>{
public:
  Layer<T>* otherInputLayer; //the tensor to add to inputLayer (see Layer parent class)
  void forward();

  float alpha, beta;
  cudnnTensorDescriptor_t tensorDesc;
  cudnnConvolutionDescriptor_t convDesc;

  AddLayer3D(Layer<T>* otherInputLayer);
  ~AddLayer3D();

  void calculateTensorDims();

  virtual std::string type(){return this->layerType;}

private:
  std::string layerType = "AddLayer3D";
};

template<typename T>
class UpscaleLayer3D : public Layer<T>{
public:
  //THE UPSCALING IS SET IN cudnnSetConvolutionDescriptor

  //if you end up making kernel versions, a version with int upsampling factors,
  //and one with input and output sizes (with generally non-int upsampling factors) might be best
  UpscaleLayer3D(std::vector<int> _upscaleFactor);
  ~UpscaleLayer3D();

  void forward();

  std::vector<int> upscaleFactor;

  void calculateTensorDims();

  virtual std::string type(){return this->layerType;}

  cudaError_t upscaleLayer3DKernelWrapper(T *input, T *output, int *outputDims, int *upscale);
private:
  std::string layerType = "UpscaleLayer3D";
  float alpha;
  cudnnFilterDescriptor_t weightDesc;
  T* weights;
  cudnnConvolutionDescriptor_t convDesc;
  T* convData;
  cudnnTensorDescriptor_t convTensorDesc;
  cudnnConvolutionFwdAlgo_t algorithm;
  T* workspace;
  size_t workspaceSizeInBytes;
  float beta;
};

template<typename T>
///
/// \brief Scales tensor by a scalar
///
class ScalingLayer3D: public Layer<T>{

public:
  //THIS IS FOR MULTIPLICATION BY A SCALAR, you use it when you divide by the standard deviation

  T scaleFactor;

  ScalingLayer3D(T _scaleFactor);
  ScalingLayer3D();

  void forward();

  void calculateTensorDims();

  //there may be a CUDNN alternative to this
  cudaError_t multiplyTensorByScalarWrapper(T *tensor, T *tensorOut, size_t tensorSize, T scalar);

  virtual std::string type(){return this->layerType;}
private:
  std::string layerType = "ScalingLayer3D";
};


template<typename T>
///
/// \brief Parent of ScaleByNormLayer3D and ScaleByInverseNormLayer3D. Helper class to avoid function duplication; instances can only be created by its children
///
class NormScalingLayer3D: public ScalingLayer3D<T>{

public:
  //this is stored in the scaleFactor of the ScalingLayer
  /*
  ///
  /// \brief The norm will be stored in this variable
  ///
  T norm;*/

  ///
  /// \brief In a DefaultNet, we want to scale back up using the norm of the DefaultNet input, before the first ScaleByInverseNormLayer3D layer.
  /// \brief This norm is stored in the ScaleByInverseNormLayer3D, and normPointer points to that value.
  ///
  T* normPointer;

  ///
  /// \brief Returns the norm of the input tensor. (???)
  /// \return the norm
  ///
  T getNorm();
/*
  ///
  /// \brief this points to either
  ///
  void (*reduceSummationWrapper)();
*/

  void reduceSummationWrapper(T* tensor, T* tensorOut, size_t tensorSize, size_t tensorOutSize);
  void reduceSummationVarianceWrapper(T *tensor, T* tensorOut, size_t tensorSize, size_t tensorOutSize, T average);

  virtual std::string type(){return this->layerType;}

private:
  std::string layerType = "NormScalingLayer3D";

protected:
  NormScalingLayer3D();
};

template<typename T>
///
/// \brief Scales tensor by the inverse of its standard deviation
///
class ScaleByInverseNormLayer3D: public NormScalingLayer3D<T>{

public:

  void forward();

  virtual std::string type(){return this->layerType;}

private:
  std::string layerType = "ScaleByInverseNormLayer3D";
};

template<typename T>
///
/// \brief Scales tensor by the inverse of its standard deviation
///
class ScaleByNormLayer3D: public NormScalingLayer3D<T>{

public:
  void forward();

  virtual std::string type(){return this->layerType;}

private:
  std::string layerType = "ScaleByNormLayer3D";
};

template<typename T>
class ConvBiasLayer3D: public Layer<T>{
  //Each dimension of the bias tensor A must match the corresponding dimension of the destination tensor C or must be equal to 1.
public:

  void makeNet();
  void forward();

  void * convData; //again, should this be T*?
  void * biasData;

  T alpha = 1;
  T alpha2 = 0; //might not be needed
  T beta = 0;

  cudnnTensorDescriptor_t convTensorDesc;
  cudnnConvolutionDescriptor_t convDesc;
  cudnnFilterDescriptor_t filterDesc;

  cudnnConvolutionFwdAlgo_t algorithm;

  std::vector<int> filterDims;
  std::vector<int> filterStrides;

  std::vector<int> biasDims;
  std::vector<int> biasStrides;

  ConvBiasLayer3D(std::vector<int> filterSize, std::vector<T> filterData, std::vector<int> biasSize, std::vector<T> biasDataCPU, std::string activationMode = "ReLU", double _coef = 0.0);

  void* workspace;
  size_t workspaceSizeInBytes;

  cudnnTensorDescriptor_t biasTensorDesc; //otherTensorDesc

  cudnnActivationDescriptor_t activationDescriptor;
  static std::unordered_map<std::string, cudnnActivationMode_t> activationModeMap;
  double coef;

  //this corresponds to the z tensor in cudnnConvolutionBiasActivationForward, which is not used
  cudnnTensorDescriptor_t uselessDesc;
  void* uselessData;

  void calculateTensorDims();
  void postPrep();

  cudnnActivationMode_t getActivationMode();

  virtual std::string type(){return this->layerType;}

private:
  std::string layerType = "ConvBiasLayer3D";

private:
  cudnnActivationMode_t activationMode;
};



#endif // CNN_H
