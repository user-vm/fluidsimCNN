#ifndef TOYCNN_H
#define TOYCNN_H

#include <defines.h>
#include <cudnn.h>
#include <vector>
#include <string>

//this might need a second version if you want the ToyCNN to still work
class ToyMaxPoolLayer3D
{
public:
  //int size[3] = {0,0,0}, stride[3] = {0,0,0};

  std::vector<int> size;
  std::vector<int> stride;
  ToyMaxPoolLayer3D(std::vector<int> size_, std::vector<int> stride_);
  ToyMaxPoolLayer3D();

  //MaxPoolLayer3D(int size_[], int stride_[]); //this crap gives them an initial value when the constructor is called, even though they are const
  //MaxPoolLayer3D();
};

class ToyCNN
{
public:
  void getTensorValuesFromFile(std::string fileName);

  cudnnHandle_t cudnnHandle;
  //cublasHandle_t cublasHandle;

  ToyMaxPoolLayer3D poolDown, poolUp, poolEqual;

  cudaChannelFormatDesc hiddenDesc;
  cudnnPoolingDescriptor_t poolDownDesc, poolUpDesc, poolEqualDesc;
  cudaArray* hidden;

  cudnnTensorDescriptor_t hiddenTensor, targetTensor, dataTensor, nothingPoolingTensor, nothingDataTensor;//,nothingTensor for dy if up-pooling doesn't work //poolUpTensor is the target

  dim3 inputSize, hiddenLayerSize;

  //float *dataToy, *hiddenToy; //this is to check whether float* works; cudaArray might not

  //std::unique_ptr<float*> dataToyHost;
  //std::unique_ptr<float*> hiddenToyHost;

  std::vector<float> dataToyHost;
  std::vector<float> hiddenToyHost;
  std::vector<float> targetToyHost;

  //std::vector<float> targetToyTensor;

  const dim3 dataToyHostSize = dim3(64,64,64);
  const dim3 hiddenToyHostSize = dim3(32,32,32);
  const dim3 targetToyHostSize = dim3(64,64,64);

  ///
  /// \brief ToyCNN constructor
  /// \details Downsamples by max pooling two to one, then upsamples without interpolation
  /// \details Can use forward pooling for downsampling (cudnnPoolingForward) and backwards pooling for downsampling (cudnnPoolingBackward)
  /// \param poolDown
  /// \param poolUp
  ///
  ToyCNN(dim3 inputSize_, dim3 hiddenSize_);

  ~ToyCNN();

  void ForwardPropagation(cudaArray *data, cudaArray *target);
};
/*
class CNN
{
public:
  cudnnHandle_t cudnnHandle;
  cublasHandle_t cublasHandle;

  cudnnTensorDescriptor_t dataTensor, conv1Tensor, conv1BiasTensor, pool1Tensor,
      conv2Tensor, conv2BiasTensor, pool2Tensor, fc1Tensor, fc2Tensor;
  cudnnFilterDescriptor_t conv1filterDesc, conv2filterDesc;
  cudnnConvolutionDescriptor_t conv1Desc, conv2Desc;
  cudnnConvolutionFwdAlgo_t conv1algo, conv2algo;
  cudnnConvolutionBwdFilterAlgo_t conv1bwfalgo, conv2bwfalgo;
  cudnnConvolutionBwdDataAlgo_t conv2bwdalgo;
  cudnnPoolingDescriptor_t poolDesc;
  cudnnActivationDescriptor_t fc1Activation;

  int m_gpuid;
  int m_batchSize;
  size_t m_workspaceSize;

  FullyConnectedLayer& ref_fc1, &ref_fc2;

  // Disable copying
  CNN& operator=(const CNN&) = delete;
  CNN(const CNN&) = delete;

  CNN();
  ~CNN();

};
*/

class MaxPoolLayer
{
public:
  int size, stride;
  MaxPoolLayer(int size_, int stride_); //this crap gives them an initial value when the constructor is called, even though they are const
};

class ConvBiasLayer
{
public:
  int in_channels, out_channels, kernel_size;
  int in_width, in_height, out_width, out_height;

  std::vector<float> pconv, pbias;

  ///
  /// \brief ConvBiasLayer
  /// \details Constructor
  /// \param in_channels_
  /// \param out_channels_
  /// \param kernel_size_
  /// \param in_w_
  /// \param in_h_
  ///
  ConvBiasLayer(int in_channels_, int out_channels_, int kernel_size_,
                int in_w_, int in_h_);

  ///
  /// \brief LoadConvLayerFromFile
  /// \details Read the model from the file somehow
  /// \param fileprefix
  /// \return
  ///
  bool LoadConvLayerFromFile(const char *fileprefix);

};

#endif // TOYCNN_H
