#ifndef TOYCNN_H
#define TOYCNN_H

#include <defines.h>
#include <cudnn.h>
#include <vector>
#include <string>

//TODO: this might need a second version if you want the ToyCNN to still work
class ToyMaxPoolLayer3D
{
public:

  std::vector<int> size;
  std::vector<int> stride;
  ToyMaxPoolLayer3D(std::vector<int> size_, std::vector<int> stride_);
  ToyMaxPoolLayer3D();
};

class ToyCNN
{
public:
  void getTensorValuesFromFile(std::string fileName);

  cudnnHandle_t cudnnHandle;

  ToyMaxPoolLayer3D poolDown, poolUp, poolEqual;

  cudaChannelFormatDesc hiddenDesc;
  cudnnPoolingDescriptor_t poolDownDesc, poolUpDesc, poolEqualDesc;
  cudaArray* hidden;

  cudnnTensorDescriptor_t hiddenTensor, targetTensor, dataTensor, nothingPoolingTensor, nothingDataTensor;//,nothingTensor for dy if up-pooling doesn't work //poolUpTensor is the target

  dim3 inputSize, hiddenLayerSize;

  std::vector<float> dataToyHost;
  std::vector<float> hiddenToyHost;
  std::vector<float> targetToyHost;

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

class MaxPoolLayer
{
public:
  int size, stride;
  MaxPoolLayer(int size_, int stride_); //this  gives them an initial value when the constructor is called, even though they are const
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
