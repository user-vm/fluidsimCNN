#ifndef DEFINES_H
#define DEFINES_H

#include<cudnn.h>
#include<sstream>
#include <helper_cuda.h>
#include <iostream>

//should probably put these in a shared header somewhere
#define MAXTEXTURESIZE_X 128
#define MAXTEXTURESIZE_Y 128
#define MAXTEXTURESIZE_Z 128

//maximum texture size is 128x128x128
#define BLOCKSIZE_X 32
#define BLOCKSIZE_Y 32
#define BLOCKSIZE_Z 2

#define THREADSPERBLOCK_X 4
#define THREADSPERBLOCK_Y 4
#define THREADSPERBLOCK_Z 64

#define BLOCK_SIZE_DIVISION 1024
#define BLOCK_SIZE 1024

#define USE_CNN 1
#define USE_CNN_CUDA_ADVECTION 0 //1

#define useJacobiCuda 0

//change nothing between here

#if defined USE_CNN
#if USE_CNN==0
#define USE_TOY 1
#else
#define USE_TOY 0
#endif
#else
#define USE_TOY 1
#define USE_CNN 0
#endif

//and here

//taken from https://github.com/tbennun/cudnn-training/blob/master/lenet.cu
#define checkCUDNN(status) do {                                        \
    std::stringstream _error;                                          \
    std::string errorStr;                                              \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      std::cout << "CUDNN failure: " << cudnnGetErrorString(status)<< " in file "<<__FILE__<<" on line "<<__LINE__<<std::endl;      \
      exit(1); \
    }                                                                  \
} while(0)

#define VERBOSE false
#define PRINTWARNINGS true

#define infoMsg(...) if(VERBOSE) printf(__VA_ARGS__)
#define warnMsg(...) if(PRINTWARNINGS) printf(__VA_ARGS__)
#define errorMsg(...) printf(__VA_ARGS__)

template< typename T >
T checkPrint(T result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    }

    return result;
}

#define printCudaErrors(val)           checkPrint ( (val), #val, __FILE__, __LINE__ )

#define VELOCITY_MIN -1e+6f
#define VELOCITY_MAX 1e+6f

#endif // DEFINES_H
