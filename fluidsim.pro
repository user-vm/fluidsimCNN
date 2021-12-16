TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

QMAKE_CC = gcc

QMAKE_CFLAGS += -std=c99
QMAKE_CFLAGS += -std=c++11

isEmpty(CUDA_DIR){
    CUDA_DIR = /usr/local/cuda
}

SOURCES += bstrlib.c Fluid3d.cpp pez.c pez.linux.c Utility.cpp \ #Fluid3d.cu
    cnn.cpp \
    toycnn.cpp
HEADERS += bstrlib.h gl3.h pez.h Utility.h vmath.hpp \
    cnn.h \
    defines.h \
    toycnn.h
OTHER_FILES += Fluid.glsl Light.glsl Raycast.glsl Jacobi_original.glsl README.md Fluid_not_staggered.glsl Raycast_original.glsl

CUDA_SOURCES += Fluid3d.cu cnn.cu# <-- same dir for this small example

#isEmpty(GL_INCLUDE_PATH){
#    GL_INCLUDE_PATH = /usr/include/GL
#}

SYSTEM_TYPE = 64            # '32' or '64', depending on your system
CUDA_ARCH = sm_35           # (tested with sm_30 on my comp) Type of CUDA architecture, for example 'compute_10', 'compute_11', 'sm_10'
NVCC_OPTIONS = --use_fast_math


# include paths
INCLUDEPATH += $$CUDA_DIR/include inc #cudnn/include #only use the inc folder AFTER removing the GL folder with ancient headers

# set TORCH_DIR to the torch install directory
#isEmpty(TORCH_DIR){
#    INCLUDEPATH += /home/vmiu/torch/install/include #this is thpp (include as thpp/xxx.h)
#}
#else {
#    INCLUDEPATH += TORCH_DIR/install/include #or maybe this is thpp
#}

# library directories
QMAKE_LIBDIR += $$CUDA_DIR/lib64/
QMAKE_LIBDIR += /usr/lib64/nvidia
#QMAKE_LIBDIR += cudnn/lib64 #this may need to be moved to the other cuda stuff ($$CUDA_DIR/lib64)

CUDA_OBJECTS_DIR = ./

# Add the necessary libraries
CUDA_LIBS = -lGL -lGLU -lX11 -lcudart -lGLEW -lcudnn -lcublas #-lglut # <-- changed this # -lglut?

# The following makes sure all path names (which often include spaces) are put between quotation marks
CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')
#LIBS += $$join(CUDA_LIBS,'.so ', '', '.so') <-- didn't need this
LIBS += $$CUDA_LIBS # <-- needed this


# SPECIFY THE R PATH FOR NVCC (this caused me a lot of trouble before)
QMAKE_LFLAGS += -Wl,-rpath,$$CUDA_DIR/lib # <-- added this
NVCCFLAGS = -Xlinker -rpath,$$CUDA_DIR/lib # <-- and this <--THIS IS PROBABLY ONLY FOR OSX, CHECK THE MARCHING CUBES MAKEFILE

# Configuration of the Cuda compiler
CONFIG(debug, debug|release) {
    # Debug mode
    #DESTDIR = build/debug
    cuda_d.input = CUDA_SOURCES
    cuda_d.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
    #remove CUDA_VISIBLE_DEVICES if using a signel GPU, for both debug and release
    cuda_d.commands = CUDA_VISIBLE_DEVICES=1 $$CUDA_DIR/bin/nvcc -D_DEBUG $$NVCC_OPTIONS $$CUDA_INC $$NVCC_LIBS --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda_d.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda_d
}
else{
    # Release mode
    #DESTDIR = build/release
    cuda.input = CUDA_SOURCES
    cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
    #remove CUDA_VISIBLE_DEVICES if using a single GPU, for both debug and release
    cuda.commands = CUDA_VISIBLE_DEVICES=1 $$CUDA_DIR/bin/nvcc $$NVCC_OPTIONS $$CUDA_INC $$NVCC_LIBS --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda
}

DISTFILES += \
    out_torch_order.bin
