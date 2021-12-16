Attempt at implementing a low-resolution real-time 3D fluid simulator with a machine learning-based projection step, using OpenGL-CUDA interop and CUDNN. Code is not in working order, and the technique did not work when using the same neural network weights as the offline 3D fluid simulator it was based on.

Based on:
- the offline machine learning-based 3D fluid simulator from Tompson et al. 2016 (from [here](https://github.com/google/FluidNet))
- a pure OpenGL fluid simulator by Philip Rideout (from [here](https://github.com/prideout/fluidsim))

Uses CUDNN 5, may need the cudnn folder (containing include/ and lib64/) copied or symlinked into this folder, or CUDA_DIR set.

main() is in pez.linux.c
Also see Fluid3d.cpp and cnn.cpp

Tested on Centos 7, Windows operation would likely require significant modifications.

My academic poster on this technique can be seen [here](https://drive.google.com/file/d/1bMUHaDJsNCUS4h3JtZ-cVSRxLWOaHjGU/view?usp=sharing).