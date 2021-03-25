# VGG-TensorRT
A small project to demonstrate how to use TensorRT to reconstruct a deep learning model from scratch.
## To run project for inference.</br>
- First, make sure you are using a CUDA-supported GPU.
- Install [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) and [TensorRT](https://developer.nvidia.com/nvidia-tensorrt-download).
- Use this repo to generate weight of VGG model https://github.com/wang-xinyu/pytorchx/tree/master/vgg
- In the root of this project, create a folder named `Weights`. Then, put the generated weight file to the folder.
- Install [Cmake](https://cmake.org/install/).
- After those steps. Try to run the C++ file (main.cpp):

```
cmake
make
./vgg
```
