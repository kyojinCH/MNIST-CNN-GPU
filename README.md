# MNIST-CNN-GPU
A hands-on project showcasing a manually implemented CNN inference pipeline for the MNIST dataset using CUDA. Includes custom kernels for convolution, pooling, and dense layers, focusing on GPU acceleration and memory management without high-level frameworks. Ideal for learning CUDA and CNN fundamentals.
This repository follows the structure of a lab assignment aimed at implementing LeNet-5 inference from scratch using CUDA. It includes tasks such as:
- Part 1: Matrix operations (multiplication, addition) to practice CUDA kernel development.
- Part 2: Implementing the initial convolution and subsampling layers.
- Part 3: Training the model in Python using a provided notebook and exporting weights for CUDA.

---
# LeNet-5 MNIST Inference (C++/CUDA) (PART 3à

This repository demonstrates an **inference-only** implementation of a LeNet-5 style convolutional neural network for MNIST digit classification, all in **C++** with **CUDA** kernels. The network architecture consists of two convolution layers (C1 & C3), each followed by 2×2 mean-pooling (S2 & S4), then three fully-connected layers. Final predictions are produced via CPU softmax.

---

## Table of Contents

1. [Overview](#overview)  
2. [Project Structure](#project-structure)  
3. [Network Architecture](#network-architecture)  
4. [How It Works](#how-it-works)  
   - [Reading MNIST Files](#reading-mnist-files)  
   - [Loading Weights](#loading-weights)  
   - [Inference Pipeline](#inference-pipeline)
5. [Memory & Performance Considerations](#memory--performance-considerations)  
   - [GPU Memory Layout](#gpu-memory-layout)  
   - [Multi-channel Convolution](#multi-channel-convolution)  
6. [Building and Running](#building-and-running)  


---

## Overview

- **Objective**: Perform forward-pass inference for MNIST digits (28×28) using a simplified LeNet-5 CNN architecture.  
- **Implementation**:  
  - **C++** for host code.  
  - **CUDA** kernels for convolution, pooling, dense, etc.  
  - Final softmax on the CPU.  
- **Weights** come from `.txt` files (exported by Keras/TensorFlow).  
- The second convolution accumulates across **6** input channels, consistent with LeNet-5.

---

## Project Structure

From the provided screenshot, an approximate layout might be:

```bash
YourRepo/ 
├── exported_weights/   # .txt weight/bias files for each layer 
  ├── layer_0_0.txt 
  ├── layer_0_1.txt 
  ├── layer_2_0.txt 
  ├── layer_2_1.txt 
  ├── layer_5_0.txt 
  ├── layer_5_1.txt 
  ├── layer_6_0.txt 
  ├── layer_6_1.txt 
  ├── layer_7_0.txt 
  └── layer_7_1.txt
├── outputs/             # for executables
  ├── mnist_infer.exe
  ├── mnist_infer.exp
  └── mnist_infer.lib
├── LeNet5.ipynb         # Notebook for weight extraction or experimentation
├── main.cu              # Entry point for the inference pipeline
├── test_mnist.cu        # Implements CUDA kernels & utility code 
├── test_mnist.h         # Declarations of kernels & CPU helpers 
└── train-images.rar     # Compressed MNIST images
```

**Key Files**:

- **`main.cu`**: Reads an MNIST image, calls CUDA kernels in order, does final classification.  
- **`test_mnist.cu`** and **`test_mnist.h`**: House the CUDA kernel definitions and CPU utility functions.  
- **`exported_weights/`**: Directory containing `.txt` weight & bias files for each layer.  

---

## Network Architecture

A typical LeNet-5 style network:
```bash
[28 x 28, single-channel] 
   -> (C1: 6 filters, 5x5, same padding) -> [28 x 28 x 6] -> addBias + tanh 
   -> (S2: 2x2 mean-pool) -> [14 x 14 x 6] 
   -> (C3: 16 filters, 5x5, valid, multi-channel) -> [10 x 10 x 16] -> addBias + tanh 
   -> (S4: 2x2 mean-pool) -> [5 x 5 x 16]
   -> Flatten -> Dense(120) -> bias -> tanh -> Dense(84) -> bias -> tanh -> Dense(10) 
   -> bias -> CPU softmax -> final digit.
```


---

## How It Works

### Reading MNIST Files

- **`train-images.idx3-ubyte`** has a 16-byte header plus the image data.  
- We parse the header, then skip to `(16 + chosen_index*(28*28))` to load a single 28×28 image.  
- We store the image in a 2D array, scale `[0..255]` → `[0..1]`.

### Loading Weights

- Each layer’s kernel/bias is in a `.txt` file. For instance:
  - `layer_2_0.txt` = 2400 values = 6×16×5×5.  
- The code loads them with `loadTxtDoubles(...)` into a `std::vector<double>`, then copies to GPU with `cudaMemcpy`.

### Inference Pipeline

1. Convert the single image to floats in `[0..1]`.  
2. **C1**: single-channel “same” conv → `(6,28,28)` → addBias3D → tanh.  
3. **S2**: 2×2 mean-pool → `(6,14,14)`.  
4. **C3**: multi-channel “valid” conv → `(16,10,10)` → addBias3D → tanh.  
5. **S4**: 2×2 pool → `(16,5,5)`.  
6. Flatten → Dense(120) → Dense(84) → Dense(10) → CPU softmax → predicted digit.

---

## Memory & Performance Considerations

### GPU Memory Layout

- Each feature map is flattened in row-major: `channel*(rows*cols) + row*cols + col`.  
- E.g., S2 output `[6,14,14] = 1176` doubles.

### Multi-channel Convolution

- `cudaConv2D_valid_multiChan(...)` sums over `in_channels=6`.  
- Indexing must ensure we multiply each channel’s patch with the corresponding 5×5 slice in the weight array.



---

## Building and Running

1. **Clone** the repo and place `train-images.idx3-ubyte` plus `.txt` weight files in the same directory structure.  
2. **Compile** with `nvcc`:
   ```bash
   nvcc main.cu test_mnist.cu -o mnist_infer
   ```
3. **Run** with:
   ```bash
   ./mnist_infer 'digit_image_index'
   ```

   
PS: When running the executable, do press enter to allow the code to execute until the next step.
