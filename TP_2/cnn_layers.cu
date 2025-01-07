#include "cnn_layers.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Function to print a character with a background color
void charBckgrndPrint(const char* str, float rgb) {
    printf("\033[48;2;%d;%d;%dm", (int)(rgb * 255), (int)(rgb * 255), (int)(rgb * 255));
    printf("%s\033[0m", str);
}

// Function to print an image with colors
void imgColorPrint(int height, int width, float* img) {
    int row, col;
    const char* str = "  ";
    for (row = 0; row < height; row++) {
        for (col = 0; col < width; col++) {
            charBckgrndPrint(str, img[row * width + col]);
        }
        printf("\n");
    }
}

// Function to initialize a 2D matrix with random values
void Matrix2DInitRand(float* M, int n, int p) {
    srand(1);
    for (int i = 0; i < n * p; i++) {
        M[i] = float(2.0 * rand() / RAND_MAX - 1);
    }
}

// Function to initialize a 3D matrix with zeros
void Matrix3DInitZero(float* M, int f, int n, int p) {
    for (int i = 0; i < n * p * f; i++) {
        M[i] = 0;
    }
}

// Function to initialize a 3D matrix with random values
void Matrix3DInitRand(float* M, int f, int n, int p) {
    srand(1);
    for (int i = 0; i < n * p * f; i++) {
        M[i] = float(2.0 * rand() / RAND_MAX - 1);
    }
}

// Function to print a 2D matrix
void Matrix2DPrint(float* M, int n, int p) {
    printf("Matrix shape is: (%d, %d)\n", n, p);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            // Print each value with fixed width and precision
            printf("%7.3f ", M[i * p + j]);
        }
        printf("\n"); // New line after each row
    }
    printf("\n"); // Extra line for better readability
}

// CUDA Kernel for 2D Convolution
__global__ void cudaConv2D(float* M, float* kernel, float* Mout, int M_lines, int M_cols, 
                           int kernel_size, int nb_kernel, int Mout_lines, int Mout_cols) {
    int lig = blockIdx.x;
    int col = threadIdx.x;

    float s = 0.0;
    if (lig < Mout_lines && col < Mout_cols) {
        int tot = M_lines * M_cols;
        for (int kernel_lig = 0; kernel_lig < kernel_size; kernel_lig++) {
            for (int kernel_col = 0; kernel_col < kernel_size; kernel_col++) {
                for (int n_k = 0; n_k < nb_kernel; n_k++) {
                    s += M[(lig + kernel_lig) * M_cols + col + kernel_col + n_k * tot] 
                         * kernel[kernel_lig * kernel_size + kernel_col + n_k * kernel_size * kernel_size];
                }
            }
        }
        Mout[lig * Mout_cols + col] = s;
    }
}

// CUDA Kernel for Mean Pooling
__global__ void cudaMeanPool(float* M, float* Mout, int M_lines, int M_cols, int M_prof, 
                             int meanpool_size, int Mout_lines, int Mout_cols) {
    int lig = 2 * blockIdx.x;
    int col = 2 * threadIdx.x;

    float s;
    int tot_meanpool = meanpool_size * meanpool_size;
    int tot_M = M_lines * M_cols;
    int tot_Mout = Mout_lines * Mout_cols;

    for (int n_prof = 0; n_prof < M_prof; n_prof++) {
        s = 0.0;
        for (int i = 0; i < meanpool_size; i++) {
            for (int j = 0; j < meanpool_size; j++) {
                s += M[(lig + i) * M_cols + col + j + n_prof * tot_M] / tot_meanpool;
            }
        }
        Mout[blockIdx.x * Mout_cols + threadIdx.x + n_prof * tot_Mout] = s;
    }
}

// CUDA Kernel for Tanh Activation Function
__global__ void activation_tanh(float* M, int M_lines, int M_cols, int M_prof, float* Mout) {
    int lig = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (lig < M_lines && col < M_cols) {
        int index = lig * M_cols + col;
        Mout[index] = tanh(M[index]);
    }
}
