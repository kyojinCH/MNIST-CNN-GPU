#ifndef CNN_LAYERS_H
#define CNN_LAYERS_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/**
 * @brief Prints a string with a background color corresponding to the given intensity.
 * @param str The string to print.
 * @param rgb The grayscale intensity (between 0 and 1) for the background color.
 */
void charBckgrndPrint(char* str, float rgb);

/**
 * @brief Prints a colored image representation using background colors for values.
 * @param height The height of the image.
 * @param width The width of the image.
 * @param img Pointer to the 2D image data stored as a 1D array.
 */
void imgColorPrint(int height, int width, float* img);

/**
 * @brief Initializes a 2D matrix with random values between -1 and 1.
 * @param M Pointer to the 2D matrix stored as a 1D array.
 * @param n The number of rows.
 * @param p The number of columns.
 */
void Matrix2DInitRand(float* M, int n, int p);

/**
 * @brief Initializes a 3D matrix with zeros.
 * @param M Pointer to the 3D matrix stored as a 1D array.
 * @param f The number of feature maps (depth).
 * @param n The number of rows.
 * @param p The number of columns.
 */
void Matrix3DInitZero(float* M, int f, int n, int p);

/**
 * @brief Initializes a 3D matrix with random values between -1 and 1.
 * @param M Pointer to the 3D matrix stored as a 1D array.
 * @param f The number of feature maps (depth).
 * @param n The number of rows.
 * @param p The number of columns.
 */
void Matrix3DInitRand(float* M, int f, int n, int p);

/**
 * @brief Prints a 2D matrix to the console.
 * @param M Pointer to the 2D matrix stored as a 1D array.
 * @param n The number of rows.
 * @param p The number of columns.
 */
void Matrix2DPrint(float* M, int n, int p);

/**
 * @brief Performs 2D convolution on the input matrix using the provided kernels.
 * @param M Pointer to the input matrix (stored as a 1D array).
 * @param kernel Pointer to the convolution kernels.
 * @param Mout Pointer to the output matrix.
 * @param M_lines Number of rows in the input matrix.
 * @param M_cols Number of columns in the input matrix.
 * @param kernel_size Size of the convolution kernels.
 * @param nb_kernel Number of convolution kernels (depth).
 * @param Mout_lines Number of rows in the output matrix.
 * @param Mout_cols Number of columns in the output matrix.
 */
__global__ void cudaConv2D(float* M, float* kernel, float* Mout, int M_lines, int M_cols, 
                           int kernel_size, int nb_kernel, int Mout_lines, int Mout_cols);

/**
 * @brief Performs 2D mean pooling (subsampling) with a specified pooling size.
 * @param M Pointer to the input matrix (stored as a 1D array).
 * @param Mout Pointer to the output matrix.
 * @param M_lines Number of rows in the input matrix.
 * @param M_cols Number of columns in the input matrix.
 * @param M_prof Depth of the input matrix.
 * @param meanpool_size Size of the pooling window (e.g., 2x2).
 * @param Mout_lines Number of rows in the output matrix.
 * @param Mout_cols Number of columns in the output matrix.
 */
__global__ void cudaMeanPool(float* M, float* Mout, int M_lines, int M_cols, int M_prof, 
                             int meanpool_size, int Mout_lines, int Mout_cols);

/**
 * @brief Applies the tanh activation function to the input matrix.
 * @param M Pointer to the input matrix (stored as a 1D array).
 * @param M_lines Number of rows in the input matrix.
 * @param M_cols Number of columns in the input matrix.
 * @param M_prof Depth of the input matrix.
 * @param Mout Pointer to the output matrix to store the activated values.
 */
__global__ void activation_tanh(float* M, int M_lines, int M_cols, int M_prof, float* Mout);

#endif // CNN_LAYERS_H
