#ifndef TEST_MNIST_H
#define TEST_MNIST_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <limits>
#include <fstream>
#include <vector>
#include <string>

/**
 * @file test_mnist.h
 * @brief Header declaring CPU-side helper functions and CUDA kernels used for an MNIST-like test/inference.
 *
 * This file provides function declarations for:
 * - Loading data (weights) from text files
 * - Printing images in pseudo-color
 * - Initializing and zeroing arrays
 * - CUDA kernels for convolution, mean-pooling, activation, dense, and bias-add
 */

/*---------------------------------------------------------------
  Constants
---------------------------------------------------------------*/
#define WIDTH  28   ///< The width dimension of the MNIST image (28).
#define HEIGHT 28   ///< The height dimension of the MNIST image (28).

const std::string MNIST_PATH = "C:\\Users\\Mon PC\\DataspellProjects\\ENSEA\\3rd_year\\Hardware for Signal Processing\\train-images.idx3-ubyte";
const std::string WEIGHTS_FOLDER_PATH = "C:\\Users\\Mon PC\\DataspellProjects\\ENSEA\\3rd_year\\Hardware for Signal Processing\\exported_weights";
/*---------------------------------------------------------------
  CPU-side Helper Functions
---------------------------------------------------------------*/

/**
 * @brief Loads all double-precision values from a text file, which can be spaced or line-separated.
 * @param filename A string containing the path to the text file holding numeric values.
 * @return A std::vector<double> with all values read from the file.
 */
std::vector<double> loadTxtDoubles(const std::string& filename);

/**
 * @brief Pauses program execution until the user presses [Enter].
 * @param message A message displayed to the console before waiting.
 */
void waitForEnter(const char* message = "Press Enter to continue...");

/**
 * @brief Prints a 2-character-wide string with a background color determined by the given RGB array.
 * @param str The string to be printed (usually 2 spaces).
 * @param rgb An integer array [3] specifying the (R, G, B) values, each in [0..255].
 *
 * This function uses ANSI escape codes to set the background color, prints `str`,
 * then resets the terminal color.
 */
void charBckgrndPrint(const char* str, int rgb[3]);

/**
 * @brief Prints a single-channel image in pseudo-color by replicating each intensity into an RGB background.
 * @param height The number of rows in the image (e.g. 28).
 * @param width  The number of columns in the image (e.g. 28).
 * @param img    A 2D array [height][width], where each pixel is an integer [0..255].
 *
 * This function calls `charBckgrndPrint(...)` for each pixel, effectively coloring
 * the console background to match the intensity.
 */
void imgColorPrint(int height, int width, int** img);

/**
 * @brief Initializes a 3D (feature-map) matrix with zeros.
 * @param M   Pointer to a 1D array storing 3D data in row-major fashion.
 * @param fm  The number of feature maps (depth).
 * @param n   The number of rows in each feature map.
 * @param p   The number of columns in each feature map.
 *
 * This sets all `fm * n * p` entries to 0.0.
 */
void Matrix3DInitZero(double* M, int fm, int n, int p);

/*---------------------------------------------------------------
  CUDA Kernels
---------------------------------------------------------------*/

/**
 * @brief Performs a single-channel 2D convolution with "same" padding.
 * @param in_data     Pointer to input data array of shape [in_rows * in_cols].
 * @param kernel      Pointer to kernel array of shape [kernel_size * kernel_size].
 * @param out_data    Pointer to output data array of shape [out_rows * out_cols].
 * @param in_rows     The number of rows in the input image.
 * @param in_cols     The number of columns in the input image.
 * @param kernel_size The dimension of the (square) kernel (e.g. 5 for a 5x5).
 * @param nb_filters  The number of filters (unused here if single in-channel).
 * @param out_rows    The number of rows in the output image (same as input if "same" padding).
 * @param out_cols    The number of columns in the output image (same as input if "same" padding).
 *
 * Each thread typically corresponds to one output pixel (row, col), and the kernel
 * accumulates values from the corresponding neighborhood in the input, applying
 * zero-padding if the neighborhood extends beyond input boundaries.
 */
__global__ void cudaConv2D_same(double* in_data, double* kernel, double* out_data,
                                int in_rows, int in_cols,
                                int kernel_size, int nb_filters,
                                int out_rows, int out_cols);

/**
 * @brief Performs a multi-channel 2D convolution with "valid" padding for the second layer.
 * @param in_data      Pointer to input data of shape [in_channels * in_rows * in_cols].
 * @param kernel       Pointer to kernel array of shape [out_channels, in_channels, kernel_size, kernel_size].
 * @param out_data     Pointer to the output array of shape [out_channels * out_rows * out_cols].
 * @param in_rows      The number of rows in each input channel (e.g. 14).
 * @param in_cols      The number of columns in each input channel (e.g. 14).
 * @param kernel_size  The dimension of the square kernel (e.g. 5).
 * @param in_channels  The number of input channels (e.g. 6).
 * @param out_channels The number of output channels/filters (e.g. 16).
 * @param out_rows     The output's row dimension (e.g. 10 if valid).
 * @param out_cols     The output's column dimension (e.g. 10 if valid).
 *
 * Each thread corresponds to one (out_channel, out_row, out_col). The kernel loops
 * over all input channels to accumulate the final output for that thread's position.
 * "Valid" padding means no outside boundaries are used (so output is smaller).
 */
__global__ void cudaConv2D_valid_multiChan(const double* __restrict__ in_data,
                                           const double* __restrict__ kernel,
                                           double* __restrict__ out_data,
                                           int in_rows, int in_cols,
                                           int kernel_size,
                                           int in_channels,
                                           int out_channels,
                                           int out_rows, int out_cols);

/**
 * @brief Performs mean-pooling (2D subsampling) with a specified window size (e.g. 2x2).
 * @param in_data    Pointer to input data [in_channels * in_rows * in_cols].
 * @param out_data   Pointer to output data [in_channels * out_rows * out_cols].
 * @param in_rows    Number of rows in each input channel.
 * @param in_cols    Number of columns in each input channel.
 * @param in_channels How many feature maps (channels) are in the input.
 * @param pool_size  The size of the pooling window in each dimension (e.g. 2).
 * @param out_rows   The number of rows in the output (in_rows / pool_size).
 * @param out_cols   The number of columns in the output (in_cols / pool_size).
 *
 * Each thread typically computes one (channel, out_row, out_col) by averaging over
 * the corresponding pool_size x pool_size region in the input.
 */
__global__ void cudaMeanPool(double* in_data, double* out_data,
                             int in_rows, int in_cols, int in_channels,
                             int pool_size, int out_rows, int out_cols);

/**
 * @brief Applies the tanh activation function to a 3D tensor.
 * @param M       Pointer to the input data [M_prof * M_ligne * M_colonne].
 * @param M_ligne Number of rows per feature map.
 * @param M_colonne Number of columns per feature map.
 * @param M_prof  Number of feature maps (channels).
 * @param Mout    Pointer to the output data (same shape) to store tanh(M).
 *
 * Each thread updates one element of (channel, row, col) by applying tanh to M.
 */
__global__ void activation_tanh(double* M, int M_ligne, int M_colonne,
                                int M_prof, double* Mout);

/**
 * @brief Dense layer kernel (matrix-vector multiply) in double precision.
 * @param A      Pointer to the weight matrix of shape [out_dim x in_dim].
 * @param in_vec Pointer to the input vector (length in_dim).
 * @param out_vec Pointer to the output vector (length out_dim).
 * @param in_dim  Size of the input vector.
 * @param out_dim Size of the output vector (number of rows in A).
 *
 * Each thread typically computes one output index out_idx by dot-producting `A[out_idx, :]`
 * with `in_vec`.
 */
__global__ void Dense(double* A, double* in_vec, double* out_vec,
                      int in_dim, int out_dim);

/**
 * @brief Adds a bias array to a 3D tensor [out_channels x out_height x out_width].
 * @param out          Pointer to the 3D output array in row-major shape.
 * @param bias         Pointer to the bias array, length = out_channels.
 * @param out_channels Number of feature maps (channels) in the output.
 * @param out_height   The height dimension of each feature map.
 * @param out_width    The width dimension of each feature map.
 *
 * Each thread updates `out[channel, row, col] += bias[channel]`.
 */
__global__ void addBias3D(double* out, const double* bias,
                          int out_channels, int out_height, int out_width);

/**
 * @brief Adds a bias array to a 1D dense output vector.
 * @param out    Pointer to the output vector (length = `length`).
 * @param bias   Pointer to the bias array, also length = `length`.
 * @param length The dimension of out/bias arrays.
 *
 * Each thread does `out[i] += bias[i]` for its assigned index i.
 */
__global__ void addBias1D(double* out, const double* bias, int length);

#endif // TEST_MNIST_H
