#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Function declarations

// CPU Functions

/**
 * @brief Initializes a matrix with random values between -1 and 1.
 * 
 * @param M Pointer to the matrix to be initialized.
 * @param n Number of rows in the matrix.
 * @param p Number of columns in the matrix.
 */
void MatrixInit(float* M, int n, int p);

/**
 * @brief Prints the elements of a matrix to the console.
 * 
 * @param M Pointer to the matrix to be printed.
 * @param n Number of rows in the matrix.
 * @param p Number of columns in the matrix.
 */
void MatrixPrint(float* M, int n, int p);

/**
 * @brief Performs element-wise addition of two matrices on the CPU.
 * 
 * @param M1 Pointer to the first input matrix.
 * @param M2 Pointer to the second input matrix.
 * @param Mout Pointer to the output matrix to store the result.
 * @param n Number of rows in the matrices.
 * @param p Number of columns in the matrices.
 */
void MatrixAdd(float* M1, float* M2, float* Mout, int n, int p);

/**
 * @brief Performs matrix multiplication of two square matrices (NxN) on the CPU.
 * 
 * @param M1 Pointer to the first input matrix.
 * @param M2 Pointer to the second input matrix.
 * @param Mout Pointer to the output matrix to store the result.
 * @param n Dimension (rows and columns) of the square matrices.
 */
void MatrixMult(float* M1, float* M2, float* Mout, int n);


// GPU Functions

/**
 * @brief Performs element-wise addition of two matrices on the GPU.
 * 
 * @param M1 Pointer to the first input matrix on the device.
 * @param M2 Pointer to the second input matrix on the device.
 * @param Mout Pointer to the output matrix on the device to store the result.
 * @param n Number of columns in the matrices.
 * @param p Number of rows in the matrices.
 */
__global__ void cudaMatrixAdd(float* M1, float* M2, float* Mout, int n, int p);

/**
 * @brief Performs matrix multiplication of two square matrices (NxN) on the GPU.
 * 
 * @param M1 Pointer to the first input matrix on the device.
 * @param M2 Pointer to the second input matrix on the device.
 * @param Mout Pointer to the output matrix on the device to store the result.
 * @param n Dimension (rows and columns) of the square matrices.
 */
__global__ void cudaMatrixMult(float* M1, float* M2, float* Mout, int n);

#endif // MATRIX_OPS_H
