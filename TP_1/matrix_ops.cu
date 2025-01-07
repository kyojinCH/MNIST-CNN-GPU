#include "matrix_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Créer une matrice sur CPU
void MatrixInit(float* M, int n, int p) {
    for (int i = 0; i < n * p; i++) {
        M[i] = float(2.0 * rand() / RAND_MAX - 1);
    }
}

// Afficher une matrice sur CPU
void MatrixPrint(float* M, int n, int p) {
    printf("Matrix shape is: (%d, %d)\n", n, p);
    for (int iy = 0; iy < p; iy++) {
        for (int ix = 0; ix < n; ix++) {
            printf("%f  ", M[iy * n + ix]);
        }
        printf("\n");
    }
}

// Addition de deux matrices sur CPU
void MatrixAdd(float* M1, float* M2, float* Mout, int n, int p) {
    for (int iy = 0; iy < p; iy++) {
        for (int ix = 0; ix < n; ix++) {
            Mout[iy * n + ix] = M1[iy * n + ix] + M2[iy * n + ix];
        }
    }
}

// Multiplication de deux matrices NxN sur CPU
void MatrixMult(float* M1, float* M2, float* Mout, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += M1[i * n + k] * M2[k * n + j];
            }
            Mout[i * n + j] = sum;
        }
}

// Addition de deux matrices sur GPU
__global__ void cudaMatrixAdd(float* M1, float* M2, float* Mout, int n, int p) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = iy * n + ix;
    if (ix < n && iy < p) {
        Mout[idx] = M1[idx] + M2[idx];
    }
}

// Multiplication de deux matrices NxN sur GPU
__global__ void cudaMatrixMult(float* M1, float* M2, float* Mout, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    float sum = 0.0f;
    for (int k = 0; k < n; k++) {
        sum += M1[j * n + k] * M2[k * n + i];
    }
    if (i < n && j < n) {
        Mout[j * n + i] = sum;
    }
}