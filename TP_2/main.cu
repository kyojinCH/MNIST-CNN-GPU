#include "cnn_layers.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    // Input Data Initialization
    float* raw_data = (float*)malloc(sizeof(float) * 32 * 32);
    float* C1_data = (float*)malloc(sizeof(float) * 6 * 28 * 28);
    float* S1_data = (float*)malloc(sizeof(float) * 6 * 14 * 14);
    float* C1_kernel = (float*)malloc(sizeof(float) * 6 * 5 * 5);

    Matrix2DInitRand(raw_data, 32, 32);
    Matrix3DInitZero(C1_data, 6, 28, 28);
    Matrix3DInitRand(C1_kernel, 6, 5, 5);

    float *d_raw, *d_C1, *d_kernel, *d_S1;
    cudaMalloc(&d_raw, sizeof(float) * 32 * 32);
    cudaMalloc(&d_C1, sizeof(float) * 6 * 28 * 28);
    cudaMalloc(&d_kernel, sizeof(float) * 6 * 5 * 5);
    cudaMalloc(&d_S1, sizeof(float) * 6 * 14 * 14);

    cudaMemcpy(d_raw, raw_data, sizeof(float) * 32 * 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, C1_kernel, sizeof(float) * 6 * 5 * 5, cudaMemcpyHostToDevice);

    cudaConv2D<<<28, 28>>>(d_raw, d_kernel, d_C1, 32, 32, 5, 6, 28, 28);
    cudaMeanPool<<<14, 14>>>(d_C1, d_S1, 28, 28, 6, 2, 14, 14);

    cudaMemcpy(S1_data, d_S1, sizeof(float) * 6 * 14 * 14, cudaMemcpyDeviceToHost);
    printf("Mean Pooling Output:\n");
    Matrix2DPrint(S1_data, 14, 14);

    cudaFree(d_raw);
    cudaFree(d_C1);
    cudaFree(d_kernel);
    cudaFree(d_S1);
    free(raw_data);
    free(C1_data);
    free(C1_kernel);
    free(S1_data);
    system("pause");
    return 0;
}

// nvcc main.cu cnn_layers.cu -o .\outputs\main -ccbin "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.42.34433\bin\Hostx64\x64"