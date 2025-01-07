// nvcc main.cu matrix_ops.cu -o main.exe -ccbin "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.42.34433\bin\Hostx64\x64"
// I was using cuda 11.8
#include "matrix_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <iomanip>
#include <iostream>

#define N 1500
#define P 1500

using namespace std;
using namespace std::chrono;

int main() {
    int nBytes = N * P * sizeof(float);

    // Allocate CPU memory
    float *M1, *M2, *Mout, *Mout2;
    M1 = (float*)malloc(nBytes);
    M2 = (float*)malloc(nBytes);
    Mout = (float*)malloc(nBytes);
    Mout2 = (float*)malloc(nBytes);

    // Result table variables
    float cpuAddTime, cpuMultTime, gpuAddTime, gpuMultTime;

    // Initialize matrices
    MatrixInit(M1, N, P);
    MatrixInit(M2, N, P);

    cout << "Starting Matrix Operations...\n";

    // CPU Matrix Addition
    auto cpuAddStart = high_resolution_clock::now();
    MatrixAdd(M1, M2, Mout, N, P);
    auto cpuAddEnd = high_resolution_clock::now();
    cpuAddTime = duration<float>(cpuAddEnd - cpuAddStart).count();
    cout << "CPU Matrix Addition Completed: " << cpuAddTime << " seconds\n";

    // CPU Matrix Multiplication
    auto cpuMultStart = high_resolution_clock::now();
    MatrixMult(M1, M2, Mout, N);
    auto cpuMultEnd = high_resolution_clock::now();
    cpuMultTime = duration<float>(cpuMultEnd - cpuMultStart).count();
    cout << "CPU Matrix Multiplication Completed: " << cpuMultTime << " seconds\n";

    // Allocate GPU memory
    float *d_M1, *d_M2, *d_Mout;
    cudaMalloc((void**)&d_M1, nBytes);
    cudaMalloc((void**)&d_M2, nBytes);
    cudaMalloc((void**)&d_Mout, nBytes);

    // Copy data to GPU
    cudaMemcpy(d_M1, M1, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2, M2, nBytes, cudaMemcpyHostToDevice);

    // Set grid and block sizes
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (P + block.y - 1) / block.y);

    // GPU Matrix Addition
    auto gpuAddStart = high_resolution_clock::now();
    cudaMatrixAdd<<<grid, block>>>(d_M1, d_M2, d_Mout, N, P);
    cudaDeviceSynchronize();
    auto gpuAddEnd = high_resolution_clock::now();
    gpuAddTime = duration<float>(gpuAddEnd - gpuAddStart).count();
    cout << "GPU Matrix Addition Completed: " << gpuAddTime << " seconds\n";

    cudaMemcpy(Mout2, d_Mout, nBytes, cudaMemcpyDeviceToHost);

    // GPU Matrix Multiplication
    auto gpuMultStart = high_resolution_clock::now();
    cudaMatrixMult<<<grid, block>>>(d_M1, d_M2, d_Mout, N);
    cudaDeviceSynchronize();
    auto gpuMultEnd = high_resolution_clock::now();
    gpuMultTime = duration<float>(gpuMultEnd - gpuMultStart).count();
    cout << "GPU Matrix Multiplication Completed: " << gpuMultTime << " seconds\n";

    cudaMemcpy(Mout2, d_Mout, nBytes, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_M1);
    cudaFree(d_M2);
    cudaFree(d_Mout);
    free(M1);
    free(M2);
    free(Mout);
    free(Mout2);

    // Print summary table
    cout << "\n---------------------------\n";
    cout << "   Matrix Operation Times   \n";
    cout << "---------------------------\n";
    cout << left << setw(30) << "Operation" << setw(15) << "Time (s)\n";
    cout << "---------------------------\n";
    cout << left << setw(30) << "CPU Matrix Addition" << setw(15) << cpuAddTime << "\n";
    cout << left << setw(30) << "CPU Matrix Multiplication" << setw(15) << cpuMultTime << "\n";
    cout << left << setw(30) << "GPU Matrix Addition" << setw(15) << gpuAddTime << "\n";
    cout << left << setw(30) << "GPU Matrix Multiplication" << setw(15) << gpuMultTime << "\n";
    cout << "---------------------------\n";

    system("pause");
    return 0;
}
