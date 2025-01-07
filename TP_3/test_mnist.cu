#include "test_mnist.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <ctime>
#include <limits>
#include <cstdlib>

// -------------- CPU Helpers --------------

std::vector<double> loadTxtDoubles(const std::string& filename)
{
    std::vector<double> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        exit(1);
    }
    double val;
    while (file >> val) {
        data.push_back(val);
    }
    file.close();
    return data;
}

void waitForEnter(const char* message)
{
    std::cout << "\n" << message << std::endl;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}

void charBckgrndPrint(const char* str, int rgb[3])
{
    printf("\033[48;2;%d;%d;%dm", rgb[0], rgb[1], rgb[2]);
    printf("%s\033[0m", str);
}

void imgColorPrint(int height, int width, int** img)
{
    // We'll treat each pixel intensity as grayscale, replicate to (R=G=B).
    // If you truly wanted [height][width][3], you'd store that differently.
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int g = img[i][j]; // [0..255]
            int rgb[3] = {g, g, g};
            charBckgrndPrint("  ", rgb);
        }
        printf("\n");
    }
}

// Initialize 3D = 6,28,28 with zero
void Matrix3DInitZero(double* M, int fm, int n, int p)
{
    for (int i = 0; i < fm*n*p; i++) {
        M[i] = 0.0;
    }
}

// -------------- CUDA Kernels --------------

// Single-channel "same" conv (for first conv layer)
__global__ void cudaConv2D_same(double* in_data, double* kernel, double* out_data,
                                int in_rows, int in_cols,
                                int kernel_size, int nb_filters,
                                int out_rows, int out_cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // output col
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // output row

    if(row < out_rows && col < out_cols) {
        double sumVal = 0.0;
        int half_k = kernel_size/2;
        for(int kr=0; kr<kernel_size; kr++){
            for(int kc=0; kc<kernel_size; kc++){
                int in_r = row + (kr - half_k);
                int in_c = col + (kc - half_k);
                if(in_r >= 0 && in_r < in_rows && in_c >= 0 && in_c < in_cols){
                    // single input channel => just index directly
                    double px = in_data[in_r * in_cols + in_c];
                    double w  = kernel[kr*kernel_size + kc]; // up to 6 filters if extended
                    sumVal += px*w;
                }
            }
        }
        // we do up to 6 filters if we do multi-k approach, but let's keep it single
        // for c1, we have 1 in-channel × 6 out-ch. So typically you either do:
        // out_ch in [0..5], do sumVal, but here you do 1 pass per out_ch?
        // We'll assume nb_filters=6 isn't used. Or you run a loop inside?
        out_data[row*out_cols + col] = sumVal;
    }
}

// 2) Multi-channel valid conv (for second conv)
__global__ void cudaConv2D_valid_multiChan(const double* __restrict__ in_data,
                                           const double* __restrict__ kernel,
                                           double* __restrict__ out_data,
                                           int in_rows, int in_cols,
                                           int kernel_size,
                                           int in_channels,
                                           int out_channels,
                                           int out_rows, int out_cols)
{
    // map each thread to (out_ch, out_row, out_col)
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    int out_ch  = blockIdx.z * blockDim.z + threadIdx.z;

    if ((out_ch < out_channels) && (out_row < out_rows) && (out_col < out_cols)) {
        double sum = 0.0;

        // sum across 6 input channels
        for(int c=0; c<in_channels; c++){
            for(int kr=0; kr<kernel_size; kr++){
                for(int kc=0; kc<kernel_size; kc++){
                    int in_r = out_row + kr; // valid => no offset
                    int in_c = out_col + kc;
                    double val_in = in_data[ c*(in_rows*in_cols)
                                           + in_r*in_cols
                                           + in_c ];
                    double val_k  = kernel[ out_ch*(in_channels*kernel_size*kernel_size)
                                          + c*(kernel_size*kernel_size)
                                          + kr*kernel_size
                                          + kc ];
                    sum += val_in*val_k;
                }
            }
        }
        out_data[out_ch*(out_rows*out_cols) + out_row*out_cols + out_col] = sum;
    }
}

// MeanPool
__global__ void cudaMeanPool(double* in_data, double* out_data,
                             int in_rows, int in_cols, int in_channels,
                             int pool_size, int out_rows, int out_cols)
{
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    if(out_row < out_rows && out_col < out_cols){
        for(int ch=0; ch<in_channels; ch++){
            double accum=0.0;
            for(int r=0; r<pool_size; r++){
                for(int c=0; c<pool_size; c++){
                    int in_r = out_row*pool_size + r;
                    int in_c = out_col*pool_size + c;
                    accum += in_data[ch*(in_rows*in_cols) + in_r*in_cols + in_c];
                }
            }
            accum /= (double)(pool_size*pool_size);
            out_data[ch*(out_rows*out_cols) + out_row*out_cols + out_col] = accum;
        }
    }
}

// Tanh
__global__ void activation_tanh(double* M, int M_ligne, int M_colonne, int M_prof, double* Mout)
{
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    if(row < M_ligne && col < M_colonne){
        int planeSize = M_ligne*M_colonne;
        for(int ch=0; ch<M_prof; ch++){
            int idx = ch*planeSize + row*M_colonne + col;
            Mout[idx] = tanh(M[idx]);
        }
    }
}

// Dense
__global__ void Dense(double* A, double* in_vec, double* out_vec,
                      int in_dim, int out_dim)
{
    int out_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(out_idx < out_dim){
        double sum=0.0;
        for(int i=0; i<in_dim; i++){
            sum += A[out_idx*in_dim + i]*in_vec[i];
        }
        out_vec[out_idx] = sum;
    }
}

// Bias 3D
__global__ void addBias3D(double* out, const double* bias,
                          int out_channels, int out_height, int out_width)
{
    int c   = blockIdx.z;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    if(c<out_channels && row<out_height && col<out_width){
        int idx = c*(out_height*out_width) + row*out_width + col;
        out[idx] += bias[c];
    }
}

// Bias 1D
__global__ void addBias1D(double* out, const double* bias, int length)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<length){
        out[i] += bias[i];
    }
}
