#include "test_mnist.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <limits>
#include <vector>
#include <string>
#include <errno.h>

int main(int argc, char* argv[])
{
    int chosen_index = 0;
    if (argc > 1) {
        chosen_index = atoi(argv[1]);
    }
    std::cout << "Chosen index: " << chosen_index << "\n";

    waitForEnter("STEP 1: Reading MNIST image...");

    // Allocate single-channel 2D
    int** img = (int**)malloc(HEIGHT*sizeof(int*));
    for(int i=0; i<HEIGHT; i++){
        img[i] = (int*)malloc(WIDTH*sizeof(int));
    }

    // Open MNIST file
    FILE* fptr = nullptr;
    errno_t err = fopen_s(&fptr, MNIST_PATH.c_str(), "rb");
    if(err != 0){
        std::cerr<<"Cannot open MNIST file. Error code="<<err<<"\n";
        return 1;
    }

    unsigned int magic, nbImg, nbRows, nbCols;
    fread(&magic,  sizeof(int), 1, fptr);
    fread(&nbImg,  sizeof(int), 1, fptr);
    fread(&nbRows, sizeof(int), 1, fptr);
    fread(&nbCols, sizeof(int), 1, fptr);

    std::cout<<"Nb Magic : "<<magic<<"\n";
    std::cout<<"Nb Img   : "<<nbImg<<"\n";
    std::cout<<"Nb Rows  : "<<nbRows<<"\n";
    std::cout<<"Nb Cols  : "<<nbCols<<"\n\n";

    // Seek to chosen image
    fseek(fptr, 16 + chosen_index*(WIDTH*HEIGHT), SEEK_SET);

    // Read single-channel intensities
    for(int i=0; i<HEIGHT; i++){
        for(int j=0; j<WIDTH; j++){
            unsigned char val;
            fread(&val, sizeof(unsigned char), 1, fptr);
            img[i][j] = (int)val;
        }
    }
    fclose(fptr);

    // Optional color print
    imgColorPrint(HEIGHT, WIDTH, img);

    waitForEnter("STEP 1 complete.");

    // STEP 2: Load Weights
    waitForEnter("STEP 2: Loading CNN & Dense layer weights/bias...");
    // c1 => [1,6,5,5]=150
    std::vector<double> c1_kernel = loadTxtDoubles(WEIGHTS_FOLDER_PATH + "\\layer_0_0.txt");
    std::vector<double> c1_bias   = loadTxtDoubles(WEIGHTS_FOLDER_PATH + "\\layer_0_1.txt");

    // c3 => [6,16,5,5]=2400
    std::vector<double> c3_kernel = loadTxtDoubles(WEIGHTS_FOLDER_PATH + "\\layer_2_0.txt");
    std::vector<double> c3_bias   = loadTxtDoubles(WEIGHTS_FOLDER_PATH + "\\layer_2_1.txt");

    // Dense(120), Dense(84), Dense(10)
    std::vector<double> d1_kernel = loadTxtDoubles(WEIGHTS_FOLDER_PATH + "\\layer_5_0.txt");
    std::vector<double> d1_bias   = loadTxtDoubles(WEIGHTS_FOLDER_PATH + "\\layer_5_1.txt");

    std::vector<double> d2_kernel = loadTxtDoubles(WEIGHTS_FOLDER_PATH + "\\layer_6_0.txt");
    std::vector<double> d2_bias   = loadTxtDoubles(WEIGHTS_FOLDER_PATH + "\\layer_6_1.txt");

    std::vector<double> d3_kernel = loadTxtDoubles(WEIGHTS_FOLDER_PATH + "\\layer_7_0.txt");
    std::vector<double> d3_bias   = loadTxtDoubles(WEIGHTS_FOLDER_PATH + "\\layer_7_1.txt");

    waitForEnter("All weights loaded. Press Enter to proceed...");

    // STEP 3: Prepare arrays
    double* raw_data = (double*)malloc(sizeof(double)*28*28);
    // scale [0..255]->[0..1]
    for(int i=0; i<HEIGHT; i++){
        for(int j=0; j<WIDTH; j++){
            raw_data[i*WIDTH + j] = (double)img[i][j]/255.0;
        }
    }

    // Buffers
    double* C1_out = (double*)malloc(sizeof(double)*6*28*28);
    Matrix3DInitZero(C1_out, 6, 28, 28);

    double* S2_out = (double*)malloc(sizeof(double)*6*14*14);
    Matrix3DInitZero(S2_out, 6, 14, 14);

    double* C3_out = (double*)malloc(sizeof(double)*16*10*10);
    Matrix3DInitZero(C3_out, 16, 10, 10);

    double* S4_out = (double*)malloc(sizeof(double)*16*5*5);
    Matrix3DInitZero(S4_out, 16, 5, 5);

    double* D1_out = (double*)malloc(sizeof(double)*120);
    double* D2_out = (double*)malloc(sizeof(double)*84);
    double* D3_out = (double*)malloc(sizeof(double)*10);

    waitForEnter("STEP 3 done. Press Enter to proceed...");

    // STEP 4: GPU & Copy
    double *d_raw_data=nullptr,
           *d_C1_kernel=nullptr, *d_C1_out=nullptr, *d_C1_bias=nullptr, *d_S2_out=nullptr,
           *d_C3_kernel=nullptr, *d_C3_out=nullptr, *d_C3_bias=nullptr, *d_S4_out=nullptr,
           *d_d1_kernel=nullptr, *d_d1_bias=nullptr, *d_D1_out=nullptr,
           *d_d2_kernel=nullptr, *d_d2_bias=nullptr, *d_D2_out=nullptr,
           *d_d3_kernel=nullptr, *d_d3_bias=nullptr, *d_D3_out=nullptr;

    // Allocate
    cudaMalloc(&d_raw_data,  28*28*sizeof(double));
    cudaMalloc(&d_C1_kernel, c1_kernel.size()*sizeof(double));
    cudaMalloc(&d_C1_out,    6*28*28*sizeof(double));
    cudaMalloc(&d_C1_bias,   c1_bias.size()*sizeof(double));
    cudaMalloc(&d_S2_out,    6*14*14*sizeof(double));

    // Copy
    cudaMemcpy(d_raw_data, raw_data, 28*28*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_kernel, c1_kernel.data(), c1_kernel.size()*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_bias,   c1_bias.data(),   c1_bias.size()*sizeof(double),   cudaMemcpyHostToDevice);

    // C3
    cudaMalloc(&d_C3_kernel, c3_kernel.size()*sizeof(double));
    cudaMalloc(&d_C3_out,    16*10*10*sizeof(double));
    cudaMalloc(&d_C3_bias,   c3_bias.size()*sizeof(double));
    cudaMalloc(&d_S4_out,    16*5*5*sizeof(double));
    cudaMemcpy(d_C3_kernel, c3_kernel.data(), c3_kernel.size()*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C3_bias,   c3_bias.data(),   c3_bias.size()*sizeof(double),   cudaMemcpyHostToDevice);

    // Dense(120)
    cudaMalloc(&d_d1_kernel, d1_kernel.size()*sizeof(double));
    cudaMalloc(&d_d1_bias,   d1_bias.size()*sizeof(double));
    cudaMalloc(&d_D1_out,    120*sizeof(double));
    cudaMemcpy(d_d1_kernel, d1_kernel.data(), d1_kernel.size()*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d1_bias,   d1_bias.data(),   d1_bias.size()*sizeof(double),   cudaMemcpyHostToDevice);

    // Dense(84)
    cudaMalloc(&d_d2_kernel, d2_kernel.size()*sizeof(double));
    cudaMalloc(&d_d2_bias,   d2_bias.size()*sizeof(double));
    cudaMalloc(&d_D2_out,    84*sizeof(double));
    cudaMemcpy(d_d2_kernel, d2_kernel.data(), d2_kernel.size()*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d2_bias,   d2_bias.data(),   d2_bias.size()*sizeof(double),   cudaMemcpyHostToDevice);

    // Dense(10)
    cudaMalloc(&d_d3_kernel, d3_kernel.size()*sizeof(double));
    cudaMalloc(&d_d3_bias,   d3_bias.size()*sizeof(double));
    cudaMalloc(&d_D3_out,    10*sizeof(double));
    cudaMemcpy(d_d3_kernel, d3_kernel.data(), d3_kernel.size()*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d3_bias,   d3_bias.data(),   d3_bias.size()*sizeof(double),   cudaMemcpyHostToDevice);

    waitForEnter("STEP 4 done. Press Enter to run the pipeline...");

    // STEP 5: Inference
    // (A) C1: single-channel same -> bias -> tanh
    {
        dim3 blockDim(28,28);
        dim3 gridDim(1,1);
        cudaConv2D_same<<<gridDim, blockDim>>>(d_raw_data, d_C1_kernel, d_C1_out,
                                               28,28, 5, 6,
                                               28,28);
        cudaDeviceSynchronize();

        // addBias3D
        {
            dim3 block(8,8,6);
            dim3 grid((28+7)/8, (28+7)/8, 6);
            addBias3D<<<grid, block>>>(d_C1_out, d_C1_bias, 6,28,28);
            cudaDeviceSynchronize();
        }

        // tanh
        {
            dim3 blockAct(28,28);
            dim3 gridAct(1,1);
            activation_tanh<<<gridAct, blockAct>>>(d_C1_out, 28,28, 6, d_C1_out);
            cudaDeviceSynchronize();
        }
    }

    // (B) S2: pool => shape=(6,14,14)
    {
        dim3 blockDim2(14,14);
        dim3 gridDim2(1,1);
        cudaMeanPool<<<gridDim2, blockDim2>>>(d_C1_out, d_S2_out,
                                              28,28, 6,
                                              2,
                                              14,14);
        cudaDeviceSynchronize();
    }

    // (C) C3 multi-chan => (16,10,10)
    {
        // Now we do multi-channel valid conv:
        // in_data shape = (6,14,14), out_data shape = (16,10,10)
        // kernel shape = [16,6,5,5] => 2400
        dim3 blockDim3(10,10,1);
        dim3 gridDim3(1,1,16);
        cudaConv2D_valid_multiChan<<<gridDim3, blockDim3>>>(
            /* in_data=*/ d_S2_out,
            /* kernel=*/  d_C3_kernel,
            /* out_data=*/d_C3_out,
            /* in_rows=*/ 14,
            /* in_cols=*/ 14,
            /* kernel_size=*/ 5,
            /* in_channels=*/ 6,
            /* out_channels=*/16,
            /* out_rows=*/ 10,
            /* out_cols=*/ 10
        );
        cudaDeviceSynchronize();

        // addBias3D
        {
            dim3 block(10,10,16);
            dim3 grid(1,1,1);
            addBias3D<<<grid, block>>>(d_C3_out, d_C3_bias, 16,10,10);
            cudaDeviceSynchronize();
        }

        // tanh
        {
            dim3 blockAct(10,10);
            dim3 gridAct(1,1);
            activation_tanh<<<gridAct, blockAct>>>(d_C3_out, 10,10,16, d_C3_out);
            cudaDeviceSynchronize();
        }
    }

    // (D) S4 => 2×2 => (16,5,5)
    {
        dim3 blockPool(5,5);
        dim3 gridPool(1,1);
        cudaMeanPool<<<gridPool, blockPool>>>(d_C3_out, d_S4_out,
                                              10,10,16,
                                              2,
                                              5,5);
        cudaDeviceSynchronize();
    }

    // (E) Flatten => 16*5*5=400 => Dense(120)
    double h_S4_out[16*5*5];
    cudaMemcpy(h_S4_out, d_S4_out, sizeof(double)*16*5*5, cudaMemcpyDeviceToHost);

    // (F) Dense1(120)
    {
        double h_d1_in[400];
        for(int i=0; i<400; i++){
            h_d1_in[i] = h_S4_out[i];
        }
        double* d_d1_in = nullptr;
        cudaMalloc(&d_d1_in, 400*sizeof(double));
        cudaMemcpy(d_d1_in, h_d1_in, 400*sizeof(double), cudaMemcpyHostToDevice);

        dim3 blockD1(128);
        dim3 gridD1((120 + blockD1.x - 1)/blockD1.x);
        Dense<<<gridD1, blockD1>>>(d_d1_kernel, d_d1_in, d_D1_out, 400, 120);
        cudaDeviceSynchronize();

        // addBias1D
        {
            dim3 blockB1(128);
            dim3 gridB1((120 + blockB1.x - 1)/blockB1.x);
            addBias1D<<<gridB1, blockB1>>>(d_D1_out, d_d1_bias, 120);
            cudaDeviceSynchronize();
        }
        cudaFree(d_d1_in);
    }

    // (G) Dense2(84)
    {
        dim3 blockD2(128);
        dim3 gridD2((84 + blockD2.x - 1)/blockD2.x);
        Dense<<<gridD2, blockD2>>>(d_d2_kernel, d_D1_out, d_D2_out, 120, 84);
        cudaDeviceSynchronize();

        // addBias1D
        {
            dim3 blockB2(128);
            dim3 gridB2((84 + blockB2.x - 1)/blockB2.x);
            addBias1D<<<gridB2, blockB2>>>(d_D2_out, d_d2_bias, 84);
            cudaDeviceSynchronize();
        }
    }

    // (H) Dense3(10)
    {
        dim3 blockD3(128);
        dim3 gridD3((10 + blockD3.x - 1)/blockD3.x);
        Dense<<<gridD3, blockD3>>>(d_d3_kernel, d_D2_out, d_D3_out, 84, 10);
        cudaDeviceSynchronize();

        // addBias1D
        {
            dim3 blockB3(128);
            dim3 gridB3((10 + blockB3.x - 1)/blockB3.x);
            addBias1D<<<gridB3, blockB3>>>(d_D3_out, d_d3_bias, 10);
            cudaDeviceSynchronize();
        }
    }

    waitForEnter("Press Enter to retrieve final prediction...");

    // Copy final to host
    double h_pred[10];
    cudaMemcpy(h_pred, d_D3_out, 10*sizeof(double), cudaMemcpyDeviceToHost);

    // CPU Softmax
    double sumExp=0.0;
    for(int i=0; i<10; i++){
        h_pred[i] = exp(h_pred[i]);
        sumExp += h_pred[i];
    }
    std::cout<<"\nFinal probability distribution:\n";
    for(int i=0; i<10; i++){
        h_pred[i] /= (sumExp + 1e-15);
        std::cout<<"prob["<< i <<"] = "<< h_pred[i]<<"\n";
    }

    // Argmax
    int argMax=0;
    double maxVal = h_pred[0];
    for(int i=1; i<10; i++){
        if(h_pred[i] > maxVal){
            maxVal = h_pred[i];
            argMax = i;
        }
    }
    std::cout<<"\nPredicted digit="<< argMax <<" with prob="<< maxVal <<"\n";

    waitForEnter("Inference done. Press Enter to clean up...");

    // Cleanup
    cudaFree(d_raw_data);
    cudaFree(d_C1_kernel);
    cudaFree(d_C1_out);
    cudaFree(d_C1_bias);
    cudaFree(d_S2_out);

    cudaFree(d_C3_kernel);
    cudaFree(d_C3_out);
    cudaFree(d_C3_bias);
    cudaFree(d_S4_out);

    cudaFree(d_d1_kernel);
    cudaFree(d_d1_bias);
    cudaFree(d_D1_out);

    cudaFree(d_d2_kernel);
    cudaFree(d_d2_bias);
    cudaFree(d_D2_out);

    cudaFree(d_d3_kernel);
    cudaFree(d_d3_bias);
    cudaFree(d_D3_out);

    free(raw_data);
    free(C1_out);
    free(S2_out);
    free(C3_out);
    free(S4_out);
    free(D1_out);
    free(D2_out);
    free(D3_out);

    // free 2D image
    for(int i=0; i<HEIGHT; i++){
        free(img[i]);
    }
    free(img);

    std::cout<<"All resources freed. Exiting.\n";
    return 0;
}
