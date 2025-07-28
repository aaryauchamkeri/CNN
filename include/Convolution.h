#pragma once
#include "tensor3d.h"
#include "tensor4d.h"
#include <utility>

using namespace std;

class Convolution {
public:
    Tensor4D kernels;
    vector<float> biases;
    tuple<int,int,int> in_dim;
    int kern_size;
    float lr;
    Tensor4D input_cache;

    Convolution(tuple<int,int,int> in_dim, int kern_size, int depth, int out, float lr);

    Tensor4D forward(Tensor4D input);

    void backward(Tensor4D& d_out);
};