#pragma once

#include "tensor3d.h"
#include "tensor4d.h"

#include <vector>

using namespace std;

// 2x2 maxpooling layer
class MaxPool {
private:
    inline bool is_close(float a, float b, float eps = 1e-5f);
public:
    Tensor4D in_cache;
    // learning rate is irrelevant
    MaxPool();

    Tensor4D forward(Tensor4D input);


    Tensor4D backward(Tensor4D d_out);
};