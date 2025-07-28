#pragma once

#include "tensor4d.h"
#include "tensor3d.h"

using namespace std;

class ReLU {
public:

    Tensor4D in_cache;

    ReLU();

    Tensor4D forward(Tensor4D input);

    Tensor4D backward(Tensor4D d_out);
};