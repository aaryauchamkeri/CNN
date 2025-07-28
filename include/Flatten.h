#pragma once

#include "tensor4d.h"
#include "tensor3d.h"

#include "vector"

using namespace std;

class Flatten {
public:
    Flatten();

    vector<float> forward(Tensor4D& in); 
};