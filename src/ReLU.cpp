#include "ReLU.h"
#include "tensor4d.h"
#include "tensor3d.h"

#include "vector"

using namespace std;

ReLU::ReLU() {}

// we dont want inplace because we might need the previous output for later
Tensor4D ReLU::forward(Tensor4D input) {
    in_cache = input;
    for(Tensor3D& x : input) {
        for(vector<vector<float>>& y : x) {
            for(vector<float>& z : y) {
                for(float& a : z) {
                    a = max(a, 0.0f);
                }
            }
        }
    }

    return input;
}

Tensor4D ReLU::backward(Tensor4D d_out) {
    for(int i = 0; i < d_out.size(); i++) {
        for(int j = 0; j < d_out[i].size(); j++) {
            for(int k = 0; k < d_out[i][j].size(); k++) {
                for(int d = 0; d < d_out[i][j][k].size(); d++) {
                    if(in_cache[i][j][k][d] <= 0.0f) {
                        d_out[i][j][k][d] = 0.0f;
                    }
                }
            }
        }
    }
    return d_out;
}