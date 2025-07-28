#include "MaxPool.h"

#include "tensor3d.h"
#include "tensor4d.h"

#include <vector>

using namespace std;

// 2x2 maxpooling layer
inline bool MaxPool::is_close(float a, float b, float eps) {
    return fabs(a - b) < eps;
}

MaxPool::MaxPool() {}

Tensor4D MaxPool::forward(Tensor4D input) {
    Tensor4D output;
    in_cache = input;
    // the tensor 3d has a 1 at the 0th dimension
    for(Tensor3D img : input) {
        Tensor3D maxpooled(1);
        for(int i = 0; i < img[0].size(); i += 2) {
            vector<float> cur;
            for(int j = 0; j < img[0][i].size(); j += 2) {
                float maxval = max({img[0][i][j], img[0][i+1][j], img[0][j+1][i], img[0][j+1][i+1]});
                cur.push_back(maxval);
            }
            maxpooled[0].push_back(cur);
        }
        output.push_back(maxpooled);
    }   

    return output;
}


Tensor4D MaxPool::backward(Tensor4D d_out) {
    Tensor4D grads;

    for(int i = 0; i < d_out.size(); i++) {
        Tensor3D& cur = d_out[i];
        Tensor3D& in = in_cache[i];
        Tensor3D grad(1, vector<vector<float>>(in[0].size(), vector<float>(in[0][0].size(), 0)));
        for(int j = 0; j < cur[0].size(); j++) {
            for(int k = 0; k < cur[0][0].size(); k++) {
                float& mxval = cur[0][j][k];
                for(int d = 0; d < 4; d++) {
                    int row = d/2;
                    int col = d%2;
                    if(is_close(in[0][j*2 + row][k*2 + col], mxval)) {
                        grad[0][j*2 + row][k*2 + col] = 1*cur[0][j][k];
                    }
                }
            }
        }
        grads.push_back(grad);
    }

    return grads;
}