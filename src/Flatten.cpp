#include "Flatten.h"

#include "tensor4d.h"
#include "tensor3d.h"

#include "vector"

using namespace std;

Flatten::Flatten() {}

vector<float> Flatten::forward(Tensor4D& in) {
    vector<float> out;

    for(Tensor3D& x : in) {
        for(vector<vector<float>>& y : x) {
            for(vector<float>& z : y) {
                for(float a : z) {
                    out.push_back(a);
                }
            }
        }
    }

    return out;
}