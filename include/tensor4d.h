#pragma once
#include <vector>
#include <cassert>
#include <iostream>

using namespace std;

using Tensor4D = vector<vector<vector<vector<float>>>>;

inline int get_out_channels(const Tensor4D& t) {
    return t.size();
}

inline int get_in_channels(const Tensor4D& t) {
    return t[0].size();
}

inline int get_kernel_height(const Tensor4D& t) {
    return t[0][0].size();
}

inline int get_kernel_width(const Tensor4D& t) {
    return t[0][0][0].size();
}

inline Tensor4D create_tensor4d(int out_channels, int in_channels, int height, int width, float val = 0.0f) {
    return Tensor4D(out_channels, vector<vector<vector<float>>>(in_channels, vector<vector<float>>(height, vector<float>(width, val))));
}

inline Tensor4D rand_tensor4d(int out_channels, int in_channels, int height, int width) {
    Tensor4D t = create_tensor4d(out_channels, in_channels, height, width);
    for (int oc = 0; oc < out_channels; ++oc)
        for (int ic = 0; ic < in_channels; ++ic)
            for (int h = 0; h < height; ++h)
                for (int w = 0; w < width; ++w)
                    t[oc][ic][h][w] = static_cast<float>(rand());
    return t;
}

inline Tensor4D add_tensors4d(const Tensor4D& a, const Tensor4D& b) {
    Tensor4D out = a;
    for (int oc = 0; oc < get_out_channels(a); ++oc)
        for (int ic = 0; ic < get_in_channels(a); ++ic)
            for (int h = 0; h < get_kernel_height(a); ++h)
                for (int w = 0; w < get_kernel_width(a); ++w)
                    out[oc][ic][h][w] += b[oc][ic][h][w];
    return out;
}

inline void print_shape(const Tensor4D& in) {
    cout << "(" << in.size() << ", " << in[0].size() << ", " << in[0][0].size() << ", " << in[0][0][0].size() << ")" << endl;
}