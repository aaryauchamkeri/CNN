#pragma once
#include <vector>
#include <cassert>
#include <iostream>

using namespace std;

using Tensor3D = vector<vector<vector<float>>>;

inline int get_channels(const Tensor3D& t) {
    return t.size();
}

inline int get_height(const Tensor3D& t) {
    return t[0].size();
}

inline int get_width(const Tensor3D& t) {
    return t[0][0].size();
}

inline Tensor3D create_tensor(int channels, int height, int width, float val = 0.0f) {
    return Tensor3D(channels, vector<vector<float>>(height, vector<float>(width, val)));
}

inline Tensor3D rand_tensor(int channels, int width, int height) {
    Tensor3D x(channels, vector<vector<float>>(width, vector<float>(height, rand()/RAND_MAX)));
    return x;
}

inline Tensor3D add_tensors(const Tensor3D& a, const Tensor3D& b) {
    Tensor3D out = a;
    for (int c = 0; c < get_channels(a); ++c)
        for (int h = 0; h < get_height(a); ++h)
            for (int w = 0; w < get_width(a); ++w)
                out[c][h][w] += b[c][h][w];
    return out;
}

inline void print_tensor(const Tensor3D& t) {
    int h = get_height(t);
    int w = get_width(t);
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j)
            std::cout << (t[0][i][j] > 0.5f ? '#' : ' ');
        std::cout << '\n';
    }
}

inline void print_shape3d(const Tensor3D& in) {
    cout << "(" << in.size() << ", " << in[0].size() << ", " << in[0][0].size() << ")" << endl;
}
