#include "Softmax.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>

using namespace std;

Softmax::Softmax() {}

vector<float> Softmax::forward(vector<float>& in) {
    float max_val = *max_element(in.begin(), in.end());
    float exp_sum = 0.0f;
    vector<float> softmax(in.size());

    for (float x : in) {
        exp_sum += exp(x - max_val);
    }
    
    for (int i = 0; i < in.size(); i++) {
        softmax[i] = exp(in[i] - max_val) / exp_sum;
    }

    this->out_cache = softmax;

    return softmax;
}

vector<float> Softmax::backward(int label) {
    vector<float> grad = out_cache;
    grad[label] -= 1.0f;
    return grad;
}