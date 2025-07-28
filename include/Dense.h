#pragma once
#include <vector>

using namespace std;

class Dense {
private:
    float dotProduct(vector<float>& f, vector<float>& s);

public:
    float lr;
    vector<vector<float>> matrix;
    vector<float> in_cache; // flat vector of 1352 elements
    vector<float> out_cache; // flat vector of 10 elements

    Dense(float lr);

    vector<float> forward(vector<float>& in);

    // returns the gradient respect to the inputs
    // we perform gradient descent on the weights by finding the gradient respect to the weights
    vector<float> backward(vector<float>& d_out);
};