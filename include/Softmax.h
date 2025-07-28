#pragma once

#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>

using namespace std;

class Softmax {
private:
    vector<float> out_cache;
public:

    Softmax();

    vector<float> forward(vector<float>& in);

    vector<float> backward(int label);
};