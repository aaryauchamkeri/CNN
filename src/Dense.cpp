#include "Dense.h"
#include <vector>

using namespace std;

float Dense::dotProduct(vector<float>& f, vector<float>& s) {
    float sum = 0.0f;
    for(int i = 0; i < f.size(); i++) {
        sum += (f[i] * s[i]);
    }
    return sum;
}
Dense::Dense(float lr) { // since we are just doing mnist rn we will just hardcode dimensions
    this->lr = lr;
    matrix = vector<vector<float>>(10, vector<float>(1352));
    for(vector<float>& x : matrix) {
        for(float& val : x) {
            val = ((float)rand())/RAND_MAX;
        }
    }
}

vector<float> Dense::forward(vector<float>& in) {
    vector<float> output;
    this->in_cache = in;

    for(int i = 0; i < matrix.size(); i++) {
        output.push_back(dotProduct(matrix[i], in));
    }

    this->out_cache = output;
    return output;
}

// returns the gradient respect to the inputs
// we perform gradient descent on the weights by finding the gradient respect to the weights
vector<float> Dense::backward(vector<float>& d_out) {

    // this performs the gradient descent
    for(int i = 0; i < matrix.size(); i++) {
        vector<float>& cur = matrix[i];
        for(int j = 0; j < cur.size(); j++) {
            cur[j] -= (in_cache[j]*d_out[i]*lr);
        }
    }

    // now we need to return the derivative respect to the inputs
    vector<float> grads(1352);
    for(int i = 0; i < matrix.size(); i++) {
        for(int j = 0; j < matrix[i].size(); j++) {
            grads[j] += (d_out[i] * matrix[i][j]);
        }
    }

    return grads;
}