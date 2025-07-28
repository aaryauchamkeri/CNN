#include "Convolution.h"
#include "tensor3d.h"
#include "tensor4d.h"
#include <utility>

using namespace std;

Convolution::Convolution(tuple<int,int,int> in_dim, int kern_size, int depth, int out, float lr) {
    this->in_dim = in_dim;
    this->kern_size = kern_size;
    this->lr = lr;
    kernels = rand_tensor4d(out, depth, kern_size, kern_size);
    biases = vector<float>(out);
    for(float& x : biases) x = rand()/((float)RAND_MAX); // random biases
}

// applying cross correlation
Tensor4D Convolution::forward(Tensor4D input) {
    this->input_cache = input;
    Tensor4D out;
    int out_r = input[0][0].size() - kern_size + 1;
    int out_c = input[0][0][0].size() - kern_size + 1;

    for(int i = 0; i < kernels.size(); i++) {
        Tensor3D& kernel_block = kernels[i];
        Tensor3D sum_tensors = create_tensor(1, out_r, out_c);
        for(int j = 0; j < kernel_block.size(); j++) {
            vector<vector<float>> kernel = kernel_block[j];
            vector<vector<float>> current_img = input[j][0];
            Tensor3D cur_tensor = create_tensor(1, out_r, out_c);
            for(int k = 0; k < out_r; k++) {
                for(int d = 0; d < out_c; d++) {
                    float sum = 0.0f;
                    for(int s = 0; s < kern_size*kern_size; s++) {
                        int crow = s/kern_size;
                        int ccol = s%kern_size;
                        sum += (current_img[k + crow][d + ccol] * kernel[crow][ccol]);
                    }
                    cur_tensor[0][k][d] = sum;
                }
            }
            sum_tensors = add_tensors(sum_tensors, cur_tensor);
        }

        for(vector<vector<float>>& v : sum_tensors) {
            for(vector<float>& v1 : v) {
                for(float& val : v1) {
                    val += biases[i];
                }
            }
        }
        out.push_back(sum_tensors);
    }

    return out;
}

// the shape of d_out is 8x26x26
void Convolution::backward(Tensor4D& d_out) {
    // there are a corresponding kernels
    for(int i = 0; i < d_out.size(); i++) {
        // dim of img is 1x26x26
        Tensor3D& img_d = d_out[i];
        vector<vector<float>>& vec2d = img_d[0]; // 26x26
        Tensor3D& cur_k = kernels[i]; // the current kernel;
        vector<vector<float>>& kernel_cur_vec2d = cur_k[0];
        for(int j = 0; j < vec2d.size(); j++) {
            for(int k = 0; k < vec2d[j].size(); k++) {
                for(vector<vector<float>>& kernel : cur_k) {
                    for(int d = 0; d < kern_size*kern_size; d++) {
                        int row = d/kern_size;
                        int col = d%kern_size;
                        // we are not using batches of images
                        kernel[row][col] -= (vec2d[j][k] * input_cache[0][0][j + row][k + col] * lr);
                    }
                }

                biases[i] -= (lr * vec2d[j][k]);
            }
        }
    }
}