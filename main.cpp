#include "tensor3d.h"
#include "tensor4d.h"
#include "MNIST_loader.h"
#include "Convolution.h"
#include "ReLU.h"
#include "MaxPool.h"
#include "Flatten.h"
#include "Dense.h"
#include "Softmax.h"

#include <iostream>
#include <cmath>
#include <algorithm>

using namespace std;

Tensor4D unflatten(const vector<float>& flat) {
    Tensor4D out(8, Tensor3D(1, vector<vector<float>>(13, vector<float>(13))));
    int index = 0;
    for (int c = 0; c < 8; c++) {
        for (int i = 0; i < 13; i++) {
            for (int j = 0; j < 13; j++) {
                out[c][0][i][j] = flat[index++];
            }
        }
    }
    return out;
}

int main() {
    Tensor4D train_images;
    vector<int> train_labels;

    MnistLoader::load("../data/train-images-idx3-ubyte", "../data/train-labels-idx1-ubyte", train_images, train_labels, 1000);

    int epochs = 30;
    float lr = 0.05;

    Convolution conv({1, 28, 28}, 3, 1, 16, lr);
    ReLU relu;
    MaxPool pool;
    Flatten flatten;
    Dense dense(lr);
    Softmax softmax;


    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0;
        int correct = 0;

        for (int i = 0; i < train_images.size(); ++i) {
            Tensor4D x = {train_images[i]};
            Tensor4D conv_out = conv.forward(x);
            Tensor4D relu_out = relu.forward(conv_out);
            Tensor4D pool_out = pool.forward(relu_out);
            vector<float> flat = flatten.forward(pool_out);
            vector<float> logits = dense.forward(flat);
            vector<float> probs = softmax.forward(logits);

            int label = train_labels[i];
            float loss = -log(probs[label] + 1e-8f);
            total_loss += loss;

            int pred = max_element(probs.begin(), probs.end()) - probs.begin();
            if (pred == label) correct++;
            vector<float> d_softmax = softmax.backward(label);
            vector<float> d_dense = dense.backward(d_softmax);
            Tensor4D d_unflat = unflatten(d_dense);           
            Tensor4D d_pool = pool.backward(d_unflat);
            Tensor4D d_relu = relu.backward(d_pool);
            conv.backward(d_relu);
        }

        float avg_loss = total_loss / train_images.size();
        float accuracy = 100.0f * correct / train_images.size();
        cout << "Epoch " << (epoch + 1) << ": Loss = " << avg_loss
             << ", Accuracy = " << accuracy << "%" << endl;
    }

    return 0;
}