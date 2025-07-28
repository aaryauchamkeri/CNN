#pragma once

#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <stdexcept>

#include "tensor3d.h"

using namespace std;

// all images in MNIST are 28x28 and gray scale so only one channel
class MnistLoader {
public:
    static void load(const string& image_path, const string& label_path,
                     vector<Tensor3D>& images, vector<int>& labels, int limit = -1);
};