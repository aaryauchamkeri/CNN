#include "MNIST_loader.h"
#include "tensor3d.h"
#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <stdexcept>

using namespace std;

// all images in MNIST are 28x28 and gray scale so only one channel
void MnistLoader::load(const string& image_path, const string& label_path,
                 vector<Tensor3D>& images, vector<int>& labels, int limit) {
    ifstream image_file(image_path, std::ios::binary);
    ifstream label_file(label_path, std::ios::binary);

    if(!image_file.is_open() || !label_file.is_open()) {
        throw runtime_error("Failed to open file");
    }

    image_file.ignore(16);
    label_file.ignore(8);

    int cnt = 0;
    while(!image_file.eof() && (limit < 0 || cnt < limit)) {
        unsigned char label_byte;
        label_file.read((char*)&label_byte, 1);
        if(label_file.eof()) break;
        Tensor3D image = create_tensor(1, 28, 28);

        for(int i = 0; i < 28; i++) {
            for(int j = 0; j < 28; j++) {
                unsigned char pixel;
                image_file.read((char*)&pixel, 1);
                if(image_file.eof()) break;
                image[0][i][j] = static_cast<float>(pixel)/255.0f;
            }
        }
        
        images.push_back(image);
        labels.push_back(label_byte);
    }
}