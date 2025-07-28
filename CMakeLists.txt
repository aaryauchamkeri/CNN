cmake_minimum_required(VERSION 3.10)
project (CNN LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
add_executable(cnn main.cpp 
                    src/Dense.cpp 
                    src/Convolution.cpp 
                    src/Flatten.cpp 
                    src/MaxPool.cpp 
                    src/MNIST_loader.cpp 
                    src/ReLU.cpp 
                    src/Softmax.cpp
                )