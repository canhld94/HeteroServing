# something ![CMake CI](https://github.com/canhld94/mec-inference-server/workflows/CMake%20CI/badge.svg)

## Introduction

This project build an inferecen server with OpenVino FPGA backend. Currently only support object detection with SSD and YoLov3. Faster R-CNN is comming soon

## Requirements

Dependencies

```
openvino==2019R1.1: deep learning framework
boost==1.73.0: socket and IPC, networking, HTTP parsing and serializing, JSON parsing and serializing
spdlog==1.7.0: logging and debugging
glfags==2.2.2: argv parsing
gtest==1.10.0: testing
```

The following package are required to build the project

```
GCC>=5
CMake>=3.13
Conan
```

If you don't want to use conan, mannually add cmake modules to the Cmake files.

## Directory structure

```
.
├── client                  >> Client
│   └── imgs                >> Sample images
├── docs                    >> Doccument
│   ├── apis                >> API doccument
│   └── dev                 >> Note I write during development of the project
├── server                  >> Server
│   ├── config              >> Server configuration
│   ├── libs                >> Libararies that implement components of the server
│   ├── parallel            >> Server runs in parallel (concurrency) modek
│   └── reactor             >> Server runs in reactor model
└── test
```

## How to build

Make sure you install CMake and Conan

```SH
git clone https://github.com/canhld94/something.git
cd something
mkdir bin
mkdir build && cd build
conan install ..
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
cmake install
```

Every binary file, include conan package binaries will be installed in the `bin` folder.

## How to run the server

1. Configure your [server file](server/config/README.md)

2. Go to bin folder and run the server

```SH
cd bin
./parallel_server -f <path to your config file>
```

3. On another terminal, go to run client folder and run client, the result will be written to file "testing.jpg"

```SH
python client.py <path to your image> <ip:port>
```
