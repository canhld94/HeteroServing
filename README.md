# something

Everybody (include [Circle CI](https://twitter.com/circleci/status/951635852974854144?lang=en)) tell me I shoud build something, so I build something

## Introduction

This project build an inference server with ~~Intel FPGA~~ Intel CPU, Intel FPGA, and NVIDIA GPU backend. Currently the inference engine supports object detection object detection models (`SSD`, `YoLov3`*, and `Faster R-CNN` family); and the server support REST API. At a glance:

*Request*

```SH
curl --location --request POST 'xxx.xxx.xxx.xxx:8080/inference' \
--header 'Content-Type: image/jpeg' \
--data-binary '@/C:/Users/CanhLD/Desktop/Demo/AirbusDrone.jpg'
```

*Return*

```JSON
{
    "predictions": [
        {
            "label_id": "1",
            "label": "plane",
            "confidences": "0.998418033",
            "detection_box": [
                "182",
                "806",
                "291",
                "919"
            ]
        },
        {
            "label_id": "1",
            "label": "plane",
            "confidences": "0.997635841",
            "detection_box": [
                "26",
                "182",
                "137",
                "309"
            ]
        }
    ]
}
```

> **_NOTE:_**  I do not implement yolo for GPU

## Requirements

The server object and protocol object depends on folowing packages. I strongly recomend install them with [Conan](https://conan.io/), so you do not need to modify the CMake files.

```
boost==1.73.0: socket and IPC, networking, HTTP parsing and serializing, JSON parsing and serializing
spdlog==1.7.0: logging and debugging
glfags==2.2.2: argv parsing
gtest==1.10.0: testing
```

For inference engine, I implemented CPU and FPGA inference with [Intel OpenVino](https://docs.openvinotoolkit.org/2019_R1.1/index.html), and GPU inference with [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt). Please refer to their doccuments to install the framework.

```
openvino==2019R1.1
tensorrt==7.0.2
```

The following package are required to build the project. If you don't want to use Conan, mannually add cmake modules to the CMake files.

```
GCC>=5
CMake>=3.13
Conan
```

## Directory structure

```
.
├── client                  >> Client samples
│   └── imgs                >> Sample images
├── docs                    >> Doccument
│   ├── apis                >> API doccument
│   └── dev                 >> Note I write during development of the project
├── server                  >> Server
│   ├── config              >> Server configuration
│   ├── libs                >> Libararies that implement components of the server
│   ├── parallel            >> Server runs in parallel (concurrency) model
│   └── reactor             >> Server runs in reactor model
└── test
```

## How to build the project

Make sure you have CMake and Conan

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

1. Understand the difference between [parallel server]() and [reactor server]()

2. Configure your [server file](server/config/README.md)

3. Go to bin folder and run the server

```SH
cd bin
./parallel_server -f <path to your config file>
```

This will start the server, and the endpoint for inferencing is `/inference`. Send any image to the endpoint and server will return detection result in JSON format.

3. On another terminal, go to run client folder and run client, the result will be written to file "testing.jpg"

```SH
python simple_client.py
```
