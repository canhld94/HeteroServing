# Read me for an easy life

1. What is this project?  
This project build the demo for MEC project in 2020. The main objective is building an FPGA inference server running deep learning algorithms on Intel FPGA and expose RESTful APIs to users.

2. Important notes
Language: C++  
Build Tool: CMake  
Testing: GTest
Library: STL, Boost, GFlags

3. Directory structure

```Text
demo_apps/
├── bin                                       // Binary
│   └── lib
├── build                                     // Build directory
│   ├── client_server_beast
│   ├── CMakeFiles
│   ├── format_reader
│   ├── intel64
│   └── ssd_standalone
├── client_server_beast                       // Testing server
│   ├── docs
│   ├── http
│   └── ws
├── cmake-modules                             // CMAKE modules
├── common                                    // Common headers
│   ├── os
│   ├── samples
│   └── ultis
├── format_reader                             // Lib format_reader from intel, better don't touch
├── ssd_inference_server                      // SSD Inference server
└── ssd_standalone                            // Standalone SSD inference application running on FPGA

```

What have been done and what to learn up to

## ---- 2020 / 07 / 23 ----

- handle image data from client using `http::string_body` --> `char` array in JPEG form
- create the inference api which response the desired format
- handle JSON with `boost::property_tree`
- create the struture for `ssdFPGA.h`
- now ready to integrate the FPGA inference engine

## ---- 2020 / 07 / 22 ----

- Finish creating a http server that support all basic desired APIs
- JSON response format for inference request
- _Requirement:_
  - JSON parsing for response: how to do it elegantly?
  - Image container:
    - How to send it to server, how the server will handle it?
    - The req created in the server is default basic string, we need to change it to file somehow
    - Should we use opencv?

## ---- 2020 / 07 / 21 ----

- Look into inside of Beast: the message container, body concept
- Adding resolver to the target request
- Should refer implementation from Tensorflow serving
- _Requirement:_
  - Asio
  - Structure of code

## ---- 2020 / 07 / 20 ----

- Cmake project: how cmake work, what is the structure of a CMake project
- Boost Beast: how to build websocket and http server with boost
- Finish building http sync server with boost and support POST
- _Requirement:_
  - Clear definition of the APT (get,post)
  - HTTP RFC7023
  - Boost Beast

## ---- 2020 / 07 / 15 ----

- OpenVino setup running environment
- Docker
