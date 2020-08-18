// Copyright (C) 2020 canhld@.kaist.ac.kr
// SPDX-License-Identifier: Apache-2.0
//
// C include
#ifndef _SSDFPGA_H_
#define _SSDFPGA_H_
#include <stdio.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <errno.h>
// C++
#include <gflags/gflags.h>
#include <functional>
#include <iostream>
#include <fstream>
#include <random>
#include <memory>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include <iterator>
#include <mutex>

#include <memory>
#include <thread>
#include <csignal>

#include <inference_engine.hpp>

#include <samples/ocv_common.hpp>
#include <samples/slog.hpp>

#include <ext_list.hpp>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>

#include "ultis.h"

// we use stl for most of our container 
// just avoid so many mess std::
using std::cout;
using std::string;
using std::vector;
using std::endl;

#ifdef _WIN32
#include <os/windows/w_dirent.h>
#else
#include <dirent.h>
#endif

/*
    This header implement three namespace:
    - common: common data structures
    - helper: utilities function that will be use by the main namespace
    - ncl: implement the core inference logic
    Why so much namespace? Who know, I like it
*/

namespace common {
    /*
    OpenVino Plugin class, hold all variable of the plugin
    */

    /*
    Light weight CNN class that hold all metadata we need
    This will be fill during construction of ssdFPGA;
    */
    class lwCNN {
    public:
        string name;
        string description;
        int num_layers;
        int num_params;
        int input_size;
        int output_size;
    };

    /*
    Bounding box class the represent a prediction
    The bounding box is (c[0],c[1]) --> bottom left, (c[2],c[3]) --> top right
    */

    class bbox {
    public:
        int label_id;       // label id
        string label;       // class name
        float prop;        // confidence score
        int c[4];           // coordinates of bounding box
    };
} // namespace commom


namespace helper {
    using namespace common;
    using namespace InferenceEngine; // namespace inference engine

/*  
    This function initilize the environment for our application
    _device must be pass when construct
*/
    void init_plugin (string& device, InferencePlugin& plugin) {
        cout << "Inference Engine version: " << GetInferenceEngineVersion() << endl;
        // Loading Plugin
        plugin = PluginDispatcher().getPluginByDevice(device);
        // Adding CPU extension
        if (device.find("CPU") != string::npos) { // has CPU, load extension
            plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());
        }
        return;
    }

    void load_network ( string& xml, 
                        string& l,
                        vector<string>& labels,
                        CNNNetwork& network ) { // get the network
        // Read the network
        CNNNetReader netReader;
        netReader.ReadNetwork(xml);
        // Loading weight
        string bin = xml;
        for (int i = 0; i < 3; ++i) bin.pop_back();
        bin += "bin";
        netReader.ReadWeights(bin);
        network = netReader.getNetwork();
        // load the labels
        std::ifstream inputFile(l);
        std::copy(std::istream_iterator<std::string>(inputFile),
                  std::istream_iterator<std::string>(),
                  std::back_inserter(labels));
        return;
    }


    void init_IO(CNNNetwork &network) {

        // Input Blob

        auto inputInfo = InputsDataMap(network.getInputsInfo());
        if (inputInfo.size() != 1) {
            throw std::logic_error("This demo accepts networks having only one input");
        }
        InputInfo::Ptr& input = inputInfo.begin()->second;
        input->setPrecision(Precision::U8);
        // input->getInputData()->setLayout(Layout::NCHW);

        // Output Blob

        // auto outputInfo = OutputsDataMap(network.getOutputsInfo());
        // if (outputInfo.size() != 1) {
        //     throw std::logic_error("This demo accepts networks having only one output");
        // }
        // DataPtr& output = outputInfo.begin()->second;
        // const SizeVector outputDims = output->getTensorDesc().getDims();
        // const int objectSize = outputDims[3];
        // if (objectSize != 7) {
        //     throw std::logic_error("Output should have 7 as a last dimension");
        // }
        // if (outputDims.size() != 4) {
        //     throw std::logic_error("Incorrect output dimensions for SSD");
        // }
        // output->setPrecision(Precision::FP32);
        // output->setLayout(Layout::NCHW);
    }

    void load_plugin(InferencePlugin& plugin,
                     CNNNetwork& network,
                     ExecutableNetwork &exe_network) {
        try {
            exe_network = plugin.LoadNetwork(network, {});
        }
        catch(const std::exception& e) {
            std::cout << e.what() << '\n';
        }
        
    }

    void frameToBlob(const cv::Mat& frame,
                 InferRequest::Ptr& inferRequest,
                 const std::string& inputName) {
                     
        Blob::Ptr frameBlob = inferRequest->GetBlob(inputName);
        matU8ToBlob<uint8_t>(frame, frameBlob);
    }

    /*
        !Testing: return a random bbox vector
    */
    vector<bbox> run_test(const char *data) {
        int n = 3;
        vector<bbox> ret(n);
        for (int i = 0; i < n; ++i) {
            // generate a fake bbox
            ret[i].label_id = rand()%91;
            ret[i].label = "lalaland_" + std::to_string(i);
            ret[i].prop = 100.0/(rand()%100);
            for (int j = 0; j < 4; ++j) {
                ret[i].c[j] = rand()%1000;
            }
        }
        return ret;
    }
}

namespace ncl {
    using namespace helper;
    using namespace common;
    using namespace InferenceEngine;


    /*
        SSD class that implement everything we need
        // ? Should we make it a singleton class
    */

    class ssdFPGA {
    private:
        vector<string> _labels;                                 // path to labels file
        InferenceEngine::InferencePlugin _plugin;               // OpenVino inference plugin
        InferenceEngine::CNNNetwork _network;                   // The logical CNN network
        InferenceEngine::ExecutableNetwork _exe_network;        // The actual object which will excute the request
        // InferenceEngine::InputsDataMap _inputInfor;
        // InferenceEngine::OutputsDataMap _outputInfor;
    public:
        // !deprecated
        // lock with std::lock_guard<std::mutex> lock(m);
        // std::lock_guard is RAII, so it will release when run out of scope, no need to unlock
        // ! Never use m->lock() and m->unlock(), use std lock guards instead
        // ! Why? what if your thread throws an exception before unlock m?
        // ! Deadlock cannot occur, but livelock can, scheduler need?
        std::mutex m;
        // default constructor
        ssdFPGA();
        // constructor for running in producer-consumer model
        ssdFPGA(string _device, string _xml, string _l, int _detach);
        // default destructor
        ~ssdFPGA();
        // run inference with input is stream from network
        // return a vector of bbox
        vector<bbox> run(const char*, int size);
        // get the metadata
        lwCNN get();
    };
//-------------------------------------------------------------------------

    ssdFPGA::ssdFPGA() {
        
    }

    ssdFPGA::ssdFPGA(string _device, string _xml, string _l, int _detach) {
        // we should follow the RAII
        // Acquire proper resouces during constructor
        // TODO: discover and lock FPGA card, reprogramming the device and ready to launch
        try
        {
            // InferenceEngine::ExecutableNetwork aux_exe_network;
            PROFILE_RELEASE("Init Pulgin",
            init_plugin(_device,_plugin);
            );
            PROFILE_RELEASE("Load Network",
            load_network(_xml,_l,_labels,_network);
            )
            PROFILE_RELEASE("Init IO Blob",
            init_IO(_network);
            )
            PROFILE_RELEASE("Create Excutable Network",
            load_plugin(_plugin,_network,_exe_network);            
            )
            // PROFILE_RELEASE("Create Auxialry Excutable Network",
            // load_plugin(_plugin,_network,aux_exe_network);            
            // )
        }
        catch(const std::exception& e)
        {
            std::cout << e.what() << '\n';
        }
    }

    ssdFPGA::~ssdFPGA() {
        // release every resource
        // TODO: release any resource that we accquire before
    }

    lwCNN ssdFPGA::get() {
        // TODO
        lwCNN ret;
        return ret;
    }


    vector<bbox> ssdFPGA::run(const char *data, int size) {
        // return run_test(data);
        vector<bbox> ret; // return value
        try {
            // decode out image
            PROFILE_RELEASE("Decode Image and Prepare Blobs",
            cv::Mat frame = cv::imdecode(cv::Mat(1,size,CV_8UC3, (unsigned char*) data),cv::IMREAD_UNCHANGED);
            const int width = frame.size().width;
            const int height = frame.size().height;
            )

            PROFILE_RELEASE("Do Inference",
            // create new request
            InferRequest::Ptr infer_request = _exe_network.CreateInferRequestPtr();
            auto inputInfor = _exe_network.GetInputsInfo();
            std::cout << inputInfor.begin()->first << std::endl;
            auto outputInfor = _exe_network.GetOutputsInfo();
            std::cout << outputInfor.begin()->first << std::endl;
            frameToBlob(frame, infer_request, inputInfor.begin()->first);
            infer_request->Infer();
            )

            PROFILE_RELEASE("Processing Network Output",
            CDataPtr &output = outputInfor.begin()->second;
            auto outputName = outputInfor.begin()->first;
            const SizeVector outputDims = output->getTensorDesc().getDims();
            const int maxProposalCount = outputDims[2];
            const int objectSize = outputDims[3];
            const float *detections = infer_request->GetBlob(outputName)->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
            for (int i = 0; i < maxProposalCount; i++) {
                float image_id = detections[i * objectSize + 0];
                if (image_id < 0) {
                    break;
                }
                float confidence = detections[i * objectSize + 2];
                auto label_id = static_cast<int>(detections[i * objectSize + 1]);
                int xmin = detections[i * objectSize + 3] * width;
                int ymin = detections[i * objectSize + 4] * height;
                int xmax = detections[i * objectSize + 5] * width;
                int ymax = detections[i * objectSize + 6] * height;
                auto label = _labels[label_id-1];

                if (confidence > 0.45) {
                    bbox d;
                    d.prop = confidence;
                    d.label_id = label_id;
                    d.label = label;
                    d.c[0] = xmin;
                    d.c[1] = ymin;
                    d.c[2] = xmax;
                    d.c[3] = ymax;
                    ret.push_back(d);
                }
            }
            )
            return ret;
        }
        catch (const cv::Exception &e) {
            // let not opencv silly exception terminate our program
            std::cerr << "Error: " << e.what() << std::endl;
            return ret;
        }
    }
} // namespace ncl
#endif