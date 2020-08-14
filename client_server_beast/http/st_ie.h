
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

namespace st {
namespace ie {

    class inference_engine {
    private:
        vector<string> labels;                                 // path to labels file
        InferenceEngine::InferencePlugin plugin;               // OpenVino inference plugin
        InferenceEngine::CNNNetwork network;                   // The logical CNN network
        InferenceEngine::ExecutableNetwork exe_network;        // The actual object which will excute the request
        InferenceEngine::InputsDataMap inputblob;
        InferenceEngine::OutputsDataMap outputblob;
    public:
        inference_engine() {};
        inference_engine(std::string& model, std::string& device, std::string& label) {

        };
        inference_engine(inference_engine& ) 
    };

    /*
        Generic object detection interface
    */

    class object_detection {

    };

    /*
        SSD inferencer
    */

    class ssd : public object_detection {

    };

    /*
        Yolo inferencer
    */

    class yolo : public object_detection {

    };

    /*
        Fast r cnn inferencer
    */

    class fast_r_cnn : public object_detection {

    };

    /*
        classification generic interface
    */

    class classification : public inference_engine {

    };

    /*
        Resnet inferencer
    */

    class resnet101 : public classification {

    };

    /*
        Segmentation generic interface
    */

    class segmentation : public inference_engine {
        
    };

    /*
        MaskRCNN inferencer
    */

    class mask_r_cnn :  public segmentation {

    };
}
}