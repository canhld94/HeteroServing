#ifndef _ST_IE_H_
#define _ST_IE_H_
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

    /**
     * @brief 
     * 
     */
    struct bbox {
        int label_id;       //!< label id
        std::string label;       //!< class name
        float prop;         //!< confidence score
        int c[4];           //!< coordinates of bounding box
    };


    using namespace InferenceEngine;

    class inference_engine {
    private:
        /**
         * @brief OpenVino Inference Plugin
         * 
         */
        InferenceEngine::InferencePlugin plugin;               
        /**
         * @brief The logical CNN network
         * @details The object doesn't actually do the inferece request but hold
         * the information of the network object such as architecture, weight, and
         * device plugin. Excutable networks can be created from this 
         * network. Note that the executable networks are independent, e.g. they don't
         * share any resouce. So we can create an executable network from a network, then
         * reload the network with different architect or weights, then create another 
         * executable network.
         * 
         */
        InferenceEngine::CNNNetwork network;
    protected:
        /**
         * @brief The excutable CNN network
         * @details The actual CNN network that will excute the inference request.
         * With CPU, as much as excutable CNN network can be created from a logical 
         * network. With FPGAs, number of excutable networks is number of availble FPGAs
         * on the system. Several inference requests can be created within one executable
         * network, but the ordered of execution is not guarantee FCFS
         * 
         */
        InferenceEngine::ExecutableNetwork exe_network;
        /**
         * @brief Vector of labels
         * 
         */
        std::vector<std::string> labels;
        /**
         * @brief 
         * 
         */
        std::shared_ptr<spdlog::logger> log;
    public:
        /**
         * @brief Construct a new inference engine object
         * 
         */
        inference_engine() {
            log = spdlog::basic_logger_mt("IELog","logs/IE.txt");
            log->set_pattern("[%H:%M:%S %z] [%n] [%^---%L---%$] [thread %t] %v");
            log->info("Log started!");
        }
        /**
         * @brief Initilize the device plugin
         * 
         * @param device 
         */
        void init_plugin(const std::string& device) {
            log->info("Init new plugin {}",device);
            std::cout << "Inference Engine version: " << GetInferenceEngineVersion() << endl;
            // Loading Plugin
            plugin = PluginDispatcher().getPluginByDevice(device);
            // Adding CPU extension
            if (device.find("CPU") != std::string::npos) { // has CPU, load extension
                plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());
            }
            log->info("Pluggin inited");
            return;
        }
        /**
         * @brief Load the model to network
         * 
         * @param model 
         */
        void load_network(const std::string& model) {
            log->info("Loading model from {}", model);
                // Read the network
            CNNNetReader netReader;
            netReader.ReadNetwork(model);
            // Loading weight
            std::string bin = model;
            for (int i = 0; i < 3; ++i) 
                bin.pop_back();
            bin += "bin";
            netReader.ReadWeights(bin);
            network = netReader.getNetwork();
            log->info("Model loaded");
        }
        /**
         * @brief Init the input of the model
         * @details This function is virtual and should be overrided for each network because each type of network
         * has different output format
         * @param precision 
         */
        virtual void init_IO(Precision p) {
            auto inputInfo = InputsDataMap(network.getInputsInfo());
            InputInfo::Ptr& input = inputInfo.begin()->second;
            input->setPrecision(p);
        }
        /**
         * @brief Create an executable network from the logical netowrk
         * 
         * @param extension 
         */
        void load_plugin(std::map<std::string,std::string> extension) {
            log->info("Creating new executable network");
            std::chrono::time_point<std::chrono::system_clock> start;
            std::chrono::time_point<std::chrono::system_clock> end;
            std::chrono::duration<double,std::milli> elapsed_mil;
            start = std::chrono::system_clock::now();
            try {
                exe_network = plugin.LoadNetwork(network, {});
            }
            catch(const std::exception& e) {
                std::cout << e.what() << '\n';
                exit(1);
            }
            end = std::chrono::system_clock::now();
            elapsed_mil = end - start;
            log->info("Creating new executable network in {} ms",elapsed_mil.count());
        }
        /**
         * @brief Get the labels object
         * 
         * @param label 
         */
        void get_labels(const std::string& label) {
            std::ifstream inputFile(label);
            std::copy(std::istream_iterator<std::string>(inputFile),
            std::istream_iterator<std::string>(),
            std::back_inserter(labels));
        }
        /**
         * @brief Construct a new inference engine object
         * @details This is a convinient constructor that ensembles all of above methods.
         * @param model 
         * @param device 
         * @param label 
         */
        inference_engine(const std::string& device, const std::string& model, const std::string& label) {
            log = spdlog::basic_logger_mt("IELog","logs/IE.txt");
            log->set_pattern("[%H:%M:%S %z] [%n] [%^---%L---%$] [thread %t] %v");
            log->info("Log started!");
            init_plugin(device);
            load_network(model);
            init_IO(Precision::U8);
            load_plugin({});
            get_labels(label);         
        }
        using ptr = std::shared_ptr<inference_engine>;
    };

    /**
     * @brief 
     * 
     */
    class object_detection : public inference_engine {
            public:
            using inference_engine::inference_engine;
            /**
             * @brief Run detection
             * @details This funtion is virtual and must be overrided in each detector 
             * @return std::vector<bbox> 
             */
            virtual std::vector<bbox> run(const char* data, int size) {
                return {};
            }
            using ptr = std::shared_ptr<object_detection>;
    };
    /**
     * @brief 
     * 
     */
    class ssd : public object_detection {
    private:
        void frameToBlob(const cv::Mat& frame,
                    InferRequest::Ptr& inferRequest,
                    const std::string& inputName) {
                        
            Blob::Ptr frameBlob = inferRequest->GetBlob(inputName);
            matU8ToBlob<uint8_t>(frame, frameBlob);
        }

    public:
        using object_detection::object_detection;
        /**
         * @brief 
         * 
         * @param precision 
         */
        // void init_IO(std::string& precision) override {

        // }
        std::vector<bbox> run(const char* data, int size) override {
            std::vector<bbox> ret; // return value
            try {
                // decode out image
                cv::Mat frame = cv::imdecode(cv::Mat(1,size,CV_8UC3, (unsigned char*) data),cv::IMREAD_UNCHANGED);
                const int width = frame.size().width;
                const int height = frame.size().height;

                // create new request
                InferRequest::Ptr infer_request = exe_network.CreateInferRequestPtr();
                auto inputInfor = exe_network.GetInputsInfo();
                std::cout << inputInfor.begin()->first << std::endl;
                auto outputInfor = exe_network.GetOutputsInfo();
                std::cout << outputInfor.begin()->first << std::endl;
                frameToBlob(frame, infer_request, inputInfor.begin()->first);
                infer_request->Infer();
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
                    auto label = labels[label_id-1];

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
                return ret;
            }
            catch (const cv::Exception &e) {
                // let not opencv silly exception terminate our program
                std::cerr << "Error: " << e.what() << std::endl;
                return ret;
            }
        }

        using ptr = std::shared_ptr<object_detection>;
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
} // namespace st
} // namespace ie

#endif