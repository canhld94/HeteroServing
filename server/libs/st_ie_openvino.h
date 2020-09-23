/***************************************************************************************
 * Copyright (C) 2020 canhld@.kaist.ac.kr
 * SPDX-License-Identifier: Apache-2.0
 * @b About: This file implement of Intel OpenVino Inference Engine
 ***************************************************************************************/
#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <chrono>


#include <inference_engine.hpp>
#include <opencv2/opencv.hpp>
#include <ext_list.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include "st_exception.h"
#include "st_ie_base.h"

// OpenVino Inference Engine
using namespace InferenceEngine;


namespace st {
namespace ie {
    /**
    * @brief Sets image data stored in cv::Mat object to a given Blob object.
    * @param orig_image - given cv::Mat object with an image data.
    * @param blob - Blob object which to be filled by an image data.
    * @param batchIndex - batch index of an image inside of the blob.
    */
    template <typename T>
    void matU8ToBlob(const cv::Mat& orig_image, InferenceEngine::Blob::Ptr& blob, int batchIndex = 0) {
        InferenceEngine::SizeVector blobSize = blob->getTensorDesc().getDims();
        const size_t width = blobSize[3];
        const size_t height = blobSize[2];
        const size_t channels = blobSize[1];
        T* blob_data = blob->buffer().as<T*>();

        cv::Mat resized_image(orig_image);
        if (static_cast<int>(width) != orig_image.size().width ||
                static_cast<int>(height) != orig_image.size().height) {
            cv::resize(orig_image, resized_image, cv::Size(width, height));
        }

        int batchOffset = batchIndex * width * height * channels;

        for (size_t c = 0; c < channels; c++) {
            for (size_t  h = 0; h < height; h++) {
                for (size_t w = 0; w < width; w++) {
                    blob_data[batchOffset + c * width * height + h * width + w] =
                            resized_image.at<cv::Vec3b>(h, w)[c];
                }
            }
        }
    }
    /**
     * @brief Map opencv map to blob
     * 
     * @param frame 
     * @param inferRequest 
     * @param inputName 
     */
    void frameToBlob(const cv::Mat& frame,
            InferRequest::Ptr& inferRequest,
            const std::string& inputName) {   
        Blob::Ptr frameBlob = inferRequest->GetBlob(inputName);
        matU8ToBlob<uint8_t>(frame, frameBlob);
    }

    /**
     * @brief Output of an inference request
     * 
     */
    struct network_output {
        InferRequest::Ptr infer_request;
        int width;
        int height;
    };

    /**
     * @brief OpenVino inference engine
     * 
     */
    class openvino_inference_engine : public inference_engine {
    public:
        /****************************************************************/
        /*  Inference engine public interface implementation            */
        /****************************************************************/

        std::vector<bbox> run_detection(const char* data, int size) final {
            auto net_out = do_infer(data,size);
            return detection_parser(net_out);
        }

        std::vector<int> run_classification(const char* data, int size) final {
            auto net_out = do_infer(data,size);
            return classification_parser(net_out);
        }

        /**
         * @brief Parse detection output of a inference request, network specific
         * 
         * @param net_out 
         * @return std::vector<bbox> 
         */
        virtual std::vector<bbox> detection_parser (network_output& net_out) {
            return {};
        }
        /**
         * @brief Parse classification output of a inference request, network specific
         * 
         * @param net_out 
         * @return std::vector<int> 
         */
        virtual std::vector<int> classification_parser (network_output& net_out) {
            return {};
        }

        using ptr = std::shared_ptr<openvino_inference_engine>;

    private:
        /**
         * @brief OpenVino Inference Plugin
         * 
         */
        InferenceEngine::InferencePlugin plugin;

    protected:
        /**
         * @brief The logical CNN network
         * @details The object doesn't actually do the inferece request but hold
         * the information of the network object such as architecture, weight, and
         * device plugin. Excutable networks can be created from this network.
         * Note that the executable networks are independent, e.g. they don't
         * share any resouce. So we can create an executable network from a
         * network, then reload the network with different architect or weights,
         * then create another executable network.
         */
        InferenceEngine::CNNNetwork network;
        /**
         * @brief The excutable CNN network
         * @details The actual CNN network that will excute the inference request.
         * With CPU, as much as excutable CNN network can be created from a
         * logical network. With FPGAs, number of excutable networks is number of
         * availble FPGAs on the system. Several inference requests can be created
         * within one executable network, but the ordered of execution is not
         * guarantee FCFS
         */
        InferenceEngine::ExecutableNetwork exe_network;
        /**
         * @brief Inference engine is the most important components in the system,
         * so it will have
         * a dedicated log
         *
         */
        std::shared_ptr<spdlog::logger> log;
        /**
         * @brief Initilize the device plugin
         * 
         * @param device 
         */
        void init_plugin(const std::string& device) {
            log->info("Init new {} plugin",device);
            std::cout << "Inference Engine version: " << GetInferenceEngineVersion() << std::endl;
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
         * @brief Perform sanity check for a network
         * @details This function is virutal and should be overrided for each network
         */
        virtual void IO_sanity_check () {}
        /**
         * @brief Init the input of the model
         * @details Most of network only have one single image tensor input, some like
         * faster r-cnn may have two (im_info), but it doesn't important here
         * @param precision 
         */
        void init_IO(Precision p, InferenceEngine::Layout layout) {
            IO_sanity_check(); // if wrong network, it will fail at this stage
            auto input_info = InputsDataMap(network.getInputsInfo());
            for (auto &item : input_info) {
                if (item.second->getInputData()->getTensorDesc().getDims().size() == 4) {
                    // this is image tensor
                    item.second->setPrecision(p);
                    item.second->setLayout(layout);
                }
                else if (item.second->getInputData()->getTensorDesc().getDims().size() == 2) {
                    // this is image info 
                    item.second->setPrecision(Precision::FP32);
                }
            }
        }
        /**
         * @brief Do inference and return the infered request
         * @details 
         * @param data 
         * @param size 
         * @return InferRequest::Ptr 
         */
        network_output do_infer(const char* data, int size) {
            try {
                std::chrono::time_point<std::chrono::system_clock> start;
                std::chrono::time_point<std::chrono::system_clock> end;
                std::chrono::duration<double,std::milli> elapsed_mil;

                // decode out image
                start = std::chrono::system_clock::now();
                cv::Mat frame = cv::imdecode(cv::Mat(1,size,CV_8UC3, (unsigned char*) data),cv::IMREAD_UNCHANGED);
                const int width = frame.size().width;
                const int height = frame.size().height;
                end = std::chrono::system_clock::now();
                elapsed_mil = end - start;
                log->debug("Decode image in {} ms", elapsed_mil.count());

                // create new request
                start = std::chrono::system_clock::now();
                auto input_info = exe_network.GetInputsInfo();
                InferRequest::Ptr infer_request = exe_network.CreateInferRequestPtr();
                int input_width = -1, input_height = -1;
                // prepare input blob
                for (auto it = input_info.begin(); it != input_info.end(); it++) {
                        auto name = it->first;
                        auto input = it->second;
                        if (input->getTensorDesc().getDims().size() == 4) {
                                input_width = input->getTensorDesc().getDims()[2];
                                input_height = input->getTensorDesc().getDims()[3];
                        }
                }
                for (auto it = input_info.begin(); it != input_info.end(); it++) {
                    auto name = it->first;
                    auto input = it->second;
                    if (input->getTensorDesc().getDims().size() == 4) {
                            frameToBlob(frame, infer_request, name);
                    }
                    else if (input->getTensorDesc().getDims().size() == 2) {
                        Blob::Ptr input2 = infer_request->GetBlob(name);
                        float *p = input2->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
                        assert(input_width > 0);
                        assert(input_height > 0);
                        p[0] = static_cast<float>(input_width);
                        p[1] = static_cast<float>(input_height);
                    }
                }
                // do inference
                infer_request->Infer();
                end = std::chrono::system_clock::now(); // sync mode only
                elapsed_mil = end - start;
                log->debug("Create and do inference request in {} ms", elapsed_mil.count());
                return {infer_request,width,height};
            }
            catch (const cv::Exception &e) {
                // let not opencv silly exception terminate our program
                std::cerr << "Error: " << e.what() << std::endl;
                return {nullptr,-1,-1};
            }
        }
        /**
         * @brief Get the layer object, only for YOLO
         * 
         * @param name 
         * @return CNNLayerPtr 
         */
        CNNLayerPtr get_layer(const char* name) {
            return network.getLayerByName(name);
        }
    };

    /**
     * @brief SSD object detection network
     */
    class openvino_ssd : public openvino_inference_engine {
    public:
        /**
         * @brief Construct a new SSD object
         * @details This is a convinient constructor that ensembles constructing methods.
         * @param model 
         * @param device 
         * @param label 
         */
        openvino_ssd(const std::string& device, const std::string& model, const std::string& label) {
            std::ostringstream ss;
            ss << "IELog" << rand();
            std::string log_name = ss.str();
            std::cout << log_name << std::endl;
            log = spdlog::basic_logger_mt(log_name.c_str(),"logs/IE.txt");
            log->set_pattern("[%H:%M:%S %z] [%n] [%^---%L---%$] [thread %t] %v");
            log->info("Log started!");
            init_plugin(device);
            load_network(model);
            init_IO(Precision::U8, Layout::NCHW);
            load_plugin({});
            set_labels(label);         
        }
        // detection parser implementation for ssd
        std::vector<bbox> detection_parser(network_output &net_out ) final {
            std::vector<bbox> ret = {}; // return value
            try {
                std::chrono::time_point<std::chrono::system_clock> start;
                std::chrono::time_point<std::chrono::system_clock> end;
                std::chrono::duration<double,std::milli> elapsed_mil;
                start = std::chrono::system_clock::now();
                auto infer_request = net_out.infer_request;
                auto width = net_out.width;
                auto height = net_out.height;
                auto outputInfo = OutputsDataMap(network.getOutputsInfo());
                auto output_name = outputInfo.begin()->first;
                auto blob = infer_request->GetBlob(output_name);
                const float *detections = blob->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
                auto dims = blob->dims();
                const int maxProposalCount = dims[1];
                const int objectSize = dims[0];
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
                end = std::chrono::system_clock::now(); // sync mode only
                elapsed_mil = end - start;
                log->debug("Parsing network output in {} ms", elapsed_mil.count());
                return ret;
            }
            catch (const cv::Exception &e) {
                std::cerr << "Error: " << e.what() << std::endl;
                return ret;
            }
        }

        using ptr = std::shared_ptr<openvino_ssd>;

    protected:
        // IO_snaity_check for SSD
        void IO_sanity_check() final {
            // Input Blob
            auto input_info = InputsDataMap(network.getInputsInfo());
            if (input_info.size() != 1) {
                throw std::logic_error("SSD has only one input");
            }
            // Output Blob
            auto output_info = OutputsDataMap(network.getOutputsInfo());
            if (output_info.size() != 1) {
                throw std::logic_error("SSD has only one output");
            }
            auto output = output_info.begin()->second;
            const SizeVector outputDims = output->getTensorDesc().getDims();
            const int objectSize = outputDims[3];
            if (outputDims.size() != 4) {
                throw std::logic_error("Incorrect output dimensions for SSD");
            }
            if (objectSize != 7) {
                throw std::logic_error("SSD should have 7 as a last dimension");
            }
        }
    }; // class openvino_ssd

    /**
     * @brief YOLO v3 object detection network
     * 
     */
    class openvino_yolo : public openvino_inference_engine {
    public:
        /**
         * @brief Construct a new YOLO v3 object
         * @details This is a convinient constructor that ensembles all constructing methods.
         * @param model 
         * @param device 
         * @param label 
         */
        openvino_yolo(const std::string& device, const std::string& model, const std::string& label) {
            std::ostringstream ss;
            ss << "IELog" << rand();
            std::string log_name = ss.str();
            std::cout << log_name << std::endl;
            log = spdlog::basic_logger_mt(log_name.c_str(),"logs/IE.txt");
            log->set_pattern("[%H:%M:%S %z] [%n] [%^---%L---%$] [thread %t] %v");
            log->info("Log started!");
            init_plugin(device);
            load_network(model);
            init_IO(Precision::U8, Layout::NCHW);
            load_plugin({});
            set_labels(label);         
        }
        // detection parser implementation for yolo
        std::vector<bbox> detection_parser(network_output &net_out ) final {
            std::vector<bbox> ret;
            try {
                std::chrono::time_point<std::chrono::system_clock> start;
                std::chrono::time_point<std::chrono::system_clock> end;
                std::chrono::duration<double,std::milli> elapsed_mil;
                start = std::chrono::system_clock::now();
                auto infer_request = net_out.infer_request;
                auto width = net_out.width;
                auto height = net_out.height;        
                // process YOLO output, it's quite complicated though
                auto input_info = exe_network.GetInputsInfo();
                auto output_info = exe_network.GetOutputsInfo();        
                start = std::chrono::system_clock::now();
                unsigned long resized_im_h = input_info.begin()->second.get()->getDims()[0];
                unsigned long resized_im_w = input_info.begin()->second.get()->getDims()[1];
                std::vector<detection_object> objects;
                // Parsing outputs
                for (auto &output : output_info) {
                    auto output_name = output.first;
                    CNNLayerPtr layer = get_layer(output_name.c_str());
                    Blob::Ptr blob = infer_request->GetBlob(output_name);
                    parse_yolov3_output(layer, blob, resized_im_h, resized_im_w, height, width, 0.5, objects);
                }
                // Filtering overlapping boxes
                std::sort(objects.begin(), objects.end(), std::greater<detection_object>());
                for (size_t i = 0; i < objects.size(); ++i) {
                    if (objects[i].confidence == 0)
                        continue;
                    for (size_t j = i + 1; j < objects.size(); ++j)
                        if (intersection_over_union(objects[i], objects[j]) >= 0.4)
                            objects[j].confidence = 0;
                }
                // Get the bboxes
                for (auto &object : objects) {
                    bbox d;
                    if (object.confidence < 0.5)
                        continue;
                    auto label_id = object.class_id+1;
                    auto label = labels[label_id-1];
                    float confidence = object.confidence;
                    d.prop = confidence;
                    d.label_id = label_id;
                    d.label = label;
                    d.c[0] = object.xmin;
                    d.c[1] = object.ymin;
                    d.c[2] = object.xmax;
                    d.c[3] = object.ymax;
                    ret.push_back(d);
                }
                end = std::chrono::system_clock::now(); // sync mode only
                elapsed_mil = end - start;
                log->debug("Parsing network output in {} ms", elapsed_mil.count());
                return ret;
            }
            catch(const cv::Exception& e) {
                std::cerr << e.what() << '\n';
                return ret;
            }
        }

        using ptr = std::shared_ptr<openvino_yolo>;

    private:
        /**
         * @brief Yolo detection object 
         * 
         */
        struct detection_object {
            int xmin, ymin, xmax, ymax, class_id;
            float confidence;

            detection_object(double x, double y, double h, double w, int class_id, float confidence, float h_scale, float w_scale) {
                this->xmin = static_cast<int>((x - w / 2) * w_scale);
                this->ymin = static_cast<int>((y - h / 2) * h_scale);
                this->xmax = static_cast<int>(this->xmin + w * w_scale);
                this->ymax = static_cast<int>(this->ymin + h * h_scale);
                this->class_id = class_id;
                this->confidence = confidence;
            }

            bool operator <(const detection_object &s2) const {
                return this->confidence < s2.confidence;
            }
            bool operator >(const detection_object &s2) const {
                return this->confidence > s2.confidence;
            }
        };
        /**
         * @brief 
         * 
         * @param side 
         * @param lcoords 
         * @param lclasses 
         * @param location 
         * @param entry 
         * @return int 
         */
        static int entry_index(int side, int lcoords, int lclasses, int location, int entry) {
            int n = location / (side * side);
            int loc = location % (side * side);
            return n * side * side * (lcoords + lclasses + 1) + entry * side * side + loc;
        }
        /**
         * @brief IOU
         * 
         * @param box_1 
         * @param box_2 
         * @return double 
         */
        double intersection_over_union(const detection_object &box_1, const detection_object &box_2) {
            double width_of_overlap_area = fmin(box_1.xmax, box_2.xmax) - fmax(box_1.xmin, box_2.xmin);
            double height_of_overlap_area = fmin(box_1.ymax, box_2.ymax) - fmax(box_1.ymin, box_2.ymin);
            double area_of_overlap;
            if (width_of_overlap_area < 0 || height_of_overlap_area < 0)
                area_of_overlap = 0;
            else
                area_of_overlap = width_of_overlap_area * height_of_overlap_area;
            double box_1_area = (box_1.ymax - box_1.ymin)  * (box_1.xmax - box_1.xmin);
            double box_2_area = (box_2.ymax - box_2.ymin)  * (box_2.xmax - box_2.xmin);
            double area_of_union = box_1_area + box_2_area - area_of_overlap;
            return area_of_overlap / area_of_union;
        }
        /**
         * @brief Parsing YOLO output
         * 
         * @param layer 
         * @param blob 
         * @param resized_im_h 
         * @param resized_im_w 
         * @param original_im_h 
         * @param original_im_w 
         * @param threshold 
         * @param objects 
         */
        void parse_yolov3_output(const CNNLayerPtr &layer, const Blob::Ptr &blob, const unsigned long resized_im_h,
                            const unsigned long resized_im_w, const unsigned long original_im_h,
                            const unsigned long original_im_w,
                            const double threshold, std::vector<detection_object> &objects) {
            // --------------------------- Validating output parameters -------------------------------------
            if (layer->type != "RegionYolo")
                throw std::runtime_error("Invalid output type: " + layer->type + ". RegionYolo expected");
            const int out_blob_h = static_cast<int>(blob->getTensorDesc().getDims()[2]);
            const int out_blob_w = static_cast<int>(blob->getTensorDesc().getDims()[3]);
            if (out_blob_h != out_blob_w)
                throw std::runtime_error("Invalid size of output " + layer->name +
                " It should be in NCHW layout and H should be equal to W. Current H = " + std::to_string(out_blob_h) +
                ", current W = " + std::to_string(out_blob_h));
            // --------------------------- Extracting layer parameters -------------------------------------
            auto num = layer->GetParamAsInt("num");
            try { num = layer->GetParamAsInts("mask").size(); } catch (...) {}
            auto coords = layer->GetParamAsInt("coords");
            auto classes = layer->GetParamAsInt("classes");
            std::vector<float> anchors = {10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0, 198.0, 373.0, 326.0};            
            try { anchors = layer->GetParamAsFloats("anchors"); } catch (...) {}
            auto side = out_blob_h;
            int anchor_offset = 0;
            std::cout << side << std::endl;
            switch (side) {
                case 13:
                    anchor_offset = 2 * 6;
                    break;
                case 26:
                    anchor_offset = 2 * 3;
                    break;
                case 52:
                    anchor_offset = 2 * 0;
                    break;
                default:
                    throw std::runtime_error("Invalid output size");
            }
            auto side_square = side * side;
            const float *output_blob = blob->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
            // --------------------------- Parsing YOLO Region output -------------------------------------
            for (int i = 0; i < side_square; ++i) {
                int row = i / side;
                int col = i % side;
                for (int n = 0; n < num; ++n) {
                    int obj_index = entry_index(side, coords, classes, n * side * side + i, coords);
                    int box_index = entry_index(side, coords, classes, n * side * side + i, 0);
                    float scale = output_blob[obj_index];
                    if (scale < threshold)
                        continue;
                    double x = (col + output_blob[box_index + 0 * side_square]) / side * resized_im_w;
                    double y = (row + output_blob[box_index + 1 * side_square]) / side * resized_im_h;
                    double height = std::exp(output_blob[box_index + 3 * side_square]) * anchors[anchor_offset + 2 * n + 1];
                    double width = std::exp(output_blob[box_index + 2 * side_square]) * anchors[anchor_offset + 2 * n];
                    for (int j = 0; j < classes; ++j) {
                        int class_index = entry_index(side, coords, classes, n * side_square + i, coords + 1 + j);
                        float prob = scale * output_blob[class_index];
                        if (prob < threshold)
                            continue;
                        detection_object obj(x, y, height, width, j, prob,
                                static_cast<float>(original_im_h) / static_cast<float>(resized_im_h),
                                static_cast<float>(original_im_w) / static_cast<float>(resized_im_w));
                        objects.push_back(obj);
                    }
                }
            }
        }
    }; // class openvino_yolo

    /**
     * @brief 
     * 
     */
    class openvino_frcnn : public openvino_inference_engine {
    public:
        /**
         * @brief Construct a new openvino frcnn object
         * 
         * @param device 
         * @param model 
         * @param label 
         */
        openvino_frcnn(const std::string& device, const std::string& model, const std::string& label) {
            std::ostringstream ss;
            ss << "IELog" << rand();
            std::string log_name = ss.str();
            std::cout << log_name << std::endl;
            log = spdlog::basic_logger_mt(log_name.c_str(),"logs/IE.txt");
            log->set_pattern("[%H:%M:%S %z] [%n] [%^---%L---%$] [thread %t] %v");
            log->info("Log started!");
            init_plugin(device);
            load_network(model);
            init_IO(Precision::U8, Layout::NCHW);
            load_plugin({});
            set_labels(label);         
        }
        
        // frcnn detection parser implementation
        std::vector<bbox> detection_parser(network_output &net_out ) final {
            std::vector<bbox> ret = {}; // return value
            try {
                std::chrono::time_point<std::chrono::system_clock> start;
                std::chrono::time_point<std::chrono::system_clock> end;
                std::chrono::duration<double,std::milli> elapsed_mil;
                start = std::chrono::system_clock::now();
                auto infer_request = net_out.infer_request;
                auto width = net_out.width;
                auto height = net_out.height;
                auto outputInfo = OutputsDataMap(network.getOutputsInfo());
                auto output_name = outputInfo.begin()->first;
                auto blob = infer_request->GetBlob(output_name);
                const float *detections = blob->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
                auto dims = blob->dims();
                const int maxProposalCount = dims[1];
                const int objectSize = dims[0];
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
                end = std::chrono::system_clock::now(); // sync mode only
                elapsed_mil = end - start;
                log->debug("Parsing network output in {} ms", elapsed_mil.count());
                return ret;
            }
            catch (const cv::Exception &e) {
                std::cerr << "Error: " << e.what() << std::endl;
                return ret;
            }
        }

        using ptr = std::shared_ptr<openvino_frcnn>;

    protected:
        // IO sanity check for frcnn
        void IO_sanity_check() final {
            // Input Blob
            auto input_info = InputsDataMap(network.getInputsInfo());
            if (input_info.size() != 2) {
                throw std::logic_error("Current version of Faster R-CNN accepts networks having two inputs: image_tensor and image_info");
            }
            // Output Blob
            auto output_info = OutputsDataMap(network.getOutputsInfo());
            // Faster R-CNN has two head: cls_score and bbox_pred; however they are all unified in a 
            // detection output layer which do nms
            if (output_info.size() != 1) {
                throw std::logic_error("Faster R-CNN must have one outputs");
            }
            auto output = output_info.begin()->second;
            const SizeVector outputDims = output->getTensorDesc().getDims();
            const int objectSize = outputDims[3];
            if (objectSize != 7) {
                throw std::logic_error("Faster R-CNN should have 7 as a last dimension");
            }
            if (outputDims.size() != 4) {
                throw std::logic_error("Incorrect output dimensions for Faster R-CNN");
            }
        }
    };  // class openvino_frcnn
}   // namespace ie
}   // namespace st
