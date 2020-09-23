/***************************************************************************************
 * Copyright (C) 2020 canhld@.kaist.ac.kr
 * SPDX-License-Identifier: Apache-2.0
 * @b About: This file provide factory method to create inference engine
 ***************************************************************************************/
#pragma once
#include <string>
#include "st_ie_base.h"
#include "st_ie_openvino.h"
#include "st_ie_tensorrt.h"


namespace st {
namespace ie {
    /**
     * @brief Factory class to create different type of inference engine
     * 
     */
    class ie_factory {
    public:
        /**
         * @brief Create a inference engine object
         * 
         * @param model_name 
         * @param device_name 
         * @param model 
         * @param label 
         * @return inference_engine::ptr 
         */
        inference_engine::ptr create_inference_engine(const std::string& model_name, 
                                                      const std::string& device_name, 
                                                      const std::string& model, 
                                                      const std::string& label) {
            auto mcode = str2mcode(model_name);
            auto dcode = str2dcode(device_name);
            if (dcode != device_code::GPU) {
                return create_openvino_engine(mcode, dcode, model, label);
            }
            else {
                return create_tensorrt_engine(mcode,model,label);
            }
        }
    private:
        enum class model_code {
            SSD = 0,
            YOLOV3 = 1,
            RCNN = 2,
        };

        enum class device_code {
            CPU = 0,
            FPGA = 1,
            GPU = 2,
        };
        // convert string to model code
        model_code str2mcode(const std::string& model_name) {
            std::string model(model_name);
            std::transform(model_name.begin(), model_name.end(), model.begin(),
                           [](unsigned char c) {return std::tolower(c);});
            if (model == "ssd") {
                return model_code::SSD;
            }
            else if (model == "yolov3") {
                return model_code::YOLOV3;
            }
            else if (model == "rcnn") {
                return model_code::RCNN;
            }
            else {
                throw st::exception::ie_not_implemented();
            }
        }
        // convert string to device code
        device_code str2dcode(const std::string& device_name) {
            std::string dev(device_name);
            std::transform(device_name.begin(), device_name.end(), dev.begin(),
                           [](unsigned char c) {return std::tolower(c);});
            if (dev == "cpu") {
                return device_code::CPU;
            }
            else if (dev == "fpga") {
                return device_code::FPGA;
            }
            else if (dev == "gpu") {
                return device_code::GPU;
            }
            else {
                throw st::exception::ie_not_implemented();
            }
        }
        // create openvino inference engine
        inference_engine::ptr create_openvino_engine(model_code type, 
                                                      device_code dev, 
                                                      const std::string& model, 
                                                      const std::string& label) {
            std::string plugin;
            switch (dev) {
            case device_code::CPU :
                plugin = "CPU";
                break;
            case device_code::FPGA :
                plugin = "HETERO:FPGA,CPU";
                break;
            default:
                throw st::exception::ie_not_implemented();
                break;
            }
            switch (type) {
            case model_code::SSD :
                return  std::make_shared<openvino_ssd>(plugin,model,label);
                break;
            case model_code::YOLOV3 :
                return std::make_shared<openvino_yolo>(plugin,model,label);
                break;
            case model_code::RCNN :
                return std::make_shared<openvino_frcnn>(plugin,model,label);
                break;
            default :
                throw st::exception::ie_not_implemented();
            }
        }
        // create tensorrt inference engine object
        inference_engine::ptr create_tensorrt_engine(model_code type,
                                                     const std::string& model, 
                                                     const std::string& label) {
            switch (type) {
            // case model_code::SSD :
            //     return  std::make_shared<ssd>(plugin,model,label);
            //     break;
            // case model_code::YOLOV3 :
            //     return std::make_shared<yolo>(plugin,model,label);
            //     break;
            // case model_code::RCNN :
            //     return std::make_shared<faster_r_cnn>(plugin,model,label);
            //     break;
            default :
                throw st::exception::ie_not_implemented();
            }
        }
    };
}
}