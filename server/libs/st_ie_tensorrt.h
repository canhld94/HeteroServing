/***************************************************************************************
 * Copyright (C) 2020 canhld@.kaist.ac.kr
 * SPDX-License-Identifier: Apache-2.0
 * @b About: This file implement of NVIDIA Tensorrt Inference Engine
 ***************************************************************************************/

#pragma once

#include <vector>
#include <string>
#include <memory>

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvUffParser.h"

#include "st_ie_base.h"
using namespace st::ie;
using namespace nvinfer1;

namespace st {
namespace ie {
    class tensorrt_inference_engine : public inference_engine {
    public: 
        /*****************************************************/
        /*  Implement of inference engine public interface   */
        /*****************************************************/
        std::vector<bbox>  run_detection (const char* data, int size) override {
            return {};
        }

        std::vector<int> run_classification (const char* data, int size) override {
            return {};
        }

        using ptr = std::shared_ptr<tensorrt_inference_engine>;
    protected:
        std::shared_ptr<ICudaEngine> engine;

        

    };

    class tensorrt_ssd : public tensorrt_inference_engine {

    };

    class tensorrt_yolo : public tensorrt_inference_engine {

    };

    class tensorrt_frcnn : public tensorrt_inference_engine {

    };
}
}


