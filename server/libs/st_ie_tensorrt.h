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
    class logger : public ILogger {
        void log(Severity serverity, const char *msg) {
            if (serverity != Severity::kINFO) {
                std::cout << msg << std::endl;
            }
        }
    } glogger; // global logger

    struct trt_object_deleter {
    template <typename T>
        void operator()(T* obj) const {
            if (obj)
            {
                obj->destroy();
            }
        }
    };
    template <class T>
    using trt_unique_ptr = std::unique_ptr<T,trt_object_deleter>;

    // template <class T>
    // using trt_shared_ptr = std::shared_ptr<T,trt_object_deleter>;

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

        virtual std::vector<bbox> detection_parser (trt_unique_ptr<IExecutionContext> infer_request) {
            return {};
        }

        virtual std::vector<int> classification_parser (trt_unique_ptr<IExecutionContext> infer_request) {
            return {};
        }

        using ptr = std::shared_ptr<tensorrt_inference_engine>;
    protected:
        // unique_ptr
        // --> pass to function --> always move
        // --> return from function --> not neccessary if copy elision is possible
        trt_unique_ptr<ICudaEngine> engine;

        virtual void build_engine(std::string uff_file) {
            // create builder
            auto builder = createInferBuilder(glogger);
            // create network
            trt_unique_ptr<INetworkDefinition> network{builder->createNetworkV2(0U)};
            // create parser
            auto parser = nvuffparser::createUffParser();
            // declare the network input and output
            // so we must know the input size of the network
            // why don't we interpret it from the uff file? 
            // if then, we need specify it in the server config (json file)
            // input name, input size (NCHW), output name 

            // parse the network
            parser->parse(uff_file.c_str(), *network, DataType::kFLOAT);
            // create the cuda engine
            trt_unique_ptr<IBuilderConfig> config{builder->createBuilderConfig()};
            config->setMaxWorkspaceSize(1<<20); // 2GB
            engine = trt_unique_ptr<ICudaEngine>{builder->buildEngineWithConfig(*network,*config)};
        }

        virtual trt_unique_ptr<IExecutionContext> do_infer(const char* data, int size) {
            // create execution context with memory allocation for all 
            // activations (laten features)
            trt_unique_ptr<IExecutionContext> infer_request{engine->createExecutionContext()};

            // prepare input and output buffer, just like in ovn
            // but here we need to handle it ourself, i.e. allocate and dealocate the input and
            // output blob memory --> RAII buffer

            // should be able to get the output of network from this request
            // may be we should return a buffer?
            return infer_request;
        }

    };

    class tensorrt_ssd : public tensorrt_inference_engine {

    };

    class tensorrt_yolo : public tensorrt_inference_engine {

    };

    class tensorrt_frcnn : public tensorrt_inference_engine {

    };
}
}


