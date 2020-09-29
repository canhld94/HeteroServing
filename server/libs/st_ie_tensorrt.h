/***************************************************************************************
 * Copyright (C) 2020 canhld@.kaist.ac.kr
 * SPDX-License-Identifier: Apache-2.0
 * @b About: This file implement of NVIDIA Tensorrt Inference Engine
 ***************************************************************************************/

#pragma once

#include <vector>
#include <string>
#include <memory>

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvUffParser.h>

#include "st_ie_base.h"
#include "st_ie_common.h"
#include "st_ie_buffer.h"

using namespace st::ie;
using namespace nvinfer1;
using namespace nvuffparser;

class logger glogger;

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

        virtual void build_engine(std::string uff_file, DataType Tp) {
            // create builder
            auto builder = trt_unique_ptr<IBuilder>{createInferBuilder(glogger)};
            assert(builder);
            // create network
            trt_unique_ptr<INetworkDefinition> network{builder->createNetworkV2(0U)};
            // create parser
            auto parser = trt_unique_ptr<IUffParser>{nvuffparser::createUffParser()};
            assert(parser);
            // declare the network input and output
            // so we must know the input size of the network
            // why don't we interpret it from the uff file? 
            // if then, we need specify it in the server config (json file)
            // input name, input size (NCHW), output name 
            for (auto &node : input_node) {
                parser->registerInput(node.c_str(),DimsCHW(C,H,W),UffInputOrder::kNCHW);
            }
            for (auto &node : output_nodes) {
                parser->registerOutput(node.c_str());
            }
            // parse the network
            parser->parse(uff_file.c_str(), *network, Tp);
            // create the cuda engine
            trt_unique_ptr<IBuilderConfig> config{builder->createBuilderConfig()};
            assert(config);
            builder->setMaxBatchSize(N);
            config->setMaxWorkspaceSize(1<<20); // 2GB
            // enable fp16
            if (fp16) {
                config->setFlag(BuilderFlag::kFP16);
            }
            engine = trt_shared_ptr<ICudaEngine>{builder->buildEngineWithConfig(*network,*config),trt_object_deleter()};
            assert(engine);
        }

        virtual std::unique_ptr<buffer_manager> do_infer(const char* data, int size) {
            // create execution context with memory allocation for all 
            // activations (laten features)
            trt_unique_ptr<IExecutionContext> infer_request{engine->createExecutionContext()};
            std::unique_ptr<buffer_manager> iobuf{new buffer_manager(engine,N)};
            // prepare input and output buffer, just like in ovn
            // but here we need to handle it ourself, i.e. allocate and dealocate the input and
            // output blob memory --> RAII buffer
            cv::Mat frame = cv::imdecode(cv::Mat(1,size,CV_8UC3, (unsigned char*) data),cv::IMREAD_UNCHANGED);
            // should be able to get the output of network from this request
            // may be we should return a buffer? --> yes
            return iobuf;
        }

        trt_shared_ptr<ICudaEngine> engine;
        int N,C,H,W;                                // input shape of the models
        bool fp16;
        std::vector<std::string> input_node;
        std::vector<std::string> output_nodes;
    };

    class tensorrt_ssd : public tensorrt_inference_engine {

    };

    class tensorrt_yolo : public tensorrt_inference_engine {

    };

    class tensorrt_frcnn : public tensorrt_inference_engine {

    };
}
}


