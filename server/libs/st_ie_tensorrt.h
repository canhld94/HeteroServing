/***************************************************************************************
 * Copyright (C) 2020 canhld@.kaist.ac.kr
 * SPDX-License-Identifier: Apache-2.0
 * @b About: This file implement of NVIDIA Tensorrt Inference Engine
 ***************************************************************************************/

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <fstream>
#include <exception>
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvUffParser.h>
#include <NvOnnxParser.h>

#include "st_ie_base.h"
#include "st_ie_buffer.h"
#include "st_ie_common.h"

using namespace st::ie;
using namespace nvinfer1;
using namespace nvuffparser;
using namespace nvonnxparser;

class logger glogger;

namespace st {
namespace ie {

class tensorrt_inference_engine : public inference_engine {
 public:
  /*****************************************************/
  /*  Implement of inference engine public interface   */
  /*****************************************************/
  std::vector<bbox> run_detection(const char* data, int size) final {
    auto iobuf = do_infer(data,size);
    return detection_parser(std::move(iobuf));
  }

  std::vector<int> run_classification(const char* data, int size) final {
    return {};
  }

  virtual std::vector<bbox> detection_parser(
      std::unique_ptr<buffer_manager>&& iobuf) {
    return {};
  }

  virtual std::vector<int> classification_parser(
      std::unique_ptr<buffer_manager>&& iobuf) {
    return {};
  }

  using ptr = std::shared_ptr<tensorrt_inference_engine>;

 protected:
  // unique_ptr
  // --> pass to function --> always move
  // --> return from function --> not neccessary if copy elision is possible

  virtual void build_engine(std::string serialized_model) {
    // create builder
    auto runtime = createInferRuntime(glogger);
    assert(runtime);
    initLibNvInferPlugins(&glogger, "");
    std::ifstream fi(serialized_model.c_str(),std::ios::binary | std::ios::ate);
    auto sz = fi.tellg();
    fi.seekg(0,std::ios::beg);
    std::vector<char> buf(sz);
    if (fi.read(buf.data(),sz)) {
      engine = trt_make_shared<ICudaEngine>(
          runtime->deserializeCudaEngine(buf.data(), sz, nullptr));
      assert(engine);
      std::cout << "Engine support " << engine->getMaxBatchSize() << " max batch size" << std::endl; 
      std::cout << "Engine require " << engine->getDeviceMemorySize() << " bytes for activations" << std::endl; 
    }
    else {
      throw std::logic_error("Cannot load serialized model from " + serialized_model);
    }
  }

  virtual std::unique_ptr<buffer_manager> do_infer(const char* data, int size) {
    // create execution context with memory allocation for all
    // activations (laten features)
    trt_shared_ptr<IExecutionContext> context = trt_make_shared<IExecutionContext>(engine->createExecutionContext());
    assert(context);
    std::unique_ptr<buffer_manager> iobuf{new buffer_manager(context)};
    std::cout << "Create unique buffer with context" << std:: endl;
    // prepare input and output buffer, just like in ovn
    // but here we need to handle it ourself, i.e. allocate and dealocate the
    // input and output blob memory --> RAII buffer
    cv::Mat frame = cv::imdecode(
        cv::Mat(1, size, CV_8UC3, (unsigned char*)data), cv::IMREAD_UNCHANGED);
    for (int ix = 0; ix < engine->getNbBindings(); ++ix) {
      if (engine->bindingIsInput(ix)) {
        iobuf->fill_input(ix, frame);
      }
    }
    iobuf->memcpy_input_htod();
    auto bindings = iobuf->get_bindings();
    context->execute(1,bindings.data());
    // should be able to get the output of network from this request
    // may be we should return a buffer? --> yes
    return iobuf;
  }

  trt_shared_ptr<ICudaEngine> engine;
};

class tensorrt_ssd : public tensorrt_inference_engine {
  public:
  tensorrt_ssd(const std::string &serialized_model, const std::string &label) {
    // parser with models here
    build_engine(serialized_model);
    set_labels(label);
  }
  std::vector<bbox> detection_parser (std::unique_ptr<buffer_manager>&& iobuf) final {
    return {};
  }
};

class tensorrt_yolo : public tensorrt_inference_engine {};

class tensorrt_frcnn : public tensorrt_inference_engine {};
}  // namespace ie
}  // namespace st
