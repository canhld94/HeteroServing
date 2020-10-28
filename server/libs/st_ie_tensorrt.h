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
#include "st_logging.h"

using namespace nvinfer1;
using namespace nvuffparser;
using namespace nvonnxparser;


namespace st {
namespace ie {

class trtlogger glogger;


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
  trt_shared_ptr<ICudaEngine> engine; // static
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
      console->info("Engine support {} max batch size", engine->getMaxBatchSize()); 
      console->info("Engine require {}  bytes for activations",engine->getDeviceMemorySize()); 
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
    console->debug("Create unique buffer with context");
    // prepare input and output buffer, just like in ovn
    // but here we need to handle it ourself, i.e. allocate and dealocate the
    // input and output blob memory --> RAII buffer
    cv::Mat frame = cv::imdecode(
        cv::Mat(1, size, CV_8UC3, (unsigned char*)data), cv::IMREAD_UNCHANGED);
    iobuf->set_im_size(frame.size().width, frame.size().height);
    for (int ix = 0; ix < engine->getNbBindings(); ++ix) {
      if (engine->bindingIsInput(ix)) {
        iobuf->fill_input(ix, frame);
      }
    }
    iobuf->memcpy_input_htod();
    auto bindings = iobuf->get_bindings();
    console->debug("Invoking inferencing");
    context->execute(1,bindings.data());
    iobuf->memcpy_output_dtoh();
    // should be able to get the output of network from this request
    // may be we should return a buffer? --> yes
    return iobuf;
  }
};

class tensorrt_ssd : public tensorrt_inference_engine {
  public:
  tensorrt_ssd(const std::string &serialized_model, const std::string &label) {
    // parser with models here
    build_engine(serialized_model);
    set_labels(label);
  }
  std::vector<bbox> detection_parser (std::unique_ptr<buffer_manager>&& _iobuf) final {
    auto iobuf = std::move(_iobuf);
    std::vector<bbox> ret;
    // get the right output
    int ix = 0;
    for (ix = 0; ix < engine->getNbBindings(); ++ix) {
      if (!engine->bindingIsInput(ix) && engine->getBindingDimensions(ix).d[2] == 7) break;
    }
    float* detections = (float*) iobuf->get_buffer(true,ix);
    auto dims = engine->getBindingDimensions(ix);
    const int maxProposalCount = dims.d[1];
    const int objectSize = dims.d[2];
    std::cout << maxProposalCount << " " << objectSize << std::endl; 
    auto sz = iobuf->get_im_size();
    const int width = sz.first, height = sz.second;
      for (int i = 0; i < maxProposalCount; i++) {
        float image_id = detections[i * objectSize + 0];
        for (int j = 0; j < 7; ++j) {
          std::cout << detections[i * objectSize + j] << " ";
        }
        std::cout << std::endl;
        // continue;
        if (image_id < 0) {
          break;
        }
        float confidence = detections[i * objectSize + 2];
        auto label_id = static_cast<int>(detections[i * objectSize + 1]);
        if (label_id <= 0) continue;
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
};

class tensorrt_yolo : public tensorrt_inference_engine {};

class tensorrt_frcnn : public tensorrt_inference_engine {};
}  // namespace ie
}  // namespace st
