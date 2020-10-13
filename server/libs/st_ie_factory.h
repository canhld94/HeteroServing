/***************************************************************************************
 * Copyright (C) 2020 canhld@.kaist.ac.kr
 * SPDX-License-Identifier: Apache-2.0
 * @b About: This file provide factory method to create inference engine
 ***************************************************************************************/
#pragma once
#include <memory>
#include <string>
#include <unordered_map>
#include "st_ie_base.h"
#include "st_ie_openvino.h"
#include "st_ie_tensorrt.h"

namespace st {
namespace ie {

enum class model_code {
  SSD = 0,
  YOLOV3 = 1,
  RCNN = 2,
};
// convert string to model code
model_code str2mcode(const std::string& model_name) {
  std::string model;
  std::transform(model_name.begin(), model_name.end(),
                 std::back_inserter(model),
                 [](unsigned char c) { return std::tolower(c); });
  if (model == "ssd") {
    return model_code::SSD;
  } else if (model == "yolov3") {
    return model_code::YOLOV3;
  } else if (model == "rcnn") {
    return model_code::RCNN;
  } else {
    throw std::logic_error("Model [" + model_name + "] has not yet implemented");
  }
}

// create openvino inference engine
inference_engine::ptr create_openvino_engine(const std::string& plugin,
                                             const std::string& model_name,
                                             const std::string& model,
                                             const std::string& label) {
  auto type = str2mcode(model_name);
  switch (type) {
    case model_code::SSD:
      return std::make_shared<openvino_ssd>(plugin, model, label);
      break;
    case model_code::YOLOV3:
      return std::make_shared<openvino_yolo>(plugin, model, label);
      break;
    case model_code::RCNN:
      return std::make_shared<openvino_frcnn>(plugin, model, label);
      break;
    default:
      return nullptr;
  }
}
// create tensorrt inference engine object
inference_engine::ptr create_tensorrt_engine(const std::string& model_name,
                                             const std::string& model,
                                             const std::string& label) {
  auto type = str2mcode(model_name);
  switch (type) {
    case model_code::SSD:
      return std::make_shared<tensorrt_ssd>(model, label);
      break;
    // case model_code::YOLOV3 :
    //     return std::make_shared<yolo>(plugin,model,label);
    //     break;
    // case model_code::RCNN :
    //     return std::make_shared<faster_r_cnn>(plugin,model,label);
    //     break;
    default:
      return nullptr;
  }
}

/**
 * @brief Main creator class
 *
 */
class inference_engine_creator {
 public:
  virtual inference_engine::ptr create(const std::string& model_name,
                                       const std::string& model,
                                       const std::string& label) = 0;
};

class cpu_inference_engine_creator : public inference_engine_creator {
 public:
  inference_engine::ptr create(const std::string& model_name,
                               const std::string& model,
                               const std::string& label) final {
    std::string plugin = "CPU";
    return create_openvino_engine(plugin, model_name, model, label);
  }
};

class intel_fpga_inference_engine_creator : public inference_engine_creator {
 public:
  inference_engine::ptr create(const std::string& model_name,
                               const std::string& model,
                               const std::string& label) final {
    std::string plugin = "HETERO:FPGA,CPU";
    return create_openvino_engine(plugin, model_name, model, label);
  }
};

class nvidia_gpu_inference_engine_creator : public inference_engine_creator {
 public:
  inference_engine::ptr create(const std::string& model_name,
                               const std::string& model,
                               const std::string& label) final {
    return create_tensorrt_engine(model_name, model, label);
  }
};

class xilinx_gpu_inference_engine_creator : public inference_engine_creator {
 public:
  inference_engine::ptr create(const std::string& model_name,
                               const std::string& model,
                               const std::string& label) final {
    throw std::logic_error("Xilinx FPGA inference engine has not yet implemented");
  }
};

/**
 * @brief Factory class to create different type of inference engine
 *
 */
class ie_factory {
 public:
  ie_factory() {
    Register("intel cpu", new cpu_inference_engine_creator());
    Register("intel fpga", new intel_fpga_inference_engine_creator());
    Register("nvidia gpu", new nvidia_gpu_inference_engine_creator());
  }
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
    auto it = registry.find(device_name);
    if (it == registry.end()) {
      throw std::logic_error("Creator of [" + device_name + "] not found in registry");
    }
    auto creator = it->second;
    return creator->create(model_name, model, label);
  }

 private:
  std::unordered_map<std::string, inference_engine_creator*> registry;
  void Register(std::string plugin, inference_engine_creator* creator) {
    auto it = registry.find(plugin);
    if (it != registry.end()) {
      throw std::logic_error("Creator of [" + plugin + "] has already been in registry");
    }
    registry.insert({plugin, creator});
  }
};
}
}