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
#include "st_utils.h"
#include "st_logging.h"

using namespace st::log;

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
  virtual inference_engine::ptr create(JSON& conf) = 0;
};

/**
 * @brief Creator for intel cpu
 * 
 */
class intel_cpu_inference_engine_creator : public inference_engine_creator {
 public:
  inference_engine::ptr create(JSON& conf) final {
    std::string plugin = "CPU";
    // get the model spec, we know that its size must be positive
    auto& model = conf.get_child("model");
    const std::string& name = model.get<std::string>("name");
    const std::string& graph = model.get<std::string>("graph");
    const std::string& label = model.get<std::string>("label");
    return create_openvino_engine(plugin, name, graph, label);
  }
};
/**
 * @brief Creator for intel fpga
 * 
 */
class intel_fpga_inference_engine_creator : public inference_engine_creator {
 public:
  inference_engine::ptr create(JSON& conf) final {
    std::string plugin = "HETERO:FPGA,CPU";
    // get the model spec, we know that its size must be positive
    auto& model = conf.get_child("model");
    const std::string& name = model.get<std::string>("name");
    const std::string& graph = model.get<std::string>("graph");
    const std::string& label = model.get<std::string>("label");
    return create_openvino_engine(plugin, name, graph, label);
  }
};
/**
 * @brief Creator for nvidia gpu 
 * 
 */
class nvidia_gpu_inference_engine_creator : public inference_engine_creator {
 public:
  inference_engine::ptr create(JSON& conf) final {
    // get the model spec, we know that its size must be positive
    auto& model = conf.get_child("model");
    const std::string& name = model.get<std::string>("name");
    const std::string& graph = model.get<std::string>("graph");
    const std::string& label = model.get<std::string>("label");
    return create_tensorrt_engine(name, graph, label);
  }
};
/**
 * @brief creator for xilinx device
 * 
 */
class xilinx_fpga_inference_engine_creator : public inference_engine_creator {
 public:
  inference_engine::ptr create(JSON& conf) final {
    throw std::logic_error("Xilinx FPGA inference engine has not yet implemented");
  }
};

/**
 * @brief Factory class to create different type of inference engine
 *
 */
class ie_factory {
 public:
  typedef std::unordered_map<std::string, inference_engine_creator*> factory_map;
  ie_factory() {
    // register your plugin here and do not modify anything outside of this
    Register("intel cpu", new intel_cpu_inference_engine_creator());
    Register("intel fpga", new intel_fpga_inference_engine_creator());
    Register("nvidia gpu", new nvidia_gpu_inference_engine_creator());
  }
  /**
   * @brief Create a inference engine object
   *
   * @param conf JSON object that hold parameter to create the inference engine
   * @return inference_engine::ptr
   */
  inference_engine::ptr create_inference_engine(JSON& conf) {
    const std::string& device = conf.get<std::string>("device");
    auto &registry = get_registry();
    auto it = registry.find(device);
    if (it == registry.end()) {
      throw std::logic_error("Creator of [" + device + "] not found in registry");
    }
    auto creator = it->second;
    return creator->create(conf);
  }
  /**
   * @brief Registor a device with a creator, should be done at compile time by now
   * 
   * @param device 
   * @param creator 
   */
  void Register(std::string device, inference_engine_creator* creator) {
    auto &registry = get_registry();
    auto it = registry.find(device);
    if (it != registry.end()) {
      throw std::logic_error("Creator of [" + device + "] has already been in registry");
    }
    ie_log->info("Adding creator of [{}] to the registry", device);
    registry.insert({device, creator});
  }

 private:
  static factory_map& get_registry() {
    static factory_map registry;
    return registry;
  }
};
}
}