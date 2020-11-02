/***************************************************************************************
 * Copyright (C) 2020 canhld@.kaist.ac.kr
 * SPDX-License-Identifier: Apache-2.0
 * @b About: This file implement RAII buffers when runing tensorrt inference engine
 ***************************************************************************************/

#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <inference_engine.hpp>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "st_message_queue.h"

using namespace InferenceEngine;
using namespace nvinfer1;

namespace st {
namespace ie {
/**
 * @brief Bouding box object
 * @details Basic bouding box object that can use in any recognition task
 */
struct bbox {
  int label_id;       //!< label id
  std::string label;  //!< class name
  float prop;         //!< confidence score
  int c[4];           //!< coordinates of bounding box
};

/**
 * @brief Message template that can hold object detection result
 *
 * @tparam simple_bell
 */
template <class simple_bell>
using obj_detection_msg =
    st::sync::message<const char*, int, std::vector<bbox>*, simple_bell>;

/**
 * @brief Object detection message queue that can be used to exchange object
 * detection message
 *
 * @tparam simple_bell
 */
template <class simple_bell>
using object_detection_mq =  st::sync::blocking_queue<obj_detection_msg<simple_bell>>;

/**
* @brief Sets image data stored in cv::Mat object to a given Blob object.
* @param orig_image - given cv::Mat object with an image data.
* @param blob - Blob object which to be filled by an image data.
* @param batchIndex - batch index of an image inside of the blob.
*/
template <typename T>
void matU8ToBlob(const cv::Mat& orig_image, Blob::Ptr& blob,
                 int batchIndex = 0) {
  SizeVector blobSize = blob->getTensorDesc().getDims();
  const size_t width = blobSize[3];
  const size_t height = blobSize[2];
  const size_t channels = blobSize[1];
  // std::cout << width << " - " << height << std::endl;
  T* blob_data = blob->buffer().as<T*>();

  cv::Mat resized_image(orig_image);
  if (static_cast<int>(width) != orig_image.size().width ||
      static_cast<int>(height) != orig_image.size().height) {
    cv::resize(orig_image, resized_image, cv::Size(width, height));
  }

  int batchOffset = batchIndex * width * height * channels;

  for (size_t c = 0; c < channels; c++) {
    for (size_t h = 0; h < height; h++) {
      for (size_t w = 0; w < width; w++) {
        blob_data[batchOffset + c * width * height + h * width + w] =
            resized_image.at<cv::Vec3b>(h, w)[c];
      }
    }
  }
}
/**
 * @brief Map opencv map to openvino blob
 *
 * @param frame
 * @param inferRequest
 * @param inputName
 */
void frameToBlob(const cv::Mat& frame, InferRequest::Ptr& inferRequest,
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
 * @brief Simple logging that will just print to screen when error
 *
 */
class trtlogger : public ILogger {
  void log(Severity serverity, const char* msg) override {
    if (serverity != Severity::kINFO) {
      std::cout << msg << std::endl;
    }
  }
};
/**
 * @brief return size of type corresponding to tensorrt type
 * 
 * @param Tp 
 * @return int 
 */
int trt_type_size(DataType Tp) {
  switch (Tp) {
    case DataType::kFLOAT:
      return 4;
      break;
    case DataType::kHALF:
      return 2;
      break;
    case DataType::kINT32:
      return 4;
      break;
    case DataType::kINT8:
      return 1;
      break;
    case DataType::kBOOL:
      return 1;
      break;
    default:
      return 0;
      break;
  }
  return 0;
}

/**
 * @brief Dealocator to tensorRT objects
 *
 */
struct trt_object_deleter {
  template <typename T>
  void operator()(T* obj) const {
    if (obj) {
      obj->destroy();
    }
  }
};

// Smart pointer for tensorrt object

template <class T>
using trt_unique_ptr = std::unique_ptr<T, trt_object_deleter>;

template <class T>
using trt_shared_ptr = std::shared_ptr<T>;

template<class T>
inline trt_shared_ptr<T> trt_make_shared(T* ptr) {
  return trt_shared_ptr<T>(ptr, trt_object_deleter());
}

}
}