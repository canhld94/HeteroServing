

#pragma once

#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>
#include <NvInfer.h>


using namespace InferenceEngine;
using namespace nvinfer1;

namespace st {
namespace ie {
/**
 * @brief Bouding box object
 * @details Basic bouding box object that can use in any recognition task
 */
struct bbox {
  int label_id;      //!< label id
  std::string label; //!< class name
  float prop;        //!< confidence score
  int c[4];          //!< coordinates of bounding box
};
/**
* @brief Sets image data stored in cv::Mat object to a given Blob object.
* @param orig_image - given cv::Mat object with an image data.
* @param blob - Blob object which to be filled by an image data.
* @param batchIndex - batch index of an image inside of the blob.
*/
template <typename T>
void matU8ToBlob(const cv::Mat &orig_image, Blob::Ptr &blob,
                 int batchIndex = 0) {
  SizeVector blobSize = blob->getTensorDesc().getDims();
  const size_t width = blobSize[3];
  const size_t height = blobSize[2];
  const size_t channels = blobSize[1];
  T *blob_data = blob->buffer().as<T *>();

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
 * @brief Map opencv map to blob
 *
 * @param frame
 * @param inferRequest
 * @param inputName
 */
void frameToBlob(const cv::Mat &frame, InferRequest::Ptr &inferRequest,
                 const std::string &inputName) {
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
class logger : public ILogger {
  void log(Severity serverity, const char *msg) override {
    if (serverity != Severity::kINFO) {
      std::cout << msg << std::endl;
    }
  }
};
/**
 * @brief Dealocator to tensorRT objects
 *
 */
struct trt_object_deleter {
  template <typename T> void operator() (T *obj) const {
    if (obj) {
      obj->destroy();
    }
  }
};

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

template <class T>
using trt_unique_ptr = std::unique_ptr<T, trt_object_deleter>;

template <class T> using trt_shared_ptr = std::shared_ptr<T>;
}
}