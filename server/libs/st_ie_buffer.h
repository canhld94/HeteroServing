

#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <exception>
#include <memory>
#include <opencv2/opencv.hpp>
#include <st_ie_half.h>
#include <string>
#include <vector>

using namespace nvinfer1;
using namespace half_float;

namespace st {
namespace ie {
/**
 * @brief RAII generic buffer
 *
 */
class generic_buffer {
public:
  generic_buffer() : data(nullptr), size(0), nbytes(0){};
  virtual ~generic_buffer(){};
  generic_buffer(generic_buffer &&other)
      : data(other.data), size(other.size), nbytes(other.nbytes) {
    other.data = nullptr;
    other.size = 0;
    other.nbytes = 0;
  }
  generic_buffer &operator=(generic_buffer &&rhs) {
    if (&rhs != this) {
      data = rhs.data;
      size = rhs.size;
      nbytes = rhs.nbytes;
      rhs.data = nullptr;
      rhs.size = 0;
      rhs.nbytes = 0;
    }
    return this;
  }
  void *get_data() { return data; }
  const void *get_const_data() const { return data; }
  size_t get_size() { return size; }
  size_t get_nbytes() { return nbytes; }
  virtual bool fill_from_u8mat(cv::Mat &mat){};
  using ptr = std::unique_ptr<generic_buffer>;

private:
  void *data;
  size_t size;
  size_t nbytes;
};
/**
 * @brief RAII generic buffer implementation
 * @tparam Allocator
 * @tparam Deleter
 */
template <class Allocator, class Deleter, typename T>
class flat_buffer : public generic_buffer {
public:
  using value_type = T;
  flat_buffer() { generic_buffer(); }
  flat_buffer(size_t _size)
      : data(nullptr), size(_size), nbytes(size * sizeof(T)) {
    if (!alloc_fn(&data, nbytes)) {
      throw std::bad_alloc();
    }
  }
  ~flat_buffer() override { free_fn(data); }
  bool fill_from_u8mat(cv::Mat &orig_image, Dims4 &dim,
                       int batch_id = 0) override {
    assert(data != nullptr);
    // default NCHW
    int batch_size = dim.d[0], channels = dim.d[1], height = dim.d[2],
        width = dim.d[3];
    // assert(size == batch_size * channels * height * width);
    // resize the mat data to chw
    cv::Mat resized_image(orig_image);
    if (height != orig_image.size().height ||
        width != orig_image.size().width) {
      cv::resize(orig_image, resized_image, cv::Size(w, h));
    }
    // 3 channel, nchw
    // fill the buffer in order C->H->W
    size_t batch_offset = batch_id * c * h * w;
    for (size_t c = 0; c < channels; ++c) {
      for (size_t h = 0; h < height; ++h) {
        for (size_t w = 0; h < width; ++w) {
          data[batch_offset + c * width * height + h * width + w] =
              resized_image.at<Vec3b>(h, w)[c];
        }
      }
    }
    return true;
  }

private:
  static Allocator alloc_fn;
  static Deleter free_fn;
};
struct host_allocator {
  bool operator()(void **data, size_t size) {
    *data = malloc(size);
    return data != nullptr;
  }
};
struct host_deleter {
  void operator()(void *data) { free(data); }
};
struct gpu_allocator {
  bool operator()(void **data, size_t size) { return cudaMalloc(data, size); }
};
struct gpu_deleter {
  bool operator()(void *data) { cudaFree(data); }
};
template <typename T>
using host_buffer = flat_buffer<host_allocator, host_deleter, T>;
using gpu_buffer = flat_buffer<gpu_allocator, gpu_deleter, T>;

class buffer_factory {
  generic_buffer *create_buffer_ptr(bool is_host, DataType Tp, size_t size) = 0;
}

class host_buffer_factory : public buffer_factory {
  generic_buffer *create_buffer_ptr(bool is_host, DataType Tp,
                                    size_t size) override {
    switch (Tp) {
    case (Datatype::kFLOAT):
      return new host_buffer<float>(size);
      break;
    case (DataType::kINT32):
      return new host_buffer<int>(size);
      break;
    case (DataType::kHALF):
      retur new host_buffer<half>(size);
      break;
    default:
      return nullptr;
    }
  }
}

class gpu_buffer_factory : public buffer_factory {
  generic_buffer *create_buffer_ptr(bool is_host, DataType Tp,
                                    size_t size) override {
    switch (Tp) {
    case (Datatype::kFLOAT):
      return new gpu_buffer<float>(size);
      break;
    case (DataType::kINT32):
      return new gpu_buffer<int>(size);
      break;
    case (DataType::kHALF):
      return new gpu_buffer<half>(size);
      break;
    default:
      return nullptr;
    }
  }
}

/**
 * @brief TensorRT IO blob
 * @details every IO blob in trt have two buffer: buffer on host
 * and buffer on gpu
 *
 */
struct blob {
  generic_buffer::ptr host_mem;
  generic_buffer::ptr gpu_mem;
  blob(): host_mem(nullptr), gpu_mem(nullptr) {};
  blob(generic_buffer::ptr &&_host_mem, generic_buffer&& _gpu_mem):
      host_mem(std::move(_host_mem)), gpu_mem(std::move(_gpu_mem)) {};
  using ptr = std::unique_ptr<blob>;
}

/**
 * @brief manage the input and output buffer of the network
 * @details this is quite similar to blob object in openvino
 */
class buffer_manager {
public:
  buffer_manager(trt_shared_ptr<ICudaEngine> _engine, int _batch_size) :
                engine(_engine), batch_size(_batch_size) {
    unique_ptr<buffer_factory> host_factory{new host_buffer_factory};
    unique_ptr<buffer_factory> gpu_factory{new gpu_buffer_factory};
    // fill the blob vector
    for (int i = 0; i < engine->getNbBindings(); ++i) {
      
    }

  }
  // iterate all input blobs, copy data from host to gpu
  void memcpy_input_htod() {}
  // iterate all output blobs, copy data from gpu to host
  void memcpy_output_dtoh() {}

private:
  // hold all input and output blobs
  trt_shared_ptr<ICudaEngine> engine;
  int batch_size;
  std::vector<blob::ptr> blobs;
};
}
}