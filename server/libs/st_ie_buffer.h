

#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <st_ie_half.h>
#include <exception>
#include <memory>
#include <opencv2/opencv.hpp>
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
  generic_buffer(generic_buffer&& other)
      : data(other.data), size(other.size), nbytes(other.nbytes) {
    other.data = nullptr;
    other.size = 0;
    other.nbytes = 0;
  }
  generic_buffer& operator=(generic_buffer&& rhs) {
    if (&rhs != this) {
      data = rhs.data;
      size = rhs.size;
      nbytes = rhs.nbytes;
      rhs.data = nullptr;
      rhs.size = 0;
      rhs.nbytes = 0;
    }
    return *this;
  }
  void* get_data() const { return data; }
  const void* get_const_data() const { return data; }
  size_t get_size() { return size; }
  size_t get_nbytes() { return nbytes; }
  virtual void fill_from_mat(cv::Mat& orig_image, Dims& dim, int batch_id) {};
  using ptr = std::unique_ptr<generic_buffer>;

 protected:
  void* data;
  size_t size;
  size_t nbytes;
};

struct host_allocator {
  bool operator()(void** data, size_t size) {
    *data = malloc(size);
    return data != nullptr;
  }
};
struct host_deleter {
  void operator()(void* data) { free(data); }
};
struct gpu_allocator {
  bool operator()(void** data, size_t size) { return cudaMalloc(data, size); }
};
struct gpu_deleter {
  void operator()(void* data) { cudaFree(data); }
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
  flat_buffer(size_t _size) {
    size = _size, nbytes = size * sizeof(T);
    if (!alloc_fn(&data, nbytes)) {
      throw std::bad_alloc();
    }
  }
  ~flat_buffer() override {
    // debug
    std::cout << "buffer free" << std::endl;
    free_fn(data); 
    }
  void fill_from_mat(cv::Mat& orig_image, Dims& dim,
                     int batch_id) override {
    assert(data != nullptr);
    assert(dim.nbDims == 4);
    // default NCHW
    size_t batch_size = dim.d[0], channels = dim.d[1], height = dim.d[2],
        width = dim.d[3];
    assert(batch_size == 1);
    assert(size == batch_size * channels * height * width);
    // resize the mat data to chw
    cv::Mat resized_image(orig_image);
    if (static_cast<int>(height) != orig_image.size().height ||
        static_cast<int>(width) != orig_image.size().width) {
      cv::resize(orig_image, resized_image, cv::Size(width, height));
    }
    // 3 channel, nchw
    // fill the buffer in order C->H->W
    size_t batch_offset = batch_id * channels * height * width;
    auto typed_data = (value_type*) data;
    for (size_t c = 0; c < channels; ++c) {
      for (size_t h = 0; h < height; ++h) {
        for (size_t w = 0; h < width; ++w) {
          typed_data[batch_offset + c * width * height + h * width + w] =
              resized_image.at<cv::Vec3b>(h, w)[c];
        }
      }
    }
    return;
  }

 protected:
  Allocator alloc_fn;
  Deleter free_fn;
};

template <typename T>
using host_buffer = flat_buffer<host_allocator, host_deleter, T>;
template <typename T>
using gpu_buffer = flat_buffer<gpu_allocator, gpu_deleter, T>;

class buffer_factory {
 public:
  virtual generic_buffer* create_buffer_ptr(DataType Tp, size_t size) = 0;
};

class host_buffer_factory : public buffer_factory {
 public:
  generic_buffer* create_buffer_ptr(DataType Tp, size_t size) override {
    switch (Tp) {
      case (DataType::kFLOAT):
        return new host_buffer<float>(size);
        break;
      case (DataType::kINT32):
        return new host_buffer<int>(size);
        break;
      case (DataType::kHALF):
        return new host_buffer<half>(size);
        break;
      default:
        return nullptr;
    }
  }
};

class gpu_buffer_factory : public buffer_factory {
  generic_buffer* create_buffer_ptr(DataType Tp, size_t size) override {
    switch (Tp) {
      case (DataType::kFLOAT):
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
};

/**
* @brief TensorRT IO blob
* @details every IO blob in trt have two buffer: buffer on host
* and buffer on gpu
*
*/
struct blob {
  generic_buffer::ptr host_mem;
  generic_buffer::ptr gpu_mem;
  blob() : host_mem(nullptr), gpu_mem(nullptr){};
  blob(generic_buffer* _host_mem, generic_buffer* _gpu_mem)
      : host_mem(_host_mem), gpu_mem(_gpu_mem) {
    // debug
    std::cout << "Blob moved" << std::endl;
  };
};

/**
* @brief manage the input and output buffer of the network
* @details this is quite similar to blob object in openvino
*/
class buffer_manager {
 public:
  buffer_manager(trt_shared_ptr<IExecutionContext> _context)
      : context(_context) {
    std::unique_ptr<buffer_factory> host_factory{new host_buffer_factory};
    std::unique_ptr<buffer_factory> gpu_factory{new gpu_buffer_factory};
    // fill the blob vector
    int k = engine.getNbBindings();
    blobs.reserve(k);
    for (int i = 0; i < k; ++i) {
      // get type of bindings and binding names
      auto dtype = engine.getBindingDataType(i);
      // auto name = engine.getBindingName(i);
      // calcuate the size of bindings
      size_t vol = 1;
      auto dims = context->getBindingDimensions(i);
      for (int i = 0; i < dims.nbDims; ++i) {
        vol *= dims.d[i];
      }
      auto host_ptr = host_factory->create_buffer_ptr(dtype, vol);
      auto gpu_ptr = gpu_factory->create_buffer_ptr(dtype, vol);
      blobs.emplace_back(host_ptr, gpu_ptr);
      gpu_bindings.push_back(gpu_ptr);
    }
  }
  // get an input blob by name and fill data
  void fill_input(std::string tensorname, cv::Mat& img) {
    int index = engine.getBindingIndex(tensorname.c_str());
    if (index == -1) {
      throw std::logic_error("TRT network has no tensor name " + tensorname);
    }
    Dims dim = context->getBindingDimensions(index);
    blobs[index].host_mem->fill_from_mat(img, dim, 0);  // currently support batch 1
  }
  // iterate all input blobs, copy data from host to gpu
  void memcpy_input_htod() { memcpy_buffer(true, true, false); }
  // iterate all output blobs, copy data from gpu to host
  void memcpy_output_dtoh() { memcpy_buffer(false, false, false); }
  // get device binding for executions
  std::vector<void*> get_bindings() const { return gpu_bindings; }

 private:
  // get the buffer associcate to tensorname
  void* get_buffer(const bool is_host, std::string tensorname) const {
    int index = engine.getBindingIndex(tensorname.c_str());
    if (index == -1) {
      return nullptr;
    }
    return is_host ? blobs[index].host_mem->get_data()
                   : blobs[index].gpu_mem->get_data();
  }
  // copy data from host to device and device to host
  void memcpy_buffer(const bool is_input, const bool htod, const bool async,
                     const cudaStream_t& stream = 0) {
    for (int i = 0; i < engine.getNbBindings(); ++i) {
      void* src_ptr =
          htod ? blobs[i].host_mem->get_data() : blobs[i].gpu_mem->get_data();
      void* dst_ptr =
          htod ? blobs[i].gpu_mem->get_data() : blobs[i].host_mem->get_data();
      const size_t nbytes = blobs[i].host_mem->get_nbytes();
      const cudaMemcpyKind cuda_memcpy_kind =
          htod ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
      if ((is_input && engine.bindingIsInput(i)) ||
          (!is_input && !engine.bindingIsInput(i))) {
        if (async) {
          cudaMemcpyAsync(src_ptr, dst_ptr, nbytes, cuda_memcpy_kind, stream);
        } else {
          cudaMemcpy(src_ptr, dst_ptr, nbytes, cuda_memcpy_kind);
        }
      }
    }
  }
  trt_shared_ptr<IExecutionContext> context;
  const ICudaEngine& engine = context->getEngine();
  std::vector<blob> blobs;     // hold all input and output blobs
  std::vector<void*> gpu_bindings;  // device bindings for engine execution
};
}  // namespace ie
}  // namespace st