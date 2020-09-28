tensorRT IO procedure:

- Prepare input blobs:
  - Each input have one blob
  - Blobs may have batch size > 1
  - Layout (currently NCHW)
- Prepare output blobs:
  - Each ouput have one blob
- All I/O blobs are indicated in vector
  - How to know the mapping blob <-> input and blob <-> output
  - OpenVino solution:
    - Create InputBlobMap that hold all input blobs
    - Create OutputBlobMap that hold all output blobs
    - Associate an inference request with an input blob amp and output blob map
  - TensorRT
    - Get binding?


TensorRT bindings are an array of pointers to the input and output buffers for the network. In earlier versions of TensorRT, the size of the input and output tensors had to be specified during engine creation time. Since TensorRT 6, the inputs and outputs can have dynamic shapes (unknown during engine creation time). To perform inference with dynamic shapes, we have to specify the actual dimensions before we run the inference. This is implemented in this PR.
