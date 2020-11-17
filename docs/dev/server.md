# Server Notes

## Server Architecture

The basic components of the server are:

- Server object which listening for incoming requests

- Protocol object which handle the message incoming messages

- The inference engine which implement the deep learning inference neural network

<p align="center">
  <img width="500" height="300" src="../imgs/inference_server.jpg">
</p>

The server is thread-based: each object is running within it own thread. The implementation can be done as easy as normal synchronous server: we have one shared inference engine object, and for each new request, we create an inference context and launch the inference engine. As long as all context is thread-safe, there is no thing to worry and the deep learning framework (openvino, tensorrt) will handle the request properly. However, in my project, I use an Arria 10 development kit, which is not thread-safe (i.e. the hardware is not thread-safe and will crash the program if there are more than two thread trying to access it at the same time). Therefore, I developed the server in producer-consumer fashion: the are some dedicated threads that run the inference engine and they will get the job from a task queue, which will be populated by protocol threads.

<p align="center">
  <img width="500" height="340" src="../imgs/server.jpg">
</p>


While I developed the queue to deal with the FPGA, I found it work well for other devices, so I decided to use it as the default model for my server. The procedure of handling an incoming connection is as follow:

<p align="center">
  <img width="500" height="250" src="../imgs/server_impl.jpg">
</p>

Currently, I assume all device run a same models, therefore they can get the job from a same queue. I also take some effort to make different queue for each device, so [they can run different models](/server/_experimental/st_server_reactor.cpp). However, I stopped it as it adds extra complexity to the architecture. If we want to make a complete serving platform that can serve different models on different devices, we can use this project as the back-end and write the other routines (scheduler, load-balancer) as front-end service.

Currently, I create new thread to handle each client without adding any constrain on maximum number of concurrent clients. This sometime make the application run with enormous number of protocol threads. If you want to limit number of thread the system can use, you can make a thread pool and submit the connection to the pool when there is a new client.

## Inference Engine Class Hierarchy

I aim to develop a server that runs different device. Therefore, all inference engines, regardless the back-end device or the frameworks, must has one public interface:

```CPP
class inference_engine {
  public: 
  vector<bbox> run_detection(const char*, int sz) = 0;
}
```

Currently, the class hierarchy for inference engine, the factory, and the creators is as follow:


<p align="center">
  <img width="500" height="320" src="../imgs/inference_engine.jpg">
</p>

When start the server, depending on the configuration in the [configure file](../../server/config/README.md), the factory will select the proper creator to create the inference engine. If you want to add new back-end device or framework, do the following things:

1. Create your inference engine that inheritate the `inference_engine` interface

2. Create a creator for your inference engine. The creator should consume a [Boost Property Tree](https://www.boost.org/doc/libs/1_65_1/doc/html/property_tree.html) object and return a shared pointer to the created inference engine

3. Manually register your new creator with the factory; you should do it during the default constructor of the factory, i.e.

```CPP
  ie_factory() {
    Register("intel cpu", new intel_cpu_inference_engine_creator());
    Register("intel fpga", new intel_fpga_inference_engine_creator());
    Register("nvidia gpu", new nvidia_gpu_inference_engine_creator());
    // if you create new inference engine, register your creator here and 
    // do not modify anything outside of this constructor
  }
```
