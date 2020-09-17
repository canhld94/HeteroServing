# Dev Logs

What have been done and what to learn up to

## ---- 2020 / 09 / 17 ----

- __What've been done__
  - Reactor server compiled and worked

- _Problem_:
  - Reactor server crashs after some request

- __Backlog__:
  - Implement reactor models that support running different models on different devices
  - Add FCN16s to model server
  - Add Faster RCNN DOTA to model server


## ---- 2020 / 09 / 11 ----

- __What've been done__
  - Fix the structure of reactor server: listening -> http parser -> inference -> sender

- _Problem_:
  - Queue type between inference and sender
  - Inference will response to sender the output blob as well as the parser, but the parser
  is a method of the inference engine. If we wrap it with std::function, the queue may
  possibly be couple to one type of inference engine, which is just bad.

- __Backlog__:
  - Implement reactor models that support running different models on different devices
  - Add FCN16s to model server
  - Add Faster RCNN DOTA to model server


## ---- 2020 / 09 / 10 ----

- __Backlog__:
  - Add FCN16s to model server
  - Add Faster RCNN DOTA to model server
  - Implement reactor models that support running different models on different devices

## ---- 2020 / 09 / 09 ----

- __Backlog__:
  - Add FCN16s to model server
  - Implement reactor models that support running different models on different devices

## ---- 2020 / 09 / 07 ----

- __What've been done__
  - Fix minor bugs and finishing the parallel server; it's quite stable and can run for beta testing now
  - Define the routine of reactor server

- _Problem_:
  - Parallel server can run only 1 models on all devices

- __Backlog__:
  - Implement reactor models that support running different models on different devices

- __Roadmap__:
  - Adding support for NVIDIA  TensorRT

## ---- 2020 / 08 / 27 ----

- _Problem_:
  - Faster RCNN output parsing is quite compilcated, specially in the current version of OpenVino _Solved_
  
- __Backlog__:
  - Implement Faster RCNN _done_
  - Check whether general Faster RCNN model can be converted to OpenVino Model or not (compute graph) _done_

- __Road map__:
  - Implement Reactor model _*important_
  - Implement generic inference engine that support other type of model
  - Implement generic http server that can run with other type of inference engine

## ---- 2020 / 08 / 21 ----

- Fix problem with crashing on yolov3 fpga, it's very likely because fpga run FP16 but the model is FP32
- _Problem_:

- __Road map__:
  - Implement generic inference engine that support other type of model _in progress_
  - Implement YoloV2 and Faster RCNN _important_
  - Implement generic http server that can run with other type of inference engine

## ---- 2020 / 08 / 19 ----

- _Problem_:
  - YOLO crash with FPGA server
  - tiny yolo and yolo is not compatible

- __Road map__:
  - Implement generic inference engine that support other type of model _in progress_
  - Implement Yolo and Faster RCNN _important_ --> done Yolo
  - Implement generic http server that can run with other type of inference engine

## ---- 2020 / 08 / 17 ----

- __Road map__:
  - Implement generic inference engine that support other type of model _important_
  - Implement generic http server that can run with other type of inference engine

## ---- 2020 / 08 / 12 ----

- Fix the deadlock problem, it's likely a system error when we shutdown the FPGA server incorrectly
- Implement queue and event-loop with STL, get rid of tbb concurrent queue
- __Road map__:
  - Implement generic inference engine that support other type of model _important_
  - Implement generic http server that can run with other type of inference engine
  - Building other API

## ---- 2020 / 08 / 11 ----

- Problem: deadlock
- It seem that deadlock is caused by internal openvino synchronization, not by our program (this would be the worst case)
- But why everything was fine when we run it only in 1 threads, need to check
  - __IT DOES HAPPEN EVEN WITH SINGLE THREAD__ Did I check it before?
  - It worth note that current producer-consumer model is much faster and more stable than single thread model
  - System error?
  
## ---- 2020 / 08 / 10 ----

- Problem: at the second request:
  - http worker send the request but somehow do not wait for the inference worker but wake up and send respone to the client
  - At the same time, the inference worker still recieve the data that http_worker passed to it and do the inference on this data, but soon when http worker is destroyed, all of it data is also deleted. So, inference worker run on invalid data and crash because of segfault
- Reason: http worker doesn't return because client send keep alive signal, so the key is always 1 after first inference
- Solution: http worker clear key after recieving result --> __WORKED__ but it lead to even worse situation

## ---- 2020 / 08 / 10 ----

- Problems: http worker accquire lock __before__ inference worker and wait for inference worker to broadcast its key, but inference worker can't broadcast key if they can't accquire lock --> deadlock
- Solution maybe maybe maybe: each http worker have its own cv

## ---- 2020 / 08 / 07 ----

- Solve the crashing problem
  - Root of the problem: FPGA driver send signal to wrong thread, it should be send to inference thread, but they always send it to listening worker (main thread). The main thread will call the interrupt handler in the mmd runtime lib, thus make it access to device and cause crashing (recall that openvino will always crash if more than one thread access to FPGA)
  - __FIX__: Main thread will run inference, while another will run listening --> __WORKED__
- _Problem_: Server with FPGA back-end hang when during stress test -> error in FPGA device or dead-lock in the task queue?
- _What next?_:
  - Solving the problem when FPGA-backed server hang
  - refactoring the inference engine --> __learn__

## ---- 2020 / 08 / 06 ----

- Inference worker and listening worker (main thread cannot run together) --> __why__
- When FPGA is invoked, inference thread crashs
- This is not exception error but failed system call (system error), that make the whole process crashs
- _What next?_:
  - refactoring the inference engine --> __learn__

## ---- 2020 / 08 / 05 ----

- Inferences worker crash with FPGA inference while loading plugin --> __why__
- _What next?_:
  - refactoring the inference engine --> __learn__

## ---- 2020 / 08 / 04 ----

- Finish implement http-workers, and run with sync server but only CPU device
- _What next?_:
  - refactoring the inference engine
  - Implement inference worker
  - Implement producer-consumer model temporlary use intel TBB queue

## ---- 2020 / 08 / 02 ----

- Resoving the reading from socket problem, it's actually in the client size, not server size
- Adding simple implementation of concurrent queue, adding test and CMAKE CI on github
- _What next?:_
  - Impelment producer-consumer model for FPGA inderence __learn__ --> on going
  - Implement classes of workers: http-workers, inference-workers, websocket-workers

## ---- 2020 / 07 / 29 ----

- TODO: make ssdFPGA class singleton? -> no
- Instead, we make shared_ptr from one object and pass it to worker threads to ignore global variable
- Implement http server in async-model -> done -> still reading take longer than ususal
- _What next?_:
  - Test with fast http server to see if it improve reading performance __fast__
  - Thread-safe queue implementation __learn__
  - Need more investigate on `asio` and `beast` parsing to understand the server code __learn__
  - Impelment producer-consumer model for FPGA inderence __learn__

## ---- 2020 / 07 / 27 ----

- TODO: implement test client --> Done
- Implement thread-safe queue solution for FPGA inference?
- _What next?_:
  - structure of the cluster? proxy, nodes, ..etc
  - Other type of http server in boost (async, SSL, Auth, thread-pool)
  - _Remember_: Bug in `read` that increas TTFB time in the client size!!!

## ---- 2020 / 07 / 25 ----

- So much, first working version
- Problem:
  - With FPGA: Device can't be access with multi-thread, so the current server solution is not working
  - Without FPGA: TTFB and content download time is higher than ussual, idk

## ---- 2020 / 07 / 23 ----

- handle image data from client using `http::string_body` --> `char` array in JPEG form
- create the inference api which response the desired format
- handle JSON with `boost::property_tree`
- create the struture for `ssdFPGA.h`
- now ready to integrate the FPGA inference engine

## ---- 2020 / 07 / 22 ----

- Finish creating a http server that support all basic desired APIs
- JSON response format for inference request
- _Requirement:_
  - JSON parsing for response: how to do it elegantly?
  - Image container:
    - How to send it to server, how the server will handle it?
    - The req created in the server is default basic string, we need to change it to file somehow
    - Should we use opencv?

## ---- 2020 / 07 / 21 ----

- Look into inside of Beast: the message container, body concept
- Adding resolver to the target request
- Should refer implementation from Tensorflow serving
- _Requirement:_
  - Asio
  - Structure of code

## ---- 2020 / 07 / 20 ----

- Cmake project: how cmake work, what is the structure of a CMake project
- Boost Beast: how to build websocket and http server with boost
- Finish building http sync server with boost and support POST
- _Requirement:_
  - Clear definition of the APT (get,post)
  - HTTP RFC7023
  - Boost Beast

## ---- 2020 / 07 / 15 ----

- OpenVino setup running environment
- Docker
