# Note on implementation of concurent queue

__INPORTANT__: when I say _concurrent_, I mean thread-concurrent, not process-concurrent. They are a very, very different story.

Queuing is interesting, and it is also far more complicated than its appearance. As my daily job, I use `std::queue` most of the time. But in this project, we need a queue that work in `producer-consumer` manner, i.e. a multi-thread queue. The idea is: because openvino fpga only work with single threads, let make a `worker` thread that hold the inference instance and use a queue to recieve request from other http threads.

At the first glance, I think I can just use a `std::queue`, wrap it with the same interface, and protect it with a `std::mutex`.It's indeed a solution, but not a good solution. After reading a blog (what's a coincidence) about _unbounded, thread-safe queue implementation in Golang_, I realize that it's an interesting problem even in C++, and there's lock-free approach for it.

## Design phylosophy

1. MPMC: multi-producers, multi-consumers
2. Unbounded: No 
3. Thread-safe: sure
4. Exception safe: no-throw guarantee
5. Fully abstraction: no expicit lock, all handled insize the queue
6. Tagged?

### Thread-safe

### Exception-safety