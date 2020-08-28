# Note on http with boost::beast

## Container

### Measage and header

The container for HTTP message in beast is `message`, `request` and `response` use the same template from `message` with different template argument

```C++
/*message class*/
template <
    bool isRequest,
    class body,
    class Fields = fields>
class message;

/*request*/
template <class Body, class Fields = fields>
using request = message<true,body,Fields>

/*response*/
template <class Body, class Fields = fields>
using response = message<false,body,Fields>
```

`Header` is an independent template

```C++
template <bool isRequest, class Fields = fields>
class header;

template <class Fields>
using request_header = header<true,Fields>

template <class Fields>
using response_header = header<false, Fields>
```

Inheritance relationship among `Fields`,`headers`, `message`, `request`, `response`, `request_header`, and `response_header`

```C++
                         ::response_header<isRequest=false>
                         ::request_header<isRequest=true>
Fields::header<isRequest>::message<isRequest,Body>::request<isRequest=true>
                                             ::response<isRequest=false>
```

### Body

`Body` is an _concept_ in beast. It defines the `message::body` member, and also include algorithm for transfering data in and out. The algorithms are used during parsing and serialization. Note that `body` is just a _concept_, not a class. From the `body` concept, beast implement specific classes to handle with different type of body: _string\_body_, _file\_body_, _dynamic\_body_  
Example of `body`  

```C++
struct canhBody { // actually not my body
    // the type of the container that will carry the body
    struct value_type;
    // the algorihtm for reading during parsing
    class reader;
    // the algorithm for writing during serialization
    class writer;
    // the body's payload size
    static  
    std::uint64_t size(value_type const& body);
}
```

That said, whatever class you write that support the class member `value_type`, `reader`, `writer`, and method `size()`, is sufficient to use as `Body` in Beast. Neverthless, these member classes should fulfill some requirements, but we will not discuss it here in detail.  
Important Body that we will use frequently in this project  
[**string_body**](string_body)

```C++
template <
    class CharT,
    class Traits = std::char_traits<CharT>,
    class Allocator = std::allocator<CharT>>
    class basic_string_body {
        using value_type = std::basic_string<charT, Traits, Allocator>;
        class reader {
            ...
        }
        class writer {
            ...
        }
        static  
        std::uint64_t size(value_type const& body) {
            ...
        }
    }
using string_body = basic-string_body<std::char>
```

As its name, `string_body` is a string. It use `std::string` as the container, and use _pre\_defined_ implementation of `read` and `write` in the STL as `reader` and `writer`. Cool, right? Nooooo. `string_body` is _just_ a string, but most of RESTful API should return **JSON**. We need a method to write and parse the JSON object ourself :(. In general, `string_body` is an array of bytes, so it is also suitable to handle small `octet-stream` objects.

In `body_string`, the type of body is `std::string`, so it's very easy to work with.

[**file_body**](file_body)

[**dynamic_body**](dynamic_body)

`dynamic_body` implement dynamic buffers to handle objects. This is quite close to some of STL container (closest to `deque` as the author say). However, the implementation is messy and the logic of `read` and `write` to data is ambgious. Even the author doesn't recommend using it. Therefore, we may pass it by now.

## Parser

## Async model

Using sync model is simple, however if we want to do it in a professional way, async is the road to go.

Async model refers to async __threading__ model:

- In sync mode: your main thread listen for the connections, once it recieve a connection, it create a worker thread, transers the connection ownership (the socket) to this worker thread and this worker will run the application code.

```C++ Example usecase of non-blocking model
// endpoint to the new connection
tcp::socket sock{ioc};
// accept new request, block untill we get a connection
acceptor.accept(socket);
// create new worker to handle the request, pass the ownership of endpoint to the workder
std::thread t{std::bin(&my_fking_awesome_handler,std::move(sock))};
// no wait, the worker should done the job w/o any relation to the main thread
t.detach();
```

- In async mode: your main threads spawns a group of worker threads, each thread may do different jobs. E.g. some thread will listening to incomming connection and handling them, some will perform dedicated tasks. These worker work independenly, but they still share resources (make_shared).

```C++ Example usecase of blocking model
// listening and handling incomming connection
void listen_and_handler();
// do some dedicated task
void do_fking_awesome_task();
// group of thread that will accept the connection
std::vector<std::thread> vt;
vt.reserve(10);
for ()

```

When using async model, we have better control of your server and can do fine-grained optimization for each thread. However, we can also encounter some problems:

- Thread communication: if each thread run independently without co-operation, async is no better than sync model. When they do, we should think about how do they communicate and how they co-operate to finish a task efficiently.
- We will need some "thread-safe" data strutures. Sometime using only locks is not enough.

Async mode refer to __http IO__ mode:

- In sync mode: most of operator is blocking: `accept` is blocking until we recieve a connection, `read` is blocking untill we finish reading the stream,...
- In async mode: operator is non-blocking, but it will be associate with a handler when the operator return.

As a rule of thumbs, we ususally use sync IO mode with sync model and async IO mode and async model, simply because they fit each other