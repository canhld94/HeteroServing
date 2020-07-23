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

## Protocol
