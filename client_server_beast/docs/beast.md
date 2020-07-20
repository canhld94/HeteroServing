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

`Body` is an _concept_ in beast. It defines the `message::body` member, and also include algorithm for transfering data in and out. The algorithms are used during parsing and serialization

Note that `body` is just a _concept_, not a class. From the `body` concept, beast implement specific classes to handle with different type of body: _string\_body_, _file\_body_, _dynamic\_body_


## Protocol
