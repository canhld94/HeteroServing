
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>
#include <boost/asio/dispatch.hpp>
#include <boost/asio/strand.hpp>
#include <boost/config.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <sstream>
#include <fstream>
#include <chrono>
#include <mutex>

namespace beast = boost::beast;         // from <boost/beast.hpp>
namespace http = beast::http;           // from <boost/beast/http.hpp>
namespace net = boost::asio;            // from <boost/asio.hpp>
namespace bpt = boost::property_tree;   // from <boots/property_tree>
using tcp = boost::asio::ip::tcp;       // from <boost/asio/ip/tcp.hpp>
typedef bpt::ptree JSON;                // just hiding the ugly name
using std::cout;
using std::endl;
using std::ofstream;

#include <ultis.h> // from ncl
#include <ssdFPGA.h> // from ncl
#include <concurrent_queue.h> // from ncl;

using ncl::bbox;
using ncl::ssdFPGA;

/* 
    message struct, use between http worker and inferences worker
    ussage in http_worker:
        std::vector<bbox> predictions;
        msg m(data,key,&predictions);
*/
typedef struct message {
    const char *data;               // the pointer that hold actual data
    int size;                       // size of the data
    std::vector<bbox> *predictions; // the prediction, inference engine will write the result here
    int key;                        // key of the message, int is enought in current system
    message(const char* _data, int _size, std::vector<bbox> *_predictions, int _key):
        data(_data), size(_size), key(_key), predictions(_predictions) { };
}  msg;


/* 
    This is the C++11 equivalent of a generic lambda.
    The function object is used to send an HTTP message. 
*/
template<class Stream>
struct send_lambda
{
    Stream& stream_;
    bool& close_;
    beast::error_code& ec_;

    explicit
    send_lambda(
        Stream& stream,
        bool& close,
        beast::error_code& ec)
        : stream_(stream)
        , close_(close)
        , ec_(ec)
    {
    }

    template<bool isRequest, class Body, class Fields>
    void
    operator()(http::message<isRequest, Body, Fields>&& msg) const
    {
        // Determine if we should close the connection after
        close_ = msg.need_eof();

        // We need the serializer here because the serializer requires
        // a non-const file_body, and the message oriented version of
        // http::write only works with const messages.
        http::serializer<isRequest, Body, Fields> sr{msg};
        http::write(stream_, sr, ec_);
    }
};

/*
    General worker thread
    TODO: add logger
*/
class worker {

}

/*
    Inference worker that will run the inference engine
    CPU or GPU backend: multi-workers is availble 
    FPGA backend: there is one and only one worker, other while system will crash

*/
class inference_worker : public worker {
public:
    void run() { 
        // start listening to the queue
        // TODO: implement the queue, inference worker will sleep while queue is empty()
        msg m = TaskQueue->front(); //! find other way to do it
        TaskQueue->pop();
        *m.predictions = Ie->run(m.data, m.size);
        // Push to queue and notify the http_worker
        EventQueue.push(m.key);
    }
private:
    std::shared_ptr<ssdFPGA> Ie; // pointer to inference engine
    std::shared_ptr<ncl::concurrent_queue<msg>> TaskQueue; // pointer to the task queue
    std::shared_ptr<ncl::concurrent_queue<int>> EventQueue; // pointer to the event queue
};

/*
    http worker that will handler the request
*/

class http_worker : public worker {
public:


private:
    tcp::socket socket;
    send_lambda<tcp::socket> sender;

private:
    std::shared_ptr<ncl::concurrent_queue<msg>> TaskQueue;
    std::shared_ptr<ncl::concurrent_queue<int>> EventQueue;
};

/*
    websocket worker, worker that deal with websocket season (e.g. stream)
*/

class ws_worker : public worker {

}