
#ifndef NCL_WORKERS_H
#define NCL_WORKERS_H
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
#include <ctime>
#include <mutex>
#include <condition_variable>

namespace beast = boost::beast;         // from <boost/beast.hpp>
namespace http = beast::http;           // from <boost/beast/http.hpp>
namespace net = boost::asio;            // from <boost/asio.hpp>
namespace bpt = boost::property_tree;   // from <boots/property_tree>
using tcp = boost::asio::ip::tcp;       // from <boost/asio/ip/tcp.hpp>
typedef bpt::ptree JSON;                // just hiding the ugly name
using std::cout;
using std::endl;
using std::ofstream;

#include <ultis.h>                  // from ncl
#include <ssdFPGA.h>                // from ncl
#include <concurrent_queue.h>       // from ncl;
#include <tbb/concurrent_queue.h>   // Intel tbb concurent queue

using tbb::concurrent_bounded_queue;        // TODO: replace with own queue
using ncl::bbox;
using ncl::ssdFPGA;

/* 
    message struct, use between http worker and inferences worker
    ussage in http_worker:
        std::vector<bbox> predictions;
        msg m(data,key,&predictions);
*/
class msg {
public:
    const char *data;               // the pointer that hold actual data
    int size;                       // size of the data
    std::vector<bbox> *predictions; // the prediction, inference engine will write the result here
    std::string key;                // key of the message, int is enought in current system
    msg(): data(nullptr), size(-1), predictions(nullptr), key("") {}
    msg(const char* _data, int _size, std::vector<bbox>* _predictions, std::string _key):
        data(_data), size(_size), predictions(_predictions), key(_key) { }
};


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
    Return mime_type base on the path of the string
*/

beast::string_view
mime_type(beast::string_view path)
{
    using beast::iequals;
    auto const ext = [&path]
    {
        auto const pos = path.rfind(".");
        if(pos == beast::string_view::npos)
            return beast::string_view{};
        return path.substr(pos);
    }();
    if(iequals(ext, ".htm"))  return "text/html";
    if(iequals(ext, ".html")) return "text/html";
    if(iequals(ext, ".php"))  return "text/html";
    if(iequals(ext, ".css"))  return "text/css";
    if(iequals(ext, ".txt"))  return "text/plain";
    if(iequals(ext, ".js"))   return "application/javascript";
    if(iequals(ext, ".json")) return "application/json";
    if(iequals(ext, ".xml"))  return "application/xml";
    if(iequals(ext, ".swf"))  return "application/x-shockwave-flash";
    if(iequals(ext, ".flv"))  return "video/x-flv";
    if(iequals(ext, ".png"))  return "image/png";
    if(iequals(ext, ".jpe"))  return "image/jpeg";
    if(iequals(ext, ".jpeg")) return "image/jpeg";
    if(iequals(ext, ".jpg"))  return "image/jpeg";
    if(iequals(ext, ".gif"))  return "image/gif";
    if(iequals(ext, ".bmp"))  return "image/bmp";
    if(iequals(ext, ".ico"))  return "image/vnd.microsoft.icon";
    if(iequals(ext, ".tiff")) return "image/tiff";
    if(iequals(ext, ".tif"))  return "image/tiff";
    if(iequals(ext, ".svg"))  return "image/svg+xml";
    if(iequals(ext, ".svgz")) return "image/svg+xml";
    return "application/text";
}

/*
    General worker thread
    TODO: add logger
*/
class worker {
public:
    // worker();
    // virtual ~worker();
};

/*
    Inference worker that will run the inference engine
    CPU or GPU backend: multi-workers is availble 
    FPGA backend: there is one and only one worker, other while system will crash
    Exception-safe

*/
class inference_worker : public worker {
private:
    std::shared_ptr<ssdFPGA> Ie;
    std::shared_ptr<concurrent_bounded_queue<msg>> TaskQueue; 
    std::shared_ptr<std::condition_variable> cv;
    std::shared_ptr<std::mutex> mtx;
    std::shared_ptr<std::string> key;
public:
    inference_worker() = delete;
    inference_worker(std::shared_ptr<ssdFPGA> _Ie, std::shared_ptr<concurrent_bounded_queue<msg>> _TaskQueue, 
                    std::shared_ptr<std::condition_variable> _cv, std::shared_ptr<std::mutex> _mtx, 
                    std::shared_ptr<std::string> _key):
                    Ie(_Ie), TaskQueue(_TaskQueue), cv(_cv), mtx(_mtx), key(_key)
                    {
                        std::cout << "init inference worker" << std::endl;
                    }
    void run() { 
        // start listening to the queue
        try {
            for (;;) {
                std::cout << "IE: wating for new task" << std::endl;
                msg m; //! find other way to do it
                TaskQueue->pop(m);
                std::lock_guard<std::mutex> lk(*mtx);
                *m.predictions = Ie->run(m.data, m.size);
                // Push to queue and notify the http_worker
                *key = m.key;
                cv->notify_all();
            }
        }
        catch(const std::exception& e) {
            std::cerr << e.what() << '\n';
        }
    }
};

/*
    http worker that will handler the request
    In sync mode, each time when server receive  request, it will create a new 
    thread that run http worker class
*/

class http_worker : public worker {
private:
    tcp::acceptor& acceptor;                                // the acceptor, needed to init our socket
    tcp::socket sock{acceptor.get_executor()};              // the endpoint socket, passed from main thread
    std::shared_ptr<ssdFPGA> Ie;                            // pointer to inference engine in case we use CPU inference
    void *data;                                             // pointer to data, i.e dashboard
    std::shared_ptr<concurrent_bounded_queue<msg>> TaskQueue; 
    std::shared_ptr<std::condition_variable> cv;
    std::shared_ptr<std::mutex> mtx;
    std::shared_ptr<std::string> key;
public:
    http_worker() = delete;
    /* Most frequently used constructor */
    http_worker(tcp::acceptor& _acceptor, tcp::socket&& _sock, std::shared_ptr<ssdFPGA> _Ie, void *_data, 
                std::shared_ptr<concurrent_bounded_queue<msg>> _TaskQueue, 
                std::shared_ptr<std::condition_variable> _cv, std::shared_ptr<std::mutex> _mtx, 
                std::shared_ptr<std::string> _key):
                acceptor(_acceptor), sock(std::move(_sock)), Ie(_Ie), data(_data), TaskQueue(_TaskQueue),
                cv(_cv), mtx(_mtx), key(_key)
                {}
    /* Default destructor */
    ~http_worker() {}
    /* Start the worker */
    void start() {
        session_handler();
    }
private:
    /* 
        This funtion send a error to the client
        Depend on the error code, server should send different message 
    */
    template <class Body, class Alocator>
    http::response<http::string_body> 
    error_message(http::request<Body, http::basic_fields<Alocator>> &req, http::status status, beast::string_view why) {
        http::response<http::string_body> res{status, req.version()};
        res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
        res.set(http::field::content_type, "text/html");
        res.keep_alive(req.keep_alive());
        res.body() = std::string(why);
        res.prepare_payload();
        return res;
    } //! error_message

    /* 
        This function resolve the request target to route it
        to proper resource. Raise ec::no_such_file if the 
        resource doesn't exist
    */

    std::string
    request_resolve (beast::string_view const &target, beast::error_code &ec) {
        // Now do it as simple as possible
        // Assume the request to the server is always in form `/{resource}`
        // current supported resources
        static const std::set<std::string> resources = {
            "/",
            "metadata",
            "inference"
        };
        if (target.empty() 
        || target[0] != '/'
        || target.find("..") != beast::string_view::npos)
        return "";
        std::string ret;
        if (target.size() == 1) { // send header
            ret = "/";
        }
        else {
            // string_view has no null-terminated, therefore it cannot implicitly 
            // convert from string_view to string
            ret = static_cast<std::string>(target.substr(1,target.size()));
        }
        if (resources.find(ret) == resources.end()) {
            // raise no_such_file error
            ec = beast::errc::make_error_code(beast::errc::no_such_file_or_directory);
        }
        return ret;
    } //! request_sovle

    /* 
        This function handles the request at GET /
    */

    std::string
    greeting () {
        JSON res;
        res.put<std::string>("type","greeting");
        res.put<std::string>("from","canhld@kaist.ac.kr");
        res.put<std::string>("message","welcome to NCL inference server version 1");
        JSON what_next;
        what_next.put<std::string>("API","GET /v1/ for supported API");
        what_next.put<std::string>("INFO","GET /metadata/ for model information");
        res.put_child("what next",what_next);
        std::ostringstream ss;
        bpt::write_json(ss,res);
        return ss.str();
    } //! greeting

    /* 
        This function handles the metadata request at GET /metadata 
        TODO: Implement the function with proper resource
    */

    std::string 
    metadata_request_handler () {
        std::ostringstream ss;
        ss.str("");
        ss << std::fixed << "{\n"
        << "\"from\": \"canhld@kaist.ac.kr\",\n" 
        << "\"message\": \"this is metadata request\"\n"
        << "}\n";
        return ss.str();
    } //! metadata_request_handler

    /*
        This funtion handles the inference request at POST /inference
        All request return string body, so its return type is std::string
        should we format it with JSON?
        TODO: Implement the handler to work with image data
    */

    template <class Body, class Allocator>
    std::string 
    inference_request_handler (http::request<Body,http::basic_fields<Allocator>> & req) {
        // we know this is the post method
        // now, first extact the content-type

        // boost::beast::http::header<true, boost::beast::http::basic_fields<std::allocator<char> > >&
        auto &header = req.base();
        // string body --> basic_string
        auto &body = req.body();
        beast::string_view const &content_type = header["content-type"];
        if (content_type.find("image/") == std::string::npos) {
            return "{\n\"message\":\"not an image\"\n}";
        }

        auto data = body.data();
        int size = body.size();
        std::vector<bbox> prediction;
        // exception handling in run, no need to santiny check
        if (Ie) {
            // in this case, we can run IE ourself
            prediction = Ie->run(data,size);
        }
        else {
            // need to pass to inference worker
            std::ostringstream ss;
            ss << std::this_thread::get_id();
            std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()
            );
            ss << std::to_string(ms.count());
            msg m{data,size,&prediction,ss.str()};
            cout << m.key << endl;
            TaskQueue->push(m);
            std::unique_lock<std::mutex>lk(*mtx);
            cv->wait(lk,[&](){return m.key == *key;});
        }
        int n = prediction.size();
        // create property tree and write to json
        JSON res;                   // our response
        JSON bboxes;                // predicion
        if (n > 0) {
            res.put<std::string>("status","ok");
        }
        else {
            res.put<std::string>("status","not ok");
            res.put<std::string>("why","empty detection box");
        }
        for (int i = 0; i < n; ++i) {
            // parse prediction[i] to p[i]
            bbox &pred = prediction[i];
            JSON p;
            p.put<int>("label_id",pred.label_id);
            p.put<std::string>("label",pred.label);
            p.put<float>("confidences",pred.prop);
            JSON tmp;
            for (int i = 0; i < 4; ++i) {
                JSON v;
                v.put<int>("",pred.c[i]);
                tmp.push_back({"",v});
            }
            p.put_child("detection_box",tmp);
            bboxes.push_back({"",p});
        }
        res.put_child("predictions",bboxes);
        std::ostringstream ss;
        bpt::write_json(ss,res);
        return ss.str();
    } //! inferennce_request_handler

    /* Request handler */
    template <class Body, class Allocator, class Send>
    void request_handler (http::request<Body, http::basic_fields<Allocator>>&& req, Send& sender) {
        // Make sure we can handle the method
        if( req.method() != http::verb::get &&
            req.method() != http::verb::head &&
            req.method() != http::verb::post)
            return sender(error_message(req,http::status::bad_request,"Unknown HTTP-method"));

        // Request path must be absolute and not contain "..".
        beast::error_code ec;
        std::string target = request_resolve(req.target(),ec);
        if (target.size() == 0) {
            return sender(error_message(req,http::status::bad_request,"Illegal request-target"));
        }
        
        // Handle the case where the resource doesn't exist
        if(ec == beast::errc::no_such_file_or_directory)
            return sender(error_message(req,http::status::not_found,"Not found"));

        // Creating our response with string_body
        http::string_body::value_type body;

        // Handle an unknown error
        if(ec)
            return sender(error_message(req,http::status::unknown,ec.message()));

        // Respond to HEAD request, alway just send the basic information of the server
        if(req.method() == http::verb::head)
        {
            http::response<http::empty_body> res{http::status::ok, req.version()};
            res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
            res.set(http::field::content_type, mime_type(target));
            res.content_length(0);
            res.keep_alive(req.keep_alive());
            return sender(std::move(res));
        }
        else if (req.method() == http::verb::get) {
            // Respond to GET request
            if (target == "/") {
                body = greeting();
            }
            else if (target == "metadata") {
                body = metadata_request_handler();
            }
            else {
                return sender(error_message(req,http::status::bad_request,"Illegal HTTP method"));
            }
            // Cache the size since we need it after the move
            auto const size = body.size();
            http::response<http::string_body> res{
                std::piecewise_construct,
                std::make_tuple(std::move(body)),
                std::make_tuple(http::status::ok, req.version())};
            res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
            res.set(http::field::content_type, "application/json");
            res.content_length(size);
            res.keep_alive(req.keep_alive());
            return sender(std::move(res));
        }
        else { 
            // Respond to POST request
            if (target == "inference") {
                body = inference_request_handler(req);
            }
            else {
                return sender(error_message(req,http::status::bad_request,"Illegal HTTP method"));
            }
            // Cache the size since we need it after the move
            auto const size = body.size();
            http::response<http::string_body> res{
                std::piecewise_construct,
                std::make_tuple(std::move(body)),
                std::make_tuple(http::status::ok, req.version())};
            res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
            res.set(http::field::content_type, "application/json");
            res.content_length(size);
            res.keep_alive(req.keep_alive());
            sender(std::move(res));
        }
    } //! request_handler

    /* Report a failure */
    void fail(beast::error_code ec, char const* what) {
        std::cerr << what << ": " << ec.message() << "\n";
    } //! fail

    /* Session handler */
    void session_handler () {
        bool close = false;
        beast::error_code ec;

        // send-receive buffer
        beast::flat_buffer buffer;

        // init sender, associate it with an error code that we can read later
        send_lambda<tcp::socket> sender{sock,close,ec};

        for (;;) {
            // read from socket
            http::request<http::string_body> req;
            http::read(sock, buffer, req, ec);
            // if read indicates end of stream, stop reading
            if (ec == http::error::end_of_stream) {
                break;
            }
            // if other read error, stop doing session 
            if (ec) {
                return fail(ec,"read");
            }
            // handle request
            request_handler(std::move(req),sender);
            // handler write error report by sender
            if (ec) {
                return fail(ec,"write");
            }
            if (close) {
                break;
            }
        }
        // If we can reach here, the the request is successful 
        // Shut down the socket and return
        sock.shutdown(tcp::socket::shutdown_send,ec);
        return;
    } // ! session_handler
}; //! class http_worker

/*
    websocket worker, worker that deal with websocket season (e.g. stream)
*/

class ws_worker : public worker {

};

#endif