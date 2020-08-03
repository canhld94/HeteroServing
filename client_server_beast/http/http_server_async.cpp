//
// Copyright (c) 2016-2019 Vinnie Falco (vinnie dot falco at gmail dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Official repository: https://github.com/boostorg/beast
//

//------------------------------------------------------------------------------
//
// Example: HTTP server, synchronous
//
//------------------------------------------------------------------------------
#define BOOST_ASIO_ENABLE_HANDLER_TRACKING

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
// Custom
#define PROFILER
#include <ultis.h> // from ncl
#include <ssdFPGA.h> // from ncl

namespace beast = boost::beast;         // from <boost/beast.hpp>
namespace http = beast::http;           // from <boost/beast/http.hpp>
namespace net = boost::asio;            // from <boost/asio.hpp>
namespace bpt = boost::property_tree;   // from <boots/property_tree>
using tcp = boost::asio::ip::tcp;       // from <boost/asio/ip/tcp.hpp>

typedef bpt::ptree JSON;                // just hiding the ugly name

// basic io std that we don't want to put std every time
// we don't use explicit `using namespace ...` because many methods
// are common among namespace and it may make the code hard to follow
// if we remove its namespace
using std::cout;
using std::endl;
using std::ofstream;
using ncl::bbox;


//------------------------------------------------------------------------------

/* 
    Return a reasonable mime type based on the extension of a file.
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
}

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
}

/* 
    This function handles the request at GET /
 */

std::string
greeting () {
    std::ostringstream ss;
    ss.str("");
    ss << std::fixed << "{\n"
       << "\"type\": \"greeting\",\n" 
       << "\"from\": \"canhld@kaist.ac.kr\",\n" 
       << "\"message\": \"welcome to SSD inference server version 1\",\n"
       << "\"what next\": \"GET /v1/ for supported API\"\n"
       << "}\n";
    return ss.str();
}

/* 
    This function handles the metadata request at GET /metadata 
    TODO: Implement the function with proper resource
 */

std::string 
handle_metadata_request () {
    std::ostringstream ss;
    ss.str("");
    ss << std::fixed << "{\n"
       << "\"from\": \"canhld@kaist.ac.kr\",\n" 
       << "\"message\": \"this is metadata request\"\n"
       << "}\n";
    return ss.str();
}

/*
    This funtion handles the inference request at POST /inference
    All request return string body, so its return type is std::string
    should we format it with JSON?
    TODO: Implement the handler to work with image data
*/

template <class Body, class Allocator>
std::string 
handle_inference_request (http::request<Body,http::basic_fields<Allocator>> & req,
                          const std::shared_ptr<ncl::ssdFPGA>& ie) {
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
    // lock the inference engine to prevent race condition
    // TODO: can we do it lock-free?
    // std::lock_guard<std::mutex> lock(ie->m);
    std::vector<bbox> prediction;
    // exception handling in run, no need to santiny check
    prediction = ie->run(data,size);
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
}

/* 
    This function produces an HTTP response for the given
    request. The type of the response object depends on the
    contents of the request, so the interface requires the
    caller to pass a generic lambda for receiving the response. 
*/
template<
    class Body, class Allocator,
    class Send>
void
handle_request(
    http::request<Body, http::basic_fields<Allocator>>&& req,
    const std::shared_ptr<ncl::ssdFPGA>& ie,
    Send&& send)
{

    // Make sure we can handle the method
    if( req.method() != http::verb::get &&
        req.method() != http::verb::head &&
        req.method() != http::verb::post)
        return send(error_message(req,http::status::bad_request,"Unknown HTTP-method"));

    // Request path must be absolute and not contain "..".
    beast::error_code ec;
    std::string target = request_resolve(req.target(),ec);
    if (target.size() == 0) {
        return send(error_message(req,http::status::bad_request,"Illegal request-target"));
    }
    
    // Handle the case where the resource doesn't exist
    if(ec == beast::errc::no_such_file_or_directory)
        return send(error_message(req,http::status::not_found,"Not found"));

    // Creating our response with string_body
    http::string_body::value_type body;

    // Handle an unknown error
    if(ec)
        return send(error_message(req,http::status::unknown,ec.message()));

    // Respond to HEAD request, alway just send the basic information of the server
    if(req.method() == http::verb::head)
    {
        http::response<http::empty_body> res{http::status::ok, req.version()};
        res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
        res.set(http::field::content_type, mime_type(target));
        res.content_length(0);
        res.keep_alive(req.keep_alive());
        return send(std::move(res));
    }
    else if (req.method() == http::verb::get) {
        // Respond to GET request
        if (target == "/") {
            body = greeting();
        }
        else if (target == "metadata") {
            body = handle_metadata_request();
        }
        else {
            return send(error_message(req,http::status::bad_request,"Illegal HTTP method"));
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
        return send(std::move(res));
    }
    else { 
        // Respond to POST request
        if (target == "inference") {
            PROFILE("running inference",
            body = handle_inference_request(req,ie);
            ); //! PROFILE
        }
        else {
            return send(error_message(req,http::status::bad_request,"Illegal HTTP method"));
        }
        // Cache the size since we need it after the move
        PROFILE ("construct response",
        auto const size = body.size();
        http::response<http::string_body> res{
            std::piecewise_construct,
            std::make_tuple(std::move(body)),
            std::make_tuple(http::status::ok, req.version())};
        res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
        res.set(http::field::content_type, "application/json");
        res.content_length(size);
        res.keep_alive(req.keep_alive());
        ); //! PROFILE
        PROFILE ("send",
        send(std::move(res));
        );
        return;
    }
}

//------------------------------------------------------------------------------

/* 
    Report a failure
*/
void
fail(beast::error_code ec, char const* what)
{
    std::cerr << what << ": " << ec.message() << "\n";
}

/* 
    Handles an HTTP server connection
*/
class session : public std::enable_shared_from_this<session>
{
    // This is the C++11 equivalent of a generic lambda.
    // The function object is used to send an HTTP message.
    struct send_lambda
    {
        session& self_;

        explicit
        send_lambda(session& self)
            : self_(self)
        {
        }

        template<bool isRequest, class Body, class Fields>
        void
        operator()(http::message<isRequest, Body, Fields>&& msg) const
        {
            // The lifetime of the message has to extend
            // for the duration of the async operation so
            // we use a shared_ptr to manage it.
            auto sp = std::make_shared<
                http::message<isRequest, Body, Fields>>(std::move(msg));

            // Store a type-erased version of the shared
            // pointer in the class to keep it alive.
            self_.res_ = sp;

            // Write the response
            http::async_write(
                self_.stream_,
                *sp,
                beast::bind_front_handler(
                    &session::on_write,
                    self_.shared_from_this(),
                    sp->need_eof()));
        }
    };

    beast::tcp_stream stream_;
    beast::flat_buffer buffer_;
    http::request<http::string_body> req_;
    std::shared_ptr<void> res_;
    send_lambda lambda_;
    std::shared_ptr<ncl::ssdFPGA> ie_;

public:
    // Take ownership of the stream
    session(
        tcp::socket&& socket, std::shared_ptr<ncl::ssdFPGA>& ie)
        : stream_(std::move(socket))
        , lambda_(*this)
        ,  ie_(ie)
    {
    }

    // Start the asynchronous operation
    void
    run()
    {
        // We need to be executing within a strand to perform async operations
        // on the I/O objects in this session. Although not strictly necessary
        // for single-threaded contexts, this example code is written to be
        // thread-safe by default.
        net::dispatch(stream_.get_executor(),
                      beast::bind_front_handler(
                          &session::do_read,
                          shared_from_this()));
    }

    void
    do_read()
    {
        // Make the request empty before reading,
        // otherwise the operation behavior is undefined.
        req_ = {};

        // Set the timeout.
        stream_.expires_after(std::chrono::seconds(30));

        // Read a request
        http::async_read(stream_, buffer_, req_,
            beast::bind_front_handler(
                &session::on_read,
                shared_from_this()));
    }

    void
    on_read(
        beast::error_code ec,
        std::size_t bytes_transferred)
    {
        boost::ignore_unused(bytes_transferred);

        // This means they closed the connection
        if(ec == http::error::end_of_stream)
            return do_close();

        if(ec)
            return fail(ec, "read");

        // Send the response
        handle_request(std::move(req_), ie_, lambda_);
    }

    void
    on_write(
        bool close,
        beast::error_code ec,
        std::size_t bytes_transferred)
    {
        boost::ignore_unused(bytes_transferred);

        if(ec)
            return fail(ec, "write");

        if(close)
        {
            // This means we should close the connection, usually because
            // the response indicated the "Connection: close" semantic.
            return do_close();
        }

        // We're done with the response so delete it
        res_ = nullptr;

        // Read another request
        do_read();
    }

    void
    do_close()
    {
        // Send a TCP shutdown
        beast::error_code ec;
        stream_.socket().shutdown(tcp::socket::shutdown_send, ec);

        // At this point the connection is closed gracefully
    }
};

//------------------------------------------------------------------------------

// Accepts incoming connections and launches the sessions
class listener : public std::enable_shared_from_this<listener>
{
    net::io_context& ioc_;
    tcp::acceptor acceptor_;
    std::shared_ptr<ncl::ssdFPGA> ie_;

public:
    listener(
        net::io_context& ioc,
        tcp::endpoint endpoint,
        std::shared_ptr<ncl::ssdFPGA> &ie)
        : ioc_(ioc)
        , acceptor_(net::make_strand(ioc))
        , ie_(ie)
    {
        beast::error_code ec;

        // Open the acceptor
        acceptor_.open(endpoint.protocol(), ec);
        if(ec)
        {
            fail(ec, "open");
            return;
        }

        // Allow address reuse
        acceptor_.set_option(net::socket_base::reuse_address(true), ec);
        if(ec)
        {
            fail(ec, "set_option");
            return;
        }

        // Bind to the server address
        acceptor_.bind(endpoint, ec);
        if(ec)
        {
            fail(ec, "bind");
            return;
        }

        // Start listening for connections
        acceptor_.listen(
            net::socket_base::max_listen_connections, ec);
        if(ec)
        {
            fail(ec, "listen");
            return;
        }
    }

    // Start accepting incoming connections
    void
    run()
    {
        do_accept();
    }

private:
    void
    do_accept()
    {
        // The new connection gets its own strand
        acceptor_.async_accept(
            net::make_strand(ioc_),
            beast::bind_front_handler(
                &listener::on_accept,
                shared_from_this()));
    }

    void
    on_accept(beast::error_code ec, tcp::socket socket)
    {
        if(ec)
        {
            fail(ec, "accept");
        }
        else
        {
            // Create the session and run it
            std::make_shared<session>(
                std::move(socket),ie_)->run();
        }

        // Accept another connection
        do_accept();
    }
};

//------------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    // Check command line arguments.
    if (argc != 4)
    {
        std::cerr <<
            "Usage: http-server <address> <port> <num_threads>\n" <<
            "Example:\n" <<
            "    http-server 0.0.0.0 8080 8\n";
        return EXIT_FAILURE;
    }
    auto const address = net::ip::make_address(argv[1]);
    auto const port = static_cast<unsigned short>(std::atoi(argv[2]));
    auto const threads = std::max<int>(1, std::atoi(argv[3]));
    string const _device = "CPU";
    string const _xml = "/home/canhld/workplace/MEC_FPGA_DEMO/models/object_detection/common/ssdlite_mobilenet_v2_coco_2018_05_09/saved_model_FP32.xml";
    // string const _xml = "/home/canhld/workplace/MEC_FPGA_DEMO/models/object_detection/common/ssd/300/caffe/models/VGGNet/VOC0712Plus/SSD_300x300_ft/VGG_VOC0712Plus_SSD_300x300_ft_iter_160000.xml";
    string const _l = "/home/canhld/workplace/MEC_FPGA_DEMO/models/object_detection/common/ssd.labels";
    std::shared_ptr<ncl::ssdFPGA> ie = std::make_shared<ncl::ssdFPGA>(_device,_xml, _l,0);

    // The io_context is required for all I/O
    net::io_context ioc{threads};

    // Create and launch a listening port
    std::make_shared<listener>(
        ioc,
        tcp::endpoint{address, port},
        ie)->run();
    // Run the I/O service on the requested number of threads
    std::vector<std::thread> v;
    v.reserve(threads - 1);
    for(auto i = threads - 1; i > 0; --i)
        v.emplace_back(
        [&ioc]
        {
            ioc.run();
        });
    ioc.run();

    return EXIT_SUCCESS;

}



/************************ some testing code chunk ************************/

// image content type is image/jpeg, how to pass to our container?
// assume we receive an image, first try to save it first
// beast::string_view ext = content_type.substr(6,content_type.size());
// std::string filename = "test." + std::string(ext);
// ofstream stream;
// stream.open(filename.c_str());
// stream << body;
// stream.close();
// cout << "Save file successful" << endl;