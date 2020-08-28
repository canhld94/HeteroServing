/***************************************************************************************
 * Copyright (C) 2020 canhld@.kaist.ac.kr
 * SPDX-License-Identifier: Apache-2.0
 * @b About: This file implement workers, the execution unit of the server
 ***************************************************************************************/

#pragma once 
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>
#include <boost/asio/dispatch.hpp>
#include <boost/asio/strand.hpp>
#include <boost/config.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/lexical_cast.hpp>
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
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>

namespace beast = boost::beast;         // from <boost/beast.hpp>
namespace http = beast::http;           // from <boost/beast/http.hpp>
namespace net = boost::asio;            // from <boost/asio.hpp>
namespace bpt = boost::property_tree;   // from <boots/property_tree>
using tcp = boost::asio::ip::tcp;       // from <boost/asio/ip/tcp.hpp>
typedef bpt::ptree JSON;                // just hiding the ugly name
using std::cout;
using std::endl;
using std::ofstream;

#include "st_ultis.h"
#include "st_ie.h"
#include "st_sync.h"

namespace st {
namespace worker {
    using st::ie::bbox;
    using namespace st::sync;
    /**
     * @brief This is the C++11 equivalent of a generic lambda. 
     * function object is used to send an HTTP message. 
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

    /**
     * @brief Return mime_type base on the path of the string
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

    /**
     * @brief pure abstract worker thread
     * TODO: add logger
    */
    class worker {
    public:
        /**
         * @brief Construct a new worker object
         * 
         */
        worker() {}
        /**
         * @brief Destroy the worker object
         * 
         */
        virtual ~worker() {}
        /**
         * @brief bring the worker up
         * @details each worker will have different functionality.
         * when () operator is called, they will start serving their basic function
         */
        virtual void operator()() = 0;
    };

    /**
     * @brief Inference worker that will run the inference engine
     * @details 
     * CPU or GPU backend: multi-workers is availble,
     * FPGA backend: there is one and only one worker, other while system will crash
     * 
     * @exception 
     */
    template <class IEPtr>
    class inference_worker : public worker {
    private:
        IEPtr Ie;                                           //!< pointer to inference engine
        object_detection_mq<single_bell>::ptr TaskQueue;    //!< task queue, will get job in this queue
    public:
        /**
         * @brief Construct a new inference worker object
         * 
         */
        inference_worker() = delete;
        /**
         * @brief Construct a new inference worker object
         * 
         * @param _Ie 
         * @param _TaskQueue 
         * @param _cv 
         * @param _mtx 
         * @param _key 
         */
        inference_worker(IEPtr& _Ie, object_detection_mq<single_bell>::ptr& _TaskQueue):
                        Ie(_Ie), TaskQueue(_TaskQueue)
                        {
                            spdlog::info("Init inference worker!");
                        }
        
        /**
         * @brief Destroy the inference worker object
         * 
         */
        ~inference_worker() {

        }

        /**
         * @brief 
         * 
         */
        void operator()() { 
            pthread_setname_np(pthread_self(),"IE worker");
            // start listening to the queue
            try {
                for (;;) {
                    spdlog::info("[IEW] Waiting for new task");
                    //! find other way to do it
                    auto m = TaskQueue->pop();
                    // m.bell->lock();
                    spdlog::info("[IEW] Invoke inference engine {}", TaskQueue->size());
                    *m.predictions = Ie->run(m.data, m.size);
                    spdlog::info("[IEW] Done inferencing, predidiction size = {}", m.predictions->size());
                    // Push to queue and notify the http_worker
                    spdlog::info("[IEW] signaling request thread");
                    m.bell->ring(1);
                }
            }
            catch(const std::exception& e) {
                std::cerr << e.what() << '\n';
            }
        }
    };

    /**
     * @brief http worker that will handler the request
     * @details 
     * In sync mode, each time when server receive request, 
     * it will create a newthread that run http worker class
     * 
     */
    template<class IEPtr>
    class http_worker : public worker {
    private:
        tcp::acceptor& acceptor;                                //!< the acceptor, needed to init our socket
        tcp::socket sock{acceptor.get_executor()};              //!< the endpoint socket, passed from main thread
        IEPtr Ie;                                               //!< pointer to inference engine in case we use CPU inference
        void *data;                                             //!< pointer to data, i.e dashboard
        object_detection_mq<single_bell>::ptr TaskQueue;        //!< task queue
        single_bell::ptr bell;                                  //!< notify bell
    public:
        /**
         * @brief Construct a new http worker object
         * 
         */
        http_worker() = delete;

        /**
         * @brief Construct a new http worker object
         * 
         * @param _acceptor 
         * @param _sock 
         * @param _Ie 
         * @param _data 
         * @param _TaskQueue 
         */
        http_worker(tcp::acceptor& _acceptor, tcp::socket&& _sock, IEPtr _Ie, void *_data, 
                    object_detection_mq<single_bell>::ptr& _TaskQueue):
                    acceptor(_acceptor), sock(std::move(_sock)), Ie(_Ie), data(_data), TaskQueue(_TaskQueue)
                    {
                        bell = std::make_shared<single_bell>();
                        spdlog::info("Init new http worker!");
                    }
        
        /**
         * @brief 
         * 
         * @return * Default 
         */
        ~http_worker() {}

        /**
         * @brief 
         * 
         * @return * Start 
         */
        void operator()() {
            pthread_setname_np(pthread_self(),"http worker");
            session_handler();
        }
    private:
    /**
        * @brief This funtion generate error response 
        * @details Depend on the type of error status, different responses messages are generated
        * @tparam Body 
        * @tparam Alocator 
        * @param req 
        * @param status 
        * @param why 
        * @return http::response<http::string_body> 
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


    /**
        * @brief This function resolve the request target to route it to proper resource.
        * 
        * @param target 
        * @param ec 
        * @return std::string 
        * @exception raise ec::no_such_file if the resource doesn't exist
        */
        std::string
        request_resolve (beast::string_view const &target, beast::error_code &ec) {
            // Now do it as simple as possible
            // Assume the request to the server is always in form `/{resource}`
            // current supported resources
            static const std::set<std::string> resources = {
                "/",
                "v1"
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

        /**
         * @brief 
         * 
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

        /**
         * @brief 
         * 
         *
         * TODO: Implement the function with proper resource
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

        /**
         * @brief This funtion handles the inference request at POST /inference
         * ?All request return string body, so its return type is std::string should we format it with JSON?
         * TODO: Implement the handler to work with image data 
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
                obj_detection_msg<single_bell> m{data,size,&prediction,bell};
                spdlog::debug("[HTTPW {}] enqueue my task {}", boost::lexical_cast<std::string>(std::this_thread::get_id()), TaskQueue->size());
                TaskQueue->push(m);
                spdlog::debug("[HTTPW {}] waiting for IEW", boost::lexical_cast<std::string>(std::this_thread::get_id()));
                bell->wait(1);
                spdlog::debug("[HTTPW {}] recieved data", boost::lexical_cast<std::string>(std::this_thread::get_id()));
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


        /**
         * @brief this is our handler
         * 
         * @param req 
         * @param sender 
         * @return * Request 
         */
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

        /**
         * @brief 
         * 
         * @param ec 
         * @param what 
         * @return * Report 
         */
        void fail(beast::error_code ec, char const* what) {
            std::cerr << what << ": " << ec.message() << "\n";
        } //! fail

        /**
         * @brief 
         * 
         * @return * Session 
         */
        void session_handler () {
            bool close = false;
            beast::error_code ec;

            // send-receive buffer
            beast::flat_buffer buffer;

            // init sender, associate it with an error code that we can read later
            send_lambda<tcp::socket> sender{sock,close,ec};

            for (;;) {
                // read from socket
                PROFILE_RELEASE("Read From Socket",
                    http::request<http::string_body> req;
                    http::read(sock, buffer, req, ec);
                );
                // if read indicates end of stream, stop reading
                if (ec == http::error::end_of_stream) {
                    break;
                }
                // if other read error, stop doing session 
                if (ec) {
                    return fail(ec,"read");
                }
                // handle request
                PROFILE_RELEASE("Handle Request",
                    request_handler(std::move(req),sender);
                );
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
            spdlog::info("[HTTPW] Shutdown my socket!");
            sock.shutdown(tcp::socket::shutdown_send,ec);
            return;
        } //! session_handler
    }; //! class http_worker

    /**
     * @brief listening worker that will listen to connection
     * 
     */
    template<class IEPtr>
    class listen_worker {
    private: 
        object_detection_mq<single_bell>::ptr TaskQueue;    //!< task queue
        IEPtr Ie;                                           //!< pointer to inference engine

    private:
        /**
         * @brief 
         * 
         * @param ip 
         * @param p 
         */
        void listen(const char* ip, const char* p) {
            auto const address = net::ip::make_address(ip);
            auto const port = static_cast<unsigned short>(std::stoi(p));
            // the io_contex is required to all IO - boost asio implementation
            net::io_context ioc{1}; // we have only 1 listening thread in sync model
            // the acceptor that will recieve incomming request
            cout << "Start accepting" << endl;
            tcp::acceptor acceptor{ioc,{address,port}};
            for(;;) {
                // this socket will run 
                tcp::socket sock{ioc};
                // accep, blocking until new connection
                acceptor.accept(sock);
                // launch new http worker to handle new request
                // transfer ownership of socket to the worker
                auto f = [&](tcp::socket& _sock){
                    http_worker<IEPtr> httper{acceptor, std::move(_sock), Ie, nullptr,TaskQueue};
                    httper();
                };
                std::thread{
                    std::bind(f,std::move(sock))
                    }.detach();
            }
        }
    public:
        /**
         * @brief Construct a new listen worker object
         * 
         */
        listen_worker() = delete;

        /**
         * @brief Construct a new listen worker object
         * 
         * @param _TaskQueue 
         * @param _cv 
         * @param _mtx 
         * @param _key 
         * @param _Ie 
         */
        explicit
        listen_worker(object_detection_mq<single_bell>::ptr& _TaskQueue,
                    IEPtr& _Ie):
                    TaskQueue(_TaskQueue), Ie(_Ie)
                    {}

        /**
         * @brief Destroy the listen worker object
         * 
         */
        ~listen_worker() {

        }

        /**
         * @brief 
         * 
         */
        void operator()() {
            pthread_setname_np(pthread_self(),"listen worker");
            std::cout << "Warning: no IP and address is provide" << std::endl;
            std::cout << "Use defaul address 0.0.0.0 and default port 8080" << std:: endl;
            listen("0.0.0.0","8080");

        }

        /**
         * @brief 
         * 
         * @param ip 
         * @param port 
         */
        void operator()(std::string& ip, std::string& port) {
            pthread_setname_np(pthread_self(),"listen worker");
            listen(ip.c_str(),port.c_str());
        }

        /**
         * @brief 
         * 
         */
        void destroy_ie() {
            Ie = nullptr;
        }
    }; //! class listen worker

    /**
     * @brief 
     * 
     */
    class ws_worker : public worker {

    };
} // namespace ie
} // namespace worker