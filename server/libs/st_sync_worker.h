/***************************************************************************************
 * Copyright (C) 2020 canhld@.kaist.ac.kr
 * SPDX-License-Identifier: Apache-2.0
 * @b About: This file implement workers, the execution unit of the server
 ***************************************************************************************/

#pragma once 
#include <boost/asio/dispatch.hpp>
#include <boost/asio/strand.hpp>
#include <boost/config.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/lexical_cast.hpp>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <sstream>
#include <set>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>

namespace beast = boost::beast;         // from <boost/beast.hpp>
namespace http = beast::http;           // from <boost/beast/http.hpp>
namespace net = boost::asio;            // from <boost/asio.hpp>
namespace bpt = boost::property_tree;   // from <boots/property_tree>
using tcp = boost::asio::ip::tcp;       // from <boost/asio/ip/tcp.hpp>
typedef bpt::ptree JSON;                // just hiding the ugly name

#include "st_ultis.h"
#include "st_ie_base.h"
#include "st_sync.h"

namespace st {
namespace worker {
    using st::ie::bbox;
    using namespace st::sync;
    /**
     * @brief pure abstract worker thread
    */
    class sync_worker {
    public:
        /**
         * @brief Construct a new worker object
         * 
         */
        sync_worker() {}
        /**
         * @brief Destroy the worker object
         * 
         */
        virtual ~sync_worker() {}
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
    class sync_inference_worker : public sync_worker {
    public:
        sync_inference_worker() = delete;
        /**
         * @brief Construct a new inference worker object
         * 
         * @param _Ie 
         * @param _taskq 
         * @param _cv 
         * @param _mtx 
         * @param _key 
         */
        sync_inference_worker(IEPtr& _Ie, object_detection_mq<single_bell>::ptr& _taskq):
                        Ie(_Ie), taskq(_taskq) {
                            spdlog::info("Init inference worker!");
                        }
        /**
         * @brief Destroy the inference worker object
         * 
         */
        ~sync_inference_worker() {}
        // sync worker public interface implementation
        void operator()() final { 
            pthread_setname_np(pthread_self(),"IE worker");
            // start listening to the queue
            try {
                for (;;) {
                    spdlog::debug("[IEW] Waiting for new task");
                    auto m = taskq->pop();
                    spdlog::info("[IEW] Invoke inference engine {}", taskq->size());
                    *m.predictions = Ie->run_detection(m.data, m.size);
                    spdlog::debug("[IEW] Done inferencing, predidiction size = {}", m.predictions->size());
                    // Push to queue and notify the sync_http_worker
                    spdlog::debug("[IEW] signaling request thread");
                    m.bell->ring(1);
                }
            }
            catch(const std::exception& e) {
                std::cerr << e.what() << '\n';
            }
        }

    private:
        IEPtr Ie;                                           //!< pointer to inference engine
        object_detection_mq<single_bell>::ptr taskq;        //!< task queue, will get job in this queue
    };

    /**
     * @brief http worker that will handler the request
     * @details 
     * In sync mode, each time when server receive request, 
     * it will create a newthread that run http worker class
     * 
     */
    class sync_http_worker : public sync_worker {
    public:
        sync_http_worker() = delete;

        /**
         * @brief Construct a new http worker object
         * 
         * @param _acceptor 
         * @param _sock 
         * @param _data 
         * @param _taskq 
         */
        sync_http_worker(tcp::acceptor& _acceptor, tcp::socket&& _sock, void *_data, 
                    object_detection_mq<single_bell>::ptr& _taskq):
                    acceptor(_acceptor), sock(std::move(_sock)), data(_data), taskq(_taskq)
                    {
                        bell = std::make_shared<single_bell>();
                        spdlog::info("Init new http worker!");
                    }
        /**
         * @brief 
         * 
         * @return * Default 
         */
        ~sync_http_worker() {}
        // sync worker public interface implementation
        void operator()() final {
            pthread_setname_np(pthread_self(),"http worker");
            session_handler();
        }
    private:
        // private attribute
        tcp::acceptor& acceptor;                                //!< the acceptor, needed to init our socket
        tcp::socket sock{acceptor.get_executor()};              //!< the endpoint socket, passed from main thread
        void *data;                                             //!< pointer to data, i.e dashboard
        object_detection_mq<single_bell>::ptr taskq;            //!< task queue
        single_bell::ptr bell;                                  //!< notify bell
        // private method
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
        http::response<http::string_body> 
        error_message(beast_basic_request &req, http::status status, beast::string_view why) {
            beast_basic_response res{status, req.version()};
            res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
            res.set(http::field::content_type, "text/html");
            res.keep_alive(req.keep_alive());
            res.body() = std::string(why);
            res.prepare_payload();
            return res;
        } // error_message
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
        } // request_sovle
        /**
         * @brief 
         * 
         */
        std::string greeting () {
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
        } // greeting
        /**
         * @brief 
         * 
         *
         * TODO: Implement the function with proper resource
        */
        std::string metadata_request_handler () {
            std::ostringstream ss;
            ss.str("");
            ss << std::fixed << "{\n"
            << "\"from\": \"canhld@kaist.ac.kr\",\n" 
            << "\"message\": \"this is metadata request\"\n"
            << "}\n";
            return ss.str();
        } // metadata_request_handler
        /**
         * @brief This funtion handles the inference request at POST /inference
         * ?All request return string body, so its return type is std::string should we format it with JSON?
         */
        std::string inference_request_handler (beast_basic_request & req) {
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
            // push to queue
            obj_detection_msg<single_bell> m{data,size,&prediction,bell};
            spdlog::debug("[HTTPW {}] enqueue my task {}", boost::lexical_cast<std::string>(std::this_thread::get_id()), taskq->size());
            taskq->push(m);
            spdlog::debug("[HTTPW {}] waiting for IEW", boost::lexical_cast<std::string>(std::this_thread::get_id()));
            bell->wait(1);
            spdlog::debug("[HTTPW {}] recieved data", boost::lexical_cast<std::string>(std::this_thread::get_id()));
            int n = prediction.size();
            // create property tree and write to json
            JSON res;                   // our response
            JSON bboxes;                // predicion
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
        } // inferennce_request_handler
        /**
         * @brief this is our handler
         * 
         * @param req 
         * @param sender 
         * @return * Request 
         */
        template <class Send>
        void request_handler (beast_basic_request&& req, Send& sender) {
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
                beast_empty_response res{http::status::ok, req.version()};
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
                beast_basic_response res{
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
                beast_basic_response res{
                    std::piecewise_construct,
                    std::make_tuple(std::move(body)),
                    std::make_tuple(http::status::ok, req.version())};
                res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
                res.set(http::field::content_type, "application/json");
                res.content_length(size);
                res.keep_alive(req.keep_alive());
                sender(std::move(res));
            }
        } // request_handler
        /**
         * @brief handler the session
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
                PROFILE_DEBUG("Read From Socket",
                    beast_basic_request req;
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
                PROFILE_DEBUG("Handle Request",
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
        } // session_handler
    }; // class sync_http_worker

    /**
     * @brief listening worker that will listen to connection
     * 
     */
    class sync_listen_worker : public sync_worker {
    public:
        sync_listen_worker() = delete;
        /**
         * @brief Construct a new listen worker object
         * 
         * @param _taskq 
         */
        sync_listen_worker(object_detection_mq<single_bell>::ptr& _taskq):
                    taskq(_taskq) {}
        /**
         * @brief Destroy the listen worker object
         * 
         */
        ~sync_listen_worker() {}
        // sync worker public interface implementation
        void operator()() final {
            pthread_setname_np(pthread_self(),"listen worker");
            std::cout << "Warning: no IP and address is provide" << std::endl;
            std::cout << "Use defaul address 0.0.0.0 and default port 8080" << std::endl;
            listen("0.0.0.0","8080");

        }
        /**
         * @brief additional public interface
         * 
         * @param ip 
         * @param port 
         */
        void operator()(std::string& ip, std::string& port) {
            pthread_setname_np(pthread_self(),"listen worker");
            listen(ip.c_str(),port.c_str());
        }
    private: 
        object_detection_mq<single_bell>::ptr taskq;    //!< task queue
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
            std::cout << "Start accepting" << std::endl;
            tcp::acceptor acceptor{ioc,{address,port}};
            for(;;) {
                // this socket will run 
                tcp::socket sock{ioc};
                // accep, blocking until new connection
                acceptor.accept(sock);
                std::cout << "New client: " << sock.remote_endpoint().address().to_string() << std::endl;
                // launch new http worker to handle new request
                // transfer ownership of socket to the worker
                auto f = [&](tcp::socket& _sock){
                    sync_http_worker httper{acceptor, std::move(_sock), nullptr,taskq};
                    httper();
                };
                std::thread{
                    std::bind(f,std::move(sock))
                    }.detach();
            }
        }
    }; // class listen worker

    /**
     * @brief 
     * 
     */
    class sync_ws_worker : public sync_worker {

    };
} // namespace ie
} // namespace worker