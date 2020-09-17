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
#include <inference_engine.hpp>

namespace beast = boost::beast;         // from <boost/beast.hpp>
namespace http = beast::http;           // from <boost/beast/http.hpp>
namespace net = boost::asio;            // from <boost/asio.hpp>
namespace bpt = boost::property_tree;   // from <boots/property_tree>
using namespace InferenceEngine;

using tcp = boost::asio::ip::tcp;       // from <boost/asio/ip/tcp.hpp>
typedef bpt::ptree JSON;                // just hiding the ugly name

#include "st_ultis.h"
#include "st_ie2.h"
#include "st_sync.h"
using namespace st::sync;
using namespace st::ie;


/**
 * @brief Queue between listening workers and http workers
 * @details There is only the socket need to be passed to http worker
 */
using connection_mq = blocking_queue<tcp::socket>;

/**
 * @brief Queue between http workers inference workers
 * @details the http worker must read the request first, if it's not an 
 * inference request, then handle it. If it is an inference request, then 
 * just pass the request and the socket to inference worker
 */
struct http_conn {
    tcp::socket sock;
    beast_basic_request req;
    http_conn() = delete;
    http_conn(http_conn &other) = delete;
    http_conn(tcp::socket &&_sock, beast_basic_request &&_req): 
        sock(std::move(_sock)), req(_req) {};
    using ptr = std::shared_ptr<http_conn>;
};
using http_mq = blocking_queue<http_conn::ptr>;

/**
 * @brief Queue between inference workes and post processing workers
 * @details When done, the inference worker will pass the output blob 
 * of the network as well as the parser to the post processing worker.
 * The socket must be passed as well, and the async_pp_worker need to parse
 * the blob, and write to the socket to response client
 */

struct inference_output {
    tcp::socket sock;
    network_output net_out;
    inference_engine::ptr ie;
    inference_output() = delete;
    inference_output(inference_output& other) = delete;
    inference_output(tcp::socket &&_sock,network_output &&_net_out, inference_engine::ptr _ie) :
        sock(std::move(_sock)), net_out(std::move(_net_out)), ie(std::move(_ie)) {};
    using ptr = std::shared_ptr<inference_output>;
};
using response_mq = blocking_queue<inference_output::ptr>;

namespace st {
namespace worker {

    // class session : public std::enable_shared_from_this<session> {
    // private:
    //     beast::tcp_stream stream;
    //     beast::flat_buffer buffer;
    //     beast_basic_request req;
    // public:

    // };

    class listener : public std::enable_shared_from_this<listener> {
    private:
        net::io_context &ioc;
        tcp::acceptor acceptor;
        connection_mq::ptr connq;
    public:
        // listener() = delete;
        listener(net::io_context& _ioc, tcp::endpoint endpoint,
                connection_mq::ptr &_connq) : ioc(_ioc), 
                acceptor(net::make_strand(_ioc)), connq(_connq) {
            beast::error_code ec;
            // open the acceptor
            acceptor.open(endpoint.protocol(),ec);
            if (ec) {
                fail(ec,"open");
                return;
            }

            // allow address reuse
            acceptor.set_option(net::socket_base::reuse_address(true),ec);
            if (ec) {
                fail(ec,"set option");
                return;
            }

            // bind to the server address
            acceptor.bind(endpoint,ec);
            if (ec) {
                fail(ec,"bind");
                return;
            }

            // start listening for connection
            acceptor.listen(net::socket_base::max_listen_connections,ec);
            if (ec) {
                fail(ec,"listen");
                return;
            }
        }

        // Start accepting incomming connection
        void run() {
            do_accept();
        }
    private:
        void do_accept() {
            // the new connection gets it own strand
            acceptor.async_accept(
                net::make_strand(ioc),
                beast::bind_front_handler(
                    &listener::on_accept,
                    shared_from_this()
                )
            );
        }
        void on_accept(beast::error_code ec, tcp::socket socket) {
            if (ec) {
                fail(ec,"accept");
            }
            else {
                // just push it to queue
                connq->push(std::move(socket));
                // keep acceptong
                do_accept();
            }
        }
    };
    /**
     * @brief listening worker in async mode
     * @details in async mode, there is a group of listening worker. When there
     * is incomming request, it will push the socket to the queue
     * 
     */
    class async_listening_worker : public std::enable_shared_from_this<async_listening_worker> {
    private:
        connection_mq::ptr connq;
        std::string address;
        std::string port;
        int num_threads;
    private:
        void listen(std::string &address, std::string port, int num_threads) {
            auto const ip = net::ip::make_address(address.c_str());
            auto const p = static_cast<unsigned short>(stoi(port));
            // the io_contex is required for all IO
            net::io_context ioc{num_threads};

            // create and launch a listening port
            std::make_shared<listener>(ioc,tcp::endpoint{ip,p},connq)->run();

            // run the io service on the requested threads
            std::vector<std::thread> threads;
            threads.reserve(num_threads - 1);
            for (int i = num_threads-1; i >= 0; --i) {
                threads.emplace_back([&ioc]{
                    ioc.run();
                });
            }
            // run the final threads is myself
            ioc.run();
        } 
    public:
        async_listening_worker() {};
        async_listening_worker(connection_mq::ptr &_connq, std::string _address, 
                                std::string &_port, int _num_threads) : 
                                connq(_connq), address(_address), port(_port),
                                num_threads(_num_threads) {};
        void operator () () {
            try {
                listen(address, port, num_threads);
            }
            catch (const std::exception& e) {
                std::cerr << e.what() << std::endl;
            }
        }
    };

    class async_http_worker : public std::enable_shared_from_this<async_http_worker> {
    private:
        connection_mq::ptr connq;
        http_mq::ptr cpu_taskq;
        http_mq::ptr fpga_taskq;
        http_mq::ptr gpu_taskq;
    public:
        async_http_worker() {};
        async_http_worker(connection_mq::ptr &_connq, http_mq::ptr &_cpu_taskq,
                          http_mq::ptr &_fpga_taskq, http_mq::ptr &_gpu_taskq) :
                          connq(_connq), cpu_taskq(_cpu_taskq), fpga_taskq(_fpga_taskq),
                          gpu_taskq(_gpu_taskq) {};

        void operator () () {
            try {
                beast::flat_buffer buffer;
                for (;;) {
                    auto sock = connq->pop();
                    // read from sock
                    PROFILE_DEBUG("Read From Socket",
                    beast_basic_request req;
                    beast::error_code ec;
                    bool close = false;
                    send_lambda<tcp::socket> sender{sock,close,ec};
                    http::read(sock, buffer, req, ec);
                    );
                    // if ec, just report server error
                    if (ec) {
                        sender(error_message(req,http::status::unknown,ec.message()));
                        continue;
                    }
                    // handle request
                    request_handler(std::move(req),std::move(sock));
                }
            }
            catch (const std::exception& e) {
                std::cerr << e.what() << std::endl;
            }
        }

    private:
        http::response<http::string_body> 
        error_message(beast_basic_request &req, http::status status, beast::string_view why) {
            beast_basic_response res{status, req.version()};
            res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
            res.set(http::field::content_type, "text/html");
            res.keep_alive(req.keep_alive());
            res.body() = std::string(why);
            res.prepare_payload();
            return res;
        } //! error_message

        void request_handler(beast_basic_request &&req, tcp::socket &&sock) {
            beast::error_code ec;
            bool close;
            send_lambda<tcp::socket> sender{sock,close,ec};
            // Make sure we can handle the method
            if( req.method() != http::verb::get &&
                req.method() != http::verb::head &&
                req.method() != http::verb::post)
                return sender(error_message(req,http::status::bad_request,"Unknown HTTP-method"));
            // Request path must be absolute and not contain "..".
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
                // handle post request
                if (target == "inference/cpu") {
                    if (cpu_taskq) {
                        cpu_taskq->push(std::make_shared<http_conn>(std::move(sock),std::move(req)));
                    }
                    else {
                        sender(error_message(req,http::status::not_implemented,"Service not yet implemented"));
                    }
                }
                else if (target == "inference/fpga") {
                    if (fpga_taskq) {
                        fpga_taskq->push(std::make_shared<http_conn>(std::move(sock),std::move(req)));
                    }
                    else {
                        sender(error_message(req,http::status::not_implemented,"Service not yet implemented"));
                    }
                }
                else if (target == "inference/gpu") {
                    if (gpu_taskq) {
                        gpu_taskq->push(std::make_shared<http_conn>(std::move(sock),std::move(req)));
                    }
                    else {
                        sender(error_message(req,http::status::not_implemented,"Service not yet implemented"));
                    }
                }
                else {
                    sender(error_message(req,http::status::bad_request,"Illegal HTTP method"));
                }
            }
            
        }
        std::string
        request_resolve (beast::string_view const &target, beast::error_code &ec) {
            // Now do it as simple as possible
            // Assume the request to the server is always in form `/{resource}`
            // current supported resources
            static const std::set<std::string> resources = {
                "/",
                "v1"
                "metadata",
                "inference/cpu",
                "inference/fpga",
                "inference/gpu"
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
        } //! greeting

        std::string metadata_request_handler () {
            std::ostringstream ss;
            ss.str("");
            ss << std::fixed << "{\n"
            << "\"from\": \"canhld@kaist.ac.kr\",\n" 
            << "\"message\": \"this is metadata request\"\n"
            << "}\n";
            return ss.str();
        } //! metadata_request_handler
    };

    class async_inference_worker {
    private:
        inference_engine::ptr Ie;                                       //!< pointer to inference engine
        http_mq::ptr taskq;                             //!< task queue, will get job in this queue
        response_mq::ptr resq;
    public:
        async_inference_worker() {};
        async_inference_worker(inference_engine::ptr &_Ie, http_mq::ptr &_taskq, response_mq::ptr &_resq) :
                                Ie(_Ie), taskq(_taskq), resq(_resq) {}; 
        void operator () () {
            try {
                // get the request and the socket from taskq
                auto conn = taskq->pop();
                auto &sock = conn->sock;
                auto &req = conn->req;
                auto &body = req.body();
                auto data = body.data();
                int size = body.size();
                // run the blob
                auto net_out = Ie->run(data,size);
                // push to resq
                inference_engine::ptr ie = Ie;
                resq->push(std::make_shared<inference_output>(std::move(sock),std::move(net_out),std::move(ie)));
            }
            catch (const std::exception& e) {
                std::cerr << e.what() << std::endl;
            }
        }
    };

    class async_pp_worker {
    private: 
        response_mq::ptr resq;
    public:
        async_pp_worker() {};
        async_pp_worker(response_mq::ptr &_resq) : resq(_resq) {};
        void operator () () {
            try {
                for (;;) {
                    beast::error_code ec;
                    bool close;
                    // it's ok because async_pp_worker can only access public member 
                    // of inference engine
                    auto infer_out = resq->pop();
                    auto &ie = infer_out->ie;
                    auto &net_out = infer_out->net_out;
                    auto &sock = infer_out->sock; 
                    send_lambda<tcp::socket> sender{sock,close,ec};
                    // we should parse infer_req, and move to sock
                    auto detection_out = ie->detection_parser(net_out);
                    // create response and send throught the socket
                    int n = detection_out.size();
                    // create property tree and write to json
                    JSON detections;                   // our response
                    JSON bboxes;                // predicion
                    for (int i = 0; i < n; ++i) {
                        // parse prediction[i] to p[i]
                        bbox &pred = detection_out[i];
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
                    detections.put_child("predictions",bboxes);
                    std::ostringstream ss;
                    bpt::write_json(ss,detections);
                    auto body = ss.str();
                    auto const size = body.size();
                    beast_basic_response res{
                        std::piecewise_construct,
                        std::make_tuple(std::move(body)),
                        std::make_tuple(http::status::ok, 11)};
                    res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
                    res.set(http::field::content_type, "application/json");
                    res.content_length(size);
                    // res.keep_alive(req.keep_alive());
                    sender(std::move(res));
                } 
            }
            catch(const std::exception& e) {
                std::cerr << e.what() << '\n';
            }
        }
    };

} // namespace ie
} // namespace worker