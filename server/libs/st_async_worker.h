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
    */
    class async_worker {
    public:
        /**
         * @brief Construct a new worker object
         * 
         */
        async_worker() {}
        /**
         * @brief Destroy the worker object
         * 
         */
        virtual ~async_worker() {}
        /**
         * @brief bring the worker up
         * @details each worker will have different functionality.
         * when () operator is called, they will start serving their basic function
         */
        virtual void operator()() = 0;
    };

    class async_http_worker : public async_worker {
    private:
        
        object_detection_mq<single_bell>:: cpu_taskq;

    private:
        void session_handler(tcP::socket&& sock) {
            
        }

        void listen(const char* ip, const char *p) {
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
                // handler the request
                session_handler(std::move(sock));
            }
        }
    public:

    };

} // namespace ie
} // namespace worker