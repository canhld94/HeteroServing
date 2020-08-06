
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/config.hpp>
#include <ultis.h>
#include <inference_engine.hpp>
#include <chrono>
// #include <tbb/concurrent_queue.h>
#include "workers.h"

// using tbb:concurrent_bounded_queue;

namespace beast = boost::beast;         // from <boost/beast.hpp>
namespace http = beast::http;           // from <boost/beast/http.hpp>
namespace net = boost::asio;            // from <boost/asio.hpp>

struct listen_worker {
    void operator()(const char* ip, const char* p,
                    std::shared_ptr<tbb::concurrent_bounded_queue<msg>>&TaskQueue,
                    std::shared_ptr<std::condition_variable>& cv,
                    std::shared_ptr<std::mutex>& mtx,
                    std::shared_ptr<std::string>& key) const {
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
                pthread_setname_np(pthread_self(),"http worker");
                http_worker http_worker{acceptor, std::move(_sock), nullptr, nullptr,TaskQueue, cv, mtx, key};
                http_worker.start();
            };
            std::thread{
                std::bind(f,std::move(sock))
                }.detach();
        }
    }
};

int main(int argc, char const *argv[])
{
    pthread_setname_np(pthread_self(),"listen worker");
    try
    {
        // Check command line arguments.
        if (argc != 3)
        {
            std::cerr <<
                "Usage: http-server <address> <port>\n" <<
                "Example:\n" <<
                "    http-server 0.0.0.0 8080 .\n";
            return EXIT_FAILURE;
        }
        // task queue - Not necessary used with CPU inference
        std::shared_ptr<tbb::concurrent_bounded_queue<msg>> TaskQueue = std::make_shared<tbb::concurrent_bounded_queue<msg>>();
        std::shared_ptr<std::condition_variable> cv = std::make_shared<std::condition_variable>();
        std::shared_ptr<std::mutex> mtx = std::make_shared<std::mutex>();
        std::shared_ptr<std::string> key = std::make_shared<std::string>();
        // inference engine init
        std::string const _device = "HETERO:FPGA,CPU";
        std::string const _xml = "/home/canhld/workplace/MEC_FPGA_DEMO/models/object_detection/common/ssdlite_mobilenet_v2_coco_2018_05_09/saved_model.xml";
        // string const _xml = "/home/canhld/workplace/MEC_FPGA_DEMO/models/object_detection/common/ssd/300/caffe/models/VGGNet/VOC0712Plus/SSD_300x300_ft/VGG_VOC0712Plus_SSD_300x300_ft_iter_160000.xml";
        std::string const _l = "/home/canhld/workplace/MEC_FPGA_DEMO/models/object_detection/common/ssd.labels";
        std::shared_ptr<ncl::ssdFPGA> Ie  = std::make_shared<ncl::ssdFPGA>(_device,_xml,_l,0); //! crash here
        inference_worker ie_worker{Ie, TaskQueue, cv, mtx, key};
        cout << "Start accepting" << endl;
        listen_worker ls;
        std::thread{std::bind(ls,argv[1],argv[2], TaskQueue, cv, mtx, key)}.detach();
        ie_worker.run();
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
    return 0;
}
