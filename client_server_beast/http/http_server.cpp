
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


int main(int argc, char const *argv[])
{
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
        // TODO: make it prettier
        std::string const _device = "CPU";
        std::string const _xml = "/home/canhld/workplace/MEC_FPGA_DEMO/models/object_detection/common/ssdlite_mobilenet_v2_coco_2018_05_09/saved_model_FP32.xml";
        // string const _xml = "/home/canhld/workplace/MEC_FPGA_DEMO/models/object_detection/common/ssd/300/caffe/models/VGGNet/VOC0712Plus/SSD_300x300_ft/VGG_VOC0712Plus_SSD_300x300_ft_iter_160000.xml";
        std::string const _l = "/home/canhld/workplace/MEC_FPGA_DEMO/models/object_detection/common/ssd.labels";
        std::shared_ptr<ncl::ssdFPGA> Ie = std::make_shared<ncl::ssdFPGA>(_device,_xml,_l,0);
        listen_worker listener{TaskQueue, cv, mtx, key, Ie};

        // FPGA or not
        bool FPGA = _device.find("FPGA") != std::string::npos;
        if (FPGA) {
            // we will run inference in main thread 
            // and create other thead to run listener
            listener.destroy_ie();
            std::thread{std::bind(listener,argv[1],argv[2])}.detach();
            inference_worker inferencer{Ie, TaskQueue, cv, mtx, key};
            inferencer();
        }
        else {
            // we don't need explicit inferencer thread
            // this thread will run listener
            listener(argv[1],argv[2]);
        }
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
    return 0;
}
