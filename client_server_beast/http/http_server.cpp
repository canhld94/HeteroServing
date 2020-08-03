
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/config.hpp>
#include <ultis.h>
#include "workers.h"

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
        // inference engine init
        std::string const _device = "CPU";
        std::string const _xml = "/home/canhld/workplace/MEC_FPGA_DEMO/models/object_detection/common/ssdlite_mobilenet_v2_coco_2018_05_09/saved_model_FP32.xml";
        // string const _xml = "/home/canhld/workplace/MEC_FPGA_DEMO/models/object_detection/common/ssd/300/caffe/models/VGGNet/VOC0712Plus/SSD_300x300_ft/VGG_VOC0712Plus_SSD_300x300_ft_iter_160000.xml";
        std::string const _l = "/home/canhld/workplace/MEC_FPGA_DEMO/models/object_detection/common/ssd.labels";
        std::shared_ptr<ncl::ssdFPGA> Ie = std::make_shared<ncl::ssdFPGA>(_device,_xml,_l,0);
        // network IO init
        auto const address = net::ip::make_address(argv[1]);
        auto const port = static_cast<unsigned short>(std::stoi(argv[2]));
        // the io_contex is required to all IO - boost asio implementation
        net::io_context ioc{1}; // we have only 1 listening thread in sync model
        // the acceptor that will recieve incomming request
        tcp::acceptor acceptor{ioc,{address,port}};
        for(;;) {
            // this socket will run 
            tcp::socket sock{ioc};
            // accep, blocking until new connection
            acceptor.accept(sock);
            // launch new http worker to handle new request
            // transfer ownership of socket to the worker
            auto f = [&](tcp::socket& _sock){
                http_worker worker{acceptor, std::move(_sock), Ie, nullptr};
                worker.start();
            };
            std::thread{
                std::bind(f,std::move(sock))
                }.detach();
        }
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
    return 0;
}
