
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/config.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/filesystem.hpp>
#include <inference_engine.hpp>
#include <gflags/gflags.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <ultis.h>
#include "workers.h"
namespace beast = boost::beast;         // from <boost/beast.hpp>
namespace http = beast::http;           // from <boost/beast/http.hpp>
namespace net = boost::asio;            // from <boost/asio.hpp>
namespace bpt = boost::property_tree;   // from <boots/property_tree>
namespace fs = boost::filesystem;       // from <boots/filesystem>
typedef bpt::ptree JSON;                // just hiding the ugly name


/// @brief message for help argument
constexpr char help_message[] = "Print this message.";

/// @brief message for images argument
constexpr char config_file_message[] = "Required: path to server configuration file (json)";

/// \brief Define flag for showing help message <br>
DEFINE_bool(h, false, help_message);

/// \brief Define parameter for set image file <br>
DEFINE_string(f, "../server_config/config.json", config_file_message);


/**
* \brief This function show a help message
*/
static void showUsage() {
    std::cout << std::endl;
    std::cout << "http_server [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                        " << help_message << std::endl;
    std::cout << "    -f \"<path>\"               " << config_file_message << std::endl;
    std::cout << std::endl;
}


bool ParseAndCheckCommandLine(int argc,char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
       showUsage();
       return false;
    }

    if (FLAGS_f.empty()) {
        std::cout << "WARNING: server configuration file is not provided!" << std::endl;
        std::cout << "using default ../server_config/config.json" << std::endl;
        std::cout << std::endl;
    }

    return true;
}


int main(int argc, char const *argv[])
{
    try
    {
        // Set the default logger to file logger
        auto file_logger = spdlog::basic_logger_mt("basic_logger", "logs/basic.txt");
        spdlog::set_default_logger(file_logger);
        spdlog::set_level(spdlog::level::debug);
        spdlog::info("Log started!");
        // Check command line arguments.
        if (!ParseAndCheckCommandLine(argc, const_cast<char**>(argv))) {
            return 0;
        }
        std::cout << "Loading server configuration from " << FLAGS_f << std::endl;
        if (!fs::exists(FLAGS_f)) {
            std::cout << "WARNING: server configuration file is not provided!" << std::endl;
            std::cout << "using default ../server_config/config.json" << std::endl;
            FLAGS_f = "../server_config/config.json";
        }

        // getting server configuration
        JSON config;
        bpt::read_json(FLAGS_f,config);
        std::cout << "Server configuration:" << std::endl;
        bpt::write_json(std::cout,config);
        spdlog::debug("This message should be displayed..");    
        
        // parsing 
        // server
        auto ip = config.get<std::string>("ip");
        auto port = config.get<std::string>("port");
        // model
        const auto &ie = config.get_child("inference engine").begin()->second;
        const std::string &device = ie.get<std::string>("device");
        const std::string &model = ie.get<std::string>("model");
        const std::string &labels = ie.get<std::string>("labels");

        // task queue - Not necessary used with CPU inference
        std::shared_ptr<tbb::concurrent_bounded_queue<msg>> TaskQueue = std::make_shared<tbb::concurrent_bounded_queue<msg>>();
        std::shared_ptr<std::condition_variable> cv = std::make_shared<std::condition_variable>();
        std::shared_ptr<std::mutex> mtx = std::make_shared<std::mutex>();
        std::shared_ptr<std::string> key = std::make_shared<std::string>();

        // inference engine init
        // TODO: make it prettier
        std::shared_ptr<ncl::ssdFPGA> Ie = std::make_shared<ncl::ssdFPGA>(device,model,labels,0);
        listen_worker listener{TaskQueue, cv, mtx, key, Ie};

        // FPGA or not
        bool FPGA = device.find("FPGA") != std::string::npos;
        if (FPGA) {
            // we will run inference in main thread 
            // and create other thead to run listener
            listener.destroy_ie();
            std::thread{std::bind(listener,ip,port)}.detach();
            inference_worker inferencer{Ie, TaskQueue, cv, mtx, key};
            inferencer();
        }
        else {
            // we don't need explicit inferencer thread
            // this thread will run listener
            listener(ip,port);
        }
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
    return 0;
}
