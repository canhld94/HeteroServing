/***************************************************************************************
 * Copyright (C) 2020 canhld@.kaist.ac.kr
 * SPDX-License-Identifier: Apache-2.0
 * @b About: This file is the main application
 ***************************************************************************************/

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
#include <stdlib.h> 
#include "st_ultis.h"
#include "st_async_worker.h"
#include "st_ie.h"
#include "st_exception.h"
namespace beast = boost::beast;         // from <boost/beast.hpp>
namespace http = beast::http;           // from <boost/beast/http.hpp>
namespace net = boost::asio;            // from <boost/asio.hpp>
namespace bpt = boost::property_tree;   // from <boots/property_tree>
namespace fs = boost::filesystem;       // from <boots/filesystem>
typedef bpt::ptree JSON;                // just hiding the ugly name
using namespace st::sync;
using namespace st::worker;
using namespace st::ie;
using namespace st::exception;

/// @brief message for help argument
constexpr char help_message[] = "Print this message.";

/// @brief message for images argument
constexpr char config_file_message[] = "Required: path to server configuration file (json)";

/// @brief Define flag for showing help message
DEFINE_bool(h, false, help_message);

/// @brief Define parameter for set image file
DEFINE_string(f, "../config/config.json", config_file_message);


/**
* @brief This function show a help message
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


bool parse_and_check_cmd_line(int argc,char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
       showUsage();
       return false;
    }

    if (FLAGS_f.empty()) {
        std::cout << "WARNING: server configuration file is not provided!" << std::endl;
        std::cout << "using default ../server/config/config.json" << std::endl;
        std::cout << std::endl;
    }

    return true;
}

auto static const _____ = []() {
    pthread_setname_np(pthread_self(),"main worker"); // just for debugging
};

int main(int argc, char const *argv[])
{
    try
    {
        // Set the default logger to file logger
        auto file_logger = spdlog::basic_logger_mt("basic_logger", "logs/basic.txt");
        spdlog::set_default_logger(file_logger);
        spdlog::set_level(spdlog::level::info);
        spdlog::info("Log started!");
        // Check command line arguments.
        if (!parse_and_check_cmd_line(argc, const_cast<char**>(argv))) {
            return 0;
        }
        std::cout << "Loading server configuration from " << FLAGS_f << std::endl;
        if (!fs::exists(FLAGS_f)) {
            std::cout << "WARNING: server configuration file is not exist or invalid" << std::endl;
            std::cout << "using default ../server/config/config.json" << std::endl;
            FLAGS_f = "../server/config/config.json";
        }

        // getting server configuration
        JSON config;
        bpt::read_json(FLAGS_f,config);
        std::cout << "Server configuration:" << std::endl;
        bpt::write_json(std::cout,config);
        
        // parsing 
        // server
        auto ip = config.get<std::string>("ip");
        auto port = config.get<std::string>("port");

        // inference engine
        bool fpga = false, gpu = false, cpu = false;
        std::vector<inference_engine::ptr> cpu_ies;
        std::vector<inference_engine::ptr> gpu_ies; // default: nvgpu
        std::vector<inference_engine::ptr> fpga_ies;

        const auto &ie_array = config.get_child("inference engines");
        std::unordered_set<std::string> fpga_devs; 
        ie_factory factory;
        // iterate over all devices
        for (auto it = ie_array.begin(); it != ie_array.end(); ++it) {
            // get the configuration of each device
            auto conf = it->second;
            const std::string device = conf.get<std::string>("device");
            // get the models list, pass if there is no models
            auto &model_list = conf.get_child("models");
            if (model_list.size() == 0) continue;
            // currently, one device can run only one models, so only get the begin
            auto model = conf.begin()->second;
            std::vector<inference_engine::ptr> tmp;
            const std::string name = conf.get<std::string>("name");
            // path to the model graph and weight
            const std::string &graph = model.get<std::string>("graph");
            const std::string &labels = ie.get<std::string>("labels");
            const int replicas = ie.get<std::string>("replicas");
            bool is_fpga = device.find("fpga") != std::string::npos;
            bool is_cpu = device.find("cpu") != std::string::npos;
            bool is_gpu = device.find("nvgpu") != std::string::npos; // default nv gpu
            if (is_fpga) {
                // FPGA inference worker cannot run outside of main threads
                // Therefore, current version of inference server can run at most
                // one FPGA inference worker.
                if (replicas > 1) {
                    throw fpga_overused();
                }
                // bitstream
                const std::string &bitstream = fpga_conf.get<std::string>("bitstream");
                // setenv("DLA_AOCX",bitstream.c_str(),0);
                setenv("CL_CONTEXT_COMPILER_MODE_INTELFPGA","3",0);
            }
            // create inference engines
            auto mcode = factory.str2mcode(name);
            auto dcode = factory.str2dcode(name);
            for (int i = 0; i < replicas; ++i) {
                tmp.push_back(factory.create_inference_engin(type,dcode,model,labels));
            }
            if (is_fpga) {
                fpga_ies = std::move(tmp);
            }
            if (is_cpu) {
                cpu_ies = std::move(tmp);
            }
            if (is_gpu) {
                gpu_ies = std::move(tmp);
            }
        }

        // task queue
        bool cpu = cpu_ies.size() > 0;
        bool fpga = fpga_ies.size() > 0;
        bool gpu = gpu_ies.size() > 0;
        object_detection_mq<single_bell>::ptr cpu_taskq = cpu ?
                                                        std::make_shared<object_detection_mq<single_bell>>() :
                                                        nullptr;
        object_detection_mq<single_bell>::ptr gpu_taskq = gpu ?
                                                        std::make_shared<object_detection_mq<single_bell>>() :
                                                        nullptr;
        object_detection_mq<single_bell>::ptr fpga_taskq = fpga ?
                                                        std::make_shared<object_detection_mq<single_bell>>() :
                                                        nullptr;
        
        // vector of fused listening worker
        listen_worker<inference_engine::ptr> listener{cpu_taskq};
        std::thread{std::bind(listener,ip,port)}.detach();

        // cpu inference work group   
        int cpu_num_workers = cpu_ies.size();
        std::vector<std::thread> cpu_ie_workers(num_workers);
        for (int i = 0; i < cpu_num_workers; ++i) {
            inference_worker<inference_engine::ptr> inferencer{cpu_ies[i], cpu_taskq};
            cpu_ie_workers[i] = std::thread{std::bind(inferencer)};
            cpu_ie_workers[i].detach();
        }
        // gpu inference work group
        int gpu_num_workers = gpu_ies.size();
        std::vector<std::thread> gpu_ie_workers(num_workers);
        for (int i = 0; i < gpu_num_workers; ++i) {
            inference_worker<inference_engine::ptr> inferencer{gpu_ies[i], gpu_taskq};
            gpu_ie_workers[i] = std::thread{std::bind(inferencer)};
            gpu_ie_workers[i].detach();
        }
        // fpga inference in the main threads
        if (fpga) {
            inference_worker<inference_engine::ptr> inferencer{fpga_ies[0], fpga_taskq};
            inferencer();
        }
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
    return 0;
}