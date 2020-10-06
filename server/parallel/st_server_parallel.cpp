/***************************************************************************************
 * Copyright (C) 2020 canhld@.kaist.ac.kr
 * SPDX-License-Identifier: Apache-2.0
 * @b About: This file is the main application
 ***************************************************************************************/

#include <gflags/gflags.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>
#include <stdlib.h>
#include <boost/asio/ip/tcp.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>
#include <boost/config.hpp>
#include <boost/filesystem.hpp>
#include <boost/property_tree/json_parser.hpp>
#include "st_exception.h"
#include "st_ie_base.h"
#include "st_ie_factory.h"
#include "st_sync_worker.h"
#include "st_ultis.h"
namespace beast = boost::beast;        // from <boost/beast.hpp>
namespace http = beast::http;          // from <boost/beast/http.hpp>
namespace net = boost::asio;           // from <boost/asio.hpp>
namespace bpt = boost::property_tree;  // from <boots/property_tree>
namespace fs = boost::filesystem;      // from <boots/filesystem>
typedef bpt::ptree JSON;               // just hiding the ugly name
using namespace st::sync;
using namespace st::worker;
using namespace st::ie;
using namespace st::exception;

/// @brief message for help argument
constexpr char help_message[] = "Print this message.";

/// @brief message for images argument
constexpr char config_file_message[] =
    "Required: path to server configuration file (json)";

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
  std::cout << "    -f \"<path>\"               " << config_file_message
            << std::endl;
  std::cout << std::endl;
}

bool parse_and_check_cmd_line(int argc, char* argv[]) {
  // ---------------------------Parsing and validation of input
  // args--------------------------------------
  gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
  if (FLAGS_h) {
    showUsage();
    return false;
  }

  if (FLAGS_f.empty()) {
    std::cout << "WARNING: server configuration file is not provided!"
              << std::endl;
    std::cout << "using default ../server/config/config.json" << std::endl;
    std::cout << std::endl;
  }

  return true;
}

auto static const _____ = []() {
  pthread_setname_np(pthread_self(), "main worker");  // just for debugging
};

int main(int argc, char const* argv[]) {
  try {
    // Set the default logger to file logger
    auto file_logger =
        spdlog::basic_logger_mt("basic_logger", "logs/basic.txt");
    spdlog::set_default_logger(file_logger);
    spdlog::set_level(spdlog::level::info);
    spdlog::info("Log started!");
    // Check command line arguments.
    if (!parse_and_check_cmd_line(argc, const_cast<char**>(argv))) {
      return 0;
    }
    std::cout << "Loading server configuration from " << FLAGS_f << std::endl;
    if (!fs::exists(FLAGS_f)) {
      std::cout << "WARNING: server configuration file is not exist or invalid"
                << std::endl;
      std::cout << "using default ../server/config/config.json" << std::endl;
      FLAGS_f = "../server/config/config.json";
    }

    // getting server configuration
    JSON config;
    bpt::read_json(FLAGS_f, config);
    std::cout << "Server configuration:" << std::endl;
    bpt::write_json(std::cout, config);

    // parsing
    // server
    auto ip = config.get<std::string>("ip");
    auto port = config.get<std::string>("port");
    std::cout << ip << ":" << port << std::endl;
    // inference engine
    std::vector<inference_engine::ptr> IEs;

    const auto& ie_array = config.get_child("inference engines");
    ie_factory factory;
    // iterate over all devices
    for (auto it = ie_array.begin(); it != ie_array.end(); ++it) {
      // get the configuration of each device
      auto conf = it->second;
      const std::string& device = conf.get<std::string>("device");
      // get the models list, pass if there is no models
      auto& model_list = conf.get_child("models");
      if (model_list.size() == 0) continue;
      // currently, one device can run only one models, so only get the begin
      auto model = model_list.begin()->second;
      const std::string& name = model.get<std::string>("name");
      // path to the model graph and weight
      const std::string& graph = model.get<std::string>("graph");
      const std::string& labels = model.get<std::string>("label");
      const int replicas = model.get<int>("replicas");
      bool is_fpga = device.find("fpga") != std::string::npos;
      if (is_fpga) {
        // FPGA inference worker cannot run outside of main threads
        // Therefore, current version of inference server can run at most
        // one FPGA inference worker.
        if (replicas > 1) {
          throw fpga_overused();
        }
        // bitstream
        const std::string& bitstream = conf.get<std::string>("bitstream");
        setenv("DLA_AOCX", bitstream.c_str(), 0);
        // setenv("CL_CONTEXT_COMPILER_MODE_INTELFPGA","3",0);
      }
      // create inference engines
      for (int i = 0; i < replicas; ++i) {
        if (is_fpga) {
          IEs.insert(IEs.begin(), factory.create_inference_engine(
                                      name, device, graph, labels));
        } else {
          IEs.push_back(
              factory.create_inference_engine(name, device, graph, labels));
        }
      }
    }

    // task queue - Not necessary used with CPU inference
    object_detection_mq<single_bell>::ptr TaskQueue =
        std::make_shared<object_detection_mq<single_bell>>();

    // listening worker
    sync_listen_worker listener{TaskQueue};

    // inference work group
    std::thread{std::bind(listener, ip, port)}.detach();

    // FPGA inference worker cannot run outside of main threads
    // Therefore, current version of inference server can run at most
    // one FPGA inference worker. By convention, we assume that if there
    // is a FPGA inferencer, it would be the first IE in the configuration
    // file
    int num_workers = IEs.size() - 1;
    std::vector<std::thread> ie_workers(num_workers);
    for (int i = 0; i < num_workers; ++i) {
      sync_inference_worker<inference_engine::ptr> inferencer{IEs[i + 1],
                                                              TaskQueue};
      ie_workers[i] = std::thread{std::bind(inferencer)};
      ie_workers[i].detach();
    }
    sync_inference_worker<inference_engine::ptr> inferencer{IEs[0], TaskQueue};
    inferencer();
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
  }

  return 0;
}

/*
+--------+----+----+----+----+------+------+------+------+
|        | C1 | C2 | C3 | C4 | C(5) | C(6) | C(7) | C(8) |
+--------+----+----+----+----+------+------+------+------+
| CV_8U  |  0 |  8 | 16 | 24 |   32 |   40 |   48 |   56 |
| CV_8S  |  1 |  9 | 17 | 25 |   33 |   41 |   49 |   57 |
| CV_16U |  2 | 10 | 18 | 26 |   34 |   42 |   50 |   58 |
| CV_16S |  3 | 11 | 19 | 27 |   35 |   43 |   51 |   59 |
| CV_32S |  4 | 12 | 20 | 28 |   36 |   44 |   52 |   60 |
| CV_32F |  5 | 13 | 21 | 29 |   37 |   45 |   53 |   61 |
| CV_64F |  6 | 14 | 22 | 30 |   38 |   46 |   54 |   62 |
+--------+----+----+----+----+------+------+------+------+
*/