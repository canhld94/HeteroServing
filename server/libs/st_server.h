

#include <gflags/gflags.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>
#include <stdlib.h>
#include <boost/config.hpp>
#include <boost/filesystem.hpp>
#include "st_ie_base.h"
#include "st_ie_factory.h"
#include "st_worker.h"
#include "st_utils.h"
#include "st_grpc_impl.h"
using namespace st::sync;
using namespace st::worker;
using namespace st::rpc;
using namespace st::ie;

namespace st {
class server {
  public:
    server() {};
    server(server* _actual) : actual(_actual) {};
    server(const std::string& json_file);
    virtual void run() {
      assert(actual);
      actual->run();
    }
  protected:
    JSON config;
    server(JSON& _config): config(_config) {};
  private:
    server *actual;
};

class http_server : public server {
  public:
    http_server(JSON &_config) : server(_config) {};
    virtual void run() override {
      try {
      // Set the default logger to file logger
      auto file_logger =
          spdlog::basic_logger_mt("basic_logger", "logs/basic.txt");
      spdlog::set_default_logger(file_logger);
      spdlog::set_level(spdlog::level::info);
      spdlog::info("Log started!");
      // parsing
      // server
      auto ip = config.get<std::string>("ip");
      auto port = config.get<std::string>("port");
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
        auto& model = conf.get_child("model");
        if (model.size() == 0) continue;
        const int replicas = conf.get<int>("replicas");
        bool is_fpga = device.find("fpga") != std::string::npos;
        if (is_fpga) {
          // FPGA inference worker cannot run outside of main threads
          // Therefore, current version of inference server can run at most
          // one FPGA inference worker.
          if (replicas > 1) {
            throw std::logic_error("FPGA inference engine: expected 1, got " + std::to_string(replicas));
          }
          // bitstream
          const std::string& bitstream = conf.get<std::string>("bitstream");
          setenv("DLA_AOCX", bitstream.c_str(), 0);
          // setenv("CL_CONTEXT_COMPILER_MODE_INTELFPGA","3",0);
        }
        // create inference engines
        for (int i = 0; i < replicas; ++i) {
          if (is_fpga) {
            IEs.insert(IEs.begin(), factory.create_inference_engine(conf));
          } else {
            IEs.push_back(
                factory.create_inference_engine(conf));
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
  }
    
};

class grpc_server : public server {
  public:
    grpc_server(JSON &_config) : server(_config) {};
    virtual void run() override {
        try {
        // Set the default logger to file logger
        auto file_logger =
            spdlog::basic_logger_mt("basic_logger", "logs/basic.txt");
        spdlog::set_default_logger(file_logger);
        spdlog::set_level(spdlog::level::info);
        spdlog::info("Log started!");
        // parsing
        // server
        auto ip = config.get<std::string>("ip");
        auto port = config.get<std::string>("port");
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
          auto& model = conf.get_child("model");
          if (model.size() == 0) continue;
          const int replicas = conf.get<int>("replicas");
          bool is_fpga = device.find("fpga") != std::string::npos;
          if (is_fpga) {
            // FPGA inference worker cannot run outside of main threads
            // Therefore, current version of inference server can run at most
            // one FPGA inference worker.
            if (replicas > 1) {
              throw std::logic_error("FPGA inference engine: expected 1, got " + std::to_string(replicas));
            }
            // bitstream
            const std::string& bitstream = conf.get<std::string>("bitstream");
            setenv("DLA_AOCX", bitstream.c_str(), 0);
            // setenv("CL_CONTEXT_COMPILER_MODE_INTELFPGA","3",0);
          }
          // create inference engines
          for (int i = 0; i < replicas; ++i) {
            if (is_fpga) {
              IEs.insert(IEs.begin(), factory.create_inference_engine(conf));
            } else {
              IEs.push_back(
                  factory.create_inference_engine(conf));
            }
          }
        }

        // task queue - Not necessary used with CPU inference
        object_detection_mq<single_bell>::ptr TaskQueue =
            std::make_shared<object_detection_mq<single_bell>>();

        // listening worker
        rpc_listen_worker listener{TaskQueue};

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
    }
};

inline server::server(const std::string& json_file) {
  JSON config;
  bpt::read_json(json_file, config);
  std::cout << "Server configuration:" << std::endl;
  bpt::write_json(std::cout, config);
  const std::string protocol = config.get<std::string>("protocol");
  if (protocol == "http") {
    actual = new http_server(config);
  }
  else {
    actual = new grpc_server(config);
  }
}

}