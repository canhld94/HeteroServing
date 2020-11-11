/***************************************************************************************
 * Copyright (C) 2020 canhld@.kaist.ac.kr
 * SPDX-License-Identifier: Apache-2.0
 * @b About: This file implement the service that was defined in 
 * stubs/inference_rpc.proto
 ***************************************************************************************/

#include <iostream>
#include <string>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include "stubs/inference_rpc.grpc.pb.h"
#include "stubs/inference_rpc.pb.h"
#include "st_utils.h"
#include "st_ie_common.h" 

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using st::rpc::encoded_image;
using st::rpc::detection_output;
using st::rpc::inference_rpc;
using namespace st::sync;
using namespace st::worker;
using namespace st::ie;
using namespace st::log;

namespace st {
namespace rpc {
/**
 * @brief service implementation
 * 
 */
class inference_rpc_impl final : public inference_rpc::Service {
  public:
    inference_rpc_impl(object_detection_mq<single_bell>::ptr& _taskq) : 
      inference_rpc::Service() , taskq(_taskq) {
        bell = std::make_shared<single_bell>();
      };
    virtual Status run_detection(ServerContext* context, const encoded_image* request, detection_output* response) override {
      auto data = request->data().c_str();
      int sz = request->size();
      std::vector<bbox> prediction;
      obj_detection_msg<single_bell> m{data, sz, &prediction, bell};
      rpc_log->debug("Enqueue my task, current queue size {}",
              taskq->size());
      taskq->push(m);
      rpc_log->debug("Waiting for inference engine");
      bell->wait(1);
      rpc_log->debug("Received data");
      int n = prediction.size();
      for (int i = 0; i < n; ++i) {
        bbox& pred = prediction[i];
        auto rpc_bbox = response->add_bboxes();
        rpc_bbox->set_label_id(pred.label_id);
        rpc_bbox->set_label(pred.label);
        rpc_bbox->set_prob(pred.prop);
        if (pred.c[3]) {
          st::rpc::detection_output_rectangle *rec = new st::rpc::detection_output_rectangle();
          rec->set_xmin(pred.c[0]);
          rec->set_ymin(pred.c[1]);
          rec->set_xmax(pred.c[2]);
          rec->set_ymax(pred.c[3]);
          rpc_bbox->set_allocated_box(rec);
        }
      }
      return Status::OK;
    }
  private:
  object_detection_mq<single_bell>::ptr taskq;
  single_bell::ptr bell;
}; // class inference_rpc_impl

/**
 * @brief grpc listening worker
 * 
 */
class rpc_listen_worker {
  public:
    rpc_listen_worker(object_detection_mq<single_bell>::ptr& _taskq)
        : taskq(_taskq) {}
    ~rpc_listen_worker() {}
    void operator()() {
      pthread_setname_np(pthread_self(), "rpc listener");
      rpc_log->warn("No IP and address is provide");
      rpc_log->warn("Use defaul address 0.0.0.0 and default port 8080");
      listen("0.0.0.0", "8080");
    }
    void operator()(std::string& ip, std::string& port) {
      pthread_setname_np(pthread_self(), "rpc listener");
      listen(ip.c_str(), port.c_str());
    }
  private:
    object_detection_mq<single_bell>::ptr taskq;
    void listen(const char* ip, const char* p) {
      std::string address(ip);
      std::string port(p);
      std::string binding = address + ":" + port;
      inference_rpc_impl service(taskq);
      grpc::EnableDefaultHealthCheckService(true);
      grpc::reflection::InitProtoReflectionServerBuilderPlugin();
      ServerBuilder builder;
      // Listen on the given address without any authentication mechanism.
      builder.AddListeningPort(binding, grpc::InsecureServerCredentials());
      builder.RegisterService(&service);
       // Finally assemble the server.
      std::unique_ptr<Server> server(builder.BuildAndStart());
      rpc_log->info("Server listening on {}",binding);

      // Wait for the server to shutdown. Note that some other thread must be
      // responsible for shutting down the server for this call to ever return.
      server->Wait();
  }
}; // class grpc_listen_worker
} // namespace rpc
} // namespace st
