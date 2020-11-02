/***************************************************************************************
 * Copyright (C) 2020 canhld@.kaist.ac.kr
 * SPDX-License-Identifier: Apache-2.0
 * @b About: This file define the log objects for the whole system
 ***************************************************************************************/

#pragma once

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <memory>

namespace st {
namespace log {
  // framework specific log
  static std::shared_ptr<spdlog::logger> ovn_log = spdlog::stdout_color_mt("openvino");
  static std::shared_ptr<spdlog::logger> trt_log = spdlog::stdout_color_mt("tensorrt");
  // component log
  static std::shared_ptr<spdlog::logger> server_log = spdlog::stdout_color_mt("server");
  static std::shared_ptr<spdlog::logger> http_log = spdlog::stdout_color_mt("http");
  static std::shared_ptr<spdlog::logger> rpc_log = spdlog::stdout_color_mt("grpc");
  static std::shared_ptr<spdlog::logger> ie_log = spdlog::stdout_color_mt("ie");
  // file syslog for deploy
  static std::shared_ptr<spdlog::logger> file_log = spdlog::basic_logger_mt("file_log", "logs/logs.txt");

  void init_log() {
    #ifdef NDEBUG
      spdlog::set_level(spdlog::level::info);
    #else
      spdlog::set_level(spdlog::level::debug);
    #endif
    spdlog::set_pattern("[%H:%M:%S] [tid %t] [%n] [%l] %v");
  }
} // namespace log
} // namespace st



