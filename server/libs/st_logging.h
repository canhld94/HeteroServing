
#pragma once

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <memory>

namespace st {
namespace log {
  static std::shared_ptr<spdlog::logger> console = spdlog::stdout_color_mt("console");
  static std::shared_ptr<spdlog::logger> file_log = spdlog::basic_logger_mt("server_log", "logs/logs.txt");

  void init_log() {
    #ifdef NDEBUG
      spdlog::set_level(spdlog::level::info);
    #else
      spdlog::set_level(spdlog::level::debug);
    #endif
  }
}
}



