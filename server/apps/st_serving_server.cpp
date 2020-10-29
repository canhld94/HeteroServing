/***************************************************************************************
 * Copyright (C) 2020 canhld@.kaist.ac.kr
 * SPDX-License-Identifier: Apache-2.0
 * @b About: This file is the main application
 ***************************************************************************************/

#include <gflags/gflags.h>
#include <iostream>
#include "st_server.h"
#include "st_logging.h"

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
    // Check command line arguments.
    if (!parse_and_check_cmd_line(argc, const_cast<char**>(argv))) {
      return 0;
    }
    st::log::init_log();
    st::server m_server(FLAGS_f);
    m_server.run();
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