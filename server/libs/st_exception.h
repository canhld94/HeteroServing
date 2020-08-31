/***************************************************************************************
 * Copyright (C) 2020 canhld@.kaist.ac.kr
 * SPDX-License-Identifier: Apache-2.0
 * @b About: This file implement the exception type of the project
 ***************************************************************************************/

#pragma once

#include <exception>

namespace st {
namespace exception {

    class st_exception : public std::exception {
    public:
        virtual char const * what() const noexcept override {
            return "Internal error";
        }
    };

    class ie_exception : public st_exception {
    public:
        virtual char const * what() const noexcept override {
            return "Inference engine error";
        }
    };

    class server_exception : public st_exception {
    public:
        virtual char const * what() const noexcept override {
            return "Server error";
        }
    };

    class ie_not_implemented : public ie_exception {
    public:
        char const * what() const noexcept override {
            return "Model not yet implemented";
        }
    };

    class fpga_overused : public server_exception {
    public:
        char const * what() const noexcept override {
            return "Two inference engines cannot shared fpga device";
        }
    };
} // namespace exception
} // namespace ie