#pragma once 
#include <string>
#include <iostream>


namespace st {
namespace ie {
    
    class device {
    public:
        std::string name;
        bool thread_safe;
    };

} // namespace ie
} // namespace st