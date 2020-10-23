#pragma once
#include <iostream>
#include <string>

namespace st {
namespace ie {

class device {
 public:
  std::string name;
  bool thread_safe;
};

}  // namespace ie
}  // namespace st