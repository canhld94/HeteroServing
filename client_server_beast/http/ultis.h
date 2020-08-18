// Copyright (C) 2020 canhld@.kaist.ac.kr
// SPDX-License-Identifier: Apache-2.0
//
#ifndef NCL_ULTIS_H
#define NCL_ULTIS_H

#include <type_traits>
#include <typeinfo>
#ifndef _MSC_VER
#   include <cxxabi.h>
#endif
#include <memory>
#include <string>
#include <cstdlib>
#include <chrono>

// Hacker way to measure time
//! DON'T use it recursively. If you do it recursively, only read the innermost result
std::chrono::time_point<std::chrono::system_clock> PROFILE_start;
std::chrono::time_point<std::chrono::system_clock> PROFILE_end;
std::chrono::duration<double,std::milli> PROFILE_elapsed_seconds;
#define PROFILE_DEBUG(NAME,...) \
      PROFILE_start = std::chrono::system_clock::now();\
      __VA_ARGS__ \
      PROFILE_end = std::chrono::system_clock::now(); \
      PROFILE_elapsed_seconds = PROFILE_end - PROFILE_start; \
      std::cout <<"[::PROFILE] " << NAME << ": " << PROFILE_elapsed_seconds.count() << " ms" << std::endl; \

#define PROFILE_RELEASE(NAME,...) __VA_ARGS__


namespace ncl {
  /* 
    Type name template to retrieve type of variable 
    * Example
    int &foo;
    std::cout << typename<decltype(foo)>() << std::endl; 
    -> print int &
 */

  template <class T>
  std::string
  type_name()
  {
      typedef typename std::remove_reference<T>::type TR;
      std::unique_ptr<char, void(*)(void*)> own
            (
      #ifndef _MSC_VER
          abi::__cxa_demangle(typeid(TR).name(), nullptr,nullptr, nullptr),
      #else
                  nullptr,
      #endif
          std::free
        );
      std::string r = own != nullptr ? own.get() : typeid(TR).name();
      if (std::is_const<TR>::value)
          r += " const";
      if (std::is_volatile<TR>::value)
          r += " volatile";
      if (std::is_lvalue_reference<T>::value)
          r += "&";
      else if (std::is_rvalue_reference<T>::value)
          r += "&&";
      return r;
  }

  /* 
      base64 encode to send image over network
  */
  static const std::string base64_chars =
              "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
              "abcdefghijklmnopqrstuvwxyz"
              "0123456789+/";

  bool is_base64(unsigned char c) {
    return (isalnum(c) || (c == '+') || (c == '/'));
  }

  /* 
      base64 encoding
  */

  std::string base64_encode(unsigned char const* bytes_to_encode, unsigned int in_len) {
    std::string ret;
    int i = 0;
    int j = 0;
    unsigned char char_array_3[3];
    unsigned char char_array_4[4];

    while (in_len--) {
      char_array_3[i++] = *(bytes_to_encode++);
      if (i == 3) {
        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
        char_array_4[3] = char_array_3[2] & 0x3f;

        for(i = 0; (i <4) ; i++)
          ret += base64_chars[char_array_4[i]];
        i = 0;
      }
    }

    if (i)
    {
      for(j = i; j < 3; j++)
        char_array_3[j] = '\0';

      char_array_4[0] = ( char_array_3[0] & 0xfc) >> 2;
      char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
      char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);

      for (j = 0; (j < i + 1); j++)
        ret += base64_chars[char_array_4[j]];

      while((i++ < 3))
        ret += '=';

    }

    return ret;

  }



  /* 
      base64 decoding
  */

  std::string base64_decode(std::string const& encoded_string) {
    int in_len = encoded_string.size();
    int i = 0;
    int j = 0;
    int in_ = 0;
    unsigned char char_array_4[4], char_array_3[3];
    std::string ret;

    while (in_len-- && ( encoded_string[in_] != '=') && is_base64(encoded_string[in_])) {
      char_array_4[i++] = encoded_string[in_]; in_++;
      if (i ==4) {
        for (i = 0; i <4; i++)
          char_array_4[i] = base64_chars.find(char_array_4[i]);

        char_array_3[0] = ( char_array_4[0] << 2       ) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
        char_array_3[2] = ((char_array_4[2] & 0x3) << 6) +   char_array_4[3];

        for (i = 0; (i < 3); i++)
          ret += char_array_3[i];
        i = 0;
      }
    }

    if (i) {
      for (j = 0; j < i; j++)
        char_array_4[j] = base64_chars.find(char_array_4[j]);

      char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
      char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);

      for (j = 0; (j < i - 1); j++) ret += char_array_3[j];
    }

    return ret;
  }
}
#endif