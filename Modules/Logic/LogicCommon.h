#pragma once

#ifndef BLEAK_LOGICCOMMON_H
#define BLEAK_LOGICCOMMON_H

#include <cctype>
#include <algorithm>
#include <functional>
#include <string>
#include <vector>
#include "Common.h"

namespace bleak {

template<typename RealType>
bool BitIsOn(const RealType &bit, const RealType &threshold = RealType(0), const RealType &on = RealType(1));

template<typename RealType>
std::vector<RealType> ConvertIntegerStringToBits(std::string strValue, const RealType &off = RealType(-1), const RealType &on = RealType(1));

template<typename RealType, typename IteratorType>
std::string ConvertBitsToIntegerString(IteratorType begin, IteratorType end, int iBase = 10, const RealType &threshold = RealType(0), const RealType &on = RealType(1));

template<typename RealType>
bool BitIsOn(const RealType &bit, const RealType &threshold, const RealType &on) {
  if (threshold < on)
    return bit > threshold;

  return bit < threshold;
}

template<typename RealType>
std::vector<RealType> ConvertIntegerStringToBits(std::string strValue, const RealType &off, const RealType &on) {
  Trim(strValue);

  if (strValue.empty())
    return std::vector<RealType>();

  std::vector<RealType> vBits;

  if (strValue[0] == '0') {
    // Octal or hex
    if (strValue.size() > 1 && strValue[1] == 'x') {
      // Hex
      if (strValue.size() == 2)
        return std::vector<RealType>(); // "0x" is not a valid hex number!

      auto endItr = strValue.rend();
      --endItr;
      --endItr; // Exclude "0x"

      vBits.reserve(4*(strValue.size()-2)); // 4 bits per digit

      for (auto itr = strValue.rbegin(); itr != endItr; ++itr) {
        int iDigit = 0;
        if (std::isalpha(*itr)) {
          iDigit = std::tolower(*itr) - 'a';

          if (iDigit < 0 || iDigit >= 6)
            return std::vector<RealType>(); // Invalid digit

          iDigit += 10;
        }
        else {
          iDigit = *itr - '0';

          if (iDigit < 0 || iDigit >= 10)
            return std::vector<RealType>(); // Invalid digit
        }

        vBits.push_back((iDigit & 1) ? on : off);
        vBits.push_back((iDigit & 2) ? on : off);
        vBits.push_back((iDigit & 4) ? on : off);
        vBits.push_back((iDigit & 8) ? on : off);
      }

      return vBits;
    }

    // Octal
    if (strValue.size() == 1) // Just a "0" string
      return { off };

    auto endItr = strValue.rend();
    --endItr; // Exclude '0'

    vBits.reserve(3*(strValue.size()-1)); // 3 bits per digit

    for (auto itr = strValue.rbegin(); itr != endItr; ++itr) {
      const int iDigit = *itr - '0';
      if (iDigit < 0 || iDigit >= 8)
        return std::vector<RealType>(); // Invalid digit

      vBits.push_back((iDigit & 1) ? on : off);
      vBits.push_back((iDigit & 2) ? on : off);
      vBits.push_back((iDigit & 4) ? on : off);
    }

    return vBits;
  }

  // Decimal
  std::vector<int> vDecimal;
  vDecimal.reserve(strValue.size());

  // Stored from least significant digit to most significant digit
  for (auto itr = strValue.rbegin(); itr != strValue.rend(); ++itr) {
    const int iDigit = *itr - '0';
    if (iDigit < 0 || iDigit >= 10)
      return std::vector<RealType>(); // Invalid digit

    vDecimal.push_back(iDigit);
  }

  vBits.reserve(4*vDecimal.size()); // ~4 bits per digit

  // Pop off each bit, one at a time
  while (!std::all_of(vDecimal.begin(), vDecimal.end(), std::logical_not<int>())) {
    // Calculate remainder of decimal integer
    vBits.push_back((vDecimal[0] & 1) ? on : off);

    vDecimal[0] /= 2;

    // Calculate quotient and carry
    for (size_t i = 1; i < vDecimal.size(); ++i) {
      if (vDecimal[i] & 1)
        vDecimal[i-1] += 5;

      vDecimal[i] /= 2;
    }
  }

  return vBits;
}

template<typename RealType, typename IteratorType>
std::string ConvertBitsToIntegerString(IteratorType begin, IteratorType end, int iBase, const RealType &threshold, const RealType &on) {
  switch (iBase) {
  case 8:
    {
      std::string strValue;
      const size_t length = end-begin;
      for (size_t i = 0; i < length; i += 3) {
        int iDigit = 0;

        switch (length-i) {
        default:
          iDigit |= BitIsOn(RealType(begin[i+2]), threshold, on) ? 4 : 0;
        case 2: // Fall through
          iDigit |= BitIsOn(RealType(begin[i+1]), threshold, on) ? 2 : 0;
        case 1: // Fall through
          iDigit |= BitIsOn(RealType(begin[i+0]), threshold, on) ? 1 : 0;
        case 0: // Uhh?
          break;
        }

        strValue += ('0' + iDigit);
      }

      strValue += '0'; // For octal notation
      
      std::reverse(strValue.begin(), strValue.end());

      return strValue;
    }
    break;
  case 10:
    {
      const int length = (int)(end-begin);

      if (length == 0)
        return std::string();

      std::vector<int> vDecimal(1, 0);

      for (int i = length-1; i >= 0; --i) {
        // Multiply digits by 2
        std::transform(vDecimal.begin(), vDecimal.end(), vDecimal.begin(),
          [](const int &x) -> int {
            return 2*x;
          });

        vDecimal[0] += BitIsOn(RealType(begin[i]), threshold, on) ? 1 : 0;

        // No digit should be larger than 19 after these two operations (and 18 for all but the first digit)

        if (vDecimal.back() >= 10)
          vDecimal.push_back(0);

        // Carry
        for (size_t j = vDecimal.size()-1; j > 0; --j) {
          if (vDecimal[j-1] >= 10) {
            vDecimal[j-1] -= 10;
            ++vDecimal[j];
          }
        }
      }

      std::string strValue;
      strValue.resize(vDecimal.size());

      std::transform(vDecimal.begin(), vDecimal.end(), strValue.rbegin(),
        [](const int &x) -> char {
          return '0' + x;
        });

      return strValue;
    }
    break;
  case 16:
    {
      std::string strValue;
      const size_t length = end-begin;
      for (size_t i = 0; i < length; i += 4) {
        int iDigit = 0;

        switch (length-i) {
        default:
          iDigit |= BitIsOn(RealType(begin[i+3]), threshold, on) ? 8 : 0;
        case 3: // Fall through
          iDigit |= BitIsOn(RealType(begin[i+2]), threshold, on) ? 4 : 0;
        case 2: // Fall through
          iDigit |= BitIsOn(RealType(begin[i+1]), threshold, on) ? 2 : 0;
        case 1: // Fall through
          iDigit |= BitIsOn(RealType(begin[i+0]), threshold, on) ? 1 : 0;
        case 0: // Uhh?
          break;
        }

        if (iDigit < 10)
          strValue += ('0' + iDigit);
        else
          strValue += ('a' + (iDigit-10));
      }

      strValue += "x0"; // For hex notation
      
      std::reverse(strValue.begin(), strValue.end());

      return strValue;
    }
    break;
  }

  return std::string();
}

} // end namespace bleak

#endif // !BLEAK_LOGICCOMMON_H
