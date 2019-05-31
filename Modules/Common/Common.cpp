/*-
 * Copyright (c) 2017 Nathan Lay (enslay@gmail.com)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR(S) BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "Common.h"

namespace bleak {

GeneratorType & GetGenerator() {
  static GeneratorType clGenerator;
  return clGenerator;
}

void Trim(std::string &strValue) {
  size_t p = strValue.find_first_not_of(" \t\r\n");

  if(p != std::string::npos)
    strValue.erase(0,p);
  else
    strValue.clear();

  p = strValue.find_last_not_of(" \t\r\n");

  if(p != std::string::npos && p+1 < strValue.size())
    strValue.erase(p+1);
}

std::string DirName(std::string strPath) {
#ifdef _WIN32
  while (strPath.size() > 1 && (strPath.back() == '/' || strPath.back() == '\\'))
    strPath.pop_back();

  size_t p = strPath.find_last_of("/\\");
#else // !_WIN32
  while (strPath.size() > 1 && strPath.back() == '/')
    strPath.pop_back();

  size_t p = strPath.find_last_of("/");
#endif // _WIN32

  if (p == std::string::npos)
    return std::string(".");

  strPath.erase(p);

  return strPath.size() > 0 ? strPath : std::string("/");
}

template<>
std::vector<std::string> SplitString<std::string>(const std::string &strString,const std::string &strDelim) {
  if(strDelim.empty() || strString.empty())
    return std::vector<std::string>();

  size_t p = 0,q = 0;

  std::vector<std::string> vValues;

  std::string strToken;
  while(p < strString.size()) {
    q = strString.find_first_of(strDelim,p);

    if(q != std::string::npos) {
      strToken = strString.substr(p,q-p);
    } 
    else {
      strToken = strString.substr(p);
      q = strString.size();
    }

    Trim(strToken);

    vValues.push_back(strToken);

    p = q+1;
  }

  return vValues;
}

} // end namespace bleak
