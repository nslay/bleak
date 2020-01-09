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

#pragma once

#ifndef BLEAK_COMMON_H
#define BLEAK_COMMON_H

#include <sstream>
#include <vector>
#include <string>
#include <random>

namespace bleak {

typedef std::mt19937_64 GeneratorType;

GeneratorType & GetGenerator();

void Trim(std::string &strValue);

std::string DirName(std::string strPath);

bool FileExists(const std::string &strPath);
bool IsFolder(const std::string &strPath);

void FindFiles(const char *p_cDir, const char *p_cPattern, std::vector<std::string> &vFiles, bool bRecursive = false);
void FindFolders(const char *p_cDir, const char *p_cPattern, std::vector<std::string> &vFolders, bool bRecursive = false);

template<typename ValueType>
std::vector<ValueType> SplitString(const std::string &strValue,const std::string &strDelim);

template<>
std::vector<std::string> SplitString<std::string>(const std::string &strString,const std::string &strDelim);

template<typename ValueType>
std::vector<ValueType> SplitString(const std::string &strString,const std::string &strDelim) {
  std::vector<std::string> vStringValues = SplitString<std::string>(strString,strDelim);

  std::vector<ValueType> vValues(vStringValues.size());

  std::stringstream valueStream;

  for(size_t i = 0; i < vStringValues.size(); ++i) {
    valueStream.clear();
    valueStream.str(vStringValues[i]);

    if(!(valueStream >> vValues[i]))
      return std::vector<ValueType>();
  }

  return vValues;
}


} // end namespace bleak

#endif // !BLEAK_COMMON_H
