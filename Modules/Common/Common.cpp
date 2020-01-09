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

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#elif defined(__unix__)
#include <sys/types.h>
#include <sys/stat.h>
#include <glob.h>
#else
#error "Not implemented."
#endif // _WIN32

#include <cstring>
#include <algorithm>
#include <string>
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

#ifdef _WIN32
bool FileExists(const std::string &strPath) {
  return GetFileAttributes(strPath.c_str()) != INVALID_FILE_ATTRIBUTES;
}
#endif // _WIN32

#ifdef __unix__
bool FileExists(const std::string &strPath) {
  struct stat stBuff = {};
  return stat(strPath.c_str(), &stBuff) == 0;
}
#endif // __unix__

#ifdef _WIN32
bool IsFolder(const std::string &strPath) {
  const DWORD dwFlags = GetFileAttributes(strPath.c_str());
  return dwFlags != INVALID_FILE_ATTRIBUTES ? ((dwFlags & FILE_ATTRIBUTE_DIRECTORY) != 0) : false;
}
#endif // _WIN32

#ifdef __unix__
bool IsFolder(const std::string &strPath) {
  struct stat stBuff = {};

  if (stat(strPath.c_str(), &stBuff) != 0)
    return false;

  return S_ISDIR(stBuff.st_mode) != 0;
}
#endif // __unix__

#ifdef _WIN32
void FindFiles(const char *p_cDir, const char *p_cPattern, std::vector<std::string> &vFiles, bool bRecursive) {
  std::string strPattern(p_cDir);
  strPattern += '\\';
  strPattern += p_cPattern;

  WIN32_FIND_DATA stFindData;

  memset(&stFindData, 0, sizeof(stFindData));
  
  HANDLE hFind = FindFirstFile(strPattern.c_str(), &stFindData);

  if (hFind != INVALID_HANDLE_VALUE) {
    do {
      if (!(stFindData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
        std::string strPath(p_cDir);
        strPath += '\\';
        strPath += stFindData.cFileName;

        vFiles.push_back(strPath);
      }
    } while (FindNextFile(hFind, &stFindData) != FALSE);

    FindClose(hFind);
  }

  if (bRecursive) {
    strPattern = p_cDir;
    strPattern += "\\*";

    memset(&stFindData, 0, sizeof(stFindData));

    hFind = FindFirstFile(strPattern.c_str(), &stFindData);

    if (hFind == INVALID_HANDLE_VALUE)
      return;

    do {
      if (strcmp(stFindData.cFileName, ".") == 0 || strcmp(stFindData.cFileName, "..") == 0)
        continue;

      if (stFindData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
        std::string strPath(p_cDir);
        strPath += '\\';
        strPath += stFindData.cFileName;

        FindFiles(strPath.c_str(), p_cPattern, vFiles, bRecursive);
      }
    } while (FindNextFile(hFind, &stFindData) != FALSE);

    FindClose(hFind);
  }

  return;
}
#endif // _WIN32

#ifdef __unix__
void FindFiles(const char *p_cDir, const char *p_cPattern, std::vector<std::string> &vFiles, bool bRecursive) {
  glob_t stGlob;

  std::memset(&stGlob, 0, sizeof(stGlob));

  std::string strPattern = p_cDir; 
  strPattern += '/';
  strPattern += p_cPattern;

  if (glob(strPattern.c_str(), GLOB_TILDE, nullptr, &stGlob) == 0) {
    for (size_t i = 0; i < stGlob.gl_pathc; ++i) {
      std::string strPath = stGlob.gl_pathv[i];
      if (!IsFolder(strPath))
        vFiles.push_back(std::move(strPath));
    }
  }

  globfree(&stGlob);

  if (bRecursive) {
    std::memset(&stGlob, 0, sizeof(stGlob));

    strPattern = p_cDir;
    strPattern += "/*";

    if (glob(strPattern.c_str(), GLOB_TILDE, nullptr, &stGlob) == 0) {
      for (size_t i = 0; i < stGlob.gl_pathc; ++i) {
        const std::string strPath = stGlob.gl_pathv[i];
        if (strPath != "." && strPath != ".." && IsFolder(strPath))
          FindFiles(strPath.c_str(), p_cPattern, vFiles, bRecursive);
      }
    }

    globfree(&stGlob);
  }
}
#endif // __unix__

#ifdef _WIN32
void FindFolders(const char *p_cDir, const char *p_cPattern, std::vector<std::string> &vFolders, bool bRecursive) {
  std::string strPattern(p_cDir);
  strPattern += '\\';
  strPattern += p_cPattern;

  WIN32_FIND_DATA stFindData;

  std::memset(&stFindData, 0, sizeof(stFindData));

  HANDLE hFind = FindFirstFile(strPattern.c_str(), &stFindData);

  if (hFind != INVALID_HANDLE_VALUE) {
    do {
      if ((stFindData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) && std::strcmp(stFindData.cFileName,".") != 0 && std::strcmp(stFindData.cFileName,"..") != 0) {
        std::string strPath(p_cDir);
        strPath += '\\';
        strPath += stFindData.cFileName;

        vFolders.push_back(strPath);
      }
    } while (FindNextFile(hFind, &stFindData) != FALSE);

    FindClose(hFind);
  }

  if (bRecursive) {

    strPattern = p_cDir;
    strPattern += "\\*";

    std::memset(&stFindData, 0, sizeof(stFindData));

    hFind = FindFirstFile(strPattern.c_str(), &stFindData);

    if (hFind == INVALID_HANDLE_VALUE)
      return;

    do {
      if (std::strcmp(stFindData.cFileName, ".") == 0 || std::strcmp(stFindData.cFileName, "..") == 0)
        continue;

      if (stFindData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
        std::string strPath(p_cDir);
        strPath += '\\';
        strPath += stFindData.cFileName;

        FindFiles(strPath.c_str(), p_cPattern, vFolders, bRecursive);
      }
    } while (FindNextFile(hFind, &stFindData) != FALSE);

    FindClose(hFind);
  }

  return;
}
#endif // _WIN32

#ifdef __unix___
void FindFolders(const char *p_cDir, const char *p_cPattern, std::vector<std::string> &vFolders, bool bRecursive) {
  glob_t stGlob;

  std::memset(&stGlob, 0, sizeof(stGlob));

  std::string strPattern = p_cDir; 
  strPattern += '/';
  strPattern += p_cPattern;

  if (glob(strPattern.c_str(), GLOB_TILDE, nullptr, &stGlob) == 0) {
    for (size_t i = 0; i < stGlob.gl_pathc; ++i) {
      std::string strPath = stGlob.gl_pathv[i];
      if (strPath != "." && strPath != ".." && IsFolder(strPath))
        vFolders.push_back(std::move(strPath));
    }
  }

  globfree(&stGlob);

  if (bRecursive) {
    std::memset(&stGlob, 0, sizeof(stGlob));

    strPattern = p_cDir;
    strPattern += "/*";

    if (glob(strPattern.c_str(), GLOB_TILDE, nullptr, &stGlob) == 0) {
      for (size_t i = 0; i < stGlob.gl_pathc; ++i) {
        const std::string strPath = stGlob.gl_pathv[i];
        if (strPath != "." && strPath != ".." && IsFolder(strPath))
          FindFolders(strPath.c_str(), p_cPattern, vFolders, bRecursive);
      }
    }

    globfree(&stGlob);
  }
}
#endif // __unix__

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
