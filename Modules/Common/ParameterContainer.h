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

#ifndef BLEAK_PARAMETERCONTAINER_H
#define BLEAK_PARAMETERCONTAINER_H

#include <sstream>
#include <unordered_map>
#include <vector>
#include "Common.h"
#include "IniFile.h"

namespace bleak {

class ParameterContainer {
public:
  virtual ~ParameterContainer() { }

  virtual bool HasKey(const std::string &strKey) const {
    std::string strValue;
    return GetStringValue(strValue, strKey);
  }

  template<typename ValueType>
  ValueType GetValue(const std::string &strKey, const ValueType &defaultValue) const;

  template<typename ValueType>
  bool SetValue(const std::string &strKey, const ValueType &value);

protected:
  virtual bool GetStringValue(std::string &strValue, const std::string &strKey) const = 0;
  virtual bool SetStringValue(const std::string &strKey, const std::string &strValue) {
    return false;
  }
};

class ParameterMap : public ParameterContainer {
public:
  ParameterMap()
  : m_p_clBelow(nullptr) { }

  ParameterMap(const ParameterContainer &clBelow)
  : m_p_clBelow(&clBelow) { }

  virtual ~ParameterMap() { }

protected:
  virtual bool GetStringValue(std::string &strValue, const std::string &strKey) const override;

  virtual bool SetStringValue(const std::string &strKey, const std::string &strValue) override {
    m_mValueMap[strKey] = strValue;
    return true;
  }

private:
  typedef std::unordered_map<std::string, std::string> MapType;

  MapType m_mValueMap;
  const ParameterContainer *m_p_clBelow;
};

class ParameterFile : public ParameterContainer {
public:
  ParameterFile(const std::string &strFile, const std::string &strSection) {
    m_clIniFile.Load(strFile);
    SetSection(strSection);
  }

  virtual ~ParameterFile() { }

  virtual void SetSection(const std::string &strSection) {
    m_strSection = strSection;
  }

protected:
  virtual bool GetStringValue(std::string &strValue, const std::string &strKey) const override {
    if (!m_clIniFile.HasSection(m_strSection))
      return false;

    const IniFile::Section &clSection = m_clIniFile.GetSection(m_strSection);

    strValue = clSection.GetValue<std::string>(strKey, "DEFAULT VALUE");

    return strValue != "DEFAULT VALUE";
  }

private:
  mutable IniFile m_clIniFile;
  std::string m_strSection;
};

template<typename ValueType>
ValueType ParameterContainer::GetValue(const std::string &strKey, const ValueType &defaultValue) const {
  std::string strValue;
  if (!GetStringValue(strValue, strKey))
    return defaultValue;

  std::stringstream valueStream;
  valueStream.str(strValue);

  ValueType value = ValueType();

  return !(valueStream >> value) ? defaultValue : value;
}

template<>
inline std::string ParameterContainer::GetValue<std::string>(const std::string &strKey, const std::string &strDefaultValue) const {
  std::string strValue;
  return GetStringValue(strValue, strKey) ? strValue : strDefaultValue;
}

template<typename ValueType>
bool ParameterContainer::SetValue(const std::string &strKey, const ValueType &value) {
  std::stringstream valueStream;

  if (!(valueStream << value) || !SetStringValue(strKey, valueStream.str()))
    return false;

  return true;
}

template<>
inline bool ParameterContainer::SetValue<std::string>(const std::string &strKey, const std::string &strValue) {
  return SetStringValue(strKey, strValue);
}

inline bool ParameterMap::GetStringValue(std::string &strValue,const std::string &strKey) const {
  MapType::const_iterator itr = m_mValueMap.find(strKey);

  if (itr != m_mValueMap.end()) {
    strValue = itr->second;
    return true;
  }

  if (m_p_clBelow != nullptr && m_p_clBelow->HasKey(strKey)) {
    strValue = m_p_clBelow->GetValue(strKey, std::string());
    return true;
  }

  return false;
}

} // end namespace bleak

#endif // !BLEAK_PARAMETERCONTAINER_H
