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

#ifndef BLEAK_PROPERTY_H
#define BLEAK_PROPERTY_H

#include <cstddef>
#include <vector>
#include <string>
#include <memory>
#include <functional>

namespace bleak {

// Simple properties
class Property {
public:
  enum DataType { STRING, FLOAT, INTEGER, BOOL, FLOAT_VECTOR, INTEGER_VECTOR };

  virtual ~Property() { }

  static std::unique_ptr<Property> New(std::string &strPersistentValue);
  static std::unique_ptr<Property> New(float &fPersistentValue);
  static std::unique_ptr<Property> New(int &iPersistentValue);
  static std::unique_ptr<Property> New(bool &bPersistentValue);
  static std::unique_ptr<Property> New(std::vector<float> &vPersistentValue);
  static std::unique_ptr<Property> New(std::vector<int> &vPersistentValue);

  static std::unique_ptr<Property> New(std::function<bool(std::string &)> funGetter, std::function<bool(const std::string &)> funSetter);
  static std::unique_ptr<Property> New(std::function<bool(float &)> funGetter, std::function<bool(const float &)> funSetter);
  static std::unique_ptr<Property> New(std::function<bool(int &)> funGetter, std::function<bool(const int &)> funSetter);
  static std::unique_ptr<Property> New(std::function<bool(bool &)> funGetter, std::function<bool(const bool &)> funSetter);
  static std::unique_ptr<Property> New(std::function<bool(std::vector<float> &)> funGetter, std::function<bool(const std::vector<float> &)> funSetter);
  static std::unique_ptr<Property> New(std::function<bool(std::vector<int> &)> funGetter, std::function<bool(const std::vector<int> &)> funSetter);

  virtual DataType GetDataType() const = 0;

  virtual bool SetValue(std::nullptr_t) { return false; }
  virtual bool SetValue(const char *p_cString) { return SetValue(std::string(p_cString)); }

  virtual bool SetValue(const std::string &strValue) = 0;
  virtual bool SetValue(float fValue) = 0;
  virtual bool SetValue(int iValue) = 0;
  virtual bool SetValue(bool bValue) = 0;
  virtual bool SetValue(const std::vector<float> &vValue) = 0;
  virtual bool SetValue(const std::vector<int> &vValue) = 0;

  virtual bool GetValue(std::string &strValue) const = 0;
  virtual bool GetValue(float &fValue) const = 0;
  virtual bool GetValue(int &iValue) const = 0;
  virtual bool GetValue(bool &bValue) const = 0;
  virtual bool GetValue(std::vector<float> &vValue) const = 0;
  virtual bool GetValue(std::vector<int> &vValue) const = 0;
};

} // end namespace bleak

#endif // !BLEAK_PROPERTY_H
