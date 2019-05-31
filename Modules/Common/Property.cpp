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

#include <sstream>
#include "Common.h"
#include "Property.h"

namespace bleak {

template<typename ToT, typename FromT>
class PropertyCaster {
public:
  static bool Cast(ToT &, const FromT &) { return false; }
};

template<typename T>
class PropertyCaster<T,T> {
public:
  static bool Cast(T &toValue, const T &fromValue) {
    toValue = fromValue;
    return true;
  }
};

template<>
class PropertyCaster<std::string,float> {
public:
  static bool Cast(std::string &strValue, const float &fValue) {
    strValue = std::to_string(fValue);
    return true;
  }
};

template<>
class PropertyCaster<std::string,int> {
public:
  static bool Cast(std::string &strValue, const int &iValue) {
    strValue = std::to_string(iValue);
    return true;
  }
};

template<>
class PropertyCaster<std::string, bool> {
public:
  static bool Cast(std::string &strValue, const bool &bValue) {
    strValue = (bValue ? "true" : "false");
    return true;
  }
};

template<>
class PropertyCaster<std::string, std::vector<float>> {
public:
  static bool Cast(std::string &strValue, const std::vector<float> &vValue) {
    if (vValue.empty()) {
      strValue.clear();
      return true;
    }

    std::stringstream valueStream;
    valueStream << vValue[0];
    for (size_t i = 1; i < vValue.size(); ++i)
      valueStream << ',' << vValue[i];

    strValue = valueStream.str();

    return true;
  }
};

template<>
class PropertyCaster<std::string, std::vector<int>> {
public:
  static bool Cast(std::string &strValue, const std::vector<int> &vValue) {
    if (vValue.empty()) {
      strValue.clear();
      return true;
    }

    std::stringstream valueStream;
    valueStream << vValue[0];
    for (size_t i = 1; i < vValue.size(); ++i)
      valueStream << ',' << vValue[i];

    strValue = valueStream.str();

    return true;
  }
};

template<>
class PropertyCaster<float, std::string> {
public:
  static bool Cast(float &fValue, const std::string &strValue) {
    std::stringstream valueStream;
    valueStream.str(strValue);

    if (!(valueStream >> fValue))
      return false;

    return true;
  }
};

template<>
class PropertyCaster<int, std::string> {
public:
  static bool Cast(int &iValue, const std::string &strValue) {
    std::stringstream valueStream;
    valueStream.str(strValue);

    if (!(valueStream >> iValue))
      return false;

    return true;
  }
};

template<>
class PropertyCaster<std::vector<float>, std::string> {
public:
  static bool Cast(std::vector<float> &vValues, const std::string &strValue) {
    vValues = SplitString<float>(strValue, std::string(","));
    return vValues.size() > 0;
  }
};

template<>
class PropertyCaster<std::vector<int>, std::string> {
public:
  static bool Cast(std::vector<int> &vValues, const std::string &strValue) {
    vValues = SplitString<int>(strValue, std::string(","));
    return true;
  }
};

template<>
class PropertyCaster<float, int> {
public:
  static bool Cast(float &fValue, const int &iValue) {
    fValue = (float)iValue;
    return true;
  }
};

template<>
class PropertyCaster<float, bool> {
public:
  static bool Cast(float &fValue, const bool &bValue) {
    fValue = (bValue ? 1.0f : 0.0f);
    return true;
  }
};

template<>
class PropertyCaster<float, std::vector<float>> {
public:
  static bool Cast(float &fValue, const std::vector<float> &vValue) {
    if (vValue.size() != 1)
      return false;

    fValue = vValue[0];

    return true;
  }
};

template<>
class PropertyCaster<float, std::vector<int>> {
public:
  static bool Cast(float &fValue, const std::vector<int> &vValue) {
    if (vValue.size() != 1)
      return false;

    fValue = (float)vValue[0];

    return true;
  }
};

template<>
class PropertyCaster<int, bool> {
public:
  static bool Cast(int &iValue, const bool &bValue) {
    iValue = (bValue ? 1 : 0);
    return true;
  }
};

template<>
class PropertyCaster<int, std::vector<int>> {
public:
  static bool Cast(int &iValue, const std::vector<int> &vValue) {
    if (vValue.size() != 1)
      return false;

    iValue = vValue[0];

    return true;
  }
};

template<>
class PropertyCaster<bool, std::string> {
public:
  static bool Cast(bool &bValue, const std::string &strValue) {
    std::string strTmpValue = strValue;
    Trim(strTmpValue);
    if (strTmpValue == "true") {
      bValue = true;
      return true;
    }

    if (strTmpValue == "false") {
      bValue = false;
      return true;
    }

    std::stringstream valueStream;
    valueStream.str(strTmpValue);

    int iValue = 0;
    if (!(valueStream >> iValue))
      return false;

    bValue = (iValue != 0);

    return true;
  }
};

template<>
class PropertyCaster<bool, int> {
public:
  static bool Cast(bool &bValue, const int &iValue) {
    bValue = (iValue != 0);
    return true;
  }
};

template<>
class PropertyCaster<bool, std::vector<int>> {
public:
  static bool Cast(bool &bValue, const std::vector<int> &vValue) {
    if (vValue.size() != 1)
      return false;

    bValue = (vValue[0] != 0);

    return true;
  }
};

template<>
class PropertyCaster<std::vector<float>, std::vector<int>> {
public:
  static bool Cast(std::vector<float> &vFloatValue, const std::vector<int> &vIntValue) {
    vFloatValue.resize(vIntValue.size());

    for (size_t i = 0; i < vIntValue.size(); ++i)
      vFloatValue[i] = (float)vIntValue[i];

    return true;
  }
};

template<typename ValueType>
class PropertyCaster<std::function<bool(const ValueType &)>, ValueType> {
public:
  typedef std::function<bool(const ValueType &)> SetterType;
  static bool Cast(SetterType funSetter, const ValueType &value) {
    return funSetter(value);
  }
};

template<typename ValueType>
class PropertyCaster<ValueType, std::function<bool(ValueType &)>> {
public:
  typedef std::function<bool(ValueType &)> GetterType;
  static bool Cast(ValueType &value, GetterType funGetter) {
    return funGetter(value);
  }
};

#define DECLARE_GETTERSETTER_CASTER(targetType, fromType) \
template<> \
class PropertyCaster< targetType , std::function<bool( fromType & )>> { \
public: \
  typedef std::function<bool( fromType & )> GetterType; \
  static bool Cast( targetType &value , GetterType funGetter) { \
    fromType tmpValue; \
    return funGetter(tmpValue) && PropertyCaster< targetType , fromType >::Cast(value, tmpValue); \
  } \
}; \
template<> \
class PropertyCaster<std::function<bool(const targetType &)>, fromType > { \
public: \
    typedef std::function<bool(const targetType &)> SetterType; \
    static bool Cast(SetterType funSetter, const fromType &value) { \
      targetType tmpValue; \
      return PropertyCaster< targetType , fromType >::Cast(tmpValue,value) && funSetter(tmpValue); \
  } \
}

DECLARE_GETTERSETTER_CASTER(std::string, float);
DECLARE_GETTERSETTER_CASTER(std::string, int);
DECLARE_GETTERSETTER_CASTER(std::string, bool);
DECLARE_GETTERSETTER_CASTER(std::string, std::vector<float>);
DECLARE_GETTERSETTER_CASTER(std::string, std::vector<int>);

DECLARE_GETTERSETTER_CASTER(float, std::string);
DECLARE_GETTERSETTER_CASTER(int, std::string);
DECLARE_GETTERSETTER_CASTER(std::vector<float>, std::string);
DECLARE_GETTERSETTER_CASTER(std::vector<int>, std::string);

DECLARE_GETTERSETTER_CASTER(float, int);
DECLARE_GETTERSETTER_CASTER(float, bool);
DECLARE_GETTERSETTER_CASTER(float, std::vector<float>);
DECLARE_GETTERSETTER_CASTER(float, std::vector<int>);

DECLARE_GETTERSETTER_CASTER(int, bool);
DECLARE_GETTERSETTER_CASTER(int, std::vector<int>);

DECLARE_GETTERSETTER_CASTER(bool, int);
DECLARE_GETTERSETTER_CASTER(bool, std::vector<int>);
DECLARE_GETTERSETTER_CASTER(bool, std::string);

#undef DECLARE_GETTERSETTER_CASTER

//////////////////// Property ////////////////////

template<typename ValueType>
class PropertyTraits { };

#define DECLARE_PROPERTY_TRAITS(nativeType, propertyType) \
template<> \
class PropertyTraits< nativeType > { \
public: \
  static Property::DataType GetDataType() { return Property:: propertyType ; } \
}

DECLARE_PROPERTY_TRAITS(std::string, STRING);
DECLARE_PROPERTY_TRAITS(float, FLOAT);
DECLARE_PROPERTY_TRAITS(int, INTEGER);
DECLARE_PROPERTY_TRAITS(bool, BOOL);
DECLARE_PROPERTY_TRAITS(std::vector<float>, FLOAT_VECTOR);
DECLARE_PROPERTY_TRAITS(std::vector<int>, INTEGER_VECTOR);

#undef DECLARE_PROPERTY_TRAITS

template<typename ValueType>
class PropertyTemplate : public Property {
public:
  PropertyTemplate(ValueType &value)
  : m_value(value) { }

  virtual ~PropertyTemplate() { }

  virtual DataType GetDataType() const { return PropertyTraits<ValueType>::GetDataType(); }

  virtual bool SetValue(const std::string &strValue) { return PropertyCaster<ValueType, std::string>::Cast(m_value, strValue); }
  virtual bool SetValue(float fValue) { return PropertyCaster<ValueType, float>::Cast(m_value, fValue); }
  virtual bool SetValue(int iValue) { return PropertyCaster<ValueType, int>::Cast(m_value, iValue); }
  virtual bool SetValue(bool bValue) { return PropertyCaster<ValueType, bool>::Cast(m_value, bValue); }
  virtual bool SetValue(const std::vector<float> &vValue) { return PropertyCaster<ValueType, std::vector<float>>::Cast(m_value, vValue); }
  virtual bool SetValue(const std::vector<int> &vValue) { return PropertyCaster<ValueType, std::vector<int>>::Cast(m_value, vValue); }

  virtual bool GetValue(std::string &strValue) const { return PropertyCaster<std::string, ValueType>::Cast(strValue, m_value); }
  virtual bool GetValue(float &fValue) const { return PropertyCaster<float, ValueType>::Cast(fValue, m_value); }
  virtual bool GetValue(int &iValue) const { return PropertyCaster<int, ValueType>::Cast(iValue, m_value); }
  virtual bool GetValue(bool &bValue) const { return PropertyCaster<bool, ValueType>::Cast(bValue, m_value); }
  virtual bool GetValue(std::vector<float> &vValue) const { return PropertyCaster<std::vector<float>, ValueType>::Cast(vValue, m_value); }
  virtual bool GetValue(std::vector<int> &vValue) const { return PropertyCaster<std::vector<int>, ValueType>::Cast(vValue, m_value); }

private:
  ValueType &m_value;
};

template<typename ValueType>
class PropertyFunctor : public Property {
public:
  typedef std::function<bool(ValueType &)> GetterType;
  typedef std::function<bool(const ValueType &)> SetterType;

  PropertyFunctor(GetterType funGetter, SetterType funSetter)
  : m_funGetter(funGetter), m_funSetter(funSetter) { }

  virtual ~PropertyFunctor() { }

  virtual DataType GetDataType() const { return PropertyTraits<ValueType>::GetDataType(); }

  virtual bool SetValue(const std::string &strValue) { return PropertyCaster<SetterType, std::string>::Cast(m_funSetter, strValue); }
  virtual bool SetValue(float fValue) { return PropertyCaster<SetterType, float>::Cast(m_funSetter, fValue); }
  virtual bool SetValue(int iValue) { return PropertyCaster<SetterType, int>::Cast(m_funSetter, iValue); }
  virtual bool SetValue(bool bValue) { return PropertyCaster<SetterType, bool>::Cast(m_funSetter, bValue); }
  virtual bool SetValue(const std::vector<float> &vValue) { return PropertyCaster<SetterType, std::vector<float>>::Cast(m_funSetter, vValue); }
  virtual bool SetValue(const std::vector<int> &vValue) { return PropertyCaster<SetterType, std::vector<int>>::Cast(m_funSetter, vValue); }

  virtual bool GetValue(std::string &strValue) const { return PropertyCaster<std::string, GetterType>::Cast(strValue, m_funGetter); }
  virtual bool GetValue(float &fValue) const { return PropertyCaster<float, GetterType>::Cast(fValue, m_funGetter); }
  virtual bool GetValue(int &iValue) const { return PropertyCaster<int, GetterType>::Cast(iValue, m_funGetter); }
  virtual bool GetValue(bool &bValue) const { return PropertyCaster<bool, GetterType>::Cast(bValue, m_funGetter); }
  virtual bool GetValue(std::vector<float> &vValue) const { return PropertyCaster<std::vector<float>, GetterType>::Cast(vValue, m_funGetter); }
  virtual bool GetValue(std::vector<int> &vValue) const { return PropertyCaster<std::vector<int>, GetterType>::Cast(vValue, m_funGetter); }

private:
  GetterType m_funGetter;
  SetterType m_funSetter;

  PropertyFunctor(const PropertyFunctor &) = delete;
  PropertyFunctor & operator=(const PropertyFunctor &) = delete;
};

std::unique_ptr<Property> Property::New(std::string &strPersistentValue) { return std::make_unique<PropertyTemplate<std::string>>(strPersistentValue); }
std::unique_ptr<Property> Property::New(float &fPersistentValue) { return std::make_unique<PropertyTemplate<float>>(fPersistentValue); }
std::unique_ptr<Property> Property::New(int &iPersistentValue) { return std::make_unique<PropertyTemplate<int>>(iPersistentValue); }
std::unique_ptr<Property> Property::New(bool &bPersistentValue) { return std::make_unique<PropertyTemplate<bool>>(bPersistentValue); }
std::unique_ptr<Property> Property::New(std::vector<float> &vPersistentValue) { return std::make_unique<PropertyTemplate<std::vector<float>>>(vPersistentValue); }
std::unique_ptr<Property> Property::New(std::vector<int> &vPersistentValue) { return std::make_unique<PropertyTemplate<std::vector<int>>>(vPersistentValue); }

std::unique_ptr<Property> Property::New(std::function<bool(std::string &)> funGetter, std::function<bool(const std::string &)> funSetter) { return std::make_unique<PropertyFunctor<std::string>>(funGetter, funSetter); }
std::unique_ptr<Property> Property::New(std::function<bool(float &)> funGetter, std::function<bool(const float &)> funSetter) { return std::make_unique<PropertyFunctor<float>>(funGetter, funSetter); }
std::unique_ptr<Property> Property::New(std::function<bool(int &)> funGetter, std::function<bool(const int &)> funSetter) { return std::make_unique<PropertyFunctor<int>>(funGetter, funSetter); }
std::unique_ptr<Property> Property::New(std::function<bool(bool &)> funGetter, std::function<bool(const bool &)> funSetter) { return std::make_unique<PropertyFunctor<bool>>(funGetter, funSetter); }
std::unique_ptr<Property> Property::New(std::function<bool(std::vector<float> &)> funGetter, std::function<bool(const std::vector<float> &)> funSetter) { return std::make_unique<PropertyFunctor<std::vector<float>>>(funGetter, funSetter); }
std::unique_ptr<Property> Property::New(std::function<bool(std::vector<int> &)> funGetter,std::function<bool(const std::vector<int> &)> funSetter) { return std::make_unique<PropertyFunctor<std::vector<int>>>(funGetter, funSetter); }

} // end namespace bleak
