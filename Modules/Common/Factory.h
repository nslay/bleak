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

#ifndef BLEAK_FACTORY_H
#define BLEAK_FACTORY_H

#include <functional>
#include <memory>
#include <unordered_map>

namespace bleak {

template<typename BaseType, typename... CreateArguments>
class Factory {
public:
  class Creator {
  public:
    virtual ~Creator() { }
    virtual std::shared_ptr<BaseType> Create(CreateArguments&&... args) const = 0;
  };

  template<typename OtherType>
  class CreatorTemplate: public Creator {
  public:
    virtual ~CreatorTemplate() { }
    virtual std::shared_ptr<BaseType> Create(CreateArguments&&... args) const {
      return OtherType::New(std::forward<CreateArguments>(args)...);
    }
  };

  typedef std::unordered_map<std::string, std::unique_ptr<Creator>> CreatorMapType;

  static Factory & GetInstance() {
    static Factory clSingleton;
    return clSingleton;
  }

  bool CanCreate(const std::string &strTypeName) const {
    return m_mCreators.find(strTypeName) != m_mCreators.end();
  }

  std::shared_ptr<BaseType> Create(const std::string &strTypeName, CreateArguments&&... args) const {
    auto itr = m_mCreators.find(strTypeName);
    return itr != m_mCreators.end() ? itr->second->Create(std::forward<CreateArguments>(args)...) : std::shared_ptr<BaseType>();
  }

  template<typename OtherType>
  bool Register() {
    return m_mCreators.emplace(OtherType::GetTypeName(), std::make_unique<CreatorTemplate<OtherType>>()).second;
  }

  // For cataloging purposes
  const CreatorMapType & GetCreatorMap() const {
    return m_mCreators;
  }

private:
  Factory() = default;

  Factory(const Factory &) = delete;
  Factory & operator=(const Factory &) = delete;

  CreatorMapType m_mCreators;
};

} // end namespace bleak

#endif // !BLEAK_FACTORY_H
