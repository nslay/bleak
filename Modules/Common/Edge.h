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

#ifndef BLEAK_EDGE_H
#define BLEAK_EDGE_H

#include <algorithm>
#include <memory>
#include <vector>
#include "Array.h"

namespace bleak {

template<typename RealType>
class Vertex;

template<typename RealType>
class Edge {
public:
  typedef Vertex<RealType> VertexType;
  typedef std::vector<std::pair<std::weak_ptr<VertexType>, int>> TargetVectorType;

  Edge() { }

  explicit Edge(const std::shared_ptr<VertexType> &p_clSource)
  : m_p_clSource(p_clSource) { }

  Array<RealType> & GetData() {
    return m_clData;
  }

  Array<RealType> & GetGradient() {
    return m_clGradient;
  }

  bool HasTargets() const {
    if (m_vTargets.empty())
      return false;

    return std::find_if(m_vTargets.begin(), m_vTargets.end(),
      [](const auto &p) -> bool {
        return p.first.lock() != nullptr;
      }) != m_vTargets.end();
  }

  bool HasTarget(const std::shared_ptr<VertexType> &p_clTarget) const {
    return std::find_if(m_vTargets.begin(), m_vTargets.end(), 
      [&p_clTarget](const auto &p) -> bool {
        return p.first.lock() == p_clTarget;
      }) != m_vTargets.end();
  }

  // Should not be called directly!
  void AddTarget(const std::shared_ptr<VertexType> &p_clTarget) {
    auto itr = std::find_if(m_vTargets.begin(),m_vTargets.end(),
      [&p_clTarget](const auto &p) -> bool {
        return p.first.lock() == p_clTarget;
      });

    if (itr == m_vTargets.end())
      m_vTargets.emplace_back(p_clTarget, 1);
    else
      ++(itr->second);
  }

  // Should not be called directly!
  void RemoveTarget(const std::shared_ptr<VertexType> &p_clTarget) {
    auto itr = std::find_if(m_vTargets.begin(),m_vTargets.end(),
      [&p_clTarget](const auto &p) -> bool {
        return p.first.lock() == p_clTarget;
      });

    if (itr == m_vTargets.end())
      return;

    if (--(itr->second) <= 0)
      m_vTargets.erase(itr);
  }

  void RemoveExpiredTargets() {
    auto itr = m_vTargets.begin();

    while (itr != m_vTargets.end()) {
      itr = std::find_if(itr, m_vTargets.end(),
        [](const auto &p) -> bool {
          return p.first.expired();
        });

      if (itr != m_vTargets.end())
        itr = m_vTargets.erase(itr);
    }
  }

  std::shared_ptr<VertexType> GetSource() const {
    return m_p_clSource.lock();
  }

  const TargetVectorType & GetAllTargets() const {
    return m_vTargets;
  }

  int SourceCount() const { return !m_p_clSource.expired() ? 1 : 0; }

  int TargetCount() const { 
    int iCount = 0;

    for (const auto &stPair : m_vTargets) {
      if (!stPair.first.expired())
        iCount += stPair.second;
    }

    return iCount;
  }

private:
  Array<RealType> m_clData;
  Array<RealType> m_clGradient;

  // These can't be owned by Edge!
  std::weak_ptr<VertexType> m_p_clSource;
  TargetVectorType m_vTargets;

  Edge(const Edge &) = delete;
  Edge & operator=(const Edge &) = delete;
};

} // end namespace bleak

#endif // !BLEAK_EDGE_H
