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

#ifndef BLEAK_GRAPH_H
#define BLEAK_GRAPH_H

#include <algorithm>
#include <unordered_set>
#include <memory>
#include "Vertex.h"
#include "Edge.h"
#include "Database.h"

namespace bleak {

template<typename RealType>
class Graph {
public:
  typedef Vertex<RealType> VertexType;
  typedef Edge<RealType> EdgeType;
  typedef std::vector<std::shared_ptr<VertexType>> VertexVectorType;
  typedef std::unordered_set<std::shared_ptr<VertexType>> VertexSetType;

  // Since we use hash sets of pointers to vertices and addresses are not necessarily the same between runs
  //   we will opportunistically sort the plan lexically to ensure random initializations happen in the same order.
  //   This is important for replicability!

  struct LexicalVertexCompare {
    bool operator()(const std::shared_ptr<VertexType> &p_clV1, const std::shared_ptr<VertexType> &p_clV2) const {
      return p_clV1->GetName().compare(p_clV2->GetName()) < 0;
    }
  };

  Graph() = default;

  bool Initialize(bool bForTraining) {
    if (!ComputePlan()) {
      m_vPlan.clear();
      return false;
    }

    for (const auto &p_clVertex : m_vPlan) {
      std::cout << "Info: Initializing " << p_clVertex->GetName() << " ..." << std::endl;

      if (!p_clVertex->SetSizes() || !p_clVertex->Allocate(bForTraining) || !p_clVertex->Initialize()) {
        m_vPlan.clear();
        return false;
      }
    }

    return true;
  }

  bool TestGradient() {
    bool bSuccess = true;

    for (const auto &p_clVertex : m_vPlan) {
      std::cout << "Info: Testing gradient for " << p_clVertex->GetName() << " ..." << std::endl;

      if (!p_clVertex->TestGradient()) {
        bSuccess = false;
        std::cerr << "Error: " << p_clVertex->GetName() << " failed the gradient test." << std::endl;
      }
    }

    return bSuccess;
  }

  bool TestGradient(const std::vector<std::string> &vVertexNames) {
    if (vVertexNames.empty())
      return TestGradient();

    bool bSuccess = true;

    for (const std::string &strName : vVertexNames) {
      auto p_clVertex = FindVertex(strName);

      if (!p_clVertex) {
        std::cerr << "Error: Could not find vertex with name '" << strName << "'." << std::endl;
        return false;
      }

      if (!p_clVertex->TestGradient()) {
        bSuccess = false;
        std::cerr << "Error: " << p_clVertex->GetName() << " failed the gradient test." << std::endl;
      }
    }

    return bSuccess;
  }

  bool SaveToDatabase(const std::shared_ptr<Database> &p_clDatabase) const {
    std::unique_ptr<Transaction> p_clTransaction = p_clDatabase->NewTransaction();

    if (p_clTransaction == nullptr)
      return false;

    for(const auto &p_clVertex : m_vPlan) {
      if(!p_clVertex->SaveToDatabase(p_clTransaction))
        return false;
    }

    return true;
  }

  bool LoadFromDatabase(const std::shared_ptr<Database> &p_clDatabase) {
    std::unique_ptr<Cursor> p_clCursor = p_clDatabase->NewCursor();

    if (p_clCursor == nullptr)
      return false;

    for (const auto &p_clVertex : m_vPlan) {
      if (!p_clVertex->LoadFromDatabase(p_clCursor))
        return false;
    }

    return true;
  }

  void Forward() {
    for(const auto &p_clVertex : m_vPlan)
      p_clVertex->Forward();
  }

  void Backward() {
    for(const auto &p_clVertex : m_vPlan) {
      const auto &mOutputs = p_clVertex->GetAllOutputs();

      for (const auto &clPair : mOutputs)
        clPair.second->GetGradient().Fill(RealType()); // Now we rely on the loss functions to determine that they're leaves and use 1 for their gradient
    }

    for (auto itr = m_vPlan.rbegin(); itr != m_vPlan.rend(); ++itr)
      (*itr)->Backward();
  }

  bool HasVertex(const std::shared_ptr<VertexType> &p_clVertex) const {
    return m_sVertices.find(p_clVertex) != m_sVertices.end();
  }

  std::shared_ptr<VertexType> FindVertex(const std::string &strName) const {
    for (const auto &p_clVertex : m_sVertices) {
      if (p_clVertex->GetName() == strName)
        return p_clVertex;
    }

    return std::shared_ptr<VertexType>();
  }

  void AddVertex(const std::shared_ptr<VertexType> &p_clVertex) {
    m_sVertices.insert(p_clVertex);
  }

  void RemoveVertex(const std::shared_ptr<VertexType> &p_clVertex) {
    auto itr = m_sVertices.find(p_clVertex);

    if (itr != m_sVertices.end())
      m_sVertices.erase(itr);
  }

  const VertexSetType & GetAllVertices() const {
    return m_sVertices; 
  }

  const VertexVectorType & GetPlan() const {
    return m_vPlan;
  }

  bool Empty() const {
    return m_sVertices.empty();
  }

  void Clear() {
    m_vPlan.clear();
    m_sVertices.clear();
  }

  void FindLeafVertices(VertexVectorType &vLeaves) const {
    vLeaves.clear();

    for(const auto &p_clVertex : m_sVertices) {
      if(p_clVertex->IsLeaf())
        vLeaves.push_back(p_clVertex);
    }

    std::sort(vLeaves.begin(), vLeaves.end(), LexicalVertexCompare());
  }

  void FindRootVertices(VertexVectorType &vRoots) const {
    vRoots.clear();

    for(const auto &p_clVertex : m_sVertices) {
      if(p_clVertex->IsRoot())
        vRoots.push_back(p_clVertex);
    }

    std::sort(vRoots.begin(), vRoots.end(), LexicalVertexCompare());
  }

private:
  VertexVectorType m_vPlan;
  VertexSetType m_sVertices;

  Graph(const Graph &) = delete;
  Graph & operator=(const Graph &) = delete;

  void CollectSourcesForVertex(const std::shared_ptr<VertexType> &p_clVertex, VertexVectorType &vSources) const {
    VertexSetType sSources;

    vSources.clear();

    const auto &mInputs = p_clVertex->GetAllInputs();

    for (const auto &clInputPair : mInputs) {
      std::shared_ptr<EdgeType> p_clEdge = clInputPair.second.lock();

      if (p_clEdge == nullptr)
        continue;

      std::shared_ptr<VertexType> p_clSource = p_clEdge->GetSource();

      if (p_clSource != nullptr)
        sSources.insert(p_clSource);
    }

    vSources.assign(sSources.begin(), sSources.end());

    std::sort(vSources.begin(), vSources.end(), LexicalVertexCompare());
  }

  void CollectTargetsForVertex(const std::shared_ptr<VertexType> &p_clVertex, VertexVectorType &vTargets) const {
    VertexSetType sTargets;

    vTargets.clear();

    const auto &mOutputs = p_clVertex->GetAllOutputs();

    for (const auto &clOutputPair : mOutputs) {
      const auto &vTargets = clOutputPair.second->GetAllTargets();

      for (const auto &clTargetPair : vTargets) {
        std::shared_ptr<VertexType> p_clTarget = clTargetPair.first.lock();

        if (p_clTarget != nullptr)
          sTargets.insert(p_clTarget);
      }
    }

    vTargets.assign(sTargets.begin(), sTargets.end());

    std::sort(vTargets.begin(), vTargets.end(), LexicalVertexCompare());
  }

  bool CheckDependencies(const std::shared_ptr<VertexType> &p_clVertex, const VertexSetType &sVisit) const {
    VertexVectorType vSources;
    CollectSourcesForVertex(p_clVertex, vSources);

    for (const auto &p_clSource : vSources) {
      if (sVisit.find(p_clSource) == sVisit.end())
        return false;
    }

    return true;
  }

  bool ComputePlan() {
    m_vPlan.clear();

    m_vPlan.reserve(m_sVertices.size());

    FindRootVertices(m_vPlan);

    VertexVectorType vTargets;
    VertexSetType sVisit(m_vPlan.begin(), m_vPlan.end());

    for (size_t i = 0; i < m_vPlan.size(); ++i) {
      std::shared_ptr<VertexType> p_clVertex = m_vPlan[i]; // XXX: Must be a copy since m_vPlan can resize

      vTargets.clear();
      CollectTargetsForVertex(p_clVertex, vTargets);

      for (const auto &p_clTarget : vTargets) {
        if (sVisit.find(p_clTarget) == sVisit.end() && CheckDependencies(p_clTarget, sVisit)) {
          sVisit.insert(p_clTarget);
          m_vPlan.push_back(p_clTarget);
        }
      }
    }

    return true;
  }
};

} // end namespace bleak

#endif // !BLEAK_GRAPH_H
