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

#ifndef BLEAK_OPTIMIZER_H
#define BLEAK_OPTIMIZER_H

#include <algorithm>
#include "ParameterContainer.h"
#include "Graph.h"
#include "BlasWrapper.h"

#define bleakForwardOptimizerTypedefs() \
  typedef typename SuperType::GraphType GraphType; \
  typedef typename SuperType::VertexType VertexType; \
  typedef typename SuperType::EdgeType EdgeType; \
  typedef typename SuperType::ArrayType ArrayType; \
  typedef typename SuperType::EdgeVectorType EdgeVectorType

#define bleakNewAbstractOptimizer(thisClass, superClass) \
  typedef superClass SuperType; \
  typedef thisClass SelfType; \
  using SuperType::GetGraph; \
  using SuperType::GetUpdateEdges; \
  using SuperType::GetLossEdges; \
  using SuperType::ComputeLoss; \
  using SuperType::LearningRateMultiplier; \
  using SuperType::ApplyWeightDecay

// NOTE: See note about DeclTypeHack in DeclTypeHack() below
#define bleakNewOptimizer(thisClass, superClass, typeName) \
  static constexpr const char * GetTypeName() { return typeName ; } \
  using superClass :: DeclTypeHack; \
  static std::shared_ptr< thisClass > New(const decltype(DeclTypeHack()) &p_clGraph) { \
    std::shared_ptr< thisClass > p_clOptimizer(new ( thisClass ) ( p_clGraph )); \
    return p_clOptimizer; \
  } \
  bleakNewAbstractOptimizer(thisClass, superClass)

namespace bleak {

template<typename RealType>
class Optimizer {
public:
  typedef Graph<RealType> GraphType;
  typedef typename GraphType::VertexType VertexType;
  typedef typename GraphType::EdgeType EdgeType;
  typedef Array<RealType> ArrayType;
  typedef std::vector<std::shared_ptr<EdgeType>> EdgeVectorType;

  // Graph assumed setup and ready to optimize!
  Optimizer(const std::shared_ptr<GraphType> &p_clGraph)
    : m_p_clGraph(p_clGraph), m_weightDecay(RealType()) {

    if(m_p_clGraph != nullptr) {
      CollectLossEdges();
      CollectUpdateEdges();
    }
  }

  virtual ~Optimizer() = default;

  virtual bool SetParameters(const ParameterContainer &clParams) {
    m_weightDecay = RealType(clParams.GetValue<float>("weightDecay", 0.0f));
    return true;
  }

  virtual bool Minimize() = 0;

  // This is supposed to be used to reset values as though you were to call Minimize() from iteration 0!
  virtual void Reset() { }

  virtual bool Good() const {
    return m_p_clGraph != nullptr && m_vUpdateEdges.size() > 0 && m_vLossEdges.size() > 0;
  }

protected:
  // We can't use "typedef typename" when the parent class is resolved...
  // But we have to use "typedef typename" when the parent class has unresolved template parameters...
  // But we can scope functions directly or with 'using' statements the same in both cases and so we 
  //   can use decltype() to determine the type std::shared_ptr<GraphType>.
  // We also can't use GetGraph() with decltype since it requires 'this' which can't be used outside a 
  //   function body...
  // See bleakNewOptimizer() above. This is lame!!!!!!
  static std::shared_ptr<GraphType> DeclTypeHack() {
    return std::shared_ptr<GraphType>();
  }

  const std::shared_ptr<GraphType> & GetGraph() const {
    return m_p_clGraph;
  }

  const EdgeVectorType & GetUpdateEdges() const {
    return m_vUpdateEdges;
  }

  const EdgeVectorType & GetLossEdges() const {
    return m_vLossEdges;
  }

  RealType LearningRateMultiplier(const std::shared_ptr<EdgeType> &p_clEdge) const {
    std::shared_ptr<VertexType> p_clSource = p_clEdge->GetSource();

    float fLearningRateMultiplier = 1.0f;

    if (p_clSource == nullptr || !p_clSource->GetProperty("learningRateMultiplier", fLearningRateMultiplier))
      return RealType(1);

    return RealType(fLearningRateMultiplier);
  }

  bool ShouldApplyWeightDecay(const std::shared_ptr<EdgeType> &p_clEdge) const {
    std::shared_ptr<VertexType> p_clSource = p_clEdge->GetSource();

    bool bApplyWeightDecay = true;

    if (p_clSource == nullptr || !p_clSource->GetProperty("applyWeightDecay", bApplyWeightDecay))
      return true;

    return bApplyWeightDecay;
  }

  RealType ComputeLoss() const {
    RealType sum = RealType();

    for (const auto &p_clEdge : m_vLossEdges)
      sum += *p_clEdge->GetData().data();

    return sum;
  }

  virtual void ApplyWeightDecay() {
    if (m_weightDecay == RealType(0))
      return;

    for (auto &p_clEdge : m_vUpdateEdges) {
      const ArrayType &clWeights = p_clEdge->GetData();
      ArrayType &clWeightsGradient = p_clEdge->GetGradient();

      if (!clWeights.Valid() || !clWeightsGradient.Valid() || !ShouldApplyWeightDecay(p_clEdge)) // Should not happen
        continue;

      if (GetUseGPU()) {
#ifdef BLEAK_USE_CUDA
        gpu_blas::axpy(clWeights.GetSize().Count(), RealType(this->m_weightDecay * 2), clWeights.data(GPU), 1, clWeightsGradient.data(GPU), 1);
#endif // BLEAK_USE_CUDA
      }
      else {
        cpu_blas::axpy(clWeights.GetSize().Count(), RealType(this->m_weightDecay * 2), clWeights.data(), 1, clWeightsGradient.data(), 1);
      }

      //std::transform(clWeights.begin(), clWeights.end(), clWeightsGradient.begin(), clWeightsGradient.begin(),
      //  [this](const RealType &weight, const RealType &weightGradient) -> RealType {
      //    return weightGradient + this->m_weightDecay * weight * RealType(2);
      //  });
    }
  }

private:
  std::shared_ptr<GraphType> m_p_clGraph;

  EdgeVectorType m_vUpdateEdges; // Edges to update
  EdgeVectorType m_vLossEdges; // Edges to sum to calculate loss

  RealType m_weightDecay;

  bool CollectLossEdges() {
    m_vLossEdges.clear();

    std::vector<std::shared_ptr<VertexType>> vLeaves;

    m_p_clGraph->FindLeafVertices(vLeaves);

    if(vLeaves.empty()) // Nothing to optimize?
      return false;

    for (const auto &p_clVertex : vLeaves) {
      for(const auto &clOutputPair : p_clVertex->GetAllOutputs()) {
        const std::string &strOutputName = clOutputPair.first;
        const std::shared_ptr<EdgeType> &p_clEdge = clOutputPair.second;

        const ArrayType &clData = p_clEdge->GetData();

        if (clData.Valid() && clData.GetSize().Product() == 1) {
          std::cout << "Info: Found output: " << p_clVertex->GetName() << '.' << strOutputName << std::endl;
          m_vLossEdges.push_back(p_clEdge);
        }
      }
    }

    return m_vLossEdges.size() > 0;
  }

  bool CollectUpdateEdges() {
    // Parameters should probably always be roots... even with cycles in the graph!

    m_vUpdateEdges.clear();

    std::vector<std::shared_ptr<VertexType>> vRoots;

    m_p_clGraph->FindRootVertices(vRoots);

    for (const auto &p_clVertex : vRoots) {
      for (const auto &clOutputPair : p_clVertex->GetAllOutputs()) {
        const std::string &strOutputName = clOutputPair.first;
        const std::shared_ptr<EdgeType> &p_clEdge = clOutputPair.second;

        const ArrayType &clData = p_clEdge->GetData();
        const ArrayType &clGradient = p_clEdge->GetGradient();

        if(clData.Valid() && clGradient.Valid()) {
          std::cout << "Info: Found learnable parameters: " << p_clVertex->GetName() << '.' << strOutputName << std::endl;
          m_vUpdateEdges.push_back(p_clEdge);
        }
      }
    }

    return m_vUpdateEdges.size() > 0;
  }
};

} // end namespace bleak

#endif // !BLEAK_OPTIMIZER_H
