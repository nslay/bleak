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

#ifndef BLEAK_STOCHASTICGRADIENTDESCENT_H
#define BLEAK_STOCHASTICGRADIENTDESCENT_H

#include <iostream>
#include <algorithm>
#include <vector>
#include <functional>
#include <memory>
#include "Array.h"
#include "Edge.h"
#include "IterativeOptimizer.h"

namespace bleak {

template<typename RealType>
class StochasticGradientDescent : public IterativeOptimizer<RealType> {
public:
  bleakNewOptimizer(StochasticGradientDescent, IterativeOptimizer<RealType>, "SGD");
  bleakForwardOptimizerTypedefs();

  using SuperType::GetNumBatchesPerIteration;
  using SuperType::GetLearningRate;

  typedef std::vector<std::pair<std::shared_ptr<EdgeType>, std::unique_ptr<ArrayType>>> MomentumVectorType;

  StochasticGradientDescent(const std::shared_ptr<GraphType> &p_clGraph)
    : SuperType(p_clGraph) {
    m_momentum = RealType(0.9);

    PrepareMomentums();
  }

  virtual ~StochasticGradientDescent() = default;

  virtual bool SetParameters(const ParameterContainer &clParams) override {
    if (!SuperType::SetParameters(clParams))
      return false;

    m_momentum = clParams.GetValue<RealType>("momentum", RealType(0.9));

    if (m_momentum < RealType(0)) {
      std::cerr << "Error: momentum is expected to be non-negative." << std::endl;
      return false;
    }

    return true;
  }

  virtual bool Good() const override {
    return SuperType::Good() && m_momentum >= RealType(0);
  }

  virtual void Reset() override {
    SuperType::Reset();

    for (auto &clMomentumPair : m_vEdgeAndMomentum)
      clMomentumPair.second->Fill(RealType());
  }

protected:
  virtual void IterationStart() override {
    ScaleMomentums();
  }

  virtual void CollectGradients() override {
    const RealType scale = -GetLearningRate() / RealType(GetNumBatchesPerIteration());

    for(auto &clPair : m_vEdgeAndMomentum) {
      const RealType scaleForThisEdge = LearningRateMultiplier(clPair.first) * scale;
      ArrayType &clGradient = clPair.first->GetGradient();
      ArrayType &clMomentum = *clPair.second;

      std::transform(clGradient.begin(),clGradient.end(),clMomentum.begin(),clMomentum.begin(),
        [&scaleForThisEdge](const RealType &grad,const RealType &mom) -> RealType {
        return scaleForThisEdge * grad + mom;
      });
    }
  }

  virtual void GradientUpdate() override {
    for(auto &clPair : m_vEdgeAndMomentum) {
      ArrayType &clData = clPair.first->GetData();
      ArrayType &clMomentum = *clPair.second;

      std::transform(clData.begin(),clData.end(),clMomentum.begin(),clData.begin(),std::plus<RealType>());
    }
  }

private:
  MomentumVectorType m_vEdgeAndMomentum;

  RealType m_momentum;

  bool PrepareMomentums() {
    m_vEdgeAndMomentum.clear();

    const EdgeVectorType &vUpdateEdges = GetUpdateEdges();

    if (vUpdateEdges.empty())
      return false;

    m_vEdgeAndMomentum.reserve(vUpdateEdges.size());

    for (const std::shared_ptr<EdgeType> &p_clEdge : vUpdateEdges) {
      const ArrayType &clGradient = p_clEdge->GetGradient();

      m_vEdgeAndMomentum.emplace_back(p_clEdge, std::make_unique<ArrayType>());

      auto &clPair = m_vEdgeAndMomentum.back();

      clPair.second->SetSize(clGradient.GetSize());
      clPair.second->Allocate();
      clPair.second->Fill(RealType());
    }

    return true;
  }

  void ScaleMomentums() {
    for (auto &clPair : m_vEdgeAndMomentum) {
      ArrayType &clMomentum = *clPair.second;

      std::transform(clMomentum.begin(), clMomentum.end(), clMomentum.begin(), 
        [this](const RealType &value) -> RealType {
          return this->m_momentum * value;
        });
    }
  }
};

} // end namespace bleak

#endif // BLEAK_STOCHASTICGRADIENTDESCENT_H
