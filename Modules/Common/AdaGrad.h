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

#ifndef BLEAK_ADAGRAD_H
#define BLEAK_ADAGRAD_H

#include <iostream>
#include <memory>
#include <utility>
#include <functional>
#include <numeric>
#include "Array.h"
#include "Edge.h"
#include "IterativeOptimizer.h"

namespace bleak {

template<typename RealType>
class AdaGrad : public IterativeOptimizer<RealType> {
public:
  bleakNewOptimizer(AdaGrad, IterativeOptimizer<RealType>, "AdaGrad");
  bleakForwardOptimizerTypedefs();

  using SuperType::GetNumBatchesPerIteration;
  using SuperType::GetLearningRate;

  typedef std::vector<std::tuple<std::shared_ptr<EdgeType>, std::unique_ptr<ArrayType>, std::unique_ptr<ArrayType>>> GradientHistoryVectorType;

  AdaGrad(const std::shared_ptr<GraphType> &p_clGraph)
  : SuperType(p_clGraph) {
    PrepareGradientHistory();
  }

  virtual ~AdaGrad() = default;

  virtual bool SetParameters(const ParameterContainer &clParams) override {
    if (!SuperType::SetParameters(clParams))
      return false;

    // Nothing to do?

    return true;
  }

  virtual void Reset() override {
    SuperType::Reset();

    for(auto &clTuple : m_vEdgeAndHistory) {
      std::get<1>(clTuple)->Fill(RealType());
      std::get<2>(clTuple)->Fill(RealType());
    }
  }

  virtual bool Good() const override {
    return SuperType::Good(); // Nothing to do?
  }

protected:
  virtual void CollectGradients() override {
    const RealType scale = -GetLearningRate() / RealType(GetNumBatchesPerIteration());

    for(auto &clTuple : m_vEdgeAndHistory) {
      const RealType scaleForThisEdge = LearningRateMultiplier(std::get<0>(clTuple)) * scale;
      ArrayType &clGradient = std::get<0>(clTuple)->GetGradient();
      ArrayType &clGradientSum = *std::get<1>(clTuple);
      ArrayType &clGradientHistory = *std::get<2>(clTuple);

      std::transform(clGradient.begin(),clGradient.end(),clGradientHistory.begin(),clGradientHistory.begin(),
        [](const RealType &grad,const RealType &gradHistory) -> RealType {
        //return std::sqrt(gradHistory*gradHistory + grad*grad);
        return std::hypot(gradHistory,grad);
      });

      std::transform(clGradient.begin(),clGradient.end(),clGradientSum.begin(),clGradientSum.begin(),
        [&scaleForThisEdge](const RealType &grad,const RealType &gradSum) -> RealType {
        return scaleForThisEdge * grad + gradSum;
      });
    }
  }

  virtual void GradientUpdate() override {
    for(auto &clTuple : m_vEdgeAndHistory) {
      ArrayType &clData = std::get<0>(clTuple)->GetData();
      ArrayType &clGradientSum = *std::get<1>(clTuple);
      ArrayType &clGradientHistory = *std::get<2>(clTuple);

      RealType * const p_data = clData.data();
      const RealType * const p_gradientSum = clGradientSum.data();
      const RealType * const p_gradientHistory = clGradientHistory.data();

      const size_t numElements = clData.GetSize().Product();

      for(size_t i = 0; i < numElements; ++i) {
        if(p_gradientHistory[i] > RealType(0))
          p_data[i] += p_gradientSum[i] / p_gradientHistory[i];
      }

      clGradientSum.Fill(RealType());
    }
  }

private:
  GradientHistoryVectorType m_vEdgeAndHistory;

  bool PrepareGradientHistory() {
    m_vEdgeAndHistory.clear();

    const EdgeVectorType &vUpdateEdges = GetUpdateEdges();

    if (vUpdateEdges.empty())
      return false;

    m_vEdgeAndHistory.reserve(vUpdateEdges.size());

    for (const std::shared_ptr<EdgeType> &p_clEdge : vUpdateEdges) {
      const ArrayType &clGradient = p_clEdge->GetGradient();

      m_vEdgeAndHistory.emplace_back(p_clEdge, std::make_unique<ArrayType>(), std::make_unique<ArrayType>());

      auto &clTuple = m_vEdgeAndHistory.back();

      std::get<1>(clTuple)->SetSize(clGradient.GetSize());
      std::get<1>(clTuple)->Allocate();
      std::get<1>(clTuple)->Fill(RealType());

      std::get<2>(clTuple)->SetSize(clGradient.GetSize());
      std::get<2>(clTuple)->Allocate();
      std::get<2>(clTuple)->Fill(RealType());
    }

    return true;
  }
};

} // end namespace bleak

#endif // !BLEAK_ADAGRAD_H
