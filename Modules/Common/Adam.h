/*-
 * Copyright (c) 2018 Nathan Lay (enslay@gmail.com)
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

#ifndef BLEAK_ADAM_H
#define BLEAK_ADAM_H

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
class Adam : public IterativeOptimizer<RealType> {
public:
  bleakNewOptimizer(Adam, IterativeOptimizer<RealType>, "Adam");
  bleakForwardOptimizerTypedefs();

  using SuperType::GetIteration;
  using SuperType::GetNumBatchesPerIteration;
  using SuperType::GetLearningRate;

  typedef std::vector<std::tuple<std::shared_ptr<EdgeType>, std::unique_ptr<ArrayType>, std::unique_ptr<ArrayType>, std::unique_ptr<ArrayType>>> GradientHistoryVectorType;

  Adam(const std::shared_ptr<GraphType> &p_clGraph)
  : SuperType(p_clGraph) {
    m_small = RealType(1e-8);
    m_beta1 = RealType(0.9);
    m_beta2 = RealType(0.999);

    PrepareGradientHistory();
  }

  virtual ~Adam() = default;

  virtual bool SetParameters(const ParameterContainer &clParams) override {
    if (!SuperType::SetParameters(clParams))
      return false;

    m_beta1 = clParams.GetValue<RealType>("beta1", RealType(0.9));
    m_beta2 = clParams.GetValue<RealType>("beta2", RealType(0.999));
    m_small = clParams.GetValue<RealType>("small", RealType(1e-8));

    if (m_beta1 < RealType(0) || m_beta1 >= RealType(1) || 
      m_beta2 < RealType(0) || m_beta2 >= RealType(1) ||
      m_small <= RealType(0) || m_small >= RealType(1)) {
      return false;
    }

    return true;
  }

  virtual void Reset() override {
    SuperType::Reset();

    for(auto &clTuple : m_vEdgeAndHistory) {
      std::get<1>(clTuple)->Fill(RealType());
      std::get<2>(clTuple)->Fill(RealType());
      std::get<3>(clTuple)->Fill(RealType());
    }
  }

  virtual bool Good() const override {
    return SuperType::Good() &&
      m_beta1 >= RealType(0) && m_beta1 < RealType(1) &&
      m_beta2 >= RealType(0) && m_beta2 < RealType(1) &&
      m_small > RealType(0) && m_small < RealType(1);
  }

protected:
  virtual void CollectGradients() override {
    for (auto &clTuple : m_vEdgeAndHistory) {
      ArrayType &clGradient = std::get<0>(clTuple)->GetGradient();
      ArrayType &clGradientSum = *std::get<1>(clTuple);

      std::transform(clGradient.begin(),clGradient.end(),clGradientSum.begin(),clGradientSum.begin(),std::plus<RealType>());
    }
  }

  virtual void GradientUpdate() override {
    const unsigned int uiNumBatchesPerIteration = GetNumBatchesPerIteration();

    for (auto &clTuple : m_vEdgeAndHistory) {
      const RealType scaleForThisEdge = -GetLearningRate() * LearningRateMultiplier(std::get<0>(clTuple));
      ArrayType &clData = std::get<0>(clTuple)->GetData();
      ArrayType &clGradientSum = *std::get<1>(clTuple);
      ArrayType &clGradientMoment1 = *std::get<2>(clTuple);
      ArrayType &clGradientMoment2 = *std::get<3>(clTuple);

      std::transform(clGradientSum.begin(), clGradientSum.end(), clGradientMoment1.begin(), clGradientMoment1.begin(),
        [this,uiNumBatchesPerIteration](const RealType &gradientSum, const RealType &gradientMoment1) -> RealType {
          return (RealType(1) - this->m_beta1) * (gradientSum / RealType(uiNumBatchesPerIteration)) + this->m_beta1 * gradientMoment1;
        });

      std::transform(clGradientSum.begin(), clGradientSum.end(), clGradientMoment2.begin(), clGradientMoment2.begin(),
        [this,uiNumBatchesPerIteration](const RealType &gradientSum, const RealType &gradientMoment2) -> RealType {
          return (RealType(1) - this->m_beta2) * std::pow(gradientSum / RealType(uiNumBatchesPerIteration), 2) + this->m_beta2 * gradientMoment2;
        });

      RealType * const p_data = clData.data();
      const RealType * const p_gradientSum = clGradientSum.data();
      const RealType * const p_gradientMoment1 = clGradientMoment1.data();
      const RealType * const p_gradientMoment2 = clGradientMoment2.data();

      const size_t numElements = clData.GetSize().Product();

      const RealType gradientMoment1Divisor = RealType(1) - std::pow(m_beta1, GetIteration()+1); // Indexed from 1
      const RealType gradientMoment2Divisor = RealType(1) - std::pow(m_beta2, GetIteration()+1); // Indexed from 1

      for (size_t i = 0; i < numElements; ++i) {
        const RealType gradientMoment1Hat = p_gradientMoment1[i] / gradientMoment1Divisor;
        const RealType gradientMoment2Hat = std::sqrt(p_gradientMoment2[i] / gradientMoment2Divisor) + m_small;

        p_data[i] += scaleForThisEdge * gradientMoment1Hat / gradientMoment2Hat;
      }

      clGradientSum.Fill(RealType());
    }
  }

private:
  RealType m_beta1, m_beta2;
  RealType m_small;

  GradientHistoryVectorType m_vEdgeAndHistory;

  bool PrepareGradientHistory() {
    m_vEdgeAndHistory.clear();

    const EdgeVectorType &vUpdateEdges = GetUpdateEdges();

    if (vUpdateEdges.empty())
      return false;

    m_vEdgeAndHistory.reserve(vUpdateEdges.size());

    for (const std::shared_ptr<EdgeType> &p_clEdge : vUpdateEdges) {
      const ArrayType &clGradient = p_clEdge->GetGradient();

      m_vEdgeAndHistory.emplace_back(p_clEdge, std::make_unique<ArrayType>(), std::make_unique<ArrayType>(), std::make_unique<ArrayType>());

      auto &clTuple = m_vEdgeAndHistory.back();

      std::get<1>(clTuple)->SetSize(clGradient.GetSize());
      std::get<1>(clTuple)->Allocate();
      std::get<1>(clTuple)->Fill(RealType());

      std::get<2>(clTuple)->SetSize(clGradient.GetSize());
      std::get<2>(clTuple)->Allocate();
      std::get<2>(clTuple)->Fill(RealType());

      std::get<3>(clTuple)->SetSize(clGradient.GetSize());
      std::get<3>(clTuple)->Allocate();
      std::get<3>(clTuple)->Fill(RealType());
    }

    return true;
  }
};

} // end namespace bleak

#endif // !BLEAK_ADAM_H
