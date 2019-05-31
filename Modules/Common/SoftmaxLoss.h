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

#ifndef BLEAK_SOFTMAXLOSS_H
#define BLEAK_SOFTMAXLOSS_H

#include <cmath>
#include "Softmax.h"

namespace bleak {

template<typename RealType>
class SoftmaxLoss : public Softmax<RealType> {
public:
  bleakNewVertex(SoftmaxLoss, Softmax<RealType>,
    bleakAddInput("inLabels"),
    bleakAddOutput("outLoss"),
    bleakAddProperty("penaltyWeight", m_fPenaltyWeight));

  bleakForwardVertexTypedefs();

  virtual ~SoftmaxLoss() = default;

  virtual bool SetSizes() override {
    if (!SuperType::SetSizes())
      return false;

    bleakGetAndCheckInput(p_clInLabels, "inLabels", false);

    bleakGetAndCheckOutput(p_clOutProbs, "outProbabilities", false);
    bleakGetAndCheckOutput(p_clOutLoss, "outLoss", false);

    const ArrayType &clInLabels = p_clInLabels->GetData();
    const ArrayType &clOutProbs = p_clOutProbs->GetData();

    // XXX: Not intuitive error message. Should be compared with inData.
    if (clInLabels.GetSize().GetDimension()+1 != clOutProbs.GetSize().GetDimension()) {
      std::cerr << GetName() << ": Error: Incompatible dimensions: inLabels = " << clInLabels.GetSize().GetDimension() << ", outProbabilities = " << clOutProbs.GetSize().GetDimension() << std::endl;
      return false;
    }

    if (clInLabels.GetSize()[0] != clOutProbs.GetSize()[0] || clInLabels.GetSize().SubSize(1) != clOutProbs.GetSize().SubSize(2)) {
      std::cerr << GetName() << ": Error: Incompatible sizes: inLabels = " << clInLabels.GetSize() << ", outProbabilities = " << clOutProbs.GetSize() << std::endl;
      return false;
    }

    // This vertex does not propagate derivatives from targets of 'outProbabilities'
    p_clOutProbs->GetGradient().Clear();

    ArrayType &clOutLoss = p_clOutLoss->GetData();
    ArrayType &clOutGradient = p_clOutLoss->GetGradient();

    const Size clSize = { 1 };    

    clOutLoss.SetSize(clSize);
    clOutGradient.SetSize(clSize);

    return true;
  }

  virtual bool Initialize() override {
    return SuperType::Initialize();
  }

  virtual void Forward() override {
    SuperType::Forward();

    bleakGetAndCheckInput(p_clInLabels, "inLabels");

    bleakGetAndCheckOutput(p_clOutProbs, "outProbabilities");
    bleakGetAndCheckOutput(p_clOutLoss, "outLoss");

    const ArrayType &clInLabels = p_clInLabels->GetData();
    const ArrayType &clOutProbs = p_clOutProbs->GetData();
    ArrayType &clOutLoss = p_clOutLoss->GetData();

    const int iOuterNum = clOutProbs.GetSize()[0];
    const int iNumOutputs = clOutProbs.GetSize()[1];
    const int iInnerNum = clOutProbs.GetSize().Product(2);

    const RealType * const p_inLabels = clInLabels.data();
    const RealType * const p_outProbs = clOutProbs.data();

    RealType &outLoss = *clOutLoss.data();

    outLoss = RealType();

    for (int i = 0; i < iOuterNum; ++i) {
      for (int j = 0; j < iInnerNum; ++j) {
        const int iLabel = (int)p_inLabels[iInnerNum*i + j];

        if (iLabel < 0 || iLabel >= iNumOutputs) // TODO: Warning?
          continue;

        const RealType prob = p_outProbs[(i*iNumOutputs + iLabel)*iInnerNum + j];

        if (prob > RealType(1e-30))
          outLoss -= std::log(prob);
        else
          outLoss -= -100;
      }
    }

    outLoss *= RealType(m_fPenaltyWeight)/RealType(iOuterNum);
  }

  virtual void Backward() override {
    // We do not call SuperType::Backward() since this simplifies both log loss and SuperType's gradient together

    bleakGetAndCheckInput(p_clInData, "inData");
    bleakGetAndCheckInput(p_clInLabels, "inLabels");

    bleakGetAndCheckOutput(p_clOutProbs, "outProbabilities");
    bleakGetAndCheckOutput(p_clOutLoss, "outLoss");

    const ArrayType &clInLabels = p_clInLabels->GetData();
    const ArrayType &clOutProbs = p_clOutProbs->GetData();
    const ArrayType &clOutLoss = p_clOutLoss->GetData();
    ArrayType &clOutGradient = p_clOutLoss->GetGradient();
    const ArrayType &clInData = p_clInData->GetData();
    ArrayType &clInGradient = p_clInData->GetGradient();

    if (!clInGradient.Valid())
      return; // Nothing to do

    const int iOuterNum = clOutProbs.GetSize()[0];
    const int iNumOutputs = clOutProbs.GetSize()[1];
    const int iInnerNum = clOutProbs.GetSize().Product(2);

    const RealType * const p_inLabels = clInLabels.data();
    const RealType * const p_outProbs = clOutProbs.data();
    RealType &outGradient = *clOutGradient.data();
    RealType * const p_inGradient = clInGradient.data();

    if (IsLeaf())
      outGradient = RealType(1);

    const RealType scale = RealType(m_fPenaltyWeight)*outGradient/RealType(iOuterNum);

    for (int i = 0; i < iOuterNum; ++i) {
      for (int k = 0; k < iInnerNum; ++k) {
        const int iLabel = (int)p_inLabels[iInnerNum*i + k];

        for (int j = 0; j < iNumOutputs; ++j) {
          const RealType prob = p_outProbs[(i*iNumOutputs + j)*iInnerNum + k];
          p_inGradient[(i*iNumOutputs + j)*iInnerNum + k] += scale*prob;
        }

        if (iLabel < 0 || iLabel >= iNumOutputs)
          continue;

        p_inGradient[(i*iNumOutputs + iLabel)*iInnerNum + k] -= scale;
      }
    }
  }

protected:
  SoftmaxLoss() {
    m_fPenaltyWeight = 1.0f;
  }

private:
  float m_fPenaltyWeight;
};

} // end namespace bleak

#endif // !BLEAK_SOFTMAXLOSS_H
