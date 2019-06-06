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

#ifndef BLEAK_TREEL2REGULARIZATION_H
#define BLEAK_TREEL2REGULARIZATION_H

#include <cmath>
#include <algorithm>
#include "Vertex.h"

namespace bleak {

template<typename RealType>
class TreeL2Regularization : public Vertex<RealType> {
public:
  bleakNewVertex(TreeL2Regularization, Vertex<RealType>,
    bleakAddInput("inData"),
    bleakAddOutput("outLoss"),
    bleakAddProperty("penaltyWeight", m_fPenaltyWeight));

  bleakForwardVertexTypedefs();

  virtual ~TreeL2Regularization() = default;

  virtual bool SetSizes() override {
    bleakGetAndCheckInput(p_clInData, "inData", false);
    bleakGetAndCheckOutput(p_clOutLoss, "outLoss", false);

    //std::cout << GetName() << ": Info: penaltyWeight = " << m_fPenaltyWeight << std::endl;

    const ArrayType &clInData = p_clInData->GetData();

    if (!clInData.GetSize().Valid()) {
      std::cerr << GetName() << ": Error: inData input size is invalid." << std::endl;
      return false;
    }

    // Actually, this is not true
    if (clInData.GetSize().GetDimension() < 2) {
      std::cerr << GetName() << ": Error: inData is expected to be at least 2D." << std::endl;
      return false;
    }

    Size clSize = { 1 };

    p_clOutLoss->GetData().SetSize(clSize);
    p_clOutLoss->GetGradient().SetSize(clSize);

    return true;
  }

  virtual bool Initialize() override {
    return true; // Nothing to do
  }

  virtual void Forward() override {
    bleakGetAndCheckInput(p_clInData, "inData");
    bleakGetAndCheckOutput(p_clOutLoss, "outLoss");

    const ArrayType &clInData = p_clInData->GetData();

    RealType &outLoss = *p_clOutLoss->GetData().data();

    const int iOuterNum = clInData.GetSize()[0];

    const RealType * const p_inData = clInData.data();

    outLoss = RealType();
    
    for (int i = 0; i < clInData.GetSize().Product(); ++i)
      outLoss += std::pow(p_inData[i], 2);

    outLoss *= RealType(m_fPenaltyWeight)/RealType(iOuterNum);
  }

  virtual void Backward() override {
    bleakGetAndCheckInput(p_clInData, "inData");
    bleakGetAndCheckOutput(p_clOutLoss, "outLoss");

    const ArrayType &clInData = p_clInData->GetData();
    ArrayType &clInDataGradient = p_clInData->GetGradient();

    if (!clInDataGradient.Valid())
      return; // Nothing to do

    const RealType &outLoss = *p_clOutLoss->GetData().data();
    RealType &outLossGradient = *p_clOutLoss->GetGradient().data();

    if (IsLeaf())
      outLossGradient = RealType(1);

    const int iOuterNum = clInData.GetSize()[0];

    const RealType * const p_inData = clInData.data();
    RealType * const p_inDataGradient = clInDataGradient.data();

    const RealType scale = RealType(2)*RealType(m_fPenaltyWeight)*outLossGradient/RealType(iOuterNum);
    
    for (int i = 0; i < clInData.GetSize().Product(); ++i)
      p_inDataGradient[i] += p_inData[i]*scale;
  }

protected:
  TreeL2Regularization() = default;

private:
  float m_fPenaltyWeight = 1.0f;
};

} // end namespace bleak

#endif // !BLEAK_TREEL2REGULARIZATION_H
