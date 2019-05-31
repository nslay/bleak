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

#ifndef BLEAK_L2LOSS_H
#define BLEAK_L2LOSS_H

#include <cmath>
#include "Vertex.h"

namespace bleak {

template<typename RealType>
class L2Loss : public Vertex<RealType> {
public:
  bleakNewVertex(L2Loss, Vertex<RealType>,
    bleakAddInput("inData"),
    bleakAddInput("inLabels"), // Named this way for consistency
    bleakAddOutput("outLoss"),
    bleakAddProperty("penaltyWeight", m_fPenaltyWeight));

  bleakForwardVertexTypedefs();

  virtual ~L2Loss() = default;

  virtual bool SetSizes() override {
    bleakGetAndCheckInput(p_clInData, "inData", false);
    bleakGetAndCheckInput(p_clInLabels, "inLabels", false);
    bleakGetAndCheckOutput(p_clOutLoss, "outLoss", false);

    const ArrayType &clInData = p_clInData->GetData();
    const ArrayType &clInLabels = p_clInLabels->GetData();
    ArrayType &clOutLoss = p_clOutLoss->GetData();

    if (!clInData.GetSize().Valid() || !clInLabels.GetSize().Valid()) {
      std::cerr << GetName() << ": Error: Invalid dimensions for inData and/or inLabels." << std::endl;
      return false;
    }

    if (clInData.GetSize().GetDimension() > 2 || clInLabels.GetSize().GetDimension() > 2) {
      std::cerr << GetName() << ": Error: inData and inLabels expected to be no larger than 2D." << std::endl;
      return false;
    }

    if (clInData.GetSize().Squeeze().GetDimension() != clInLabels.GetSize().Squeeze().GetDimension()) {
      std::cerr << GetName() << ": Error: Dimension mismatch between inData and inLabels." << std::endl;
      return false;
    }

    const Size clOutSize = { 1 };

    p_clOutLoss->GetData().SetSize(clOutSize);
    p_clOutLoss->GetGradient().SetSize(clOutSize);

    return true;
  }

  virtual bool Initialize() override {
    return true; // Nothing to do
  }

  virtual void Forward() override {
    bleakGetAndCheckInput(p_clInData, "inData");
    bleakGetAndCheckInput(p_clInLabels, "inLabels");
    bleakGetAndCheckOutput(p_clOutLoss, "outLoss");

    const ArrayType &clInData = p_clInData->GetData();
    const ArrayType &clInLabels = p_clInLabels->GetData();
    ArrayType &clOutLoss = p_clOutLoss->GetData();

    const int iOuterNum = clInData.GetSize()[0];
    const int iNumLabels = clInData.GetSize().Product(1);

    const RealType * const p_inData = clInData.data();
    const RealType * const p_inLabels = clInLabels.data();

    RealType &loss = *clOutLoss.data();

    loss = RealType();

    for (int i = 0; i < iOuterNum; ++i) {
      for (int j = 0; j < iNumLabels; ++j) {
        loss += std::pow(p_inData[i*iNumLabels + j] - p_inLabels[i*iNumLabels + j], 2);
      }
    }

    loss *= RealType(m_fPenaltyWeight)/RealType(iOuterNum);
  }

  virtual void Backward() override {
    bleakGetAndCheckInput(p_clInData, "inData");
    bleakGetAndCheckInput(p_clInLabels, "inLabels");
    bleakGetAndCheckOutput(p_clOutLoss, "outLoss");

    const ArrayType &clInData = p_clInData->GetData();
    ArrayType &clInDataGradient = p_clInData->GetGradient();
    const ArrayType &clInLabels = p_clInLabels->GetData();
    const ArrayType &clOutLoss = p_clOutLoss->GetData();
    ArrayType &clOutLossGradient = p_clOutLoss->GetGradient();

    if (!clInDataGradient.Valid()) // Nothing to do
      return;

    const int iOuterNum = clInData.GetSize()[0];
    const int iNumLabels = clInData.GetSize().Product(1);

    RealType * const p_inDataGradient = clInDataGradient.data();
    const RealType * const p_inData = clInData.data();
    const RealType * const p_inLabels = clInLabels.data();

    RealType &lossGradient = *clOutLossGradient.data();

    if (IsLeaf())
      lossGradient = RealType(1);

    const RealType scale = RealType(2*m_fPenaltyWeight)*lossGradient/RealType(iOuterNum);

    for (int i = 0; i < iOuterNum; ++i) {
      for (int j = 0; j < iNumLabels; ++j) {
        p_inDataGradient[i*iNumLabels + j] += scale*(p_inData[i*iNumLabels + j] - p_inLabels[i*iNumLabels + j]);
      }
    }
  }

protected:
  L2Loss() {
    m_fPenaltyWeight = 1.0f;
  }

private:
  float m_fPenaltyWeight;
};

} // end namespace bleak

#endif // !BLEAK_L2LOSS_H
