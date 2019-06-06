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

#ifndef BLEAK_HELLINGERLOSS_H
#define BLEAK_HELLINGERLOSS_H

#include <cmath>
#include "Vertex.h"

namespace bleak {

template<typename RealType>
class HellingerLoss : public Vertex<RealType> {
public:
  bleakNewVertex(HellingerLoss, Vertex<RealType>,
    bleakAddInput("inData"),
    bleakAddInput("inLabels"),
    bleakAddOutput("outLoss"),
    bleakAddProperty("small", m_fSmall),
    bleakAddProperty("penaltyWeight", m_fPenaltyWeight));

  bleakForwardVertexTypedefs();

  virtual ~HellingerLoss() = default;

  virtual bool SetSizes() override {
    bleakGetAndCheckInput(p_clInData, "inData", false);
    bleakGetAndCheckInput(p_clInLabels, "inLabels", false);
    bleakGetAndCheckOutput(p_clOutLoss, "outLoss", false);

    if (m_fSmall <= 0.0f) {
      std::cerr << GetName() << ": Error: small must be a positive value." << std::endl;
      return false;
    }

    const ArrayType &clInData = p_clInData->GetData();
    const ArrayType &clInLabels = p_clInLabels->GetData();
    ArrayType &clOutLoss = p_clOutLoss->GetData();

    if (!clInData.GetSize().Valid()) {
      std::cerr << GetName() << ": Error: Invalid dimensions for inData." << std::endl;
      return false;
    }

    if (clInData.GetSize().GetDimension() < 2) {
      std::cerr << GetName() << ": Error: inData is expected to be 2D or higher." << std::endl;
      return false;
    }

    if (clInData.GetSize()[1] < 2) {
      std::cerr << GetName() << ": Error: Invalid number of channels in " << clInData.GetSize() << ". Number of channels should be 2 or larger." << std::endl;
      return false;
    }

    if (clInData.GetSize().GetDimension() != clInLabels.GetSize().GetDimension()+1) {
      std::cerr << GetName() << ": Error: Incompatible dimensions: inLabels = " << clInLabels.GetSize().GetDimension() << ", inData = " << clInData.GetSize().GetDimension() << std::endl;
      return false;
    }

    if (clInLabels.GetSize()[0] != clInData.GetSize()[0] || clInLabels.GetSize().SubSize(1) != clInData.GetSize().SubSize(2)) {
      std::cerr << GetName() << ": Error: Incompatible sizes: inLabels = " << clInLabels.GetSize() << ", inData = " << clInData.GetSize() << std::endl;
      return false;
    }

    const Size clSize = { 1 };

    p_clOutLoss->GetData().SetSize(clSize);
    p_clOutLoss->GetGradient().SetSize(clSize);

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

    RealType &outLoss = *clOutLoss.data();

    outLoss = RealType();

    const int iOuterNum = clInData.GetSize()[0];
    const int iNumOutputs = clInData.GetSize()[1];
    const int iInnerNum = clInData.GetSize().Product(2);

    const RealType * const p_inData = clInData.data();
    const RealType * const p_inLabels = clInLabels.data();

    for (int i = 0; i < iOuterNum; ++i) {
      for (int k = 0; k < iInnerNum; ++k) {
        const int iLabel = (int)p_inLabels[iInnerNum*i + k];

        if (iLabel < 0 || iLabel >= iNumOutputs) // TODO: Warning?
          continue;

        for (int j = 0; j < iNumOutputs; ++j) {
          const RealType agree = (j == iLabel) ? RealType(1) : RealType(-1);
          const RealType likelihood = agree * p_inData[(i*iNumOutputs + j)*iInnerNum + k];
          const RealType sign = RealType((likelihood > RealType(0)) - (likelihood < RealType(0)));

          outLoss += std::pow(sign*sqrt(std::abs(likelihood)) - RealType(1), 2);
        }
      }
    }

    outLoss *= RealType(m_fPenaltyWeight)/RealType(iOuterNum);
  }

  virtual void Backward() override {
    bleakGetAndCheckInput(p_clInData, "inData");
    bleakGetAndCheckInput(p_clInLabels, "inLabels");
    bleakGetAndCheckOutput(p_clOutLoss, "outLoss");

    const ArrayType &clInData = p_clInData->GetData();
    ArrayType &clInDataGradient = p_clInData->GetGradient();
    const ArrayType &clInLabels = p_clInLabels->GetData(); // We don't backpropagate on labels
    ArrayType &clOutLossGradient = p_clOutLoss->GetGradient();

    if (!clInDataGradient.Valid())
      return; // Nothing to do

    const int iOuterNum = clInData.GetSize()[0];
    const int iNumOutputs = clInData.GetSize()[1];
    const int iInnerNum = clInData.GetSize().Product(2);

    const RealType * const p_inLabels = clInLabels.data();
    const RealType * const p_inData = clInData.data();
    RealType * const p_inDataGradient = clInDataGradient.data();

    RealType &outLossGradient = *clOutLossGradient.data();

    if (IsLeaf())
      outLossGradient = RealType(1);

    const RealType scale = RealType(m_fPenaltyWeight)*outLossGradient/RealType(iOuterNum);

    for (int i = 0; i < iOuterNum; ++i) {
      for (int k = 0; k < iInnerNum; ++k) {
        const int iLabel = (int)p_inLabels[iInnerNum*i + k];

        if (iLabel < 0 || iLabel >= iNumOutputs) // TODO: Warning?
          continue;

        for (int j = 0; j < iNumOutputs; ++j) {
          const RealType agree = (j == iLabel) ? RealType(1) : RealType(-1);
          const RealType likelihood = agree * p_inData[(i*iNumOutputs + j)*iInnerNum + k];
          const RealType sign = RealType((likelihood > RealType(0)) - (likelihood < RealType(0)));

          if (std::abs(likelihood) < RealType(m_fSmall))
            p_inDataGradient[(i*iNumOutputs + j)*iInnerNum + k] += (sign - RealType(1)/std::sqrt(RealType(m_fSmall)))*agree*scale;
          else
            p_inDataGradient[(i*iNumOutputs + j)*iInnerNum + k] += (sign - RealType(1)/std::sqrt(std::abs(likelihood)))*agree*scale;
        }
      }
    }
  }

protected:
  HellingerLoss() = default;

private:
  float m_fPenaltyWeight = 1.0f;
  float m_fSmall = 1e-4f;
};

} // end namespace bleak

#endif // !BLEAK_HELLINGERLOSS_H
