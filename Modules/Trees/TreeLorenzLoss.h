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

#ifndef BLEAK_TREELORENZLOSS_H
#define BLEAK_TREELORENZLOSS_H

#include <cmath>
#include <algorithm>
#include "Vertex.h"

namespace bleak {

template<typename RealType>
class TreeLorenzLoss : public Vertex<RealType> {
public:
  bleakNewVertex(TreeLorenzLoss, Vertex<RealType>,
    bleakAddInput("inData"),
    bleakAddInput("inLabels"),
    bleakAddOutput("outLoss"),
    bleakAddProperty("penaltyWeight", m_fPenaltyWeight),
    bleakAddProperty("purityWeight", m_fPurityWeight));

  bleakForwardVertexTypedefs();

  virtual ~TreeLorenzLoss() = default;

  virtual bool SetSizes() override {
    bleakGetAndCheckInput(p_clInData, "inData", false);
    bleakGetAndCheckInput(p_clInLabels, "inLabels", false);
    bleakGetAndCheckOutput(p_clOutLoss, "outLoss", false);

    //std::cout << GetName() << ": Info: penaltyWeight = " << m_fPenaltyWeight << std::endl;

    const ArrayType &clInData = p_clInData->GetData();
    const ArrayType &clInLabels = p_clInLabels->GetData();

    if (!clInData.GetSize().Valid() || !clInLabels.GetSize().Valid()) {
      std::cerr << GetName() << ": Error: inData and/or inLabel input sizes are invalid." << std::endl;
      return false;
    }

    if (clInData.GetSize().GetDimension() < 3) {
      std::cerr << GetName() << ": Error: inData is expected to be at least 3D." << std::endl;
      return false;
    }

    if (clInData.GetSize().Back() < 2) {
      std::cerr << GetName() << ": Error: Invalid number of classes in " << clInData.GetSize() << ". Number of classes should be 2 or larger." << std::endl;
      return false;
    }

    if (clInLabels.GetSize().GetDimension()+2 != clInData.GetSize().GetDimension()) {
      std::cerr << GetName() << ": Error: Incompatible dimensions: inLabels = " << clInLabels.GetSize().GetDimension() << ", inData = " << clInData.GetSize().GetDimension() << std::endl;
      return false;
    }

    const int iDim = clInData.GetSize().GetDimension();

    if (clInLabels.GetSize()[0] != clInData.GetSize()[0] || clInLabels.GetSize().SubSize(1) != clInData.GetSize().SubSize(2,iDim-1)) {
      std::cerr << GetName() << ": Error: Incompatible sizes: inLabels = " << clInLabels.GetSize() << ", inData = " << clInData.GetSize() << std::endl;
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
    bleakGetAndCheckInput(p_clInLabels, "inLabels");
    bleakGetAndCheckOutput(p_clOutLoss, "outLoss");

    const ArrayType &clInData = p_clInData->GetData();
    const ArrayType &clInLabels = p_clInLabels->GetData();

    RealType &outLoss = *p_clOutLoss->GetData().data();

    const RealType * const p_inData = clInData.data();
    const RealType * const p_inLabels = clInLabels.data();

    const int iDim = clInData.GetSize().GetDimension();

    const int iOuterNum = clInData.GetSize()[0];
    const int iNumTrees = clInData.GetSize()[1];
    const int iInnerNum = clInData.GetSize().Product(2,iDim-1);
    const int iNumClasses = clInData.GetSize().Back();

    outLoss = RealType();

    for (int i = 0; i < iOuterNum; ++i) {
      for (int l = 0; l < iNumTrees; ++l) {
        for (int k = 0; k < iInnerNum; ++k) {
          const int iLabel = (int)(p_inLabels[i*iInnerNum + k]);

          if (iLabel < 0 || iLabel >= iNumClasses) // TODO: Warning?
            continue;

          const RealType labelInput = p_inData[((i*iNumTrees + l)*iInnerNum + k)*iNumClasses + iLabel];

          for (int j = 0; j < iLabel; ++j) {
            const RealType jInput = p_inData[((i*iNumTrees + l)*iInnerNum + k)*iNumClasses + j];
            const RealType relu = std::max(RealType(0), RealType(1) + jInput - labelInput);
            outLoss += std::log(RealType(1) + relu*relu) + RealType(m_fPurityWeight)*std::abs(jInput);
            //outLoss += std::log(RealType(1) + relu*relu) + RealType(m_fPurityWeight)*jInput*jInput;
          }

          for (int j = iLabel+1; j < iNumClasses; ++j) {
            //const RealType relu = std::max(RealType(0), RealType(1) + p_inData[((i*iNumTrees + l)*iInnerNum + k)*iNumClasses + j] - labelInput);
            const RealType jInput = p_inData[((i*iNumTrees + l)*iInnerNum + k)*iNumClasses + j];
            const RealType relu = std::max(RealType(0), RealType(1) + jInput - labelInput);
            outLoss += std::log(RealType(1) + relu*relu) + RealType(m_fPurityWeight)*std::abs(jInput);
            //outLoss += std::log(RealType(1) + relu*relu) + RealType(m_fPurityWeight)*jInput*jInput;
          }
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
    const ArrayType &clInLabels = p_clInLabels->GetData(); // We don't backpropagate labels

    if (!clInDataGradient.Valid())
      return; // Nothing to do

    const RealType &outLoss = *p_clOutLoss->GetData().data();
    RealType &outLossGradient = *p_clOutLoss->GetGradient().data();

    if (IsLeaf())
      outLossGradient = RealType(1);

    const RealType * const p_inData = clInData.data();
    RealType * const p_inDataGradient = clInDataGradient.data();
    const RealType * const p_inLabels = clInLabels.data();

    const int iDim = clInData.GetSize().GetDimension();

    const int iOuterNum = clInData.GetSize()[0];
    const int iNumTrees = clInData.GetSize()[1];
    const int iInnerNum = clInData.GetSize().Product(2,iDim-1);
    const int iNumClasses = clInData.GetSize().Back();

    //std::cout << GetName() << ": inData size = " << clInDataGradient.GetSize() << std::endl;
    //std::cout << GetName() << ": iOuterNum = " << iOuterNum << ", iNumTrees = " << iNumTrees << ", iInnerNum = " << iInnerNum << ", iNumClasses = " << iNumClasses << std::endl;
    //std::cout << GetName() << ": purityWeight = " << m_fPurityWeight << std::endl;

    const RealType scale = RealType(m_fPenaltyWeight)*outLossGradient/RealType(iOuterNum);

    for (int i = 0; i < iOuterNum; ++i) {
      for (int l = 0; l < iNumTrees; ++l) {
        for (int k = 0; k < iInnerNum; ++k) {
          const int iLabel = (int)(p_inLabels[i*iInnerNum + k]);

          if (iLabel < 0 || iLabel >= iNumClasses) // TODO: Warning
            continue;

          const RealType labelInput = p_inData[((i*iNumTrees + l)*iInnerNum + k)*iNumClasses + iLabel];

          for (int j = 0; j < iLabel; ++j) {
            //const RealType relu = std::max(RealType(0), RealType(1) + p_inData[((i*iNumTrees + l)*iInnerNum + k)*iNumClasses + j] - labelInput);
            const RealType jInput = p_inData[((i*iNumTrees + l)*iInnerNum + k)*iNumClasses + j];
            const RealType sign = (jInput > RealType(0)) - (jInput < RealType(0));
            const RealType relu = std::max(RealType(0), RealType(1) + jInput - labelInput);
            const RealType diff = RealType(2)*scale*relu/(RealType(1) + relu*relu);
            p_inDataGradient[((i*iNumTrees + l)*iInnerNum + k)*iNumClasses + j] += diff + scale*RealType(m_fPurityWeight)*sign;
            //p_inDataGradient[((i*iNumTrees + l)*iInnerNum + k)*iNumClasses + j] += diff + scale*RealType(m_fPurityWeight)*RealType(2)*jInput;
            p_inDataGradient[((i*iNumTrees + l)*iInnerNum + k)*iNumClasses + iLabel] -= diff;
          }

          for (int j = iLabel+1; j < iNumClasses; ++j) {
            const RealType jInput = p_inData[((i*iNumTrees + l)*iInnerNum + k)*iNumClasses + j];
            const RealType sign = (jInput > RealType(0)) - (jInput < RealType(0));
            const RealType relu = std::max(RealType(0), RealType(1) + jInput - labelInput);
            const RealType diff = RealType(2)*scale*relu/(RealType(1) + relu*relu);
            p_inDataGradient[((i*iNumTrees + l)*iInnerNum + k)*iNumClasses + j] += diff + scale*RealType(m_fPurityWeight)*sign;
            //p_inDataGradient[((i*iNumTrees + l)*iInnerNum + k)*iNumClasses + j] += diff + scale*RealType(m_fPurityWeight)*RealType(2)*jInput;
            p_inDataGradient[((i*iNumTrees + l)*iInnerNum + k)*iNumClasses + iLabel] -= diff;
          }
        }
      }
    }
  }

protected:
  TreeLorenzLoss() = default;

private:
  float m_fPenaltyWeight = 1.0f;
  float m_fPurityWeight = 1.0f;
};

} // end namespace bleak

#endif // !BLEAK_TREELORENZLOSS_H
