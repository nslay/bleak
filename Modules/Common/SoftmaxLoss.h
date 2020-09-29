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
#include <vector>
#include "Softmax.h"

namespace bleak {

template<typename RealType>
class SoftmaxLoss : public Softmax<RealType> {
public:
  bleakNewVertex(SoftmaxLoss, Softmax<RealType>,
    bleakAddInput("inLabels"),
    bleakAddOutput("outLoss"),
    bleakAddProperty("gamma", m_fGamma),
    bleakAddProperty("penaltyWeights", m_vPenaltyWeights),
    bleakAddGetterSetter("penaltyWeight", &SoftmaxLoss::GetPenaltyWeight, &SoftmaxLoss::SetPenaltyWeight));

  bleakForwardVertexTypedefs();

  virtual ~SoftmaxLoss() = default;

  constexpr static RealType GetSmall() { return RealType(1e-30); }

  virtual bool TestGradient() override {
    return SuperType::TestGradient("outLoss");
  }

  bool GetPenaltyWeight(float &fPenaltyWeight) const {
    switch (m_vPenaltyWeights.size()) {
    case 0:
      fPenaltyWeight = 1.0f;
      break;
    case 1:
      fPenaltyWeight = m_vPenaltyWeights[0];
      break;
    default:
      return false;
    }

    return true;
  }

  bool SetPenaltyWeight(const float &fPenaltyWeight) {
    m_vPenaltyWeights.resize(1);
    m_vPenaltyWeights[0] = fPenaltyWeight;
    return true;
  }

  virtual bool SetSizes() override {
    if (!SuperType::SetSizes())
      return false;

    bleakGetAndCheckInput(p_clInLabels, "inLabels", false);

    bleakGetAndCheckOutput(p_clOutProbs, "outProbabilities", false);
    bleakGetAndCheckOutput(p_clOutLoss, "outLoss", false);

    const ArrayType &clInLabels = p_clInLabels->GetData();
    const ArrayType &clOutProbs = p_clOutProbs->GetData();

    if (m_vPenaltyWeights.size() > 1 && (size_t)clOutProbs.GetSize()[1] != m_vPenaltyWeights.size()) {
      std::cerr << GetName() << ": Error: Dimension mismatch between penaltyWeights and outProbabilities. Expected " << m_vPenaltyWeights.size() << " channels, but got " << clOutProbs.GetSize()[1] << " channels." << std::endl;
      return false;
    }

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

    //RealType &outLoss = *clOutLoss.data();

    //outLoss = RealType();

    RealType outLoss = RealType();

    // Pretty arbitrary... 
    if (iInnerNum > iOuterNum) {
      for (int i = 0; i < iOuterNum; ++i) {

#pragma omp parallel for reduction(+:outLoss)
        for (int j = 0; j < iInnerNum; ++j) {
          const int iLabel = (int)p_inLabels[iInnerNum*i + j];

          if (iLabel < 0 || iLabel >= iNumOutputs)
            continue;

          const RealType weight = GetPenaltyWeightForLabel(iLabel, iNumOutputs);

          const RealType prob = std::max(GetSmall(), p_outProbs[(i*iNumOutputs + iLabel)*iInnerNum + j]);

          const RealType focalLossScale = m_fGamma > 0.0f ? std::pow(RealType(1) - prob, RealType(m_fGamma)) : RealType(1);
          outLoss += -std::log(prob) * weight * focalLossScale;
        }
      }
    }
    else {

#pragma omp parallel for reduction(+:outLoss)
      for (int i = 0; i < iOuterNum; ++i) {
        for (int j = 0; j < iInnerNum; ++j) {
          const int iLabel = (int)p_inLabels[iInnerNum*i + j];

          if (iLabel < 0 || iLabel >= iNumOutputs)
            continue;

          const RealType weight = GetPenaltyWeightForLabel(iLabel, iNumOutputs);

          const RealType prob = std::max(GetSmall(), p_outProbs[(i*iNumOutputs + iLabel)*iInnerNum + j]);

          const RealType focalLossScale = m_fGamma > 0.0f ? std::pow(RealType(1) - prob, RealType(m_fGamma)) : RealType(1);
          outLoss += -std::log(prob) * weight * focalLossScale;
        }
      }
    }

    outLoss /= RealType(iOuterNum);
    //outLoss *= RealType(m_fPenaltyWeight)/RealType(iOuterNum);

    *clOutLoss.data() = outLoss;
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

    //const RealType scale = RealType(m_fPenaltyWeight)*outGradient/RealType(iOuterNum);
    const RealType scale = outGradient/RealType(iOuterNum);

    // Pretty arbitrary...
    if (iInnerNum > iOuterNum) {
      for (int i = 0; i < iOuterNum; ++i) {

#pragma omp parallel for
        for (int k = 0; k < iInnerNum; ++k) {
          const int iLabel = (int)p_inLabels[iInnerNum*i + k];

          // Do no learning on this example
          if (iLabel < 0 || iLabel >= iNumOutputs)
            continue;

          RealType weight = GetPenaltyWeightForLabel(iLabel, iNumOutputs)*scale;

          //RealType focalLossScale = RealType(1);

          if (m_fGamma > 0.0f) {
            const RealType prob = std::max(GetSmall(), p_outProbs[(i*iNumOutputs + iLabel)*iInnerNum + k]);
            //focalLossScale = std::pow(RealType(1) - prob, RealType(m_fGamma - 1.0f));
            //focalLossScale *= -RealType(m_fGamma)*prob*std::log(prob) + (RealType(1) - prob);
            weight *= std::pow(RealType(1) - prob, RealType(m_fGamma - 1.0f)) * (-RealType(m_fGamma)*prob*std::log(prob) + (RealType(1) - prob));
          }

          for (int j = 0; j < iNumOutputs; ++j) {
            const RealType prob = p_outProbs[(i*iNumOutputs + j)*iInnerNum + k];
            //p_inGradient[(i*iNumOutputs + j)*iInnerNum + k] += focalLossScale*weight*scale*prob;
            p_inGradient[(i*iNumOutputs + j)*iInnerNum + k] += weight*prob;
          }

          //p_inGradient[(i*iNumOutputs + iLabel)*iInnerNum + k] -= focalLossScale*weight*scale;
          p_inGradient[(i*iNumOutputs + iLabel)*iInnerNum + k] -= weight;
        }
      }

    }
    else {

#pragma omp parallel for
      for (int i = 0; i < iOuterNum; ++i) {
        for (int k = 0; k < iInnerNum; ++k) {
          const int iLabel = (int)p_inLabels[iInnerNum*i + k];

          // Do no learning on this example
          if (iLabel < 0 || iLabel >= iNumOutputs)
            continue;

          RealType weight = GetPenaltyWeightForLabel(iLabel, iNumOutputs)*scale;

          //RealType focalLossScale = RealType(1);

          if (m_fGamma > 0.0f) {
            const RealType prob = std::max(GetSmall(), p_outProbs[(i*iNumOutputs + iLabel)*iInnerNum + k]);
            //focalLossScale = std::pow(RealType(1) - prob, RealType(m_fGamma - 1.0f));
            //focalLossScale *= -RealType(m_fGamma)*prob*std::log(prob) + (RealType(1) - prob);
            weight *= std::pow(RealType(1) - prob, RealType(m_fGamma - 1.0f)) * (-RealType(m_fGamma)*prob*std::log(prob) + (RealType(1) - prob));
          }

          for (int j = 0; j < iNumOutputs; ++j) {
            const RealType prob = p_outProbs[(i*iNumOutputs + j)*iInnerNum + k];
            //p_inGradient[(i*iNumOutputs + j)*iInnerNum + k] += focalLossScale*weight*scale*prob;
            p_inGradient[(i*iNumOutputs + j)*iInnerNum + k] += weight*prob;
          }

          //p_inGradient[(i*iNumOutputs + iLabel)*iInnerNum + k] -= focalLossScale*weight*scale;
          p_inGradient[(i*iNumOutputs + iLabel)*iInnerNum + k] -= weight;
        }
      }
    }
  }

protected:
  SoftmaxLoss() = default;

  RealType GetPenaltyWeightForLabel(int iLabel, int iNumClasses) const {
    if (iLabel < 0 || iLabel >= iNumClasses) // No learning on these labels!
      return RealType(0);

    switch (m_vPenaltyWeights.size()) {
    case 0: // Default weight
      return RealType(1); 
    case 1: // All labels use the same weight
      return RealType(m_vPenaltyWeights[0]);
    default:
      return RealType(m_vPenaltyWeights[iLabel]);
    }

    return RealType(0); // Not reached
  }

private:
  std::vector<float> m_vPenaltyWeights = { 1.0f };
  float m_fGamma = 0.0f;
};

} // end namespace bleak

#endif // !BLEAK_SOFTMAXLOSS_H
