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

#ifndef BLEAK_BATCHNORMALIZATION_H
#define BLEAK_BATCHNORMALIZATION_H

#include <cmath>
#include "Vertex.h"

namespace bleak {

template<typename RealType>
class BatchNormalization : public Vertex<RealType> {
public:
  bleakNewVertex(BatchNormalization, Vertex<RealType>,
    bleakAddProperty("small", m_fSmall),
    bleakAddProperty("alphaMax", m_fAlphaMax),
    bleakAddInput("inMeans"),
    bleakAddInput("inVariances"),
    bleakAddInput("inData"),
    bleakAddOutput("outData"));

  bleakForwardVertexTypedefs();

  virtual ~BatchNormalization() = default;

  virtual bool SetSizes() override {
    bleakGetAndCheckInput(p_clMeans, "inMeans", false);
    bleakGetAndCheckInput(p_clVars, "inVariances", false);
    bleakGetAndCheckInput(p_clInData, "inData", false);
    bleakGetAndCheckOutput(p_clOutData, "outData", false);

    const ArrayType &clMeans = p_clMeans->GetData();
    const ArrayType &clVars = p_clMeans->GetData();
    const ArrayType &clInData = p_clInData->GetData();

    if (!clInData.GetSize().Valid() || !clMeans.GetSize().Valid() || !clVars.GetSize().Valid()) {
      std::cerr << GetName() << ": Error: Invalid dimensions for inData, inMeans, and/or inVariances." << std::endl;
      return false;
    }

    if (m_fAlphaMax < RealType(0)) {
      std::cerr << GetName() << ": Error: alphaMax is expected be non-negative." << std::endl;
      return false;
    }

    if (m_fSmall <= RealType(0)) {
      std::cerr << GetName() << ": Error: small is expected to be positive." << std::endl;
      return false;
    }

    if (p_clMeans->GetGradient().GetSize().Valid() || p_clVars->GetGradient().GetSize().Valid()) {
      std::cerr << GetName() << ": Error: inMeans and/or inVariances appears learnable. However, BatchNormalization has its own custom updates for these inputs." << std::endl;
      return false;
    }

    if (clInData.GetSize().GetDimension() < 2) {
      std::cerr << GetName() << "Error: inData is expected to be at least 2D." << std::endl;
      return false;
    }

    if (clMeans.GetSize().GetDimension() != 1 || clVars.GetSize().GetDimension() != 1) {
      std::cerr << GetName() << ": Error: inMeans and inVariances expected to be 1D." << std::endl;
      return false;
    }

    if (clMeans.GetSize()[0] != clInData.GetSize()[1] || clVars.GetSize()[0] != clInData.GetSize()[1]) {
      std::cerr << GetName() << ": Error: inMeans and inVariances are expected to have size " << clInData.GetSize()[1] << '.' << std::endl;
      return false;
    }

    //if (clInData.GetSize().Product() / clInData.GetSize()[1] < 2)
      //std::cerr << GetName() << ": Warning: When training, either batch size should be 2 or larger or each channel should be 2 or larger." << std::endl;

    p_clOutData->GetData().SetSize(clInData.GetSize());
    p_clOutData->GetGradient().SetSize(clInData.GetSize());

    return true;
  }

  virtual bool Initialize() override {
    bleakGetAndCheckInput(p_clMeans, "inMeans", false);
    bleakGetAndCheckInput(p_clVars, "inVariances", false);

    m_vMeansTmp.resize(p_clMeans->GetData().GetSize()[0]);
    m_vVarsTmp.resize(m_vMeansTmp.size());

    m_iIteration = 0;

    return true;
  }

  virtual void Forward() override {
    bleakGetAndCheckInput(p_clMeans, "inMeans");
    bleakGetAndCheckInput(p_clVars, "inVariances");
    bleakGetAndCheckInput(p_clInData, "inData");
    bleakGetAndCheckOutput(p_clOutData, "outData");

    const ArrayType &clInData = p_clInData->GetData();
    const ArrayType &clMeans = p_clMeans->GetData();
    const ArrayType &clVars = p_clVars->GetData();
    ArrayType &clOutData = p_clOutData->GetData();

    const int iOuterNum = clInData.GetSize()[0];
    const int iNumChannels = clInData.GetSize()[1];
    const int iInnerNum = clInData.GetSize().Product(2);

    const RealType * const p_inData = clInData.data();
    RealType * const p_outData = clOutData.data_no_sync();
    const RealType * const p_means = clMeans.data();
    const RealType * const p_vars = clVars.data();

    for (int i = 0; i < iOuterNum; ++i) {
      for (int k = 0; k < iInnerNum; ++k) {
        for (int j = 0; j < iNumChannels; ++j) {
          const RealType std = std::sqrt(std::abs(p_vars[j]) + RealType(m_fSmall));
          p_outData[(i*iNumChannels + j)*iInnerNum + k] = (p_inData[(i*iNumChannels + j)*iInnerNum + k] - p_means[j])/std;
        }
      }
    }
  }

  virtual void Backward() override {
    bleakGetAndCheckInput(p_clMeans, "inMeans");
    bleakGetAndCheckInput(p_clVars, "inVariances");
    bleakGetAndCheckInput(p_clInData, "inData");
    bleakGetAndCheckOutput(p_clOutData, "outData");

    const ArrayType &clInData = p_clInData->GetData();
    ArrayType &clMeans = p_clMeans->GetData();
    ArrayType &clVars = p_clVars->GetData();
    ArrayType &clInDataGradient = p_clInData->GetGradient();
    const ArrayType &clOutData = p_clOutData->GetData();
    const ArrayType &clOutDataGradient = p_clOutData->GetGradient();

    if (!clInDataGradient.Valid())
      return; // Nothing to do

    const int iOuterNum = clInData.GetSize()[0];
    const int iNumChannels = clInData.GetSize()[1];
    const int iInnerNum = clInData.GetSize().Product(2);

    const RealType * const p_inData = clInData.data();
    RealType * const p_inDataGradient = clInDataGradient.data();
    //const RealType * const p_outData = clOutData.data();
    const RealType * const p_outDataGradient = clOutDataGradient.data();
    RealType * const p_means = clMeans.data();
    RealType * const p_vars = clVars.data();

#if 0
    // XXX: Pretending we used this batch's mean/var in Forward()!

    std::fill(m_vMeansTmp.begin(),m_vMeansTmp.end(),RealType());
    std::fill(m_vVarsTmp.begin(),m_vVarsTmp.end(),RealType());

    for (int i = 0; i < iOuterNum; ++i) {
      for (int k = 0; k < iInnerNum; ++k) {
        for (int j = 0; j < iNumChannels; ++j) {
          const RealType std = std::sqrt(p_vars[j] + RealType(m_fSmall));

          m_vVarsTmp[j] += -p_outDataGradient[(i*iNumChannels + j)*iInnerNum + k] * (p_inData[(i*iNumChannels + j)*iInnerNum + k] - p_means[j]) * std::pow(std, -3)/2;
          m_vMeansTmp[j] += -p_outDataGradient[(i*iNumChannels + j)*iInnerNum + k]/std;
        }
      }
    }
#endif

    const RealType m = RealType(iOuterNum * iInnerNum);

    for (int i = 0; i < iOuterNum; ++i) {
      for (int k = 0; k < iInnerNum; ++k) {
        for (int j = 0; j < iNumChannels; ++j) {
          const RealType std = std::sqrt(std::abs(p_vars[j]) + RealType(m_fSmall));

#if 0
          p_inDataGradient[(i*iNumChannels + j)*iInnerNum + k] += p_outDataGradient[(i*iNumChannels + j)*iInnerNum + k]/std + 
            m_vVarsTmp[j] * 2*(p_inData[(i*iNumChannels + j)*iInnerNum + k] - p_means[j])/m + m_vMeansTmp[j]/m;
#else
          p_inDataGradient[(i*iNumChannels + j)*iInnerNum + k] += p_outDataGradient[(i*iNumChannels + j)*iInnerNum + k]/std;
#endif
        }
      }
    }

    if (m <= 1) {
      std::cerr << GetName() << ": Error: Batch size or number of channels is less than 2" << std::endl;
      return;
    }

    // Update the mean and variance

    std::fill(m_vMeansTmp.begin(), m_vMeansTmp.end(), RealType());
    std::fill(m_vVarsTmp.begin(), m_vVarsTmp.end(), RealType());

    for (int i = 0; i < iOuterNum; ++i) {
      for (int k = 0; k < iInnerNum; ++k) {
        for (int j = 0; j < iNumChannels; ++j) {
          const RealType delta = p_inData[(i*iNumChannels + j)*iInnerNum + k] - m_vMeansTmp[j];
          m_vMeansTmp[j] += delta/(i*iInnerNum + k + 1);
          m_vVarsTmp[j] += delta*(p_inData[(i*iNumChannels + j)*iInnerNum + k] - m_vMeansTmp[j]);
        }
      }
    }

    const RealType alpha = std::min(RealType(m_iIteration)/RealType(m_iIteration+1), RealType(m_fAlphaMax));

    for (int j = 0; j < iNumChannels; ++j) {
      m_vVarsTmp[j] = std::abs(m_vVarsTmp[j]/RealType(m-1));

      //p_means[j] = (RealType(m_iIteration)*p_means[j] + m_vMeansTmp[j])/RealType(m_iIteration+1);
      //p_vars[j] = (RealType(m_iIteration)*p_vars[j] + m_vVarsTmp[j])/RealType(m_iIteration+1);

      p_means[j] = alpha*p_means[j] + (1 - alpha)*m_vMeansTmp[j];
      p_vars[j] = alpha*p_vars[j] + (1 - alpha)*m_vVarsTmp[j];
    }

    ++m_iIteration;
  }

protected:
  BatchNormalization() = default;

private:
  float m_fSmall = 1e-7f;
  float m_fAlphaMax = 0.99f;
  int m_iIteration = 0;

  std::vector<RealType> m_vMeansTmp, m_vVarsTmp;
};

} // end namespace bleak

#endif // !BLEAK_BATCHNORMALIZATION_H
