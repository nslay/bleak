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
#include "BlasWrapper.h"
#include "Vertex.h"

// DISCLAIMER: This is NOT the original BatchNorm formulation. It is a simple running average lacking the fancy/strange gradient and train/test discrepancy.

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
    bleakGetAndCheckInput(p_clInData, "inData", false);
    bleakGetAndCheckInput(p_clMeans, "inMeans", false);
    bleakGetAndCheckInput(p_clVars, "inVariances", false);

    //m_vMeansTmp.resize(p_clMeans->GetData().GetSize()[0]);
    //m_vVarsTmp.resize(m_vMeansTmp.size());

    const int iOuterNum = p_clInData->GetData().GetSize()[0];
    const int iNumChannels = p_clInData->GetData().GetSize()[1];
    const int iInnerNum = p_clInData->GetData().GetSize().Product(2);

    const Size clOnesSize = { iOuterNum * iInnerNum };
    m_clOnes.SetSize(clOnesSize);
    m_clOnes.Allocate();
    m_clOnes.Fill(RealType(1));

    const Size clMomentSize = { iNumChannels };

    m_clVarsTmp.SetSize(clMomentSize);
    m_clVarsTmp.Allocate();

    if (p_clInData->GetGradient().Valid()) {
      m_clWork.SetSize(p_clInData->GetData().GetSize());
      m_clWork.Allocate();

      m_clMeansTmp.SetSize(clMomentSize);
      m_clMeansTmp.Allocate();
    }

    m_iIteration = 0;

    return true;
  }

  virtual bool TestGradient(const std::string &strOutputName) override {
    const int iIterationOrig = m_iIteration;
    const RealType fAlphaMaxOrig = m_fAlphaMax;

    m_iIteration = 1000000000;
    m_fAlphaMax = 1.0f;

    const bool bSuccess = SuperType::TestGradient(strOutputName);
    
    m_fAlphaMax = fAlphaMaxOrig;
    m_iIteration = iIterationOrig;

    return bSuccess;
  }

  virtual void ForwardCPU() override {
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

    if (iInnerNum == 1) {
      cpu_blas::copy(iOuterNum*iNumChannels, p_inData, 1, p_outData, 1);

      // In C++
      // clInData is BatchSize x Channels

      // Want to shift by means and then scale by standard deviations while leveraging BLAS as much as possible
      //
      // Let's use ger to shift by means
      // X - 1 * mu^T
      //
      // In Fortran
      // clInData is Channels by BatchSize
      //
      // We instead want X - mu * 1^T
      //
      cpu_blas::ger(iNumChannels, iOuterNum, RealType(-1), p_means, 1, m_clOnes.data(), 1, p_outData, iNumChannels);

      RealType * const p_varsTmp = m_clVarsTmp.data_no_sync();
      for (int j = 0; j < iNumChannels; ++j)
        p_varsTmp[j] = RealType(1)/std::sqrt(std::abs(p_vars[j]) + RealType(m_fSmall));

      for (int i = 0; i < iOuterNum; ++i)
        for (int j = 0; j < iNumChannels; ++j)
          p_outData[i*iNumChannels + j] *= p_varsTmp[j];

      // I mean... iNumChannels is probably larger!
      //for (int j = 0; j < iNumChannels; ++j) {
      //  const RealType std = std::sqrt(std::abs(p_vars[j]) + RealType(m_fSmall));
      //  cpu_blas::axpy(iOuterNum, RealType(-1), p_means + j, 0, p_outData + j, iNumChannels);
      //  cpu_blas::scal(iOuterNum, RealType(1)/std, p_outData + j, iNumChannels);
      //}   
    }
    else {
      for (int i = 0; i < iOuterNum; ++i) {
        cpu_blas::copy(iNumChannels*iInnerNum, p_inData + i*iNumChannels*iInnerNum, 1, p_outData + i*iNumChannels*iInnerNum, 1);

        for (int j = 0; j < iNumChannels; ++j) {
          const RealType std = std::sqrt(std::abs(p_vars[j]) + RealType(m_fSmall));
          cpu_blas::axpy(iInnerNum, RealType(-1), p_means + j, 0, p_outData + (i*iNumChannels + j)*iInnerNum, 1);
          cpu_blas::scal(iInnerNum, RealType(1)/std, p_outData + (i*iNumChannels + j)*iInnerNum, 1);
        }
      }
    }
  }

  virtual void BackwardCPU() override {
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

    if (iInnerNum == 1) {
      RealType * const p_varsTmp = m_clVarsTmp.data_no_sync();
      for (int j = 0; j < iNumChannels; ++j)
        p_varsTmp[j] = RealType(1)/std::sqrt(std::abs(p_vars[j]) + RealType(m_fSmall));

      for (int i = 0; i < iOuterNum; ++i)
        for (int j = 0; j < iNumChannels; ++j)
          p_inDataGradient[i*iNumChannels + j] += p_outDataGradient[i*iNumChannels + j] * p_varsTmp[j];

      //for (int j = 0; j < iNumChannels; ++j) {
      //  const RealType std = std::sqrt(std::abs(p_vars[j]) + RealType(m_fSmall));
      //  cpu_blas::axpy(iOuterNum, RealType(1)/std, p_outDataGradient + j, iNumChannels, p_inDataGradient + j, iNumChannels);
      //}
    }
    else {
      for (int i = 0; i < iOuterNum; ++i) {
        for (int j = 0; j < iNumChannels; ++j) {
          const RealType std = std::sqrt(std::abs(p_vars[j]) + RealType(m_fSmall));
          cpu_blas::axpy(iInnerNum, RealType(1)/std, p_outDataGradient + (i*iNumChannels + j)*iInnerNum, 1, p_inDataGradient + (i*iNumChannels + j)*iInnerNum, 1);
        }
      }
    }

    const RealType m = RealType(iOuterNum * iInnerNum);

    if (m < 2) {
      std::cerr << GetName() << ": Error: Batch size or number of channels is less than 2" << std::endl;
      return;
    }

    RealType * const p_work = m_clWork.data_no_sync();
    RealType * const p_meansTmp = m_clMeansTmp.data_no_sync();
    RealType * const p_varsTmp = m_clVarsTmp.data_no_sync();

    m_clMeansTmp.Fill(RealType(0));
    m_clVarsTmp.Fill(RealType(0));

    if (iInnerNum == 1) {
      // C/C++
      // X is BatchSize x Channels
      //
      // That means, I would want
      // X^T * 1/N = means
      //
      // in Fortran, that means:
      // X is Channels x BatchSize
      //
      // Hence, I actually want:
      // X * 1/N = means

      cpu_blas::gemv('N', iNumChannels, iOuterNum, RealType(1)/m, p_inData, iNumChannels, m_clOnes.data(), 1, RealType(0), p_meansTmp, 1);
      cpu_blas::copy(iOuterNum*iNumChannels, p_inData, 1, p_work, 1);
      cpu_blas::ger(iNumChannels, iOuterNum, RealType(-1), p_meansTmp, 1, m_clOnes.data(), 1, p_work, iNumChannels);

      // Square all the elements
      for (int i = 0; i < iNumChannels*iOuterNum; ++i)
        p_work[i] = std::pow(p_work[i], 2);

      cpu_blas::gemv('N', iNumChannels, iOuterNum, RealType(1)/RealType(m-1), p_work, iNumChannels, m_clOnes.data(), 1, RealType(0), p_varsTmp, 1);

      //for (int j = 0; j < iNumChannels; ++j) {
      //  cpu_blas::copy(iOuterNum, p_inData + j, iNumChannels, p_work, 1);
      //  cpu_blas::axpy(iOuterNum, RealType(1)/m, p_work, 1, p_meansTmp + j, 0);
      //  cpu_blas::axpy(iOuterNum, RealType(-1), p_meansTmp + j, 0, p_work, 1);
      //  p_varsTmp[j] = cpu_blas::dot(iOuterNum, p_work, 1, p_work, 1) / RealType(m-1);
      //}
    }
    else {
      // As opposed to iOuterNum x C x iInnerNum
      // I want C x iOuterNum x iInnerNum
      for (int i = 0; i < iOuterNum; ++i) {
        for (int j = 0; j < iNumChannels; ++j) {
          cpu_blas::copy(iInnerNum, p_inData + (i*iNumChannels + j)*iInnerNum, 1, p_work + (j*iOuterNum + i)*iInnerNum, 1);
        }
      }

      // Now X = C x (iOuterNum * iInnerNum)
      // So I would want:
      // X * 1

      cpu_blas::gemv('T', iOuterNum*iInnerNum, iNumChannels, RealType(1)/m, p_work, iOuterNum*iInnerNum, m_clOnes.data(), 1, RealType(0), p_meansTmp, 1);
      cpu_blas::ger(iOuterNum*iInnerNum, iNumChannels, RealType(-1), m_clOnes.data(), 1, p_meansTmp, 1, p_work, iOuterNum*iInnerNum);

      // Square all the elements
      for (int i = 0; i < iNumChannels*iInnerNum*iOuterNum; ++i)
        p_work[i] = std::pow(p_work[i], 2);

      cpu_blas::gemv('T', iOuterNum*iInnerNum, iNumChannels, RealType(1)/RealType(m-1), p_work, iOuterNum*iInnerNum, m_clOnes.data(), 1, RealType(0), p_varsTmp, 1);

      //for (int j = 0; j < iNumChannels; ++j) {
      //  for (int i = 0; i < iOuterNum; ++i)
      //    cpu_blas::copy(iInnerNum, p_inData + (i*iNumChannels + j)*iInnerNum, 1, p_work + i*iInnerNum, 1);
      //
      //  cpu_blas::axpy(iOuterNum*iInnerNum, RealType(1)/m, p_work, 1, p_meansTmp + j, 0);
      //  cpu_blas::axpy(iOuterNum*iInnerNum, RealType(-1), p_meansTmp + j, 0, p_work, 1);
      //  p_varsTmp[j] = cpu_blas::dot(iOuterNum*iInnerNum, p_work, 1, p_work, 1) / RealType(m-1);
      //}
    }

    const RealType alpha = std::min(RealType(m_iIteration)/RealType(m_iIteration+1), RealType(m_fAlphaMax));

    for (int j = 0; j < iNumChannels; ++j) {
      p_varsTmp[j] = std::abs(p_varsTmp[j]);

      p_means[j] = alpha*p_means[j] + (RealType(1) - alpha)*p_meansTmp[j];
      p_vars[j] = alpha*p_vars[j] + (RealType(1) - alpha)*p_varsTmp[j];
    }

    ++m_iIteration;
  }

#ifdef BLEAK_USE_CUDA
  virtual void ForwardGPU() override;
  virtual void BackwardGPU() override;
#endif // BLEAK_USE_CUDA

protected:
  BatchNormalization() = default;

private:
  float m_fSmall = 1e-7f;
  float m_fAlphaMax = 0.99f;
  int m_iIteration = 0;

  ArrayType m_clWork; // Work variable
  ArrayType m_clOnes; // Vector of 1s.
  ArrayType m_clMeansTmp;
  ArrayType m_clVarsTmp;
};

} // end namespace bleak

#endif // !BLEAK_BATCHNORMALIZATION_H
