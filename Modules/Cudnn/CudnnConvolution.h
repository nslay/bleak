/*-
 * Copyright (c) 2020 Nathan Lay (enslay@gmail.com)
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

#ifndef BLEAK_CUDNNCONVOLUTION_H
#define BLEAK_CUDNNCONVOLUTION_H

#include <iostream>
#include <algorithm>
#include <vector>
#include "Vertex.h"
#include "CudnnCommon.h"

namespace bleak {

template<typename RealType, unsigned int Dimension>
class CudnnConvolution : public Vertex<RealType> {
public:
  bleakNewAbstractVertex(CudnnConvolution, Vertex<RealType>,
    bleakAddInput("inData"),
    bleakAddInput("inWeights"),
    bleakAddInput("inBias"),
    bleakAddOutput("outData"),
    bleakAddProperty("padding", m_vPadding),
    bleakAddProperty("stride", m_vStride),
    bleakAddProperty("dilate", m_vDilate));

  bleakForwardVertexTypedefs();

  virtual ~CudnnConvolution() = default;

  static_assert(Dimension > 0, "Dimension must be larger than 0.");

  constexpr static unsigned int GetDimension() { return Dimension; }

  virtual bool SetSizes() override {
    bleakGetAndCheckInput(p_clInWeights, "inWeights", false);
    bleakGetAndCheckInput(p_clInData, "inData", false);
    bleakGetAndCheckOutput(p_clOutData, "outData", false);

    std::shared_ptr<EdgeType> p_clInBias = GetInput("inBias");

    if (m_vPadding.size() != GetDimension() || m_vStride.size() != GetDimension() || m_vDilate.size() != GetDimension()) {
      std::cerr << GetName() << ": Error: 'padding', 'stride' and 'dilate' expected to be " << GetDimension() << "D." << std::endl;
      return false;
    }

    if (*std::min_element(m_vPadding.begin(), m_vPadding.end()) < 0) {
      std::cerr << GetName() << ": Error: 'padding' expected to be non-negative." << std::endl;
      return false;
    }

    if (*std::min_element(m_vStride.begin(), m_vStride.end()) <= 0) {
      std::cerr << GetName() << ": Error: 'stride' expected to be positive." << std::endl;
      return false;
    }

    if (*std::min_element(m_vDilate.begin(), m_vDilate.end()) <= 0) {
      std::cerr << GetName() << ": Error: 'dilate' expected to be positive." << std::endl;
      return false;
    }

    const Size clInDataSize = p_clInData->GetData().GetSize();
    const Size clInWeightsSize = p_clInWeights->GetData().GetSize();

    if (!clInDataSize.Valid() || !clInWeightsSize.Valid()) {
      std::cerr << GetName() << ": Error: Invalid 'inData' (" << clInDataSize << ") or 'inWeights' size (" << clInWeightsSize << ")." << std::endl;
      return false;
    }

    if (clInDataSize.GetDimension() != 2 + GetDimension()) {
      std::cerr << GetName() << ": Error: Expected 'inData' to be BatchSize x InputChannels x ImageDimensions (expected " << 2 + GetDimension() << " dimensions but got " << clInDataSize.GetDimension() << " dimensions)." << std::endl;
      return false;
    }

    if (clInWeightsSize.GetDimension() != 2 + GetDimension()) {
      std::cerr << GetName() << ": Error: Expected 'inWeights' to be OutputChannels x InputChannels x KernelDimensions (expected " << 2 + GetDimension() << " dimensions but got " << clInWeightsSize.GetDimension() << " dimensions)." << std::endl;
      return false;
    }

    if (p_clInBias != nullptr) {
      const Size clInBiasSize = p_clInBias->GetData().GetSize();

      if (!clInBiasSize.Valid()) {
        std::cerr << GetName() << ": Error: Invalid 'inBias'." << std::endl;
        return false;
      }

      if (clInBiasSize.GetDimension() != 1) {
        std::cerr << GetName() << ": Error: Expected 'inBias' to be OutputChannels (expected 1 dimension but got " << clInBiasSize.GetDimension() << " dimensions)." << std::endl;
        return false;
      }

      if (clInBiasSize[0] != clInWeightsSize[0]) {
        std::cerr << GetName() << ": Error: Expected same number of output channels between 'inBias' and 'inWeights'." << std::endl;
        return false;
      }
    }

    if (!m_clInDataDesc.Set(clInDataSize)) {
      std::cerr << GetName() << ": Error: Could not set inData descriptor." << std::endl;
      return false;
    }

    if (!m_clWeightsDesc.Set(clInWeightsSize)) {
      std::cerr << GetName() << ": Error: Could not set inWeights descriptor." << std::endl;
      return false;
    }

    if (p_clInBias != nullptr) {
      //Size clBiasSize(clInWeightsSize.GetDimension());
      Size clBiasSize(4); // Min size of tensor descriptor stuff is 4

      clBiasSize[1] = p_clInBias->GetData().GetSize()[0];

      if (!m_clBiasDesc.Set(clBiasSize)) {
        std::cerr << GetName() << ": Error: Could not set inBias descriptor." << std::endl;
        return false;
      }

      if (!m_clActivationDesc.Set(CUDNN_ACTIVATION_IDENTITY)) {
        std::cerr << GetName() << ": Error: Could not set activation descriptor." << std::endl;
        return false;
      }
    }

    if (!m_clConvolutionDesc.Set((int)GetDimension(), m_vPadding.data(), m_vStride.data(), m_vDilate.data())) {
      std::cerr << GetName() << ": Error: Could not set convolution descriptor." << std::endl;
      return false;
    }

    Size clOutDataSize(clInDataSize);

    clOutDataSize[1] = clInWeightsSize[0];

    if (!m_clConvolutionDesc.GetOutputSize(m_clInDataDesc, m_clWeightsDesc, clOutDataSize)) {
      std::cerr << GetName() << ": Error: Could not compute output size." << std::endl;
      return false;
    }

    //std::cout << GetName() << ": Output size = " << clOutDataSize << std::endl;

    if (!m_clOutDataDesc.Set(clOutDataSize)) {
      std::cerr << GetName() << ": Error: Could not set outData descriptor." << std::endl;
      return false;
    }

    p_clOutData->GetData().SetSize(clOutDataSize);

    if (p_clInData->GetGradient().GetSize().Valid() || p_clInWeights->GetGradient().GetSize().Valid() || (p_clInBias != nullptr && p_clInBias->GetGradient().GetSize().Valid())) {
      p_clOutData->GetGradient().SetSize(clOutDataSize);
    }
    else {
      p_clOutData->GetGradient().Clear();
    }

    return true;
  }

  virtual bool Initialize() override {
    bleakGetAndCheckInput(p_clInWeights, "inWeights", false);
    bleakGetAndCheckInput(p_clInData, "inData", false);
    bleakGetAndCheckOutput(p_clOutData, "outData", false);

    std::shared_ptr<EdgeType> p_clInBias = GetInput("inBias");

    bool bUseGPU = false;
    if (!GetProperty("useGPU", bUseGPU) || !bUseGPU) {
      std::cerr << GetName() << ": Error: 'useGPU' must be set to true." << std::endl;
      return false;
    }

    if (p_clInBias != nullptr) {
      // Need to use this algorithm for cudnnConvolutionBiasActivationForward()
      if (!m_clForwardAlgo.Set(m_clInDataDesc, m_clWeightsDesc, m_clConvolutionDesc, m_clOutDataDesc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM)) {
        std::cerr << GetName() << ": Error: Failed to initialize forward algorithm with bias." << std::endl;
        return false;
      }
    }
    else {
      if (!m_clForwardAlgo.Set(m_clInDataDesc, m_clWeightsDesc, m_clConvolutionDesc, m_clOutDataDesc)) {
        std::cerr << GetName() << ": Error: Failed to initialize forward algorithm." << std::endl;
        return false;
      }
    }

    // No need to do this if we're not backpropagating
    if (p_clInData->GetGradient().Valid() && !m_clBackwardDataAlgo.Set(m_clWeightsDesc, m_clOutDataDesc, m_clConvolutionDesc, m_clInDataDesc)) {
      std::cerr << GetName() << ": Error: Failed to initialize backward data algorithm." << std::endl;
      return false;
    }

    if (p_clInWeights->GetGradient().Valid() && !m_clBackwardFilterAlgo.Set(m_clInDataDesc, m_clOutDataDesc, m_clConvolutionDesc, m_clWeightsDesc)) {
      std::cerr << GetName() << ": Error: Failed to initialize backward filter algorithm." << std::endl;
      return false;
    }

    return true;
  }

  virtual void ForwardCPU() override { }
  virtual void BackwardCPU() override { }

  virtual void ForwardGPU() override {
    bleakGetAndCheckInput(p_clInWeights, "inWeights");
    bleakGetAndCheckInput(p_clInData, "inData");
    bleakGetAndCheckOutput(p_clOutData, "outData");

    const RealType * const p_weights = p_clInWeights->GetData().data(GPU);
    const RealType * const p_inData = p_clInData->GetData().data(GPU);
    RealType * const p_outData = p_clOutData->GetData().data_no_sync(GPU);

    std::shared_ptr<EdgeType> p_clInBias = GetInput("inBias");

    if (p_clInBias != nullptr) {
      const RealType * const p_bias = p_clInBias->GetData().data(GPU);

      CudnnConvolutionBiasActivationForward<RealType> clForward(m_clInDataDesc, m_clWeightsDesc, m_clConvolutionDesc, m_clForwardAlgo, m_clOutDataDesc, m_clBiasDesc, m_clActivationDesc, m_clOutDataDesc);

      if (!clForward(RealType(1), p_inData, p_weights, RealType(0), p_outData, p_bias, p_outData))
        std::cerr << GetName() << ": Error: Forward convolution with bias failed." << std::endl;

      return;
    }

    CudnnConvolutionForward<RealType> clForward(m_clInDataDesc, m_clWeightsDesc, m_clConvolutionDesc, m_clForwardAlgo, m_clOutDataDesc);

    if (!clForward(RealType(1), p_inData, p_weights, RealType(0), p_outData))
      std::cerr << GetName() << ": Error: Forward convolution failed." << std::endl;
  }
  
  virtual void BackwardGPU() override {
    bleakGetAndCheckInput(p_clInWeights, "inWeights");
    bleakGetAndCheckInput(p_clInData, "inData");
    bleakGetAndCheckOutput(p_clOutData, "outData");

    std::shared_ptr<EdgeType> p_clInBias = GetInput("inBias");

    if (!p_clOutData->GetGradient().Valid()) // Nothing to do
      return;

    const RealType * const p_weights = p_clInWeights->GetData().data(GPU);
    RealType * const p_weightsGrad = p_clInWeights->GetGradient().data(GPU);
    const RealType * const p_inData = p_clInData->GetData().data(GPU);
    RealType * const p_inDataGrad = p_clInData->GetGradient().data(GPU);
    //const RealType * const p_outData = p_clOutData->GetData().data(GPU);
    const RealType * const p_outDataGrad = p_clOutData->GetGradient().data(GPU);


    if (p_clInBias != nullptr && p_clInBias->GetGradient().Valid()) {
      RealType * const p_biasGrad = p_clInBias->GetGradient().data(GPU);

      CudnnConvolutionBackwardBias<RealType> clBackward(m_clOutDataDesc, m_clBiasDesc);

      if (!clBackward(RealType(1), p_outDataGrad, RealType(1), p_biasGrad)) {
        std::cerr << GetName() << ": Error: Backward convolution bias failed." << std::endl;
        return;
      }
    }

    if (p_weightsGrad != nullptr) {
      CudnnConvolutionBackwardFilter<RealType> clBackward(m_clInDataDesc, m_clOutDataDesc, m_clConvolutionDesc, m_clBackwardFilterAlgo, m_clWeightsDesc);

      if (!clBackward(RealType(1), p_inData, p_outDataGrad, RealType(1), p_weightsGrad)) {
        std::cerr << GetName() << ": Error: Backward convolution filter failed." << std::endl;
        return;
      }
    }

    if (p_inDataGrad != nullptr) {
      CudnnConvolutionBackwardData<RealType> clBackward(m_clWeightsDesc, m_clOutDataDesc, m_clConvolutionDesc, m_clBackwardDataAlgo, m_clInDataDesc);

      if (!clBackward(RealType(1), p_weights, p_outDataGrad, RealType(1), p_inDataGrad)) {
        std::cerr << GetName() << ": Error: Backward convolution data failed." << std::endl;
        return;
      }
    }
  }

protected:
  CudnnConvolution()
  : m_vPadding(GetDimension(), 0), m_vStride(GetDimension(), 1), m_vDilate(GetDimension(), 1) { }

private:
  std::vector<int> m_vPadding;
  std::vector<int> m_vStride;
  std::vector<int> m_vDilate;

  CudnnTensorDescriptor<RealType> m_clInDataDesc;
  CudnnTensorDescriptor<RealType> m_clOutDataDesc;
  CudnnTensorDescriptor<RealType> m_clBiasDesc;
  CudnnFilterDescriptor<RealType> m_clWeightsDesc;
  CudnnConvolutionDescriptor<RealType> m_clConvolutionDesc;
  CudnnActivationDescriptor m_clActivationDesc;

  CudnnConvolutionFwdAlgorithm m_clForwardAlgo;
  CudnnConvolutionBwdDataAlgorithm m_clBackwardDataAlgo;
  CudnnConvolutionBwdFilterAlgorithm m_clBackwardFilterAlgo;

  CudnnConvolution(const CudnnConvolution &) = delete;
  CudnnConvolution(CudnnConvolution &&) = delete;

  CudnnConvolution & operator=(const CudnnConvolution &) = delete;
  CudnnConvolution & operator=(CudnnConvolution &&) = delete;
};

template<typename RealType>
class CudnnConvolution1D : public CudnnConvolution<RealType, 1> {
public:
  typedef CudnnConvolution<RealType, 1> WorkAroundVarArgsType;

  bleakNewVertex(CudnnConvolution1D, WorkAroundVarArgsType);
};

template<typename RealType>
class CudnnConvolution2D : public CudnnConvolution<RealType, 2> {
public:
  typedef CudnnConvolution<RealType, 2> WorkAroundVarArgsType;

  bleakNewVertex(CudnnConvolution2D, WorkAroundVarArgsType);
};

template<typename RealType>
class CudnnConvolution3D : public CudnnConvolution<RealType, 3> {
public:
  typedef CudnnConvolution<RealType, 3> WorkAroundVarArgsType;

  bleakNewVertex(CudnnConvolution3D, WorkAroundVarArgsType);
};

} // end namespace bleak

#endif // !BLEAK_CUDNNCONVOLUTION_H
