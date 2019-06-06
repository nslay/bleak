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

#ifndef BLEAK_CONVOLUTION_H
#define BLEAK_CONVOLUTION_H

#include <algorithm>
#include "Vertex.h"

namespace bleak {

// Base class for Convolutions
template<typename RealType, unsigned int Dimension>
class Convolution : public Vertex<RealType> {
public:
  bleakNewAbstractVertex(Convolution, Vertex<RealType>,
    bleakAddInput("inData"),
    bleakAddInput("inWeights"),
    bleakAddInput("inBias"),
    bleakAddOutput("outData"),
    bleakAddProperty("padding", m_vPadding),
    bleakAddProperty("stride", m_vStride),
    bleakAddProperty("dilate", m_vDilate));

  bleakForwardVertexTypedefs();

  static constexpr unsigned int GetDimension() {
    return Dimension;
  }

  virtual ~Convolution() = default;

  virtual bool SetSizes() override {
    bleakGetAndCheckInput(p_clInData, "inData", false);
    bleakGetAndCheckInput(p_clInWeights, "inWeights", false);
    bleakGetAndCheckOutput(p_clOutData, "outData", false);

    std::shared_ptr<EdgeType> p_clInBias = GetInput("inBias");

    if (m_vPadding.size() != GetDimension() || m_vStride.size() != GetDimension() || m_vDilate.size() != GetDimension()) {
      std::cerr << GetName() << ": Error: padding, stride and dilate properties are expected to be " << GetDimension() << "D." << std::endl;
      return false;
    }

    const ArrayType &clInData = p_clInData->GetData();
    const ArrayType &clInWeights = p_clInWeights->GetData();

    // TODO: Maybe this is useful?
    if (*std::min_element(m_vPadding.begin(), m_vPadding.end()) < 0) {
      std::cerr << GetName() << ": Error: Padding expected to be non-negative." << std::endl;
      return false;
    }

    if (*std::min_element(m_vStride.begin(), m_vStride.end()) <= 0) {
      std::cerr << GetName() << ": Error: Strides are expected to be positive." << std::endl;
      return false;
    }

    if (*std::min_element(m_vDilate.begin(), m_vDilate.end()) < 0) {
      std::cerr << GetName() << ": Error: Dilate expected to be non-negative." << std::endl;
      return false;
    }

    if (!clInData.GetSize().Valid() || !clInWeights.GetSize().Valid()) {
      std::cerr << GetName() << ": Error: Invalid dimensions for inData and/or inWeights." << std::endl;
      return false;
    }

    if (clInWeights.GetSize().GetDimension() != GetDimension()+1 || clInData.GetSize().GetDimension() != GetDimension() + 2) {
      std::cerr << GetName() << ": Error: Unexpected dimension for inData and/or inWeights." << std::endl;
      return false;
    }

    if (p_clInBias == nullptr) {
      std::cout << GetName() << ": Info: inBias is not connected. Bias will be assumed to be fixed as 0." << std::endl;
    }
    else {
      const ArrayType &clInBias = p_clInBias->GetData();

      if (!clInBias.GetSize().Valid()) {
        std::cerr << GetName() << ": Error: Invalid dimension for inBias." << std::endl;
        return false;
      }

      if (clInBias.GetSize().GetDimension() != 1) {
        std::cerr << GetName() << ": Error: inBias is expected to be 1D." << std::endl;
        return false;
      }

      if (clInBias.GetSize()[0] != clInWeights.GetSize()[0]) {
        std::cerr << GetName() << ": Error: Dimension mismatch between inBias and inWeights." << std::endl;
        return false;
      }
    }

    Size clOutSize(GetDimension()+2);

    clOutSize[0] = clInData.GetSize()[0];
    clOutSize[1] = clInWeights.GetSize()[0];

    for (int i = 2; i < clOutSize.GetDimension(); ++i) {
      const int iPadding = m_vPadding[i-2];
      const int iStride = m_vStride[i-2];
      const int iDilate = m_vDilate[i-2];
      const int iInputLength = 2*m_vPadding[i-2] + clInData.GetSize()[i];
      const int iKernelLength = clInWeights.GetSize()[i-1]*(1 + iDilate) - iDilate; // Simplified from K + (K-1)*D

      if (iInputLength <= iKernelLength) {
        std::cerr << GetName() << ": Error: inWeights dimensions " << clInWeights.GetSize().SubSize(1) << 
          " exceed the dimensions of inData " << clInData.GetSize().SubSize(2) << 
          ". Check inWeights or dilate property." << std::endl;

        return false;
      }

      clOutSize[i] = (iInputLength - iKernelLength) / iStride + 1;
    }

    p_clOutData->GetData().SetSize(clOutSize);
    p_clOutData->GetGradient().SetSize(clOutSize);

    return true;
  }

  virtual bool Initialize() override {
    return true; // Nothing to do
  }

protected:
  Convolution()
  : m_vPadding(GetDimension(), 0), m_vStride(GetDimension(), 1), m_vDilate(GetDimension(), 0) { }

  const std::vector<int> & GetPadding() const {
    return m_vPadding;
  }

  const std::vector<int> & GetStride() const {
    return m_vStride;
  }

  const std::vector<int> & GetDilate() const {
    return m_vDilate;
  }

private:
  std::vector<int> m_vPadding;
  std::vector<int> m_vStride;
  std::vector<int> m_vDilate;
};

template<typename RealType>
class Convolution<RealType, 0> { };

} // end namespace bleak

#endif // !BLEAK_CONVOLUTION_H
