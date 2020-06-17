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

#ifndef BLEAK_CUDNNPOOLING_H
#define BLEAK_CUDNNPOOLING_H

#include <iostream>
#include <algorithm>
#include <vector>
#include "Vertex.h"
#include "CudnnCommon.h"

namespace bleak {

template<typename RealType, unsigned int Dimension>
class CudnnPooling : public Vertex<RealType> {
public:
  bleakNewAbstractVertex(CudnnPooling, Vertex<RealType>,
    bleakAddProperty("size", m_vSize),
    bleakAddProperty("padding", m_vPadding),
    bleakAddProperty("stride", m_vStride),
    // bleakAddProperty("dilate", m_vDilate), // Not supported by cudnn
    bleakAddInput("inData"),
    bleakAddOutput("outData"));

  bleakForwardVertexTypedefs();

  constexpr static unsigned int GetDimension() { return Dimension; }

  virtual ~CudnnPooling() = default;

  // This will only do sanity checks...
  virtual bool SetSizes() override {
    bleakGetAndCheckInput(p_clInData, "inData", false);
    bleakGetAndCheckOutput(p_clOutData, "outData", false);

    if (m_vPadding.size() != GetDimension() || m_vStride.size() != GetDimension() /*|| m_vDilate.size() != GetDimension()*/) {
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

    //if (*std::min_element(m_vDilate.begin(), m_vDilate.end()) <= 0) {
    //  std::cerr << GetName() << ": Error: 'dilate' expected to be positive." << std::endl;
    //  return false;
    //}

    const Size clInDataSize = p_clInData->GetData().GetSize();

    if (!clInDataSize.Valid()) {
      std::cerr << GetName() << ": Error: Invalid 'inData' (" << clInDataSize << ")." << std::endl;
      return false;
    }

    if (clInDataSize.GetDimension() != 2 + GetDimension()) {
      std::cerr << GetName() << ": Error: Expected 'inData' to be BatchSize x InputChannels x ImageDimensions (expected " << 2 + GetDimension() << " dimensions but got " << clInDataSize.GetDimension() << " dimensions)." << std::endl;
      return false;
    }

    return true;
  }

  virtual bool Initialize() override {
    bleakGetAndCheckInput(p_clInData, "inData", false);
    bleakGetAndCheckOutput(p_clOutData, "outData", false);

    bool bUseGPU = false;
    if (!GetProperty("useGPU", bUseGPU) || !bUseGPU) {
      std::cerr << GetName() << ": Error: 'useGPU' must be set to true." << std::endl;
      return false;
    }

    return true;
  }

  virtual void ForwardCPU() override { }
  virtual void BackwardCPU() override { }

  virtual void ForwardGPU() override {
    bleakGetAndCheckInput(p_clInData, "inData");
    bleakGetAndCheckOutput(p_clOutData, "outData");

    const RealType * const p_inData = p_clInData->GetData().data(GPU);
    RealType * const p_outData = p_clOutData->GetData().data_no_sync(GPU);

    CudnnPoolingForward<RealType> clForward(m_clPoolingDesc, m_clInDataDesc, m_clOutDataDesc);

    if (!clForward(RealType(1), p_inData, RealType(0), p_outData))
      std::cerr << GetName() << ": Error: Forward pooling failed." << std::endl;
  }

  virtual void BackwardGPU() override {
    bleakGetAndCheckInput(p_clInData, "inData");
    bleakGetAndCheckOutput(p_clOutData, "outData");

    if (!p_clInData->GetGradient().Valid()) // Nothing to do!
      return;

    const RealType * const p_inData = p_clInData->GetData().data(GPU);
    RealType * const p_inDataGrad = p_clInData->GetGradient().data(GPU);
    const RealType * const p_outData = p_clOutData->GetData().data(GPU);
    const RealType * const p_outDataGrad = p_clOutData->GetGradient().data(GPU);

    CudnnPoolingBackward<RealType> clBackward(m_clPoolingDesc, m_clOutDataDesc, m_clOutDataDesc, m_clInDataDesc, m_clInDataDesc);

    if (!clBackward(RealType(1), p_outData, p_outDataGrad, p_inData, RealType(1), p_inDataGrad))
      std::cerr << GetName() << ": Error: Backward pooling failed." << std::endl;
  }

protected:
  std::vector<int> m_vSize;
  std::vector<int> m_vPadding;
  std::vector<int> m_vStride;
  //std::vector<int> m_vDilate;

  // Derived classes must set these up
  CudnnPoolingDescriptor m_clPoolingDesc;
  CudnnTensorDescriptor<RealType> m_clInDataDesc;
  CudnnTensorDescriptor<RealType> m_clOutDataDesc;

  CudnnPooling()
  : m_vPadding(GetDimension(), 0), m_vStride(GetDimension(), 1) /*, m_vDilate(GetDimension(), 1) */ { }
};

template<typename RealType, unsigned int Dimension>
class CudnnMaxPooling : public CudnnPooling<RealType, Dimension> {
public:
  typedef CudnnPooling<RealType, Dimension> WorkAroundVarArgsType;

  bleakNewAbstractVertex(CudnnMaxPooling, WorkAroundVarArgsType,
    bleakAddProperty("deterministic", m_bDeterministic));

  bleakForwardVertexTypedefs();

  using SuperType::GetDimension;
  using SuperType::m_vSize;
  using SuperType::m_vPadding;
  using SuperType::m_vStride;

  using SuperType::m_clPoolingDesc;
  using SuperType::m_clInDataDesc;
  using SuperType::m_clOutDataDesc;

  virtual ~CudnnMaxPooling() = default;

  virtual bool SetSizes() override {
    // Run sanity chcecks!
    if (!SuperType::SetSizes())
      return false;

    bleakGetAndCheckInput(p_clInData, "inData", false);
    bleakGetAndCheckOutput(p_clOutData, "outData", false);

    if (!m_clInDataDesc.Set(p_clInData->GetData().GetSize())) {
      std::cerr << GetName() << ": Error: Failed to set inData descriptor." << std::endl;
      return false;
    }

    const cudnnPoolingMode_t mode = m_bDeterministic ? CUDNN_POOLING_MAX_DETERMINISTIC : CUDNN_POOLING_MAX;

    if (!m_clPoolingDesc.Set(mode, (int)GetDimension(), m_vSize.data(), m_vPadding.data(), m_vStride.data())) {
      std::cerr << GetName() << ": Error: Failed to set pooling descriptor." << std::endl;
      return false;
    }

    Size clOutDataSize(p_clInData->GetData().GetSize());

    if (!m_clPoolingDesc.GetOutputSize(m_clInDataDesc, clOutDataSize)) {
      std::cerr << GetName() << ": Error: Could not compute output size." << std::endl;
      return false;
    }

    //std::cout << GetName() << ": Output size = " << clOutDataSize << std::endl;

    if (!m_clOutDataDesc.Set(clOutDataSize)) {
      std::cerr << GetName() << ": Error: Failed to set outData descriptor." << std::endl;
      return false;
    }

    p_clOutData->GetData().SetSize(clOutDataSize);

    if (p_clInData->GetGradient().GetSize().Valid())
      p_clOutData->GetGradient().SetSize(clOutDataSize);
    else
      p_clOutData->GetGradient().Clear();

    return true;
  }

protected:
  CudnnMaxPooling() = default;

private:
  bool m_bDeterministic = true;
};

template<typename RealType>
class CudnnMaxPooling1D : public CudnnMaxPooling<RealType, 1> {
public:
  typedef CudnnMaxPooling<RealType, 1> WorkAroundVarArgsType;
  bleakNewVertex(CudnnMaxPooling1D, WorkAroundVarArgsType);
};

template<typename RealType>
class CudnnMaxPooling2D : public CudnnMaxPooling<RealType, 2> {
public:
  typedef CudnnMaxPooling<RealType, 2> WorkAroundVarArgsType;
  bleakNewVertex(CudnnMaxPooling2D, WorkAroundVarArgsType);
};

template<typename RealType>
class CudnnMaxPooling3D : public CudnnMaxPooling<RealType, 3> {
public:
  typedef CudnnMaxPooling<RealType, 3> WorkAroundVarArgsType;
  bleakNewVertex(CudnnMaxPooling3D, WorkAroundVarArgsType);
};

template<typename RealType, unsigned int Dimension>
class CudnnMeanPooling : public CudnnPooling<RealType, Dimension> {
public:
  typedef CudnnPooling<RealType, Dimension> WorkAroundVarArgsType;

  bleakNewAbstractVertex(CudnnMeanPooling, WorkAroundVarArgsType,
    bleakAddProperty("includePadding", m_bIncludePadding));

  bleakForwardVertexTypedefs();

  using SuperType::GetDimension;
  using SuperType::m_vSize;
  using SuperType::m_vPadding;
  using SuperType::m_vStride;

  using SuperType::m_clPoolingDesc;
  using SuperType::m_clInDataDesc;
  using SuperType::m_clOutDataDesc;

  virtual ~CudnnMeanPooling() = default;

  virtual bool SetSizes() override {
    // Run sanity chcecks!
    if (!SuperType::SetSizes())
      return false;

    bleakGetAndCheckInput(p_clInData, "inData", false);
    bleakGetAndCheckOutput(p_clOutData, "outData", false);

    if (!m_clInDataDesc.Set(p_clInData->GetData().GetSize())) {
      std::cerr << GetName() << ": Error: Failed to set inData descriptor." << std::endl;
      return false;
    }

    const cudnnPoolingMode_t mode = m_bIncludePadding ? CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;

    if (!m_clPoolingDesc.Set(mode, (int)GetDimension(), m_vSize.data(), m_vPadding.data(), m_vStride.data())) {
      std::cerr << GetName() << ": Error: Failed to set pooling descriptor." << std::endl;
      return false;
    }

    Size clOutDataSize(p_clInData->GetData().GetSize());

    if (!m_clPoolingDesc.GetOutputSize(m_clInDataDesc, clOutDataSize)) {
      std::cerr << GetName() << ": Error: Could not compute output size." << std::endl;
      return false;
    }

    //std::cout << GetName() << ": Output size = " << clOutDataSize << std::endl;

    if (!m_clOutDataDesc.Set(clOutDataSize)) {
      std::cerr << GetName() << ": Error: Failed to set outData descriptor." << std::endl;
      return false;
    }

    p_clOutData->GetData().SetSize(clOutDataSize);

    if (p_clInData->GetGradient().GetSize().Valid())
      p_clOutData->GetGradient().SetSize(clOutDataSize);
    else
      p_clOutData->GetGradient().Clear();

    return true;
  }

protected:
  CudnnMeanPooling() = default;

private:
  bool m_bIncludePadding = true; // Average padding values too?
};

template<typename RealType>
class CudnnMeanPooling1D : public CudnnMeanPooling<RealType, 1> {
public:
  typedef CudnnMeanPooling<RealType, 1> WorkAroundVarArgsType;
  bleakNewVertex(CudnnMeanPooling1D, WorkAroundVarArgsType);
};

template<typename RealType>
class CudnnMeanPooling2D : public CudnnMeanPooling<RealType, 2> {
public:
  typedef CudnnMeanPooling<RealType, 2> WorkAroundVarArgsType;
  bleakNewVertex(CudnnMeanPooling2D, WorkAroundVarArgsType);
};

template<typename RealType>
class CudnnMeanPooling3D : public CudnnMeanPooling<RealType, 3> {
public:
  typedef CudnnMeanPooling<RealType, 3> WorkAroundVarArgsType;
  bleakNewVertex(CudnnMeanPooling3D, WorkAroundVarArgsType);
};

} // end namespace bleak

#endif // !BLEAK_CUDNNPOOLING_H
