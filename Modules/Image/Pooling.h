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

#ifndef BLEAK_POOLING_H
#define BLEAK_POOLING_H

#include <algorithm>
#include "Vertex.h"
#include "ImageToMatrix.h"

namespace bleak {

template<typename RealType, unsigned int Dimension>
class Pooling : public Vertex<RealType> {
public:
  bleakNewAbstractVertex(Pooling, Vertex<RealType>,
    bleakAddInput("inData"),
    bleakAddOutput("outData"),
    bleakAddProperty("padding", m_vPadding),
    bleakAddProperty("stride", m_vStride),
    bleakAddProperty("dilate", m_vDilate),
    bleakAddProperty("size", m_vSize));
  
  bleakForwardVertexTypedefs();

  typedef ImageToMatrix<RealType, Dimension> ImageToMatrixType;

  static constexpr unsigned int GetDimension() {
    return Dimension;
  }

  virtual ~Pooling() = default;

  virtual bool SetSizes() override {
    bleakGetAndCheckInput(p_clInData, "inData", false);
    bleakGetAndCheckOutput(p_clOutData, "outData", false);

    if (m_vPadding.size() != GetDimension() || m_vStride.size() != GetDimension() || m_vDilate.size() != GetDimension() || m_vSize.size() != GetDimension()) {
      std::cerr << GetName() << ": Error: padding, stride, dilate, size properties are expected to be " << GetDimension() << "D." << std::endl;
      return false;
    }

    const ArrayType &clInData = p_clInData->GetData();

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

    if (!clInData.GetSize().Valid()) {
      std::cerr << GetName() << ": Error: Invalid dimensions for inData." << std::endl;
      return false;
    }

    // inData: Batch size x Channels x Z x Y x X
    if (clInData.GetSize().GetDimension() != GetDimension() + 2) {
      std::cerr << GetName() << ": Error: Unexpected dimension for inData." << std::endl;
      return false;
    }

    Size clImageSize = clInData.GetSize().SubSize(1);
    clImageSize[0] = 1; // Process one channel at a time

    //m_clImageToMatrix.SetImageSize(p_clInData->GetData().GetSize().data()+1);
    m_clImageToMatrix.SetKernelSize(m_vSize.data());
    m_clImageToMatrix.SetStride(m_vStride.data());
    m_clImageToMatrix.SetPadding(m_vPadding.data());
    m_clImageToMatrix.SetDilate(m_vDilate.data());

    if (!m_clImageToMatrix.Good(clImageSize.data())) {
      std::cerr << GetName() << ": Error: Invalid convolution parameters (image size/kernel size/stride/padding?)." << std::endl;
      return false;
    }

    Size clOutSize(GetDimension()+2);

    clOutSize[0] = clInData.GetSize()[0]; // Batch size
    clOutSize[1] = clInData.GetSize()[1]; // Number of output channels

    const auto tmpSize = m_clImageToMatrix.ComputeOutputSize(clImageSize.data());
    std::copy(tmpSize.begin(), tmpSize.end(), clOutSize.data()+2);

    p_clOutData->GetData().SetSize(clOutSize);
    p_clOutData->GetGradient().SetSize(clOutSize);

    return true;
  }

  virtual bool Initialize() override {
    bleakGetAndCheckInput(p_clInData, "inData", false);

    Size clImageSize = p_clInData->GetData().GetSize().SubSize(1);
    clImageSize[0] = 1; // Process one channel at a time

    if (!m_clImageToMatrix.Good(clImageSize.data()))
      return false;

    m_clImageToMatrix.ComputeMatrixDimensions(m_iRows, m_iCols, clImageSize.data());

    m_vMatrix.resize(m_iRows*m_iCols, RealType(0));
    m_vIndexMatrix.resize(m_vMatrix.size(), 0);

    return true;
  }

protected:
  Pooling() = default;

  ImageToMatrixType m_clImageToMatrix;

  std::vector<RealType> m_vMatrix;
  std::vector<int> m_vIndexMatrix;

  // Row major!
  int m_iRows = 0;
  int m_iCols = 0;

private:
  std::vector<int> m_vPadding;
  std::vector<int> m_vStride;
  std::vector<int> m_vDilate;
  std::vector<int> m_vSize;
};

template<typename RealType, unsigned int Dimension>
class MaxPooling : public Pooling<RealType, Dimension> {
public:
  typedef Pooling<RealType, Dimension> WorkAroundVarArgsType;

  bleakNewAbstractVertex(MaxPooling, WorkAroundVarArgsType);

  bleakForwardVertexTypedefs();

  using SuperType::m_vMatrix;
  using SuperType::m_vIndexMatrix;
  using SuperType::m_clImageToMatrix;
  using SuperType::m_iRows;
  using SuperType::m_iCols;

  virtual ~MaxPooling() = default;

  virtual void Forward() override {
    bleakGetAndCheckInput(p_clInData, "inData");
    bleakGetAndCheckOutput(p_clOutData, "outData");

    const ArrayType &clInData = p_clInData->GetData();
    ArrayType &clOutData = p_clOutData->GetData();

    const RealType * const p_inData = clInData.data();
    RealType * const p_outData = clOutData.data();

    Size clImageSize = clInData.GetSize().SubSize(1);
    clImageSize[0] = 1; // One channel at a time!

    const int iOuterNum = clInData.GetSize()[0];
    const int iNumChannels = clInData.GetSize()[1];
    const int iInChannelSize = clInData.GetSize().Product(2);
    //const int iOutChannelSize = clOutData.GetSize().Product(2); // Same as m_iRows

    for (int i = 0; i < iOuterNum; ++i) {
      for (int j = 0; j < iNumChannels; ++j) {
        m_clImageToMatrix.ExtractMatrix(m_vMatrix.data(), p_inData + (i*iNumChannels + j)*iInChannelSize, clImageSize.data());

        for (int k = 0; k < m_iRows; ++k) {
          const RealType * const p_row = m_vMatrix.data() + k*m_iCols;
          p_outData[(i*iNumChannels + j)*m_iRows + k] = *std::max_element(p_row, p_row + m_iCols);
        }
      }
    }
  }

  virtual void Backward() override {
    bleakGetAndCheckInput(p_clInData, "inData");
    bleakGetAndCheckOutput(p_clOutData, "outData");

    const ArrayType &clInData = p_clInData->GetData();
    ArrayType &clInDataGradient = p_clInData->GetGradient();
    const ArrayType &clOutDataGradient = p_clOutData->GetGradient();

    if (!clInDataGradient.Valid())
      return; // Nothing to do

    const RealType * const p_inData = clInData.data();
    RealType * const p_inDataGradient = clInDataGradient.data();
    const RealType * const p_outDataGradient = clOutDataGradient.data();

    Size clImageSize = clInData.GetSize().SubSize(1);
    clImageSize[0] = 1; // One channel at a time!

    const int iOuterNum = clInData.GetSize()[0];
    const int iNumChannels = clInData.GetSize()[1];
    const int iInChannelSize = clInData.GetSize().Product(2);
    //const int iOutChannelSize = clOutData.GetSize().Product(2); // Same as m_iRows

    m_clImageToMatrix.ExtractIndexMatrix(m_vIndexMatrix.data(), clImageSize.data()); // This never changes

    for (int i = 0; i < iOuterNum; ++i) {
      for (int j = 0; j < iNumChannels; ++j) {
        m_clImageToMatrix.ExtractMatrix(m_vMatrix.data(), p_inData + (i*iNumChannels + j)*iInChannelSize, clImageSize.data());

        for (int k = 0; k < m_iRows; ++k) {
          const RealType * const p_row = m_vMatrix.data() + k*m_iCols;
          const int * const p_indexRow = m_vIndexMatrix.data() + k*m_iCols;

          const int iFeatureIndex = (int)(std::max_element(p_row, p_row + m_iCols) - p_row);
          const int index = p_indexRow[iFeatureIndex];

          if (index >= 0)
            p_inDataGradient[(i*iNumChannels + j)*iInChannelSize + index] += p_outDataGradient[(i*iNumChannels + j)*m_iRows + k];
        }
      }
    }
  }

protected:
  MaxPooling() = default;
};

template<typename RealType>
class MaxPooling1D : public MaxPooling<RealType, 1> {
public:
  typedef MaxPooling<RealType, 1> WorkAroundVarArgsType;
  bleakNewVertex(MaxPooling1D, WorkAroundVarArgsType);
};

template<typename RealType>
class MaxPooling2D : public MaxPooling<RealType, 2> {
public:
  typedef MaxPooling<RealType, 2> WorkAroundVarArgsType;
  bleakNewVertex(MaxPooling2D, WorkAroundVarArgsType);
};

template<typename RealType>
class MaxPooling3D : public MaxPooling<RealType, 3> {
public:
  typedef MaxPooling<RealType, 3> WorkAroundVarArgsType;
  bleakNewVertex(MaxPooling3D, WorkAroundVarArgsType);
};

} // end namespace bleak

#endif // !BLEAK_POOLING_H
