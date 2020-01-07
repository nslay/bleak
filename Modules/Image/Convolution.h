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

#include "Vertex.h"
#include "ImageToMatrix.h"
#include "BlasWrapper.h"

namespace bleak {

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

  typedef ImageToMatrix<RealType, Dimension> ImageToMatrixType;

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

    // inData: Batch size x Channels x Z x Y x X
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

    clOutSize[0] = clInData.GetSize()[0]; // Batch size
    clOutSize[1] = clInWeights.GetSize()[0]; // Number of output channels

    for (int i = 2; i < clOutSize.GetDimension(); ++i) {
      const int iPadding = m_vPadding[i-2];
      const int iStride = m_vStride[i-2];
      const int iDilate = m_vDilate[i-2];
      const int iInputLength = 2*m_vPadding[i-2] + clInData.GetSize()[i];
      const int iKernelLength = clInWeights.GetSize()[i-1]*(1 + iDilate) - iDilate; // Alternate form of K + (K-1)*D

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
    bleakGetAndCheckInput(p_clInData, "inData", false);
    bleakGetAndCheckInput(p_clInWeights, "inWeights", false);
    bleakGetAndCheckOutput(p_clOutData, "outData", false);

    //m_clImageToMatrix.SetImageSize(p_clInData->GetData().GetSize().SubSize(1).data());
    m_clImageToMatrix.SetKernelSize(p_clInWeights->GetData().GetSize().SubSize(1).data());
    m_clImageToMatrix.SetStride(m_vStride.data());
    m_clImageToMatrix.SetPadding(m_vPadding.data());
    m_clImageToMatrix.SetDilate(m_vDilate.data());

    Size clImageSize = p_clInData->GetData().GetSize().SubSize(1);
    clImageSize[0] = 1; // Process 1 channel at a time

    if (!m_clImageToMatrix.Good(clImageSize.data()))
      return false;

    // Row major 
    m_clImageToMatrix.ComputeMatrixDimensions(m_iRows, m_iCols, clImageSize.data());

    m_vMatrix.resize(m_iRows * m_iCols, RealType(0));
    m_vIndexMatrix.resize(m_vMatrix.size(), 0);

    return true;
  }

  virtual void Forward() override {
    bleakGetAndCheckInput(p_clInData, "inData");
    bleakGetAndCheckInput(p_clInWeights, "inWeights");
    bleakGetAndCheckOutput(p_clOutData, "outData");

    std::shared_ptr<EdgeType> p_clInBias = GetInput("inBias");

    const ArrayType &clInData = p_clInData->GetData();
    const ArrayType &clInWeights = p_clInWeights->GetData();
    ArrayType &clOutData = p_clOutData->GetData();

    Size clImageSize = p_clInData->GetData().GetSize().SubSize(1);
    clImageSize[0] = 1; // Process 1 channel at a time

    const RealType * const p_inData = clInData.data();
    const RealType * const p_inWeights = clInWeights.data();
    const RealType * const p_inBias = p_clInBias != nullptr ? p_clInBias->GetData().data() : nullptr;
    RealType * const p_outData = clOutData.data();

    const int iOuterNum = clInData.GetSize()[0];
    const int iInnerNum = clInData.GetSize().Product(1);
    const int iInDataNumChannels = clInData.GetSize()[1];
    const int iOutDataNumChannels = clOutData.GetSize()[1]; // Synonym for weights/bias channels
    const int iOutDataChannelSize = clOutData.GetSize().Product(2);
    const int iOutDataInnerNum = clOutData.GetSize().Product(1);
    const int iInDataChannelSize = clInData.GetSize().Product(2);

    if (p_inBias != nullptr) {
      for (int i = 0; i < iOuterNum; ++i) {
        for (int l = 0; l < iOutDataNumChannels; ++l) {
          std::fill_n(p_outData + (i*iOutDataNumChannels + l)*iOutDataChannelSize, iOutDataChannelSize, p_inBias[l]);
        }
      }
    }
    else {
      clOutData.Fill(RealType());
    }

    // In C++ (row major)
    // Weights: iOutDataNumChannels x m_iCols
    // m_vMatrix: m_iRows x m_iCols
    // C = iOutDataNumChannels x m_iRows

    // In column major that means...
    // Weights: m_iCols x iOutDataNumChannels
    // m_vMatrix: m_iCols x m_iRows
    // C = m_iRows x iOutDataNumChannels

    // So we need m_vMatrix^T * Weights = m_iRows x iOutDataNumChannels

    for (int i = 0; i < iOuterNum; ++i) {
      for (int j = 0; j < iInDataNumChannels; ++j) {
        m_clImageToMatrix.ExtractMatrix(m_vMatrix.data(), p_inData + (i*iInnerNum + j*iInDataChannelSize), clImageSize.data());
        cpu_blas::gemm('T', 'N', m_iRows, iOutDataNumChannels, m_iCols, RealType(1), m_vMatrix.data(), m_iCols, p_inWeights, m_iCols, RealType(1), clOutData.data() + i*iOutDataInnerNum, m_iRows);
      }
    }
  }

  virtual void Backward() override {
    bleakGetAndCheckInput(p_clInData, "inData");
    bleakGetAndCheckInput(p_clInWeights, "inWeights");
    bleakGetAndCheckOutput(p_clOutData, "outData");

    std::shared_ptr<EdgeType> p_clInBias = GetInput("inBias");

    const ArrayType &clInData = p_clInData->GetData();
    ArrayType &clInDataGradient = p_clInData->GetGradient();
    const ArrayType &clInWeights = p_clInWeights->GetData();
    ArrayType &clInWeightsGradient = p_clInWeights->GetGradient();
    const ArrayType &clOutData = p_clOutData->GetData();
    const ArrayType &clOutDataGradient = p_clOutData->GetGradient();

    Size clImageSize = p_clInData->GetData().GetSize().SubSize(1);
    clImageSize[0] = 1; // Process 1 channel at a time

    const RealType * const p_inData = clInData.data();
    RealType * const p_inDataGradient = clInDataGradient.data();
    const RealType * const p_inWeights = clInWeights.data();
    RealType * const p_inWeightsGradient = clInWeightsGradient.data();
    const RealType * const p_inBias = p_clInBias != nullptr ? p_clInBias->GetData().data() : nullptr;
    RealType * const p_inBiasGradient = p_clInBias != nullptr ? p_clInBias->GetGradient().data() : nullptr;
    const RealType * const p_outData = clOutData.data();
    const RealType * const p_outDataGradient = clOutDataGradient.data();

    const int iOuterNum = clInData.GetSize()[0];
    const int iInnerNum = clInData.GetSize().Product(1);
    const int iInDataNumChannels = clInData.GetSize()[1];
    const int iInWeightsChannelSize = clInWeights.GetSize().Product(1);
    const int iOutDataNumChannels = clOutData.GetSize()[1];
    const int iOutDataInnerNum = clOutData.GetSize().Product(1);
    const int iOutDataChannelSize = clOutData.GetSize().Product(2);
    const int iInDataChannelSize = clInData.GetSize().Product(2);

    if (p_inBiasGradient != nullptr) {
      for (int i = 0; i < iOuterNum; ++i) {
        for (int l = 0; l < iOutDataNumChannels; ++l) {
          for (int k = 0; k < iOutDataChannelSize; ++k)
            p_inBiasGradient[l] += p_outDataGradient[(i*iOutDataNumChannels + l)*iOutDataChannelSize + k];
        }
      }
    }

    // In C++ (row major)
    // Weights: iOutDataNumChannels x m_iCols
    // m_vMatrix: m_iRows x m_iCols
    // dC: iOutDataNumChannels x m_iRows

    // In column major that means...
    // Weights: m_iCols x iOutDataNumChannels
    // m_vMatrix: m_iCols x m_iRows
    // dC: m_iRows x iOutDataNumChannels

    // For weights:
    // dW = M*dC = m_iCols x iOutDataNumChannels
    //
    // For data:
    // dM = W*dC^T = m_iCols x m_iRows
    //

    if (p_inWeightsGradient != nullptr) {
      for (int i = 0; i < iOuterNum; ++i) {
        for (int j = 0; j < iInDataNumChannels; ++j) {
          m_clImageToMatrix.ExtractMatrix(m_vMatrix.data(), p_inData + i*iInnerNum + j*iInDataChannelSize, clImageSize.data());

          cpu_blas::gemm('N', 'N', m_iCols, iOutDataNumChannels, m_iRows, RealType(1), m_vMatrix.data(), m_iCols, p_outDataGradient + i*iOutDataInnerNum, m_iRows, RealType(1), p_inWeightsGradient, m_iCols);
        }
      }
    }

    if (p_inDataGradient != nullptr) {
      m_clImageToMatrix.ExtractIndexMatrix(m_vIndexMatrix.data(), clImageSize.data()); // This never changes

      for (int i = 0; i < iOuterNum; ++i) {
        for (int j = 0; j < iInDataNumChannels; ++j) {
          //m_clImageToMatrix.ExtractMatrix(m_vMatrix.data(), p_inData + i*iInnerNum + j*iInDataChannelSize, clImageSize.data());

          cpu_blas::gemm('N', 'T', m_iCols, m_iRows, iOutDataNumChannels, RealType(1), p_inWeights, m_iCols, p_outDataGradient + i*iOutDataInnerNum, m_iRows, RealType(0), m_vMatrix.data(), m_iCols);

          //for (int index : m_vIndexMatrix) {
          for (size_t k = 0; k < m_vIndexMatrix.size(); ++k) {
            const int index = m_vIndexMatrix[k];
            if (index >= 0)
              p_inDataGradient[i*iInnerNum + j*iInDataChannelSize + index] += m_vMatrix[k];
          }
        }
      }
    }
  }

protected:
  Convolution() = default;

private:
  std::vector<int> m_vPadding;
  std::vector<int> m_vStride;
  std::vector<int> m_vDilate;

  ImageToMatrixType m_clImageToMatrix;

  int m_iRows = 0;
  int m_iCols = 0;

  // Row major!!!!
  std::vector<RealType> m_vMatrix; // m_iRows x m_iCols
  std::vector<int> m_vIndexMatrix;
};

template<typename RealType>
class Convolution3D : public Convolution<RealType, 3> {
public:
  typedef Convolution<RealType, 3> WorkAroundVarArgsType;

  bleakNewVertex(Convolution3D, WorkAroundVarArgsType);
};

template<typename RealType>
class Convolution2D : public Convolution<RealType, 2> {
public:
  typedef Convolution<RealType, 2> WorkAroundVarArgsType;

  bleakNewVertex(Convolution2D, WorkAroundVarArgsType);
};

template<typename RealType>
class Convolution1D : public Convolution<RealType, 1> {
public:
  typedef Convolution<RealType, 1> WorkAroundVarArgsType;

  bleakNewVertex(Convolution1D, WorkAroundVarArgsType);
};

} // end namespace bleak

#endif // !BLEAK_CONVOLUTION_H
