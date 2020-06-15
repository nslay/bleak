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

#ifndef BLEAK_CONVOLUTIONNAIVE_H
#define BLEAK_CONVOLUTIONNAIVE_H

#include <algorithm>
#include "Vertex.h"

// NOTE: This is kept for reproducibility with old experiments. You should not use this.

namespace bleak {

// Base class for Convolutions
template<typename RealType, unsigned int Dimension>
class ConvolutionNaive : public Vertex<RealType> {
public:
  bleakNewAbstractVertex(ConvolutionNaive, Vertex<RealType>,
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

  virtual ~ConvolutionNaive() = default;

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

    if (*std::min_element(m_vDilate.begin(), m_vDilate.end()) <= 0) {
      std::cerr << GetName() << ": Error: Dilate expected to be positive." << std::endl;
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
      const int iKernelLength = clInWeights.GetSize()[i-1]*iDilate - (iDilate - 1); // Simplified from K + (K-1)*(D-1)

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
  ConvolutionNaive()
  : m_vPadding(GetDimension(), 0), m_vStride(GetDimension(), 1), m_vDilate(GetDimension(), 1) { }

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
class ConvolutionNaive<RealType, 0> { };

template<typename RealType>
class ConvolutionNaive2D : public ConvolutionNaive<RealType, 2> {
public:
  typedef ConvolutionNaive<RealType, 2> WorkAroundVarArgsType;

  bleakNewVertex(ConvolutionNaive2D, WorkAroundVarArgsType);

  bleakForwardVertexTypedefs();

  using SuperType::GetPadding;
  using SuperType::GetStride;
  using SuperType::GetDilate;

  virtual ~ConvolutionNaive2D() = default;

  virtual void Forward() override {
    bleakGetAndCheckInput(p_clInData, "inData");
    bleakGetAndCheckInput(p_clInWeights, "inWeights");
    bleakGetAndCheckOutput(p_clOutData, "outData");

    std::shared_ptr<EdgeType> p_clInBias = GetInput("inBias");

    const ArrayType &clInData = p_clInData->GetData();
    const ArrayType &clInWeights = p_clInWeights->GetData();
    ArrayType &clOutData = p_clOutData->GetData();

    const RealType * const p_inData = clInData.data();
    const RealType * const p_inWeights = clInWeights.data();
    const RealType * const p_inBias = p_clInBias != nullptr ? p_clInBias->GetData().data() : nullptr;
    RealType * const p_outData = clOutData.data();

    const int iOuterNum = clInData.GetSize()[0];
    const int iInNumChannels = clInData.GetSize()[1];
    const int iOutNumChannels = clInWeights.GetSize()[0];
    const int iInputHeight = clInData.GetSize()[2];
    const int iInputWidth = clInData.GetSize()[3];

    const int iOutputHeight = clOutData.GetSize()[2];
    const int iOutputWidth = clOutData.GetSize()[3];

    const int iKernelHeight = clInWeights.GetSize()[1];
    const int iKernelWidth = clInWeights.GetSize()[2];

    const std::vector<int> &vPadding = GetPadding();
    const std::vector<int> &vStride = GetStride();
    const std::vector<int> &vDilate = GetDilate();

    const int iKernelHeightDilate = iKernelHeight*vDilate[0] - (vDilate[0]-1); // Simplified from K + (K-1)*(D-1)
    const int iKernelWidthDilate = iKernelWidth*vDilate[1] - (vDilate[1]-1);

    const int yiStep = vDilate[0];
    const int xiStep = vDilate[1];

    if (p_inBias == nullptr) {
      clOutData.Fill(RealType());
    }
    else {
      const int iInnerNum = iOutputHeight * iOutputWidth;

      for (int i = 0; i < iOuterNum; ++i) {
        for (int j = 0; j < iOutNumChannels; ++j) {
          for (int k = 0; k < iInnerNum; ++k) {
            p_outData[(i*iOutNumChannels + j)*iInnerNum + k] = p_inBias[j];
          }
        }
      }
    }

    for (int i = 0; i < iOuterNum; ++i) {
      for (int j = 0; j < iOutNumChannels; ++j) {
        const int lOff = j*iKernelHeight*iKernelWidth;

        for (int yo = 0; yo < iOutputHeight; ++yo) {
          const int yiOff = yo*vStride[0] - vPadding[0];
          int yiBegin = yiOff + iKernelHeightDilate/2;
          int yiEnd = yiBegin + iKernelHeightDilate;

          const int yiBeginRef = yiBegin; // Needed to deduce kernel index

          yiBegin = std::max(0, yiBegin);
          yiEnd = std::min(iInputHeight, yiEnd);

          const int yiCorrectionSteps = (yiBegin - yiBeginRef + yiStep-1)/yiStep;

          // Align yiBegin onto the dilation grid
          yiBegin = yiBeginRef + yiCorrectionSteps*yiStep;

          for (int xo = 0; xo < iOutputWidth; ++xo) {
            const int xiOff = xo*vStride[1] - vPadding[1];
            int xiBegin = xiOff + iKernelWidthDilate/2;
            int xiEnd = xiBegin + iKernelWidthDilate;

            const int xiBeginRef = xiBegin; // Needed to deduce kernel index

            xiBegin = std::max(0, xiBegin);
            xiEnd = std::min(iInputWidth, xiEnd);

            const int xiCorrectionSteps = (xiBegin - xiBeginRef + xiStep-1)/xiStep;

            // Align xiBegin onto the dilation grid
            xiBegin = xiBeginRef + xiCorrectionSteps*xiStep;

            //const int lBegin = lOff + iKernelWidth*(yiBegin - yiBeginRef)/yiStep + (xiBegin - xiBeginRef)/xiStep;
            const int lBegin = lOff + iKernelWidth*yiCorrectionSteps + xiCorrectionSteps;

            for (int k = 0; k < iInNumChannels; ++k) {
              int l = lBegin;

              for (int yi = yiBegin; yi < yiEnd; yi += yiStep) {
                for (int xi = xiBegin; xi < xiEnd; ++l, xi += xiStep) {
                  p_outData[((i*iOutNumChannels + j)*iOutputHeight + yo)*iOutputWidth + xo] += p_inWeights[l] * p_inData[((i*iInNumChannels + k)*iInputHeight + yi)*iInputWidth + xi];
                }
              }
            }
          }
        }
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

    const RealType * const p_inData = clInData.data();
    RealType * const p_inDataGradient = clInDataGradient.data();
    const RealType * const p_inWeights = clInWeights.data();
    RealType * const p_inWeightsGradient = clInWeightsGradient.data();
    const RealType * const p_inBias = p_clInBias != nullptr ? p_clInBias->GetData().data() : nullptr;
    RealType * const p_inBiasGradient = p_clInBias != nullptr ? p_clInBias->GetGradient().data() : nullptr;
    const RealType * const p_outData = clOutData.data();
    const RealType * const p_outDataGradient = clOutDataGradient.data();

    const int iOuterNum = clInData.GetSize()[0];
    const int iInNumChannels = clInData.GetSize()[1];
    const int iOutNumChannels = clInWeights.GetSize()[0];
    const int iInputHeight = clInData.GetSize()[2];
    const int iInputWidth = clInData.GetSize()[3];

    const int iOutputHeight = clOutData.GetSize()[2];
    const int iOutputWidth = clOutData.GetSize()[3];

    const int iKernelHeight = clInWeights.GetSize()[1];
    const int iKernelWidth = clInWeights.GetSize()[2];

    const std::vector<int> &vPadding = GetPadding();
    const std::vector<int> &vStride = GetStride();
    const std::vector<int> &vDilate = GetDilate();

    const int iKernelHeightDilate = iKernelHeight*vDilate[0] - (vDilate[0]-1); // Simplified from K + (K-1)*(D-1)
    const int iKernelWidthDilate = iKernelWidth*vDilate[1] - (vDilate[1]-1);

    const int yiStep = vDilate[0];
    const int xiStep = vDilate[1];

    if (p_inBiasGradient != nullptr) {
      // Derivative with respect to bias

      const int iInnerNum = iOutputHeight * iOutputWidth;

      for (int i = 0; i < iOuterNum; ++i) {
        for (int j = 0; j < iOutNumChannels; ++j) {
          for (int k = 0; k < iInnerNum; ++k) {
            p_inBiasGradient[j] += p_outDataGradient[(i*iOutNumChannels + j)*iInnerNum + k];
          }
        }
      }
    }

    if (p_inWeightsGradient != nullptr) {
      // Derivative with respect to weights

      for (int i = 0; i < iOuterNum; ++i) {
        for (int j = 0; j < iOutNumChannels; ++j) {
          const int lOff = j*iKernelHeight*iKernelWidth;

          for (int yo = 0; yo < iOutputHeight; ++yo) {
            const int yiOff = yo*vStride[0] - vPadding[0];
            int yiBegin = yiOff + iKernelHeightDilate/2;
            int yiEnd = yiBegin + iKernelHeightDilate;

            const int yiBeginRef = yiBegin; // Needed to deduce kernel index

            yiBegin = std::max(0, yiBegin);
            yiEnd = std::min(iInputHeight, yiEnd);

            const int yiCorrectionSteps = (yiBegin - yiBeginRef + yiStep-1)/yiStep;

            // Align yiBegin onto the dilation grid
            yiBegin = yiBeginRef + yiCorrectionSteps*yiStep;

            for (int xo = 0; xo < iOutputWidth; ++xo) {
              const int xiOff = xo*vStride[1] - vPadding[1];
              int xiBegin = xiOff + iKernelWidthDilate/2;
              int xiEnd = xiBegin + iKernelWidthDilate;

              const int xiBeginRef = xiBegin; // Needed to deduce kernel index

              xiBegin = std::max(0, xiBegin);
              xiEnd = std::min(iInputWidth, xiEnd);

              const int xiCorrectionSteps = (xiBegin - xiBeginRef + xiStep-1)/xiStep;

              // Align xiBegin onto the dilation grid
              xiBegin = xiBeginRef + xiCorrectionSteps*xiStep;

              //const int lBegin = lOff + iKernelWidth*(yiBegin - yiBeginRef)/yiStep + (xiBegin - xiBeginRef)/xiStep;
              const int lBegin = lOff + iKernelWidth*yiCorrectionSteps + xiCorrectionSteps;

              const RealType &scale = p_outDataGradient[((i*iOutNumChannels + j)*iOutputHeight + yo)*iOutputWidth + xo];

              for (int k = 0; k < iInNumChannels; ++k) {
                int l = lBegin;

                for (int yi = yiBegin; yi < yiEnd; yi += yiStep) {
                  for (int xi = xiBegin; xi < xiEnd; ++l, xi += xiStep) {
                    p_inWeightsGradient[l] += p_inData[((i*iInNumChannels + k)*iInputHeight + yi)*iInputWidth + xi] * scale;
                  }
                }
              }
            }
          }
        }
      }
    }

    if (p_inDataGradient != nullptr) {
      // Derivative with respect to data

      for (int i = 0; i < iOuterNum; ++i) {
        for (int j = 0; j < iOutNumChannels; ++j) {
          const int lOff = j*iKernelHeight*iKernelWidth;

          for (int yo = 0; yo < iOutputHeight; ++yo) {
            const int yiOff = yo*vStride[0] - vPadding[0];
            int yiBegin = yiOff + iKernelHeightDilate/2;
            int yiEnd = yiBegin + iKernelHeightDilate;

            const int yiBeginRef = yiBegin; // Needed to deduce kernel index

            yiBegin = std::max(0, yiBegin);
            yiEnd = std::min(iInputHeight, yiEnd);

            const int yiCorrectionSteps = (yiBegin - yiBeginRef + yiStep-1)/yiStep;

            // Align yiBegin onto the dilation grid
            yiBegin = yiBeginRef + yiCorrectionSteps*yiStep;

            for (int xo = 0; xo < iOutputWidth; ++xo) {
              const int xiOff = xo*vStride[1] - vPadding[1];
              int xiBegin = xiOff + iKernelWidthDilate/2;
              int xiEnd = xiBegin + iKernelWidthDilate;

              const int xiBeginRef = xiBegin; // Needed to deduce kernel index

              xiBegin = std::max(0, xiBegin);
              xiEnd = std::min(iInputWidth, xiEnd);

              const int xiCorrectionSteps = (xiBegin - xiBeginRef + xiStep-1)/xiStep;

              // Align xiBegin onto the dilation grid
              xiBegin = xiBeginRef + xiCorrectionSteps*xiStep;

              //const int lBegin = lOff + iKernelWidth*(yiBegin - yiBeginRef)/yiStep + (xiBegin - xiBeginRef)/xiStep;
              const int lBegin = lOff + iKernelWidth*yiCorrectionSteps + xiCorrectionSteps;

              const RealType &scale = p_outDataGradient[((i*iOutNumChannels + j)*iOutputHeight + yo)*iOutputWidth + xo];

              for (int k = 0; k < iInNumChannels; ++k) {
                int l = lBegin;

                for (int yi = yiBegin; yi < yiEnd; yi += yiStep) {
                  for (int xi = xiBegin; xi < xiEnd; ++l, xi += xiStep) {
                    p_inDataGradient[((i*iInNumChannels + k)*iInputHeight + yi)*iInputWidth + xi] += p_inWeights[l] * scale;
                  }
                }
              }
            }
          }
        }
      }
    }
  }

protected:
  ConvolutionNaive2D() = default;

};

} // end namespace bleak

#endif // !BLEAK_CONVOLUTIONNAIVE_H
