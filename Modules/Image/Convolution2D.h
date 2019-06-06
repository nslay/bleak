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

#ifndef BLEAK_CONVOLUTION2D_H
#define BLEAK_CONVOLUTION2D_H

#include "Convolution.h"

namespace bleak {

template<typename RealType>
class Convolution2D : public Convolution<RealType, 2> {
public:
  typedef Convolution<RealType, 2> WorkAroundVarArgsType;

  bleakNewVertex(Convolution2D, WorkAroundVarArgsType);

  bleakForwardVertexTypedefs();

  using SuperType::GetPadding;
  using SuperType::GetStride;
  using SuperType::GetDilate;

  virtual ~Convolution2D() = default;

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

    const int iKernelHeightDilate = iKernelHeight*(1 + vDilate[0]) - vDilate[0]; // Simplified from K + (K-1)*D
    const int iKernelWidthDilate = iKernelWidth*(1 + vDilate[1]) - vDilate[1];

    const int yiStep = 1 + vDilate[0];
    const int xiStep = 1 + vDilate[1];

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

    const int iKernelHeightDilate = iKernelHeight*(1 + vDilate[0]) - vDilate[0]; // Simplified from K + (K-1)*D
    const int iKernelWidthDilate = iKernelWidth*(1 + vDilate[1]) - vDilate[1];

    const int yiStep = 1 + vDilate[0];
    const int xiStep = 1 + vDilate[1];

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
  Convolution2D() = default;

};

} // end namespace bleak

#endif // !BLEAK_CONVOLUTION2D_H
