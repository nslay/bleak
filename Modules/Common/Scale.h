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

#ifndef BLEAK_SCALE_H
#define BLEAK_SCALE_H

#include "Vertex.h"

namespace bleak {

template<typename RealType>
class Scale : public Vertex<RealType> {
public:
  bleakNewVertex(Scale, Vertex<RealType>,
    bleakAddInput("inWeights"),
    bleakAddInput("inBias"),
    bleakAddInput("inData"),
    bleakAddOutput("outData"));

  bleakForwardVertexTypedefs();

  virtual ~Scale() = default;

  virtual bool SetSizes() override {
    bleakGetAndCheckInput(p_clInData, "inData", false);
    bleakGetAndCheckInput(p_clInWeights, "inWeights", false);
    bleakGetAndCheckOutput(p_clOutData, "outData", false);

    std::shared_ptr<EdgeType> p_clInBias = GetInput("inBias"); // If this is not provided, the bias will be treated as though it were fixed to be 0

    const ArrayType &clInData = p_clInData->GetData();
    const ArrayType &clInWeights = p_clInWeights->GetData();

    if (!clInData.GetSize().Valid() || !clInWeights.GetSize().Valid()) {
      std::cerr << GetName() << ": Error: Invalid dimensions for inData and/or inWeights." << std::endl;
      return false;
    }

    if (clInData.GetSize().GetDimension() < 2) {
      std::cerr << GetName() << ": Error: inData is expected to be 2D or higher." << std::endl;
      return false;
    }

    if (clInWeights.GetSize().GetDimension() != 1) {
      std::cerr << GetName() << ": Error: inWeights is expected to be 1D." << std::endl;
      return false;
    }

    if (clInData.GetSize()[1] != clInWeights.GetSize()[0]) {
      std::cerr << GetName() << ": Error: inData and inWeights are expected to have the same number of channels." << std::endl;
      return false;
    }

    if (p_clInBias == nullptr) {
      std::cout << GetName() << ": Info: inBias is not connected. Bias will be assumed to be fixed as 0." << std::endl;
    }
    else {
      const ArrayType &clInBias = p_clInBias->GetData();
      if (clInBias.GetSize().GetDimension() != 1) {
        std::cerr << GetName() << ": Error: inBias is expected to be 1D." << std::endl;
        return false;
      }

      if (clInBias.GetSize()[0] != clInWeights.GetSize()[0]) {
        std::cerr << GetName() << ": Error: inWeights and inBias are expected to have the same number of outputs." << std::endl;
        return false;
      }
    }

    p_clOutData->GetData().SetSize(clInData.GetSize());
    p_clOutData->GetGradient().SetSize(clInData.GetSize());

    return true;
  }

  virtual bool Initialize() override {
    return true; // Nothing to do
  }

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
    const int iNumChannels = clInData.GetSize()[1];
    const int iInnerNum = clInData.GetSize().Product(2);

    if (p_inBias == nullptr) {
      clOutData.Fill(RealType());
    }
    else {
      for (int i = 0; i < iOuterNum; ++i) {
        for (int k = 0; k < iInnerNum; ++k) {
          for (int j = 0; j < iNumChannels; ++j) {
            p_outData[(i*iNumChannels + j)*iInnerNum + k] = p_inBias[j];
          }
        }
      }
    }

    for (int i = 0; i < iOuterNum; ++i) {
      for (int k = 0; k < iInnerNum; ++k) {
        for (int j = 0; j < iNumChannels; ++j) {
          p_outData[(i*iNumChannels + j)*iInnerNum + k] += p_inWeights[j] * p_inData[(i*iNumChannels + j)*iInnerNum + k];
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

    const ArrayType &clOutDataGradient = p_clOutData->GetGradient();

    const RealType * const p_inData = clInData.data();
    const RealType * const p_inWeights = clInWeights.data();
    const RealType * const p_inBias = p_clInBias != nullptr ? p_clInBias->GetData().data() : nullptr;
    const RealType * const p_outDataGradient = clOutDataGradient.data();

    RealType * const p_inDataGradient = clInDataGradient.Valid() ? clInDataGradient.data() : nullptr;
    RealType * const p_inWeightsGradient = clInWeightsGradient.Valid() ? clInWeightsGradient.data() : nullptr;
    RealType * const p_inBiasGradient = p_clInBias != nullptr && p_clInBias->GetGradient().Valid() ? p_clInBias->GetGradient().data() : nullptr;

    const int iOuterNum = clInData.GetSize()[0];
    const int iNumChannels = clInData.GetSize()[1];
    const int iInnerNum = clInData.GetSize().Product(2);

    if (p_inBiasGradient != nullptr) {
      for (int i = 0; i < iOuterNum; ++i) {
        for (int k = 0; k < iInnerNum; ++k) {
          for (int j = 0; j < iNumChannels; ++j) {
            p_inBiasGradient[j] += p_outDataGradient[(i*iNumChannels + j)*iInnerNum + k];
          }
        }
      }
    }

    if (p_inWeightsGradient != nullptr) {
      for (int i = 0; i < iOuterNum; ++i) {
        for (int k = 0; k < iInnerNum; ++k) {
          for (int j = 0; j < iNumChannels; ++j) {
            p_inWeightsGradient[j] += p_inData[(i*iNumChannels + j)*iInnerNum + k] * p_outDataGradient[(i*iNumChannels + j)*iInnerNum + k];
          }
        }
      }
    }

    if (p_inDataGradient != nullptr) {
      for(int i = 0; i < iOuterNum; ++i) {
        for(int k = 0; k < iInnerNum; ++k) {
          for(int j = 0; j < iNumChannels; ++j) {
            p_inDataGradient[(i*iNumChannels + j)*iInnerNum + k] += p_inWeights[j] * p_outDataGradient[(i*iNumChannels + j)*iInnerNum + k];
          }
        }
      }
    }
  }

protected:
  Scale() = default;
};

} // end namespace bleak

#endif // !BLEAK_SCALE_H
