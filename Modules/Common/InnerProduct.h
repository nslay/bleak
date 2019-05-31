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

#ifndef BLEAK_INNERPRODUCT_H
#define BLEAK_INNERPRODUCT_H

#include "Vertex.h"

namespace bleak {

template<typename RealType>
class InnerProduct : public Vertex<RealType> {
public:
  bleakNewVertex(InnerProduct, Vertex<RealType>,
    bleakAddInput("inData"),
    bleakAddInput("inWeights"),
    bleakAddInput("inBias"),
    bleakAddOutput("outData"));

  bleakForwardVertexTypedefs();

  virtual ~InnerProduct() = default;

  virtual bool SetSizes() override {
    bleakGetAndCheckInput(p_clInData, "inData", false);
    bleakGetAndCheckInput(p_clWeights, "inWeights", false);
    bleakGetAndCheckOutput(p_clOutData, "outData", false);

    std::shared_ptr<EdgeType> p_clBias = GetInput("inBias");

    // If "inBias" isn't connected, we'll assume it's 0

    const ArrayType &clInData = p_clInData->GetData();
    const ArrayType &clWeights = p_clWeights->GetData();
    ArrayType &clOutData = p_clOutData->GetData();
    ArrayType &clOutGradient = p_clOutData->GetGradient();

    if (!clInData.GetSize().Valid() || !clWeights.GetSize().Valid()) {
      std::cerr << GetName() << ": Error: Invalid dimensions for inData and/or inWeights." << std::endl;
      return false;
    }

    if (clInData.GetSize().GetDimension() < 2) {
      std::cerr << GetName() << ": Error: inData is expected to be 2D or higher." << std::endl;
      return false;
    }

    // Weights are [ $numOutputs, $numChannels ]
    // InnerProduct is computed along axis 1
    if (clWeights.GetSize().GetDimension() != 2) {
      std::cerr << GetName() << ": Error: inWeights is expected to be 2D." << std::endl;
      return false;
    }

    if (clInData.GetSize()[1] != clWeights.GetSize()[1]) {
      std::cerr << GetName() << ": Error: inData and inWeights are expected to have the same number of channels." << std::endl;
      return false;
    }

    if (p_clBias == nullptr) {
      std::cout << GetName() << ": Info: inBias is not connected. Bias will be assumed to be fixed as 0." << std::endl;
    }
    else {
      const ArrayType &clBias = p_clBias->GetData();
      if (clBias.GetSize().GetDimension() != 1) {
        std::cerr << GetName() << ": Error: inBias is expected to be 1D." << std::endl;
        return false;
      }

      if (clBias.GetSize()[0] != clWeights.GetSize()[0]) {
        std::cerr << GetName() << ": Error: inWeights and inBias are expected to have the same number of outputs." << std::endl;
        return false;
      }
    }

    Size clOutSize(clInData.GetSize());
    clOutSize[1] = clWeights.GetSize()[0]; // Number of outputs

    clOutData.SetSize(clOutSize);
    clOutGradient.SetSize(clOutSize);

    return true;
  }

  virtual bool Initialize() override {
    // Nothing to do ...
    return true;
  }

  virtual void Forward() override {
    bleakGetAndCheckInput(p_clInData,"inData");
    bleakGetAndCheckInput(p_clWeights,"inWeights");
    bleakGetAndCheckOutput(p_clOutData,"outData");

    std::shared_ptr<EdgeType> p_clBias = GetInput("inBias");

    const ArrayType &clInData = p_clInData->GetData();
    const ArrayType &clWeights = p_clWeights->GetData();
    ArrayType &clOutData = p_clOutData->GetData();

    const RealType * const p_inData = clInData.data();
    const RealType * const p_weights = clWeights.data();
    const RealType * const p_bias = p_clBias != nullptr ? p_clBias->GetData().data() : nullptr;

    RealType * const p_outData = clOutData.data();

    const int iOuterNum = clInData.GetSize()[0];
    const int iNumChannels = clInData.GetSize()[1];
    const int iNumOutputs = clWeights.GetSize()[0];
    const int iInnerNum = clInData.GetSize().Product(2);

    if (p_bias == nullptr) {
      clOutData.Fill(RealType());
    }
    else {
      for (int i = 0; i < iOuterNum; ++i) {
        for (int k = 0; k < iInnerNum; ++k) {
          for (int j = 0; j < iNumOutputs; ++j) {
            p_outData[(i*iNumOutputs + j)*iInnerNum + k] = p_bias[j];
          }
        }
      }
    }

    for (int i = 0; i < iOuterNum; ++i) {
      for(int j = 0; j < iNumOutputs; ++j) {
        for(int k = 0; k < iInnerNum; ++k) {
          for(int l = 0; l < iNumChannels; ++l) {
            p_outData[(i*iNumOutputs + j)*iInnerNum + k] += p_inData[(i*iNumChannels + l)*iInnerNum + k]*p_weights[j*iNumChannels + l];
          }
        }
      }
    }

  }

  virtual void Backward() override {
    bleakGetAndCheckInput(p_clInData,"inData");
    bleakGetAndCheckInput(p_clWeights,"inWeights");
    bleakGetAndCheckOutput(p_clOutData,"outData");

    std::shared_ptr<EdgeType> p_clBias = GetInput("inBias");

    const ArrayType &clInData = p_clInData->GetData();
    ArrayType &clInDataGradient = p_clInData->GetGradient();

    const ArrayType &clWeights = p_clWeights->GetData();
    ArrayType &clWeightsGradient = p_clWeights->GetGradient();

    const ArrayType &clOutGradient = p_clOutData->GetGradient();

    const RealType * const p_inData = clInData.data();
    const RealType * const p_weights = clWeights.data();
    const RealType * const p_bias = p_clBias != nullptr ? p_clBias->GetData().data() : nullptr;
    const RealType * const p_outDataGradient = clOutGradient.data();

    RealType * const p_inDataGradient = clInDataGradient.Valid() ? clInDataGradient.data() : nullptr;
    RealType * const p_weightsGradient = clWeightsGradient.Valid() ? clWeightsGradient.data() : nullptr;
    RealType * const p_biasGradient = p_clBias != nullptr && p_clBias->GetGradient().Valid() ? p_clBias->GetGradient().data() : nullptr;

    const int iOuterNum = clInData.GetSize()[0];
    const int iNumChannels = clInData.GetSize()[1];
    const int iNumOutputs = clWeights.GetSize()[0];
    const int iInnerNum = clInData.GetSize().Product(2);

    if (p_biasGradient != nullptr) {
      for (int i = 0; i < iOuterNum; ++i) {
        for (int k = 0; k < iInnerNum; ++k) {
          for (int j = 0; j < iNumOutputs; ++j) {
            p_biasGradient[j] += p_outDataGradient[(i*iNumOutputs + j)*iInnerNum + k];
          }
        }
      }
    }

    if (p_weightsGradient != nullptr) {
      for (int i = 0; i < iOuterNum; ++i) {
        for (int j = 0; j < iNumOutputs; ++j) {
          for(int k = 0; k < iInnerNum; ++k) {
            for(int l = 0; l < iNumChannels; ++l) {
              p_weightsGradient[j*iNumChannels + l] += p_outDataGradient[(i*iNumOutputs + j)*iInnerNum + k] * p_inData[(i*iNumChannels + l)*iInnerNum + k];
            }
          }
        }
      }
    }

    if (p_inDataGradient != nullptr) {
      for (int i = 0; i < iOuterNum; ++i) {
        for (int j = 0; j < iNumOutputs; ++j) {
          for(int k = 0; k < iInnerNum; ++k) {
            for(int l = 0; l < iNumChannels; ++l) {
              p_inDataGradient[(i*iNumChannels + l)*iInnerNum + k] += p_outDataGradient[(i*iNumOutputs + j)*iInnerNum + k] * p_weights[j*iNumChannels + l];
            }
          }
        }
      }
    }
  }

protected:
  InnerProduct() = default;
};

} // end namespace bleak

#endif // !BLEAK_INNERPRODUCT_H
