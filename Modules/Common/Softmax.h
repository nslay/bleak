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

#ifndef BLEAK_SOFTMAX_H
#define BLEAK_SOFTMAX_H

#include <cmath>
#include <algorithm>
#include "Vertex.h"

namespace bleak {

template<typename RealType>
class Softmax : public Vertex<RealType> {
public:
  bleakNewVertex(Softmax, Vertex<RealType>,
    bleakAddInput("inData"),
    bleakAddOutput("outProbabilities"));

  bleakForwardVertexTypedefs();

  virtual ~Softmax() = default;

  virtual bool SetSizes() override {
    bleakGetAndCheckInput(p_clInData, "inData", false);
    bleakGetAndCheckOutput(p_clOutProbs, "outProbabilities", false);

    const ArrayType &clInData = p_clInData->GetData();
    ArrayType &clOutProbs = p_clOutProbs->GetData();
    ArrayType &clOutGradient = p_clOutProbs->GetGradient();

    if (!clInData.GetSize().Valid()) {
      std::cerr << GetName() << ": Error: Invalid dimensions for inData." << std::endl;
      return false;
    }

    if (clInData.GetSize().GetDimension() < 2) {
      std::cerr << GetName() << ": Error: inData is expected to be 2D or higher." << std::endl;
      return false;
    }

    if (clInData.GetSize()[1] < 2) {
      std::cerr << GetName() << ": Error: Invalid number of channels in " << clInData.GetSize() << ". Number of channels should be 2 or larger." << std::endl;
      return false;
    }

    clOutProbs.SetSize(clInData.GetSize());
    clOutGradient.SetSize(clInData.GetSize());

    return true;
  }

  virtual bool Initialize() override {
    return true;
  }

  virtual void Forward() override {
    bleakGetAndCheckInput(p_clInData, "inData");
    bleakGetAndCheckOutput(p_clOutProbs, "outProbabilities");

    const ArrayType &clInData = p_clInData->GetData();
    ArrayType &clOutProbs = p_clOutProbs->GetData();

    const int iOuterNum = clInData.GetSize()[0];
    const int iNumOutputs = clInData.GetSize()[1];
    const int iInnerNum = clInData.GetSize().Product(2);

    const RealType * const p_inData = clInData.data();
    RealType * const p_outProbs = clOutProbs.data();

    for (int i = 0; i < iOuterNum; ++i) {
      for (int k = 0; k < iInnerNum; ++k) {
        RealType valueMax = p_inData[(i*iNumOutputs + 0)*iInnerNum + k];
        int jMax = 0;

        for (int j = 1; j < iNumOutputs; ++j) {
          const RealType value = p_inData[(i*iNumOutputs + j)*iInnerNum + k];

          if (value > valueMax) {
            valueMax = value;
            jMax = j;
          }
        }

        RealType sum = RealType();

        for (int j = 0; j < iNumOutputs; ++j) {
          const RealType value = p_inData[(i*iNumOutputs + j)*iInnerNum + k];
          const RealType valueExp = (j != jMax) ? std::exp(value - valueMax) : RealType(1);

          p_outProbs[(i*iNumOutputs + j)*iInnerNum + k] = valueExp;

          sum += valueExp;
        }

        if (!std::isfinite(sum)) {
          std::cerr << GetName() << ": Error: Normalizer is not finite!" << std::endl;
          return;
        }

        for (int j = 0; j < iNumOutputs; ++j)
          p_outProbs[(i*iNumOutputs + j)*iInnerNum + k] /= sum;
      }
    }
  }

  virtual void Backward() override {
    bleakGetAndCheckInput(p_clInData,"inData");
    bleakGetAndCheckOutput(p_clOutProbs, "outProbabilities");

    ArrayType &clInData = p_clInData->GetData();
    ArrayType &clInGradient = p_clInData->GetGradient();
    const ArrayType &clOutProbs = p_clOutProbs->GetData();
    const ArrayType &clOutGradient = p_clOutProbs->GetGradient();

    if (!clInGradient.Valid()) // Nothing to do
      return;

    const int iOuterNum = clInData.GetSize()[0];
    const int iNumOutputs = clInData.GetSize()[1];
    const int iInnerNum = clInData.GetSize().Product(2);

    const RealType * const p_outProbs = clOutProbs.data();
    const RealType * const p_outGradient = clOutGradient.data();
    RealType * const p_inGradient = clInGradient.data();

    for (int i = 0; i < iOuterNum; ++i) {
      for (int k = 0; k < iInnerNum; ++k) {
        RealType innerProd = RealType();

        for (int j = 0; j < iNumOutputs; ++j)
          innerProd += p_outGradient[(i*iNumOutputs + j)*iInnerNum + k]*p_outProbs[(i*iNumOutputs + j)*iInnerNum + k];

        for (int j = 0; j < iNumOutputs; ++j) {
          const RealType &prob = p_outProbs[(i*iNumOutputs + j)*iInnerNum + k];
          p_inGradient[(i*iNumOutputs + j)*iInnerNum + k] += (p_outGradient[(i*iNumOutputs + j)*iInnerNum + k] - innerProd)*prob;
        }
      }
    }
  }

protected:

  Softmax() = default;
};

} // end namespace bleak

#endif // !BLEAK_SOFTMAX_H
