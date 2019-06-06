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

#ifndef BLEAK_RANDOMHINGEFERNS_H
#define BLEAK_RANDOMHINGEFERNS_H

#include <cstdint>
#include <climits>
#include <algorithm>
#include <vector>
#include <limits>
#include <numeric>
#include <utility>
#include <random>
#include "Common.h"
#include "Vertex.h"

namespace bleak {

template<typename RealType>
class RandomHingeFerns : public Vertex<RealType> {
public:
  bleakNewVertex(RandomHingeFerns, Vertex<RealType>,
    bleakAddInput("inOrdinals"),
    bleakAddInput("inThresholds"),
    bleakAddInput("inWeights"),
    bleakAddInput("inData"),
    bleakAddOutput("outData"));

  bleakForwardVertexTypedefs();

  typedef uint32_t KeyType;

  // This limits the depth of the tree to 32 (so thresholds size should be 32 or less)
  static constexpr KeyType GetMaxTreeDepth() {
    return CHAR_BIT*sizeof(KeyType);
  }

  virtual ~RandomHingeFerns() = default;

  virtual bool SetSizes() override {
    bleakGetAndCheckInput(p_clOrdinals, "inOrdinals", false);
    bleakGetAndCheckInput(p_clThresholds, "inThresholds", false);
    bleakGetAndCheckInput(p_clWeights, "inWeights", false);
    bleakGetAndCheckInput(p_clInData, "inData", false);
    bleakGetAndCheckOutput(p_clOutData, "outData", false);

    const ArrayType &clOrdinals = p_clOrdinals->GetData();
    const ArrayType &clThresholds = p_clThresholds->GetData();
    const ArrayType &clWeights = p_clWeights->GetData();
    const ArrayType &clInData = p_clInData->GetData();
    ArrayType &clOutData = p_clOutData->GetData();
    ArrayType &clOutDataGradient = p_clOutData->GetGradient();

    if (!clOrdinals.GetSize().Valid() || !clThresholds.GetSize().Valid() || !clWeights.GetSize().Valid() || !clInData.GetSize().Valid()) {
      std::cerr << GetName() << ": Error: Invalid size for inOrdinals, inThresholds, inWeights and/or inData." << std::endl;
      return false;
    }

    if (p_clOrdinals->GetGradient().GetSize().Valid()) {
      std::cerr << GetName() << ": Error: inOrdinals appears learnable. However, RandomHingeFerns has its own custom updates for this input." << std::endl;
      return false;
    }

    // [ $batchSize, $numFeatures, ... ]
    if (clInData.GetSize().GetDimension() < 2) {
      std::cerr << GetName() << ": Error: inData is expected to be 2D or higher." << std::endl;
      return false;
    }

    if (clThresholds.GetSize().GetDimension() != 2 || clOrdinals.GetSize().GetDimension() != 2) {
      std::cerr << GetName() << ": Error: inThresholds and inOrdinals are expected to be 2D." << std::endl;
      return false;
    }

    if (clWeights.GetSize().GetDimension() < 2) {
      std::cerr << GetName() << ": Error: inWeights is expected to be 2D or higher." << std::endl;
      return false;
    }

    if (clWeights.GetSize()[0] != clThresholds.GetSize()[0] || clWeights.GetSize()[0] != clOrdinals.GetSize()[0]) {
      std::cerr << GetName() << ": Error: Dimension mismatch between inWeights, inThresholds and/or inOrdinals." << std::endl;
      return false;
    }

    if (clOrdinals.GetSize()[1] != clThresholds.GetSize()[1]) {
      std::cerr << GetName() << ": Error: Dimension mismatch between inThresholds and/or inOrdinals." << std::endl;
      return false;
    }

    if (clInData.GetSize()[1] < clThresholds.GetSize()[1]) {
      std::cerr << GetName() << ": Error: Too few inputs. Each instance is " << clInData.GetSize()[1] << " inputs, but requested tree depth is " << clThresholds.GetSize()[1] << '.' << std::endl;
      return false;
    }

    const int iOuterNum = clInData.GetSize()[0];
    const int iTreeDepth = clThresholds.GetSize()[1];
    const int iNumTrees = clWeights.GetSize()[0];
    const int iNumLeavesPerTree = clWeights.GetSize()[1];

    const KeyType numLeavesExpected = (KeyType(1) << iTreeDepth);

    if (KeyType(iTreeDepth) > GetMaxTreeDepth()) {
      std::cerr << GetName() << ": Error: inThresholds would result in trees that are " << iTreeDepth << " deep. But we are limited to trees of depth " << GetMaxTreeDepth() << '.' << std::endl;
      return false;
    }

    if (KeyType(iNumLeavesPerTree) != numLeavesExpected) {
      std::cerr << GetName() << ": Error: Dimension mismatch between inThresholds and inWeights. Each tree is expect to have " << numLeavesExpected << " weights." << std::endl;
      return false;
    }

    m_iTreeDepth = iTreeDepth;

    Size clOutSize;

    clOutSize.SetDimension(clInData.GetSize().GetDimension() + clWeights.GetSize().GetDimension() - 2);

    // [ $batchSize, $numTrees, $dataDims..., $weightsDims... ]
    auto itr = std::copy(clInData.GetSize().begin(), clInData.GetSize().end(), clOutSize.begin());
    std::copy(clWeights.GetSize().begin() + 2, clWeights.GetSize().end(), itr);

    clOutSize[1] = iNumTrees;

    clOutData.SetSize(clOutSize);
    clOutDataGradient.SetSize(clOutSize);

    return true;
  }

  virtual bool Initialize() override {
    bleakGetAndCheckInput(p_clThresholds, "inThresholds", false);

    m_iTreeDepth = p_clThresholds->GetData().GetSize()[1];

    return AssignTreeIndices();
  }

  virtual void Forward() override {
    bleakGetAndCheckInput(p_clOrdinals, "inOrdinals");
    bleakGetAndCheckInput(p_clThresholds, "inThresholds");
    bleakGetAndCheckInput(p_clWeights, "inWeights");
    bleakGetAndCheckInput(p_clInData, "inData");
    bleakGetAndCheckOutput(p_clOutData, "outData");

    const ArrayType &clOrdinals = p_clOrdinals->GetData();
    const ArrayType &clInData = p_clInData->GetData();
    const ArrayType &clThresholds = p_clThresholds->GetData();
    const ArrayType &clWeights = p_clWeights->GetData();
    ArrayType &clOutData = p_clOutData->GetData();

    const int iOuterNum = clInData.GetSize()[0];
    const int iNumChannels = clInData.GetSize()[1];
    const int iInnerDataNum = clInData.GetSize().Product(2);
    const int iTreeDepth = clThresholds.GetSize()[1];
    const int iNumTrees = clWeights.GetSize()[0];
    const int iNumLeavesPerTree = clWeights.GetSize()[1];
    const int iInnerWeightsNum = clWeights.GetSize().Product(2);
    
    const RealType * const p_ordinals = clOrdinals.data();
    const RealType * const p_inData = clInData.data();
    const RealType * const p_weights = clWeights.data();
    const RealType * const p_thresholds = clThresholds.data();
    RealType * const p_outData = clOutData.data();

    for (int i = 0; i < iOuterNum; ++i) {
      for (int j = 0; j < iNumTrees; ++j) {
        for (int k = 0; k < iInnerDataNum; ++k) {
          // p_inData[(i*iNumChannels + l)*iInnerDataNum + k]
          const auto clKeyMarginTuple = ComputeKeyAndSignedMargin(p_inData + ((i*iNumChannels + 0)*iInnerDataNum + k), 
            p_thresholds + (j*iTreeDepth + 0), p_ordinals + (j*iTreeDepth + 0), iInnerDataNum);

          const KeyType key = std::get<0>(clKeyMarginTuple);
          const RealType margin = std::get<1>(clKeyMarginTuple); // Signed margin

          for (int m = 0; m < iInnerWeightsNum; ++m) {
            p_outData[((i*iNumTrees + j)*iInnerDataNum + k)*iInnerWeightsNum + m] = std::abs(margin) * p_weights[(j*iNumLeavesPerTree + key)*iInnerWeightsNum + m];
          }
        }
      }
    }
  }

  virtual void Backward() override {
    bleakGetAndCheckInput(p_clOrdinals, "inOrdinals");
    bleakGetAndCheckInput(p_clInData, "inData");
    bleakGetAndCheckInput(p_clThresholds, "inThresholds");
    bleakGetAndCheckInput(p_clWeights, "inWeights");
    bleakGetAndCheckOutput(p_clOutData, "outData");

    const ArrayType &clOrdinals = p_clOrdinals->GetData();
    const ArrayType &clInData = p_clInData->GetData();
    ArrayType &clInDataGradient = p_clInData->GetGradient();
    const ArrayType &clThresholds = p_clThresholds->GetData();
    ArrayType &clThresholdsGradient = p_clThresholds->GetGradient();
    const ArrayType &clWeights = p_clWeights->GetData();
    ArrayType &clWeightsGradient = p_clWeights->GetGradient();
    const ArrayType &clOutData = p_clOutData->GetData();
    const ArrayType &clOutDataGradient = p_clOutData->GetGradient();

    const RealType * const p_ordinals = clOrdinals.data();
    const RealType * const p_inData = clInData.data();
    RealType * const p_inDataGradient = clInDataGradient.data();
    const RealType * const p_thresholds = clThresholds.data();
    RealType * const p_thresholdsGradient = clThresholdsGradient.data();
    const RealType * const p_weights = clWeights.data();
    RealType * const p_weightsGradient = clWeightsGradient.data();
    const RealType * const p_outData = clOutData.data();
    const RealType * const p_outDataGradient = clOutDataGradient.data();

    const int iOuterNum = clInData.GetSize()[0];
    const int iNumChannels = clInData.GetSize()[1];
    const int iInnerDataNum = clInData.GetSize().Product(2);
    const int iTreeDepth = clThresholds.GetSize()[1];
    const int iNumTrees = clWeights.GetSize()[0];
    const int iNumLeavesPerTree = clWeights.GetSize()[1];
    const int iInnerWeightsNum = clWeights.GetSize().Product(2);

    if (p_inDataGradient != nullptr) {
      for(int i = 0; i < iOuterNum; ++i) {
        for(int j = 0; j < iNumTrees; ++j) {
          for (int k = 0; k < iInnerDataNum; ++k) {
            // p_inData[(i*iNumChannels + l)*iInnerNum + k]
            const auto clKeyMarginTuple = ComputeKeyAndSignedMargin(p_inData + ((i*iNumChannels + 0)*iInnerDataNum + k), 
              p_thresholds + (j*iTreeDepth + 0), p_ordinals + (j*iTreeDepth + 0), iInnerDataNum);

            const KeyType key = std::get<0>(clKeyMarginTuple);
            const RealType margin = std::get<1>(clKeyMarginTuple); // Signed margin
            
            const int iThresholdIndex = std::get<2>(clKeyMarginTuple);
            const int iInputIndex = (int)p_ordinals[j*iTreeDepth + iThresholdIndex];
            const RealType sign = (RealType(0) < margin) - (margin < RealType(0));

            for (int m = 0; m < iInnerWeightsNum; ++m) {
              p_inDataGradient[(i*iNumChannels + iInputIndex)*iInnerDataNum + k] += sign * p_weights[(j*iNumLeavesPerTree + key)*iInnerWeightsNum + m] * p_outDataGradient[((i*iNumTrees + j)*iInnerDataNum + k)*iInnerWeightsNum + m];
            }
          }
        }
      }
    }

    if (p_thresholdsGradient != nullptr) {
      for (int i = 0; i < iOuterNum; ++i) {
        for (int j = 0; j < iNumTrees; ++j) {
          for (int k = 0; k < iInnerDataNum; ++k) {
            // p_inData[(i*iNumChannels + l)*iInnerNum + k]
            const auto clKeyMarginTuple = ComputeKeyAndSignedMargin(p_inData + ((i*iNumChannels + 0)*iInnerDataNum + k), 
              p_thresholds + (j*iTreeDepth + 0), p_ordinals + (j*iTreeDepth + 0), iInnerDataNum);

            const KeyType key = std::get<0>(clKeyMarginTuple);
            const RealType margin = std::get<1>(clKeyMarginTuple); // Signed margin
            
            const int iThresholdIndex = std::get<2>(clKeyMarginTuple);
            const RealType sign = (RealType(0) < margin) - (margin < RealType(0));

            for (int m = 0; m < iInnerWeightsNum; ++m) {
              p_thresholdsGradient[j*iTreeDepth + iThresholdIndex] += -sign * p_weights[(j*iNumLeavesPerTree + key)*iInnerWeightsNum + m] * p_outDataGradient[((i*iNumTrees + j)*iInnerDataNum + k)*iInnerWeightsNum + m];
            }
          }
        }
      }
    }

    if (p_weightsGradient != nullptr) {
      for (int i = 0; i < iOuterNum; ++i) {
        for (int j = 0; j < iNumTrees; ++j) {
          for (int k = 0; k < iInnerDataNum; ++k) {
            // p_inData[(i*iNumChannels + l)*iInnerNum + k]
            const auto clKeyMarginTuple = ComputeKeyAndSignedMargin(p_inData + ((i*iNumChannels + 0)*iInnerDataNum + k), 
              p_thresholds + (j*iTreeDepth + 0), p_ordinals + (j*iTreeDepth + 0), iInnerDataNum);


            const KeyType key = std::get<0>(clKeyMarginTuple);
            const RealType margin = std::get<1>(clKeyMarginTuple); // Signed margin

            for (int m = 0; m < iInnerWeightsNum; ++m) {
              p_weightsGradient[(j*iNumLeavesPerTree + key)*iInnerWeightsNum + m] += std::abs(margin) * p_outDataGradient[((i*iNumTrees + j)*iInnerDataNum + k)*iInnerWeightsNum + m];
            }
          }
        }
      }
    }
  }

protected:
  RandomHingeFerns() = default;

private:
  typedef std::tuple<KeyType,RealType,int> KeyMarginTupleType;

  int m_iTreeDepth = 0;

  KeyMarginTupleType ComputeKeyAndSignedMargin(const RealType *p_data, const RealType *p_thresholds, const RealType *p_ordinals, int iStride) {
    KeyType key = KeyType();
    RealType minMargin = p_data[iStride*(int)p_ordinals[0]] - p_thresholds[0];
    int iMinMarginIndex = 0;

    for (int i = 0; i < m_iTreeDepth; ++i) {
      const int j = (int)p_ordinals[i];
      const RealType margin = p_data[iStride*j] - p_thresholds[i];
      const KeyType bit = (margin > RealType(0));

      key |= (bit << i);

      if (std::abs(margin) < std::abs(minMargin)) {
        minMargin = margin;
        iMinMarginIndex = (int)i;
      }
    }

    return std::make_tuple(key, minMargin, iMinMarginIndex);
  }

  bool AssignTreeIndices() {
    bleakGetAndCheckInput(p_clInData, "inData", false);
    bleakGetAndCheckInput(p_clOrdinals, "inOrdinals", false);

    const ArrayType &clInData = p_clInData->GetData();
    ArrayType &clOrdinals = p_clOrdinals->GetData();

    const int iNumTrees = clOrdinals.GetSize()[0];
    const int iTreeDepth = clOrdinals.GetSize()[1];
    const int iNumChannels = clInData.GetSize()[1];

    if (iTreeDepth > iNumChannels || !clOrdinals.Valid())
      return false;

    RealType * const p_ordinals = clOrdinals.data();

    std::uniform_int_distribution<int> clRandomFeature(0, iNumChannels-1);

    for (int i = 0; i < iNumTrees; ++i) {
      for (int j = 0; j < iTreeDepth; ++j) {
        p_ordinals[i*iTreeDepth + j] = RealType(clRandomFeature(GetGenerator()));
      }

      // Fern decisions are invariant to ordering, so might as well optimize for memory access
      std::sort(p_ordinals + (i*iTreeDepth + 0), p_ordinals + ((i+1)*iTreeDepth + 0));
    }

    return true;
  }
};

} // end namespace bleak

#endif // !BLEAK_RANDOMHINGEFERNS_H
