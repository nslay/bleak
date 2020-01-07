/*-
 * Copyright (c) 2019 Nathan Lay (enslay@gmail.com)
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

#ifndef BLEAK_RANDOMHINGEFOREST_H
#define BLEAK_RANDOMHINGEFOREST_H

#include <random>
#include "Vertex.h"
#include "Common.h"
#include "HingeTreeCommon.h"

namespace bleak {

template<typename RealType, typename TreeTraitsType>
class RandomHingeForestTemplate : public Vertex<RealType> {
public:
  bleakNewAbstractVertex(RandomHingeForestTemplate, Vertex<RealType>,
    bleakAddInput("inOrdinals"),
    bleakAddInput("inThresholds"),
    bleakAddInput("inWeights"),
    bleakAddInput("inData"),
    bleakAddOutput("outData"));

  bleakForwardVertexTypedefs();

  typedef typename TreeTraitsType::KeyType KeyType;

  virtual ~RandomHingeForestTemplate() = default;

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

    if (!clOrdinals.GetSize().Valid() || !clThresholds.GetSize().Valid() || !clWeights.GetSize().Valid() || !clInData.GetSize().Valid()) {
      std::cerr << GetName() << ": Error: Invalid sizes for inOrdinals, inThresholds, inWeights and/or inData." << std::endl;
      return false;
    }

    if (p_clOrdinals->GetGradient().GetSize().Valid()) {
      std::cerr << GetName() << ": Error: inOrdinals appears learnable. However, RandomHingeForest/Fern has its own custom updates for this input." << std::endl;
      return false;
    }

    // [ $batchSize, $numFeatures, ... ]
    if (clInData.GetSize().GetDimension() < 2) {
      std::cerr << GetName() << ": Error: inData is expected to be 2D or higher." << std::endl;
      return false;
    }

    if (clOrdinals.GetSize().GetDimension() != 2 || clThresholds.GetSize().GetDimension() != 2) {
      std::cerr << GetName() << ": Error: inOrdinals and inThresholds are expected to be 2D." << std::endl;
      return false;
    }

    if (clWeights.GetSize().GetDimension() < 2) {
      std::cerr << GetName() << ": Error: inWeights is expected to be 2D or higher." << std::endl;
      return false;
    }

    const int iNumTrees = clOrdinals.GetSize()[0];

    if (iNumTrees != clThresholds.GetSize()[0] || iNumTrees != clWeights.GetSize()[0]) {
      std::cerr << GetName() << ": Error: Dimension mismatch between inOrdinals, inThresholds and inWeights. The first dimension is the number of trees and these should all match." << std::endl;
      return false;
    }

    m_iTreeDepth = TreeTraitsType::ComputeDepth(clWeights.GetSize()[1]);

    if (m_iTreeDepth <= 0) {
      std::cerr << GetName() << ": Error: inWeights is expected to have a power of 2 size for its second dimension." << std::endl;
      return false;
    }

    if (m_iTreeDepth > TreeTraitsType::GetMaxDepth()) {
      std::cerr << GetName() << ": Error: Tree depth is calculated to be " << m_iTreeDepth << " which exceeds the compiled-in limitation of " << TreeTraitsType::GetMaxDepth() << '.' << std::endl;
      return false;
    }

    if (clOrdinals.GetSize()[1] != clThresholds.GetSize()[1]) {
      std::cerr << GetName() << ": Error: inOrdinals and inThresholds are expected to have the same dimension." << std::endl;
      return false;
    }

    if (clThresholds.GetSize()[1] != TreeTraitsType::GetThresholdCount(m_iTreeDepth)) {
      std::cerr << GetName() << ": Error: inThresholds/inOrdinals have an unexpected number of elements (" << clThresholds.GetSize()[1] << " != " << TreeTraitsType::GetThresholdCount(m_iTreeDepth) << ")." << std::endl;
      return false;
    }

    Size clOutSize;

    clOutSize.SetDimension(clInData.GetSize().GetDimension() + clWeights.GetSize().GetDimension() - 2);

    // [ $batchSize, $numTrees, $dataDims..., $weightsDims... ]
    auto itr = std::copy(clInData.GetSize().begin(), clInData.GetSize().end(), clOutSize.begin());
    std::copy(clWeights.GetSize().begin() + 2, clWeights.GetSize().end(), itr);

    clOutSize[1] = iNumTrees;

    p_clOutData->GetData().SetSize(clOutSize);
    p_clOutData->GetGradient().SetSize(clOutSize);

    return true;
  }

  virtual bool Initialize() override {
    bleakGetAndCheckInput(p_clWeights, "inWeights", false);

    if (p_clWeights->GetData().GetSize().GetDimension() < 2 || (m_iTreeDepth = TreeTraitsType::ComputeDepth(p_clWeights->GetData().GetSize()[1])) <= 0)
      return false;

    return AssignTreeIndices();
  }

  virtual void Forward() override {
    bleakGetAndCheckInput(p_clOrdinals, "inOrdinals");
    bleakGetAndCheckInput(p_clThresholds, "inThresholds");
    bleakGetAndCheckInput(p_clWeights, "inWeights");
    bleakGetAndCheckInput(p_clInData, "inData");
    bleakGetAndCheckOutput(p_clOutData, "outData");

    const ArrayType &clOrdinals = p_clOrdinals->GetData();
    const ArrayType &clThresholds = p_clThresholds->GetData();
    const ArrayType &clWeights = p_clWeights->GetData();
    const ArrayType &clInData = p_clInData->GetData();
    ArrayType &clOutData = p_clOutData->GetData();

    const RealType * const p_ordinals = clOrdinals.data();
    const RealType * const p_thresholds = clThresholds.data();
    const RealType * const p_weights = clWeights.data();
    const RealType * const p_inData = clInData.data();
    RealType * const p_outData = clOutData.data();

    const int iOuterNum = clInData.GetSize()[0];
    const int iNumChannels = clInData.GetSize()[1];
    const int iInnerDataNum = clInData.GetSize().Product(2);
    const int iNumTrees = clWeights.GetSize()[0];
    const int iNumDecisionsPerTree = clOrdinals.GetSize()[1];
    const int iNumLeavesPerTree = clWeights.GetSize()[1];
    const int iInnerWeightsNum = clWeights.GetSize().Product(2);

    for (int i = 0; i < iOuterNum; ++i) {
      for (int j = 0; j < iNumTrees; ++j) {
        for (int k = 0; k < iInnerDataNum; ++k) {
          // p_inData[(i*iNumChannels + l)*iInnerDataNum + k]
          const auto clKeyMarginTuple = TreeTraitsType::ComputeKeyAndSignedMargin(p_inData + ((i*iNumChannels + 0)*iInnerDataNum + k), 
            p_thresholds + (j*iNumDecisionsPerTree + 0), p_ordinals + (j*iNumDecisionsPerTree + 0), m_iTreeDepth, iInnerDataNum);

          const KeyType leafKey = std::get<0>(clKeyMarginTuple);
          const RealType margin = std::get<1>(clKeyMarginTuple); // Signed margin

          for (int m = 0; m < iInnerWeightsNum; ++m) {
            p_outData[((i*iNumTrees + j)*iInnerDataNum + k)*iInnerWeightsNum + m] = std::abs(margin) * p_weights[(j*iNumLeavesPerTree + leafKey)*iInnerWeightsNum + m];
          }
        }
      }
    }

  }

  virtual void Backward() override {
    bleakGetAndCheckInput(p_clOrdinals, "inOrdinals");
    bleakGetAndCheckInput(p_clThresholds, "inThresholds");
    bleakGetAndCheckInput(p_clWeights, "inWeights");
    bleakGetAndCheckInput(p_clInData, "inData");
    bleakGetAndCheckOutput(p_clOutData, "outData");

    const ArrayType &clOrdinals = p_clOrdinals->GetData();
    const ArrayType &clThresholds = p_clThresholds->GetData();
    ArrayType &clThresholdsGradient = p_clThresholds->GetGradient();
    const ArrayType &clWeights = p_clWeights->GetData();
    ArrayType &clWeightsGradient = p_clWeights->GetGradient();
    const ArrayType &clInData = p_clInData->GetData();
    ArrayType &clInDataGradient = p_clInData->GetGradient();
    const ArrayType &clOutData = p_clOutData->GetData();
    const ArrayType &clOutDataGradient = p_clOutData->GetGradient();

    const RealType * const p_ordinals = clOrdinals.data();
    const RealType * const p_thresholds = clThresholds.data();
    RealType * const p_thresholdsGradient = clThresholdsGradient.data();
    const RealType * const p_weights = clWeights.data();
    RealType * const p_weightsGradient = clWeightsGradient.data();
    const RealType * const p_inData = clInData.data();
    RealType * p_inDataGradient = clInDataGradient.data();
    const RealType * const p_outData = clOutData.data();
    const RealType * const p_outDataGradient = clOutDataGradient.data();

    const int iOuterNum = clInData.GetSize()[0];
    const int iNumChannels = clInData.GetSize()[1];
    const int iInnerDataNum = clInData.GetSize().Product(2);
    const int iNumTrees = clWeights.GetSize()[0];
    const int iNumDecisionsPerTree = clOrdinals.GetSize()[1];
    const int iNumLeavesPerTree = clWeights.GetSize()[1];
    const int iInnerWeightsNum = clWeights.GetSize().Product(2);

    if (p_inDataGradient != nullptr) {
      for(int i = 0; i < iOuterNum; ++i) {
        for(int j = 0; j < iNumTrees; ++j) {
          for (int k = 0; k < iInnerDataNum; ++k) {
            // p_inData[(i*iNumChannels + l)*iInnerNum + k]
            const auto clKeyMarginTuple = TreeTraitsType::ComputeKeyAndSignedMargin(p_inData + ((i*iNumChannels + 0)*iInnerDataNum + k), 
              p_thresholds + (j*iNumDecisionsPerTree + 0), p_ordinals + (j*iNumDecisionsPerTree + 0), m_iTreeDepth, iInnerDataNum);

            const KeyType leafKey = std::get<0>(clKeyMarginTuple);
            const RealType margin = std::get<1>(clKeyMarginTuple); // Signed margin
            const KeyType treeIndex = std::get<2>(clKeyMarginTuple);

            const int iInputIndex = (int)p_ordinals[j*iNumDecisionsPerTree + treeIndex];
            const RealType sign = RealType((RealType(0) < margin) - (margin < RealType(0)));

            for (int m = 0; m < iInnerWeightsNum; ++m) {
              p_inDataGradient[(i*iNumChannels + iInputIndex)*iInnerDataNum + k] += sign * p_weights[(j*iNumLeavesPerTree + leafKey)*iInnerWeightsNum + m] * p_outDataGradient[((i*iNumTrees + j)*iInnerDataNum + k)*iInnerWeightsNum + m];
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
            const auto clKeyMarginTuple = TreeTraitsType::ComputeKeyAndSignedMargin(p_inData + ((i*iNumChannels + 0)*iInnerDataNum + k), 
              p_thresholds + (j*iNumDecisionsPerTree + 0), p_ordinals + (j*iNumDecisionsPerTree + 0), m_iTreeDepth, iInnerDataNum);

            const KeyType leafKey = std::get<0>(clKeyMarginTuple);
            const RealType margin = std::get<1>(clKeyMarginTuple); // Signed margin
            const KeyType treeIndex = std::get<2>(clKeyMarginTuple);

            const RealType sign = RealType((RealType(0) < margin) - (margin < RealType(0)));

            for (int m = 0; m < iInnerWeightsNum; ++m) {
              p_thresholdsGradient[j*iNumDecisionsPerTree + treeIndex] += -sign * p_weights[(j*iNumLeavesPerTree + leafKey)*iInnerWeightsNum + m] * p_outDataGradient[((i*iNumTrees + j)*iInnerDataNum + k)*iInnerWeightsNum + m];
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
            const auto clKeyMarginTuple = TreeTraitsType::ComputeKeyAndSignedMargin(p_inData + ((i*iNumChannels + 0)*iInnerDataNum + k), 
              p_thresholds + (j*iNumDecisionsPerTree + 0), p_ordinals + (j*iNumDecisionsPerTree + 0), m_iTreeDepth, iInnerDataNum);

            const KeyType leafKey = std::get<0>(clKeyMarginTuple);
            const RealType margin = std::get<1>(clKeyMarginTuple); // Signed margin

            for (int m = 0; m < iInnerWeightsNum; ++m) {
              p_weightsGradient[(j*iNumLeavesPerTree + leafKey)*iInnerWeightsNum + m] += std::abs(margin) * p_outDataGradient[((i*iNumTrees + j)*iInnerDataNum + k)*iInnerWeightsNum + m];
            }
          }
        }
      }
    }

  }

protected:
  RandomHingeForestTemplate() = default;

private:
  int m_iTreeDepth = 0;

  bool AssignTreeIndices() {
    bleakGetAndCheckInput(p_clOrdinals, "inOrdinals", false);
    bleakGetAndCheckInput(p_clInData, "inData", false);

    const ArrayType &clInData = p_clInData->GetData();
    ArrayType &clOrdinals = p_clOrdinals->GetData();

    const int iNumTrees = clOrdinals.GetSize()[0];
    const int iNumDecisionsPerTree = clOrdinals.GetSize()[1];
    const int iNumChannels = clInData.GetSize()[1];

    if (iNumDecisionsPerTree != TreeTraitsType::GetThresholdCount(m_iTreeDepth))
      return false;

    RealType * const p_ordinals = clOrdinals.data();

    std::uniform_int_distribution<int> clRandomFeature(0, iNumChannels-1);

    // Unroll loop for clarity
    for (int i = 0; i < iNumTrees; ++i) {
      for (int j = 0; j < iNumDecisionsPerTree; ++j) {
        p_ordinals[i*iNumDecisionsPerTree + j] = RealType(clRandomFeature(GetGenerator()));
      }

      if (m_iTreeDepth == iNumDecisionsPerTree) {
        // Fern decisions are invariant to ordering, so might as well optimize for memory access
        std::sort(p_ordinals + (i*m_iTreeDepth + 0), p_ordinals + ((i+1)*m_iTreeDepth + 0));
      }
    }

    return true;
  }
};

template<typename RealType>
class RandomHingeFerns : public RandomHingeForestTemplate<RealType, HingeFernCommon<RealType>> {
public:
  typedef RandomHingeForestTemplate<RealType, HingeFernCommon<RealType>> WorkAroundVarArgsType;

  bleakNewVertex(RandomHingeFerns, WorkAroundVarArgsType);
};

template<typename RealType>
class RandomHingeForest : public RandomHingeForestTemplate<RealType, HingeTreeCommon<RealType>> {
public:
  typedef RandomHingeForestTemplate<RealType, HingeTreeCommon<RealType>> WorkAroundVarArgsType;

  bleakNewVertex(RandomHingeForest, WorkAroundVarArgsType);
};

} // end namespace bleak

#endif // !BLEAK_RANDOMHINGEFOREST_H
