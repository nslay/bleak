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

#ifndef BLEAK_HINGETREECONV_H
#define BLEAK_HINGETREECONV_H

#include <cstdint>
#include <climits>
#include <random>
#include <utility>
#include "Common.h"
#include "Vertex.h"
#include "ImageToMatrix.h"
#include "HingeTreeCommon.h"

namespace bleak {

template<typename RealType, unsigned int Dimension, typename TreeTraitsType>
class HingeTreeConvTemplate : public Vertex<RealType> {
public:
  bleakNewAbstractVertex(HingeTreeConvTemplate, Vertex<RealType>,
    bleakAddInput("inData"),
    bleakAddInput("inWeights"),
    bleakAddInput("inOrdinals"),
    bleakAddInput("inThresholds"),
    bleakAddOutput("outData"),
    bleakAddProperty("padding", m_vPadding),
    bleakAddProperty("stride", m_vStride),
    bleakAddProperty("dilate", m_vDilate),
    bleakAddProperty("kernelSize", m_vKernelSize));

  bleakForwardVertexTypedefs();

  typedef ImageToMatrix<RealType, Dimension> ImageToMatrixType;
  typedef typename TreeTraitsType::KeyType KeyType;

  static constexpr unsigned int GetDimension() {
    return Dimension;
  }

  virtual ~HingeTreeConvTemplate() = default;

  virtual bool SetSizes() override {
    bleakGetAndCheckInput(p_clInData, "inData", false);
    bleakGetAndCheckInput(p_clInWeights, "inWeights", false);
    bleakGetAndCheckInput(p_clInThresholds, "inThresholds", false);
    bleakGetAndCheckInput(p_clInOrdinals, "inOrdinals", false);
    bleakGetAndCheckOutput(p_clOutData, "outData", false);

    if (m_vPadding.size() != GetDimension() || m_vStride.size() != GetDimension() || m_vDilate.size() != GetDimension() || m_vKernelSize.size() != GetDimension()) {
      std::cerr << GetName() << ": Error: padding, stride, dilate, kernelSize properties are expected to be " << GetDimension() << "D." << std::endl;
      return false;
    }

    if (p_clInOrdinals->GetGradient().GetSize().Valid()) {
      std::cerr << GetName() << ": Error: inOrdinals appears learnable. However, HingeTree/FernConv has its own custom updates for this input." << std::endl;
      return false;
    }

    const ArrayType &clInData = p_clInData->GetData(); // batchSize x numInputChannels x ...
    const ArrayType &clInWeights = p_clInWeights->GetData(); // numTrees x numInputChannels x numLeavesPerTree x ...
    const ArrayType &clInThresholds = p_clInThresholds->GetData(); // numTrees x numInputChannels x treeDepth
    const ArrayType &clInOrdinals = p_clInOrdinals->GetData(); // numTrees x x numInputChannels x treeDepth

    if (!clInData.GetSize().Valid() || !clInWeights.GetSize().Valid() || !clInThresholds.GetSize().Valid() || !clInOrdinals.GetSize().Valid()) {
      std::cerr << GetName() << ": Error: Invalid dimensions for inData, inWeights, inThresholds and/or inOrdinals." << std::endl;
      return false;
    }

    if (clInWeights.GetSize().GetDimension() < 3 || clInData.GetSize().GetDimension() != GetDimension()+2 || clInThresholds.GetSize().GetDimension() != 3 || clInOrdinals.GetSize().GetDimension() != 3) {
      std::cerr << GetName() << ": Error: Unexpected dimension for inData, inWeights, inThresholds and/or inOrdinals." << std::endl;
      return false;
    }

    m_iGroups = clInData.GetSize()[1] / clInWeights.GetSize()[1];

    if (clInWeights.GetSize()[1]*m_iGroups != clInData.GetSize()[1]) {
      std::cerr << GetName() << ": Error: inWeights channels size must divide inData channels size (" << clInWeights.GetSize()[1] << " does not divide " << clInData.GetSize()[1] << ")." << std::endl;
      return false;
    }

    if (m_iGroups != 1)
      std::cout << GetName() << ": Info: Using group convolution (groups = " << m_iGroups << ")." << std::endl;

    if ((clInWeights.GetSize()[0] % m_iGroups) != 0) {
      std::cerr << GetName() << ": Error: Groups must divide output channels (" << m_iGroups << " does not divide " << clInWeights.GetSize()[0] << ")." << std::endl;
      return false;
    }

    if (clInThresholds.GetSize() != clInOrdinals.GetSize() || clInWeights.GetSize()[0] != clInThresholds.GetSize()[0]) {
      std::cerr << GetName() << ": Error: Dimension mismatch between inThresholds, inOrdinals and/or inWeights." << std::endl;
      return false;
    }

    Size clImageSize = clInData.GetSize().SubSize(1);
    clImageSize[0] = 1; // Process one channel at a time

    //m_clImageToMatrix.SetImageSize(p_clInData->GetData().GetSize().data()+1);
    m_clImageToMatrix.SetKernelSize(m_vKernelSize.data());
    m_clImageToMatrix.SetStride(m_vStride.data());
    m_clImageToMatrix.SetPadding(m_vPadding.data());
    m_clImageToMatrix.SetDilate(m_vDilate.data());

    if (!m_clImageToMatrix.Good(clImageSize.data())) {
      std::cerr << GetName() << ": Error: Invalid convolution parameters (image size/kernel size/padding?)." << std::endl;
      return false;
    }

    m_iTreeDepth = TreeTraitsType::ComputeDepth(clInWeights.GetSize()[2]);
    if (m_iTreeDepth <= 0) {
      std::cerr << GetName() << ": Error: Number of weights is expected to be a power of 2." << std::endl;
      return false;
    }

    if (m_iTreeDepth > TreeTraitsType::GetMaxDepth()) {
      std::cerr << GetName() << ": Error: Tree depth exceeds maximum supported depth (" << TreeTraitsType::GetMaxDepth() << ")." << std::endl;
      return false;
    }

    if (clInThresholds.GetSize()[2] != TreeTraitsType::GetThresholdCount(m_iTreeDepth)) {
      std::cerr << GetName() << ": Error: Tree depth and number of leaves are incompatible." << std::endl;
      return false;
    }

    Size clOutSize(2 + GetDimension() + clInWeights.GetSize().GetDimension()-3); // Outputting more than 1 value per window!?

    clOutSize[0] = clInData.GetSize()[0]; // Batch size
    clOutSize[1] = clInWeights.GetSize()[0]; // Number of trees

    const auto tmpSize = m_clImageToMatrix.ComputeOutputSize(clImageSize.data());
    std::copy(tmpSize.begin(), tmpSize.end(), clOutSize.data()+2);

    std::copy_n(clInWeights.GetSize().data()+3, clInWeights.GetSize().GetDimension()-3, clOutSize.data()+2+GetDimension());

    p_clOutData->GetData().SetSize(clOutSize);

    if (p_clInData->GetGradient().GetSize().Valid() || p_clInWeights->GetGradient().GetSize().Valid() || p_clInThresholds->GetGradient().GetSize().Valid())
      p_clOutData->GetGradient().SetSize(clOutSize);
    else
      p_clOutData->GetGradient().Clear();

    return true;
  }

  virtual bool Initialize() override {
    bleakGetAndCheckInput(p_clInData, "inData", false);
    bleakGetAndCheckInput(p_clInWeights, "inWeights", false);
    bleakGetAndCheckInput(p_clInOrdinals, "inOrdinals", false);

    Size clImageSize = p_clInData->GetData().GetSize().SubSize(1);
    clImageSize[0] = 1; // Process 1 channel at a time

    if (!m_clImageToMatrix.Good(clImageSize.data()))
      return false;

    m_iTreeDepth = TreeTraitsType::ComputeDepth(p_clInWeights->GetData().GetSize()[2]);

    if (m_iTreeDepth < 0)
      return false;

    m_clImageToMatrix.ComputeMatrixDimensions(m_iRows, m_iCols, clImageSize.data());

    Size clMatrixSize = { m_iRows, m_iCols };

    m_clMatrix.SetSize(clMatrixSize);
    m_clMatrix.Allocate();

    m_clIndexMatrix.SetSize(clMatrixSize);
    m_clIndexMatrix.Allocate();

    m_clImageToMatrix.ExtractIndexMatrix(m_clIndexMatrix.data_no_sync(), clImageSize.data()); // This never changes

    return AssignTreeIndices();
  }

  virtual void ForwardCPU() override {
    bleakGetAndCheckInput(p_clInData, "inData");
    bleakGetAndCheckInput(p_clInWeights, "inWeights");
    bleakGetAndCheckInput(p_clInThresholds, "inThresholds");
    bleakGetAndCheckInput(p_clInOrdinals, "inOrdinals");
    bleakGetAndCheckOutput(p_clOutData, "outData");

    const ArrayType &clInData = p_clInData->GetData();
    const ArrayType &clInWeights = p_clInWeights->GetData();
    const ArrayType &clInOrdinals = p_clInOrdinals->GetData();
    const ArrayType &clInThresholds = p_clInThresholds->GetData();
    ArrayType &clOutData = p_clOutData->GetData();

    Size clImageSize = p_clInData->GetData().GetSize().SubSize(1);
    clImageSize[0] = 1; // Process 1 channel at a time

    const RealType * const p_inData = clInData.data();
    const RealType * const p_inWeights = clInWeights.data();
    const RealType * const p_inOrdinals = clInOrdinals.data();
    const RealType * const p_inThresholds = clInThresholds.data();
    RealType * const p_outData = clOutData.data_no_sync();

    const int iOuterDataNum = clInData.GetSize()[0];
    const int iInnerDataNum = clInData.GetSize().Product(1) / m_iGroups;
    const int iInNumChannels = clInData.GetSize()[1] / m_iGroups;
    const int iInChannelSize = clInData.GetSize().Product(2);
    const int iNumTrees = clInWeights.GetSize()[0] / m_iGroups;
    const int iNumDecisionsPerTree = clInOrdinals.GetSize()[2];
    const int iNumLeavesPerTree = clInWeights.GetSize()[2];
    const int iInnerWeightsNum = clInWeights.GetSize().Product(3);
    const int iOutDataImageSize = clOutData.GetSize().Product(2,2+GetDimension());

    clOutData.Fill(RealType());

    for (int i = 0; i < iOuterDataNum; ++i) {
      for (int g = 0; g < m_iGroups; ++g) {
        for (int c = 0; c < iInNumChannels; ++c) {
          m_clImageToMatrix.ExtractMatrix(m_clMatrix.data_no_sync(), p_inData + ((i*m_iGroups + g)*iInNumChannels + c)*iInChannelSize, m_clIndexMatrix.data(), clImageSize.data());

          // Iterate over output kernels
#pragma omp parallel for
          for (int j = 0; j < iNumTrees; ++j) {
            const RealType * const p_thresholds = p_inThresholds + ((g*iNumTrees + j)*iInNumChannels + c)*iNumDecisionsPerTree;
            const RealType * const p_ordinals = p_inOrdinals + ((g*iNumTrees + j)*iInNumChannels + c)*iNumDecisionsPerTree;

            // Iterate over extracted patches
            for (int k = 0; k < m_iRows; ++k) {
              const RealType * const p_row = m_clMatrix.data() + k*m_iCols;

              const auto clKeyMarginTuple = TreeTraitsType::ComputeKeyAndSignedMargin(p_row, p_thresholds, p_ordinals, m_iTreeDepth, 1);

              const KeyType key = std::get<0>(clKeyMarginTuple);
              const RealType signedMargin = std::get<1>(clKeyMarginTuple);
              const RealType margin = std::abs(signedMargin);

              const RealType * const p_leafWeights = p_inWeights + (((g*iNumTrees + j)*iInNumChannels + c)*iNumLeavesPerTree + key)*iInnerWeightsNum;

              for (int l = 0; l < iInnerWeightsNum; ++l) {
                //p_outData[((i*iNumTrees + j)*iOutDataImageSize + k)*iInnerWeightsNum + l] += p_leafWeights[l]*margin;
                p_outData[(((i*m_iGroups + g)*iNumTrees + j)*iOutDataImageSize + k)*iInnerWeightsNum + l] += p_leafWeights[l]*margin;
              }
            }
          }
        }
      }
    }
  }

  virtual void BackwardCPU() override {
    bleakGetAndCheckInput(p_clInData, "inData");
    bleakGetAndCheckInput(p_clInWeights, "inWeights");
    bleakGetAndCheckInput(p_clInThresholds, "inThresholds");
    bleakGetAndCheckInput(p_clInOrdinals, "inOrdinals");
    bleakGetAndCheckOutput(p_clOutData, "outData");

    if (!p_clOutData->GetGradient().Valid())
      return; // Nothing to do

    const ArrayType &clInData = p_clInData->GetData();
    ArrayType &clInDataGradient = p_clInData->GetGradient();
    const ArrayType &clInWeights = p_clInWeights->GetData();
    ArrayType &clInWeightsGradient = p_clInWeights->GetGradient();
    const ArrayType &clInThresholds = p_clInThresholds->GetData();
    ArrayType &clInThresholdsGradient = p_clInThresholds->GetGradient();
    const ArrayType &clInOrdinals = p_clInOrdinals->GetData();
    const ArrayType &clOutData = p_clOutData->GetData();
    const ArrayType &clOutDataGradient = p_clOutData->GetGradient();

    Size clImageSize = p_clInData->GetData().GetSize().SubSize(1);
    clImageSize[0] = 1; // Process 1 channel at a time

    const RealType * const p_inData = clInData.data();
    RealType * const p_inDataGradient = clInDataGradient.data();
    const RealType * const p_inWeights = clInWeights.data();
    RealType * const p_inWeightsGradient = clInWeightsGradient.data();
    const RealType * const p_inThresholds = clInThresholds.data();
    RealType * const p_inThresholdsGradient = clInThresholdsGradient.data();
    const RealType * const p_inOrdinals = clInOrdinals.data();
    const RealType * const p_outData = clOutData.data();
    const RealType * const p_outDataGradient = clOutDataGradient.data();

    const int iOuterDataNum = clInData.GetSize()[0];
    const int iInnerDataNum = clInData.GetSize().Product(1) / m_iGroups;
    const int iInNumChannels = clInData.GetSize()[1] / m_iGroups;
    const int iInChannelSize = clInData.GetSize().Product(2);
    const int iNumTrees = clInWeights.GetSize()[0] / m_iGroups;
    const int iNumDecisionsPerTree = clInOrdinals.GetSize()[2];
    const int iNumLeavesPerTree = clInWeights.GetSize()[2];
    const int iInnerWeightsNum = clInWeights.GetSize().Product(3);
    const int iOutDataImageSize = clOutData.GetSize().Product(2,2+GetDimension());

    if (p_inThresholdsGradient != nullptr) {
      for (int i = 0; i < iOuterDataNum; ++i) {
        for (int g = 0; g < m_iGroups; ++g) {
          for (int c = 0; c < iInNumChannels; ++c) {
            m_clImageToMatrix.ExtractMatrix(m_clMatrix.data_no_sync(), p_inData + ((i*m_iGroups + g)*iInNumChannels + c)*iInChannelSize, m_clIndexMatrix.data(), clImageSize.data());

#pragma omp parallel for
            for (int j = 0; j < iNumTrees; ++j) {
              const RealType * const p_thresholds = p_inThresholds + ((g*iNumTrees + j)*iInNumChannels + c)*iNumDecisionsPerTree;
              const RealType * const p_ordinals = p_inOrdinals + ((g*iNumTrees + j)*iInNumChannels + c)*iNumDecisionsPerTree;

              for (int k = 0; k < m_iRows; ++k) {
                const RealType * const p_row = m_clMatrix.data() + k*m_iCols;

                const auto clKeyMarginTuple = TreeTraitsType::ComputeKeyAndSignedMargin(p_row, p_thresholds, p_ordinals, m_iTreeDepth, 1);

                const KeyType key = std::get<0>(clKeyMarginTuple);
                const RealType signedMargin = std::get<1>(clKeyMarginTuple);
                const int iThresholdIndex = std::get<2>(clKeyMarginTuple);

                const RealType sign = RealType((RealType(0) < signedMargin) - (signedMargin < RealType(0)));

                const RealType * const p_leafWeights = p_inWeights + (((g*iNumTrees + j)*iInNumChannels + c)*iNumLeavesPerTree + key)*iInnerWeightsNum;

                for (int l = 0; l < iInnerWeightsNum; ++l)
                  p_inThresholdsGradient[((g*iNumTrees + j)*iInNumChannels + c)*iNumDecisionsPerTree + iThresholdIndex] += -sign * p_leafWeights[l] * p_outDataGradient[(((i*m_iGroups + g)*iNumTrees + j)*iOutDataImageSize + k)*iInnerWeightsNum + l];
              }
            }
          }
        }
      }
    }

    if (p_inWeightsGradient != nullptr) {
      for (int i = 0; i < iOuterDataNum; ++i) {
        for (int g = 0; g < m_iGroups; ++g) {
          for (int c = 0; c < iInNumChannels; ++c) {
            m_clImageToMatrix.ExtractMatrix(m_clMatrix.data_no_sync(), p_inData + ((i*m_iGroups + g)*iInNumChannels + c)*iInChannelSize, m_clIndexMatrix.data(), clImageSize.data());

#pragma omp parallel for
            for (int j = 0; j < iNumTrees; ++j) {
              const RealType * const p_thresholds = p_inThresholds + ((g*iNumTrees + j)*iInNumChannels + c)*iNumDecisionsPerTree;
              const RealType * const p_ordinals = p_inOrdinals + ((g*iNumTrees + j)*iInNumChannels + c)*iNumDecisionsPerTree;

              for (int k = 0; k < m_iRows; ++k) {
                const RealType * const p_row = m_clMatrix.data() + k*m_iCols;

                const auto clKeyMarginTuple = TreeTraitsType::ComputeKeyAndSignedMargin(p_row, p_thresholds, p_ordinals, m_iTreeDepth, 1);

                const KeyType key = std::get<0>(clKeyMarginTuple);
                const RealType signedMargin = std::get<1>(clKeyMarginTuple);
                const int iThresholdIndex = std::get<2>(clKeyMarginTuple);

                const RealType margin = std::abs(signedMargin);

                for (int l = 0; l < iInnerWeightsNum; ++l)
                  p_inWeightsGradient[(((g*iNumTrees + j)*iInNumChannels + c)*iNumLeavesPerTree + key)*iInnerWeightsNum + l] += margin * p_outDataGradient[(((i*m_iGroups + g)*iNumTrees + j)*iOutDataImageSize + k)*iInnerWeightsNum + l];
              }
            }
          }
        }
      }
    }

    if (p_inDataGradient != nullptr) {
      for (int i = 0; i < iOuterDataNum; ++i) {
        for (int g = 0; g < m_iGroups; ++g) {
          for (int c = 0; c < iInNumChannels; ++c) {
            m_clImageToMatrix.ExtractMatrix(m_clMatrix.data_no_sync(), p_inData + ((i*m_iGroups + g)*iInNumChannels + c)*iInChannelSize, m_clIndexMatrix.data(), clImageSize.data());

#pragma omp parallel for
            for (int j = 0; j < iNumTrees; ++j) {
              const RealType * const p_thresholds = p_inThresholds + ((g*iNumTrees + j)*iInNumChannels + c)*iNumDecisionsPerTree;
              const RealType * const p_ordinals = p_inOrdinals + ((g*iNumTrees + j)*iInNumChannels + c)*iNumDecisionsPerTree;

              for (int k = 0; k < m_iRows; ++k) {
                const RealType * const p_row = m_clMatrix.data() + k*m_iCols;
                const int * const p_iIndexRow = m_clIndexMatrix.data() + k*m_iCols;

                const auto clKeyMarginTuple = TreeTraitsType::ComputeKeyAndSignedMargin(p_row, p_thresholds, p_ordinals, m_iTreeDepth, 1);

                const KeyType key = std::get<0>(clKeyMarginTuple);
                const RealType signedMargin = std::get<1>(clKeyMarginTuple);
                const int iThresholdIndex = std::get<2>(clKeyMarginTuple);
                const int iFeatureIndex = (int)p_ordinals[iThresholdIndex];
                const int iImageIndex = p_iIndexRow[iFeatureIndex];

                if (iImageIndex >= 0) { // Check if it's not padding!
                  const RealType sign = RealType((RealType(0) < signedMargin) - (signedMargin < RealType(0)));

                  const RealType * const p_leafWeights = p_inWeights + (((g*iNumTrees + j)*iInNumChannels + c)*iNumLeavesPerTree + key)*iInnerWeightsNum;

                  for (int l = 0; l < iInnerWeightsNum; ++l) {
#pragma omp atomic
                    p_inDataGradient[((i*m_iGroups + g)*iInNumChannels + c)*iInChannelSize + iImageIndex] += sign * p_leafWeights[l] * p_outDataGradient[(((i*m_iGroups + g)*iNumTrees + j)*iOutDataImageSize + k)*iInnerWeightsNum + l];
                  }
                }
              }
            }
          }
        }
      }
    }
  }

#ifdef BLEAK_USE_CUDA
  virtual void ForwardGPU() override;
  virtual void BackwardGPU() override;
#endif // BLEAK_USE_CUDA

  virtual bool LoadFromDatabase(const std::unique_ptr<Cursor> &p_clCursor) override {
    if (!SuperType::LoadFromDatabase(p_clCursor))
      return false;

    bleakGetAndCheckInput(p_clInOrdinals, "inOrdinals", false);

    const int iKernelSize = m_iCols; // With channels

    const ArrayType &clInOrdinals = p_clInOrdinals->GetData();

    for (const RealType &value : clInOrdinals) {
      if (value < RealType(0) || value >= RealType(iKernelSize)) {
        std::cerr << GetName() << ": Error: Invalid ordinal index found." << std::endl;
        return false;
      }
    }

    return true;
  }

protected:
  HingeTreeConvTemplate()
  : m_vPadding(GetDimension(), 0), m_vStride(GetDimension(), 1), m_vDilate(GetDimension(), 1) { }

private:
  std::vector<int> m_vPadding;
  std::vector<int> m_vStride;
  std::vector<int> m_vDilate;
  std::vector<int> m_vKernelSize;
  int m_iGroups = 1;

  ImageToMatrixType m_clImageToMatrix;

  int m_iRows = 0;
  int m_iCols = 0;
  int m_iTreeDepth = 0;

  Array<RealType> m_clMatrix;
  Array<int> m_clIndexMatrix;

  bool AssignTreeIndices() {
    bleakGetAndCheckInput(p_clInOrdinals, "inOrdinals", false);

    ArrayType &clInOrdinals = p_clInOrdinals->GetData();

    const int iKernelSize = m_iCols; 
    //const int iNumTrees = clInOrdinals.GetSize()[0];
    //const int iTreeDepth = clInOrdinals.GetSize()[1];

    //std::cout << GetName() << ": iKernelSize = " << iKernelSize << std::endl;

    std::uniform_int_distribution<int> clUniform(0, iKernelSize-1);

    std::generate(clInOrdinals.begin(), clInOrdinals.end(), 
      [&clUniform]() -> RealType {
        return RealType(clUniform(GetGenerator()));
      });

    return true;
  }
};

template<typename RealType>
class HingeFernConv3D : public HingeTreeConvTemplate<RealType, 3, HingeFernCommon<RealType>> {
public:
  typedef HingeTreeConvTemplate<RealType, 3, HingeFernCommon<RealType>> WorkAroundVarArgsType;

  bleakNewVertex(HingeFernConv3D, WorkAroundVarArgsType);
};

template<typename RealType>
class HingeFernConv2D : public HingeTreeConvTemplate<RealType, 2, HingeFernCommon<RealType>> {
public:
  typedef HingeTreeConvTemplate<RealType, 2, HingeFernCommon<RealType>> WorkAroundVarArgsType;

  bleakNewVertex(HingeFernConv2D, WorkAroundVarArgsType);
};

template<typename RealType>
class HingeFernConv1D : public HingeTreeConvTemplate<RealType, 1, HingeFernCommon<RealType>> {
public:
  typedef HingeTreeConvTemplate<RealType, 1, HingeFernCommon<RealType>> WorkAroundVarArgsType;

  bleakNewVertex(HingeFernConv1D, WorkAroundVarArgsType);
};

template<typename RealType>
class HingeTreeConv3D : public HingeTreeConvTemplate<RealType, 3, HingeTreeCommon<RealType>> {
public:
  typedef HingeTreeConvTemplate<RealType, 3, HingeTreeCommon<RealType>> WorkAroundVarArgsType;

  bleakNewVertex(HingeTreeConv3D, WorkAroundVarArgsType);
};

template<typename RealType>
class HingeTreeConv2D : public HingeTreeConvTemplate<RealType, 2, HingeTreeCommon<RealType>> {
public:
  typedef HingeTreeConvTemplate<RealType, 2, HingeTreeCommon<RealType>> WorkAroundVarArgsType;

  bleakNewVertex(HingeTreeConv2D, WorkAroundVarArgsType);
};

template<typename RealType>
class HingeTreeConv1D : public HingeTreeConvTemplate<RealType, 1, HingeTreeCommon<RealType>> {
public:
  typedef HingeTreeConvTemplate<RealType, 1, HingeTreeCommon<RealType>> WorkAroundVarArgsType;

  bleakNewVertex(HingeTreeConv1D, WorkAroundVarArgsType);
};

} // end namespace bleak

#endif // !BLEAK_HINGETREECONV_H
