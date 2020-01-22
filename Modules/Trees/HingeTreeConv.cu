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

#include "cuda_runtime.h"
#include "HingeTreeConv.h"
#include "HingeTreeCommon.cuh"

namespace {

// From: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
#if __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

} // end anonymous namespace

namespace bleak {

namespace {

template<typename TreeTraitsTypeGPU, typename RealType>
__global__ void ForwardKernel(const RealType *d_matrix, const RealType *d_inThresholds, const RealType *d_inOrdinals, const RealType *d_inWeights, RealType *d_outData, 
    int iTreeDepth, int iThresholdStride, int iWeightsStride, int iInnerWeightsNum, int iNumTrees, int iRows, int iCols) {

  typedef typename TreeTraitsTypeGPU::KeyType KeyType;

  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  const int k = blockIdx.y * blockDim.y + threadIdx.y;

  if (j < iNumTrees && k < iRows) {
    const RealType * const d_thresholds = d_inThresholds + j*iThresholdStride;
    const RealType * const d_ordinals = d_inOrdinals + j*iThresholdStride;

    const RealType * const d_row = d_matrix + k*iCols;

    // leaf key, margin, ordinal index
    const double3 keyMarginTuple = TreeTraitsTypeGPU::ComputeKeyAndSignedMargin(d_row, d_thresholds, d_ordinals, iTreeDepth, 1);

    const KeyType key = KeyType(keyMarginTuple.x);
    const RealType signedMargin = RealType(keyMarginTuple.y);
    const RealType margin = std::abs(signedMargin);

    const RealType * const d_leafWeights = d_inWeights + (j*iWeightsStride + key)*iInnerWeightsNum;
    RealType * const d_out = d_outData + (j*iRows + k)*iInnerWeightsNum;

    for (int l = 0; l < iInnerWeightsNum; ++l)
      d_out[l] += d_leafWeights[l] * margin;
  }
}

template<typename TreeTraitsTypeGPU, typename RealType>
__global__ void BackwardThresholdsKernel(const RealType *d_matrix, const RealType *d_inThresholds, const RealType *d_inOrdinals, const RealType *d_inWeights, 
    const RealType *d_outDataGradient, RealType *d_inThresholdsGradient, int iTreeDepth, int iThresholdStride, int iWeightsStride, int iInnerWeightsNum, int iNumTrees, 
    int iRows, int iCols) {

  typedef typename TreeTraitsTypeGPU::KeyType KeyType;

  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  const int k = blockIdx.y * blockDim.y + threadIdx.y;

  if (j < iNumTrees && k < iRows) {
    const RealType * const d_thresholds = d_inThresholds + j*iThresholdStride;
    const RealType * const d_ordinals = d_inOrdinals + j*iThresholdStride;
    RealType * const d_thresholdsGradient = d_inThresholdsGradient + j*iThresholdStride;

    const RealType * const d_row = d_matrix + k*iCols;

    // leaf key, margin, ordinal index
    const double3 keyMarginTuple = TreeTraitsTypeGPU::ComputeKeyAndSignedMargin(d_row, d_thresholds, d_ordinals, iTreeDepth, 1);

    const KeyType key = KeyType(keyMarginTuple.x);
    const RealType signedMargin = RealType(keyMarginTuple.y);
    const KeyType thresholdIndex = KeyType(keyMarginTuple.z);

    const RealType sign = RealType((RealType(0) < signedMargin) - (signedMargin < RealType(0)));

    const RealType * const d_leafWeights = d_inWeights + (j*iWeightsStride + key)*iInnerWeightsNum;
    const RealType * const d_outGradient = d_outDataGradient + (j*iRows + k)*iInnerWeightsNum;

    for (int l = 0; l < iInnerWeightsNum; ++l)
      d_thresholdsGradient[thresholdIndex] += -sign * d_leafWeights[l] * d_outGradient[l];
  }
}

template<typename TreeTraitsTypeGPU, typename RealType>
__global__ void BackwardWeightsKernel(const RealType *d_matrix, const RealType *d_inThresholds, const RealType *d_inOrdinals, /*const RealType *d_inWeights,*/
    const RealType *d_outDataGradient, RealType *d_inWeightsGradient, int iTreeDepth, int iThresholdStride, int iWeightsStride, int iInnerWeightsNum, int iNumTrees, 
    int iRows, int iCols) {

  typedef typename TreeTraitsTypeGPU::KeyType KeyType;

  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  const int k = blockIdx.y * blockDim.y + threadIdx.y;

  if (j < iNumTrees && k < iRows) {
    const RealType * const d_thresholds = d_inThresholds + j*iThresholdStride;
    const RealType * const d_ordinals = d_inOrdinals + j*iThresholdStride;

    const RealType * const d_row = d_matrix + k*iCols;

    // leaf key, margin, ordinal index
    const double3 keyMarginTuple = TreeTraitsTypeGPU::ComputeKeyAndSignedMargin(d_row, d_thresholds, d_ordinals, iTreeDepth, 1);

    const KeyType key = KeyType(keyMarginTuple.x);
    const RealType signedMargin = RealType(keyMarginTuple.y);
    const RealType margin = std::abs(signedMargin);

    const RealType * const d_outGradient = d_outDataGradient + (j*iRows + k)*iInnerWeightsNum;
    RealType * const d_leafWeightsGradient = d_inWeightsGradient + (j*iWeightsStride + key)*iInnerWeightsNum;

    for (int l = 0; l < iInnerWeightsNum; ++l)
      d_leafWeightsGradient[l] += margin * d_outGradient[l];
  }
}

template<typename TreeTraitsTypeGPU, typename RealType>
__global__ void BackwardDataKernel(const RealType *d_matrix, const int *d_indexMatrix, const RealType *d_inThresholds, const RealType *d_inOrdinals, const RealType *d_inWeights, 
    const RealType *d_outDataGradient, RealType *d_inDataGradient, int iTreeDepth, int iThresholdStride, int iWeightsStride, int iInnerWeightsNum, int iNumTrees, 
    int iRows, int iCols) {

  typedef typename TreeTraitsTypeGPU::KeyType KeyType;

  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  const int k = blockIdx.y * blockDim.y + threadIdx.y;

  if (j < iNumTrees && k < iRows) {
    const RealType * const d_thresholds = d_inThresholds + j*iThresholdStride;
    const RealType * const d_ordinals = d_inOrdinals + j*iThresholdStride;

    const RealType * const d_row = d_matrix + k*iCols;
    const int * const d_iIndexRow = d_indexMatrix + k*iCols;

    // leaf key, margin, ordinal index
    const double3 keyMarginTuple = TreeTraitsTypeGPU::ComputeKeyAndSignedMargin(d_row, d_thresholds, d_ordinals, iTreeDepth, 1);

    const KeyType key = KeyType(keyMarginTuple.x);
    const RealType signedMargin = RealType(keyMarginTuple.y);
    const KeyType thresholdIndex = KeyType(keyMarginTuple.z);
    const int iFeatureIndex = (int)d_ordinals[thresholdIndex];
    const int iImageIndex = d_iIndexRow[iFeatureIndex];

    if (iImageIndex >= 0) {
      const RealType * const d_leafWeights = d_inWeights + (j*iWeightsStride + key)*iInnerWeightsNum;
      const RealType * const d_outGradient = d_outDataGradient + (j*iRows + k)*iInnerWeightsNum;

      const RealType sign = RealType((RealType(0) < signedMargin) - (signedMargin < RealType(0)));
      RealType tmpSum = RealType(0);

      for (int l = 0; l < iInnerWeightsNum; ++l)
        tmpSum += d_leafWeights[l] * d_outGradient[l];

      tmpSum *= sign;

      atomicAdd(d_inDataGradient + iImageIndex, tmpSum); // Do this just once
    }
  }
}

} // end anonymous namespace

template<typename RealType, unsigned int Dimension, typename TreeTraitsType>
void HingeTreeConvTemplate<RealType, Dimension, TreeTraitsType>::ForwardGPU() {
  typedef HingeTreeCommonGPU<TreeTraitsType> TreeTraitsTypeGPU;
  typedef typename TreeTraitsTypeGPU::KeyMarginTupleType KeyMarginTupleType;

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

  const RealType * const p_inData = clInData.data(GPU);
  const RealType * const p_inWeights = clInWeights.data(GPU);
  const RealType * const p_inOrdinals = clInOrdinals.data(GPU);
  const RealType * const p_inThresholds = clInThresholds.data(GPU);
  RealType * const p_outData = clOutData.data_no_sync(GPU);

  const int iOuterDataNum = clInData.GetSize()[0];
  const int iInnerDataNum = clInData.GetSize().Product(1);
  const int iInNumChannels = clInData.GetSize()[1];
  const int iInChannelSize = clInData.GetSize().Product(2);
  const int iNumTrees = clInWeights.GetSize()[0];
  const int iNumDecisionsPerTree = clInOrdinals.GetSize()[2];
  const int iNumLeavesPerTree = clInWeights.GetSize()[2];
  const int iInnerWeightsNum = clInWeights.GetSize().Product(3);
  //const int iOutDataImageSize = clOutData.GetSize().Product(2,2+GetDimension()); // Should be a synonym for m_iRows

  // Trees vs Patch Rows (m_iRows)
  const dim3 threadsPerBlock(16, 16);
  const dim3 numBlocks((iNumTrees + threadsPerBlock.x-1) / threadsPerBlock.x, (m_iRows + threadsPerBlock.y-1) / threadsPerBlock.y);

  //clOutData.Fill(RealType());
  cudaMemset(p_outData, 0, sizeof(RealType) * clOutData.GetSize().Count());

  const int iWeightsStride = iInNumChannels * iNumLeavesPerTree;
  const int iThresholdStride = iInNumChannels * iNumDecisionsPerTree;

  for (int i = 0; i < iOuterDataNum; ++i) {
    for (int c = 0; c < iInNumChannels; ++c) {
      m_clImageToMatrix.ExtractMatrixGPU(m_clMatrix.data_no_sync(GPU), p_inData + (i*iInNumChannels + c)*iInChannelSize, m_clIndexMatrix.data(GPU), clImageSize.data());

      ForwardKernel<TreeTraitsTypeGPU><<<numBlocks, threadsPerBlock>>>(m_clMatrix.data(GPU), 
        p_inThresholds + c*iNumDecisionsPerTree, p_inOrdinals + c*iNumDecisionsPerTree, p_inWeights + c*iNumLeavesPerTree*iInnerWeightsNum,
        p_outData + i*iNumTrees*m_iRows*iInnerWeightsNum, m_iTreeDepth, iThresholdStride, iWeightsStride, iInnerWeightsNum, iNumTrees, m_iRows, m_iCols);

      //// Iterate over output kernels
      //for (int j = 0; j < iNumTrees; ++j) {
      //  const RealType * const p_thresholds = p_inThresholds + (j*iInNumChannels + c)*iNumDecisionsPerTree;
      //  const RealType * const p_ordinals = p_inOrdinals + (j*iInNumChannels + c)*iNumDecisionsPerTree;
      //
      //  // Iterate over extracted patches
      //  for (int k = 0; k < m_iRows; ++k) {
      //    const RealType * const p_row = m_clMatrix.data() + k*m_iCols;
      //
      //    const auto clKeyMarginTuple = TreeTraitsType::ComputeKeyAndSignedMargin(p_row, p_thresholds, p_ordinals, m_iTreeDepth, 1);
      //
      //    const KeyType key = std::get<0>(clKeyMarginTuple);
      //    const RealType signedMargin = std::get<1>(clKeyMarginTuple);
      //    const RealType margin = std::abs(signedMargin);
      //
      //    const RealType * const p_leafWeights = p_inWeights + ((j*iInNumChannels + c)*iNumLeavesPerTree + key)*iInnerWeightsNum;
      //
      //    for (int l = 0; l < iInnerWeightsNum; ++l)
      //      p_outData[((i*iNumTrees + j)*iOutDataImageSize + k)*iInnerWeightsNum + l] += p_leafWeights[l]*margin;
      //  }
      //}
    }
  }
}

template<typename RealType, unsigned int Dimension, typename TreeTraitsType>
void HingeTreeConvTemplate<RealType, Dimension, TreeTraitsType>::BackwardGPU() {
  typedef HingeTreeCommonGPU<TreeTraitsType> TreeTraitsTypeGPU;
  typedef typename TreeTraitsTypeGPU::KeyMarginTupleType KeyMarginTupleType;

  bleakGetAndCheckInput(p_clInData, "inData");
  bleakGetAndCheckInput(p_clInWeights, "inWeights");
  bleakGetAndCheckInput(p_clInThresholds, "inThresholds");
  bleakGetAndCheckInput(p_clInOrdinals, "inOrdinals");
  bleakGetAndCheckOutput(p_clOutData, "outData");

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

  const RealType * const p_inData = clInData.data(GPU);
  RealType * const p_inDataGradient = clInDataGradient.data(GPU);
  const RealType * const p_inWeights = clInWeights.data(GPU);
  RealType * const p_inWeightsGradient = clInWeightsGradient.data(GPU);
  const RealType * const p_inThresholds = clInThresholds.data(GPU);
  RealType * const p_inThresholdsGradient = clInThresholdsGradient.data(GPU);
  const RealType * const p_inOrdinals = clInOrdinals.data(GPU);
  //const RealType * const p_outData = clOutData.data();
  const RealType * const p_outDataGradient = clOutDataGradient.data(GPU);

  const int iOuterDataNum = clInData.GetSize()[0];
  const int iInnerDataNum = clInData.GetSize().Product(1);
  const int iInNumChannels = clInData.GetSize()[1];
  const int iInChannelSize = clInData.GetSize().Product(2);
  const int iNumTrees = clInWeights.GetSize()[0];
  const int iNumDecisionsPerTree = clInOrdinals.GetSize()[2];
  const int iNumLeavesPerTree = clInWeights.GetSize()[2];
  const int iInnerWeightsNum = clInWeights.GetSize().Product(3);
  const int iOutDataImageSize = clOutData.GetSize().Product(2,2+GetDimension());

  // Trees vs Patch Rows (m_iRows)
  const dim3 threadsPerBlock(16, 16);
  const dim3 numBlocks((iNumTrees + threadsPerBlock.x-1) / threadsPerBlock.x, (m_iRows + threadsPerBlock.y-1) / threadsPerBlock.y);

  const int iWeightsStride = iInNumChannels * iNumLeavesPerTree;
  const int iThresholdStride = iInNumChannels * iNumDecisionsPerTree;

  if (p_inThresholdsGradient != nullptr) {
    for (int i = 0; i < iOuterDataNum; ++i) {
      for (int c = 0; c < iInNumChannels; ++c) {
        m_clImageToMatrix.ExtractMatrixGPU(m_clMatrix.data_no_sync(GPU), p_inData + (i*iInNumChannels + c)*iInChannelSize, m_clIndexMatrix.data(GPU), clImageSize.data());

        BackwardThresholdsKernel<TreeTraitsTypeGPU><<<numBlocks, threadsPerBlock>>>(m_clMatrix.data(GPU), 
          p_inThresholds + c*iNumDecisionsPerTree, p_inOrdinals + c*iNumDecisionsPerTree, p_inWeights + c*iNumLeavesPerTree*iInnerWeightsNum,
          p_outDataGradient + i*iNumTrees*m_iRows*iInnerWeightsNum, p_inThresholdsGradient + c*iNumDecisionsPerTree, m_iTreeDepth, iThresholdStride, 
          iWeightsStride, iInnerWeightsNum, iNumTrees, m_iRows, m_iCols);
      }
    }
  }

  if (p_inWeightsGradient != nullptr) {
    for (int i = 0; i < iOuterDataNum; ++i) {
      for (int c = 0; c < iInNumChannels; ++c) {
        m_clImageToMatrix.ExtractMatrixGPU(m_clMatrix.data_no_sync(GPU), p_inData + (i*iInNumChannels + c)*iInChannelSize, m_clIndexMatrix.data(GPU), clImageSize.data());

        BackwardWeightsKernel<TreeTraitsTypeGPU><<<numBlocks, threadsPerBlock>>>(m_clMatrix.data(GPU), 
          p_inThresholds + c*iNumDecisionsPerTree, p_inOrdinals + c*iNumDecisionsPerTree, /*p_inWeights + c*iNumLeavesPerTree*iInnerWeightsNum,*/
          p_outDataGradient + i*iNumTrees*m_iRows*iInnerWeightsNum, p_inWeightsGradient + c*iNumLeavesPerTree*iInnerWeightsNum, m_iTreeDepth, iThresholdStride, 
          iWeightsStride, iInnerWeightsNum, iNumTrees, m_iRows, m_iCols);
      }
    }
  }

  if (p_inDataGradient != nullptr) {
    for (int i = 0; i < iOuterDataNum; ++i) {
      for (int c = 0; c < iInNumChannels; ++c) {
        m_clImageToMatrix.ExtractMatrix(m_clMatrix.data_no_sync(GPU), p_inData + (i*iInNumChannels + c)*iInChannelSize, m_clIndexMatrix.data(GPU), clImageSize.data());

        BackwardDataKernel<TreeTraitsTypeGPU><<<numBlocks, threadsPerBlock>>>(m_clMatrix.data(GPU), m_clIndexMatrix.data(GPU),
          p_inThresholds + c*iNumDecisionsPerTree, p_inOrdinals + c*iNumDecisionsPerTree, p_inWeights + c*iNumLeavesPerTree*iInnerWeightsNum,
          p_outDataGradient + i*iNumTrees*m_iRows*iInnerWeightsNum, p_inDataGradient + (i*iInNumChannels + c)*iInChannelSize, m_iTreeDepth, iThresholdStride, 
          iWeightsStride, iInnerWeightsNum, iNumTrees, m_iRows, m_iCols);        
      }
    }
  }
}

template class HingeTreeConvTemplate<float, 1, HingeFernCommon<float>>;
template class HingeTreeConvTemplate<double, 1, HingeFernCommon<double>>;

template class HingeTreeConvTemplate<float, 2, HingeFernCommon<float>>;
template class HingeTreeConvTemplate<double, 2, HingeFernCommon<double>>;

template class HingeTreeConvTemplate<float, 3, HingeFernCommon<float>>;
template class HingeTreeConvTemplate<double, 3, HingeFernCommon<double>>;

template class HingeTreeConvTemplate<float, 1, HingeTreeCommon<float>>;
template class HingeTreeConvTemplate<double, 1, HingeTreeCommon<double>>;

template class HingeTreeConvTemplate<float, 2, HingeTreeCommon<float>>;
template class HingeTreeConvTemplate<double, 2, HingeTreeCommon<double>>;

template class HingeTreeConvTemplate<float, 3, HingeTreeCommon<float>>;
template class HingeTreeConvTemplate<double, 3, HingeTreeCommon<double>>;

} // end namespace bleak
