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

#include "RandomHingeForest.h"
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
__global__ void ForwardKernel(const RealType *d_inData, const RealType *d_inThresholds, const RealType *d_inOrdinals, const RealType *d_inWeights, RealType *d_outData, 
    int iTreeDepth, int iThresholdStride, int iWeightsStride, int iInnerWeightsNum, int iNumTrees, int iOuterNum, int iNumChannels, int iInnerDataNum) {

  typedef typename TreeTraitsTypeGPU::KeyType KeyType;

  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i < iOuterNum && j < iNumTrees && k < iInnerDataNum) {
    const RealType * const d_thresholds = d_inThresholds + j*iThresholdStride;
    const RealType * const d_ordinals = d_inOrdinals + j*iThresholdStride;

    const RealType * const d_row = d_inData + ((i*iNumChannels + 0)*iInnerDataNum + k);

    // leaf key, margin, ordinal index
    const auto keyMarginTuple = TreeTraitsTypeGPU::ComputeKeyAndSignedMargin(d_row, d_thresholds, d_ordinals, iTreeDepth, iInnerDataNum);

    const KeyType key = keyMarginTuple.leafKey;
    const RealType signedMargin = keyMarginTuple.signedMargin;
    const RealType margin = std::abs(signedMargin);

    const RealType * const d_leafWeights = d_inWeights + (j*iWeightsStride + key)*iInnerWeightsNum;
    RealType * const d_out = d_outData + ((i*iNumTrees + j)*iInnerDataNum + k)*iInnerWeightsNum;

    for (int l = 0; l < iInnerWeightsNum; ++l)
      d_out[l] = d_leafWeights[l] * margin;
  }
}

template<typename TreeTraitsTypeGPU, typename RealType>
__global__ void BackwardThresholdsKernel(const RealType *d_inData, const RealType *d_inThresholds, const RealType *d_inOrdinals, const RealType *d_inWeights, 
    const RealType *d_outDataGradient, RealType *d_inThresholdsGradient, int iTreeDepth, int iThresholdStride, int iWeightsStride, int iInnerWeightsNum, int iNumTrees, 
    int iOuterNum, int iNumChannels, int iInnerDataNum) {

  typedef typename TreeTraitsTypeGPU::KeyType KeyType;

  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i < iOuterNum && j < iNumTrees && k < iInnerDataNum) {
    const RealType * const d_thresholds = d_inThresholds + j*iThresholdStride;
    const RealType * const d_ordinals = d_inOrdinals + j*iThresholdStride;
    RealType * const d_thresholdsGradient = d_inThresholdsGradient + j*iThresholdStride;

    const RealType * const d_row = d_inData + ((i*iNumChannels + 0)*iInnerDataNum + k);

    // leaf key, margin, ordinal index
    const auto keyMarginTuple = TreeTraitsTypeGPU::ComputeKeyAndSignedMargin(d_row, d_thresholds, d_ordinals, iTreeDepth, iInnerDataNum);

    const KeyType key = keyMarginTuple.leafKey;
    const RealType signedMargin = keyMarginTuple.signedMargin;
    const KeyType thresholdIndex = keyMarginTuple.thresholdIndex;

    const RealType sign = RealType((RealType(0) < signedMargin) - (signedMargin < RealType(0)));

    const RealType * const d_leafWeights = d_inWeights + (j*iWeightsStride + key)*iInnerWeightsNum;
    const RealType * const d_outGradient = d_outDataGradient + ((i*iNumTrees + j)*iInnerDataNum + k)*iInnerWeightsNum;

    RealType tmpSum = RealType(0);
    for (int l = 0; l < iInnerWeightsNum; ++l)
      tmpSum += d_leafWeights[l] * d_outGradient[l];

    tmpSum *= -sign;

    atomicAdd(d_thresholdsGradient + thresholdIndex, tmpSum); // Do this just once

    //for (int l = 0; l < iInnerWeightsNum; ++l)
      //d_thresholdsGradient[thresholdIndex] += -sign * d_leafWeights[l] * d_outGradient[l];
  }
}

template<typename TreeTraitsTypeGPU, typename RealType>
__global__ void BackwardWeightsKernel(const RealType *d_inData, const RealType *d_inThresholds, const RealType *d_inOrdinals, /*const RealType *d_inWeights,*/
    const RealType *d_outDataGradient, RealType *d_inWeightsGradient, int iTreeDepth, int iThresholdStride, int iWeightsStride, int iInnerWeightsNum, int iNumTrees, 
    int iOuterNum, int iNumChannels, int iInnerDataNum) {

  typedef typename TreeTraitsTypeGPU::KeyType KeyType;

  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i < iOuterNum && j < iNumTrees && k < iInnerDataNum) {
    const RealType * const d_thresholds = d_inThresholds + j*iThresholdStride;
    const RealType * const d_ordinals = d_inOrdinals + j*iThresholdStride;

    const RealType * const d_row = d_inData + ((i*iNumChannels + 0)*iInnerDataNum + k);

    // leaf key, margin, ordinal index
    const auto keyMarginTuple = TreeTraitsTypeGPU::ComputeKeyAndSignedMargin(d_row, d_thresholds, d_ordinals, iTreeDepth, iInnerDataNum);

    const KeyType key = keyMarginTuple.leafKey;
    const RealType signedMargin = keyMarginTuple.signedMargin;
    const RealType margin = std::abs(signedMargin);

    const RealType * const d_outGradient = d_outDataGradient + ((i*iNumTrees + j)*iInnerDataNum + k)*iInnerWeightsNum;
    RealType * const d_leafWeightsGradient = d_inWeightsGradient + (j*iWeightsStride + key)*iInnerWeightsNum;

    for (int l = 0; l < iInnerWeightsNum; ++l) {
      atomicAdd(d_leafWeightsGradient + l, margin * d_outGradient[l]); // Really bad!
      //d_leafWeightsGradient[l] += margin * d_outGradient[l];
    }
  }
}

template<typename TreeTraitsTypeGPU, typename RealType>
__global__ void BackwardDataKernel(const RealType *d_inData, const RealType *d_inThresholds, const RealType *d_inOrdinals, const RealType *d_inWeights, 
    const RealType *d_outDataGradient, RealType *d_inDataGradient, int iTreeDepth, int iThresholdStride, int iWeightsStride, int iInnerWeightsNum, int iNumTrees, 
    int iOuterNum, int iNumChannels, int iInnerDataNum) {

  typedef typename TreeTraitsTypeGPU::KeyType KeyType;

  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i < iOuterNum && j < iNumTrees && k < iInnerDataNum) {
    const RealType * const d_thresholds = d_inThresholds + j*iThresholdStride;
    const RealType * const d_ordinals = d_inOrdinals + j*iThresholdStride;

    const RealType * const d_row = d_inData + ((i*iNumChannels + 0)*iInnerDataNum + k);

    // leaf key, margin, ordinal index
    const auto keyMarginTuple = TreeTraitsTypeGPU::ComputeKeyAndSignedMargin(d_row, d_thresholds, d_ordinals, iTreeDepth, iInnerDataNum);

    const KeyType key = keyMarginTuple.leafKey;
    const RealType signedMargin = keyMarginTuple.signedMargin;
    const KeyType thresholdIndex = keyMarginTuple.thresholdIndex;
    const int iInputIndex = (int)d_ordinals[thresholdIndex];

    const RealType * const d_leafWeights = d_inWeights + (j*iWeightsStride + key)*iInnerWeightsNum;
    const RealType * const d_outGradient = d_outDataGradient + ((i*iNumTrees + j)*iInnerDataNum + k)*iInnerWeightsNum;

    const RealType sign = RealType((RealType(0) < signedMargin) - (signedMargin < RealType(0)));
    RealType tmpSum = RealType(0);

    for (int l = 0; l < iInnerWeightsNum; ++l)
      tmpSum += d_leafWeights[l] * d_outGradient[l];

    tmpSum *= sign;

    atomicAdd(d_inDataGradient + ((i*iNumChannels + iInputIndex)*iInnerDataNum + k), tmpSum); // Do this just once

    //d_inDataGradient[(i*iNumChannels + iInputIndex)*iInnerDataNum + k] += tmpSum; // Do this just once
  }
}

} // end anonymous namespace

template<typename RealType, typename TreeTraitsType>
void RandomHingeForestTemplate<RealType, TreeTraitsType>::ForwardGPU() {
  typedef HingeTreeCommonGPU<TreeTraitsType> TreeTraitsTypeGPU;
  typedef typename TreeTraitsTypeGPU::KeyMarginTupleType KeyMarginTupleType;

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

  const RealType * const p_ordinals = clOrdinals.data(GPU);
  const RealType * const p_thresholds = clThresholds.data(GPU);
  const RealType * const p_weights = clWeights.data(GPU);
  const RealType * const p_inData = clInData.data(GPU);
  RealType * const p_outData = clOutData.data_no_sync(GPU);

  const int iOuterNum = clInData.GetSize()[0];
  const int iNumChannels = clInData.GetSize()[1];
  const int iInnerDataNum = clInData.GetSize().Product(2);
  const int iNumTrees = clWeights.GetSize()[0];
  const int iNumDecisionsPerTree = clOrdinals.GetSize()[1];
  const int iNumLeavesPerTree = clWeights.GetSize()[1];
  const int iInnerWeightsNum = clWeights.GetSize().Product(2);

  const dim3 threadsPerBlock(8,8,8);
  const dim3 numBlocks((iOuterNum + threadsPerBlock.x-1)/threadsPerBlock.x, (iNumTrees + threadsPerBlock.y-1)/threadsPerBlock.y, (iInnerDataNum + threadsPerBlock.z-1)/threadsPerBlock.z);

  ForwardKernel<TreeTraitsTypeGPU><<<numBlocks, threadsPerBlock>>>(p_inData, p_thresholds, p_ordinals, p_weights, p_outData, 
    m_iTreeDepth, iNumDecisionsPerTree, iNumLeavesPerTree, iInnerWeightsNum, iNumTrees, iOuterNum, iNumChannels, iInnerDataNum);
}

template<typename RealType, typename TreeTraitsType>
void RandomHingeForestTemplate<RealType, TreeTraitsType>::BackwardGPU() {
  typedef HingeTreeCommonGPU<TreeTraitsType> TreeTraitsTypeGPU;
  typedef typename TreeTraitsTypeGPU::KeyMarginTupleType KeyMarginTupleType;

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

  const RealType * const p_ordinals = clOrdinals.data(GPU);
  const RealType * const p_thresholds = clThresholds.data(GPU);
  RealType * const p_thresholdsGradient = clThresholdsGradient.data(GPU);
  const RealType * const p_weights = clWeights.data(GPU);
  RealType * const p_weightsGradient = clWeightsGradient.data(GPU);
  const RealType * const p_inData = clInData.data(GPU);
  RealType * p_inDataGradient = clInDataGradient.data(GPU);
  //const RealType * const p_outData = clOutData.data(GPU);
  const RealType * const p_outDataGradient = clOutDataGradient.data(GPU);

  const int iOuterNum = clInData.GetSize()[0];
  const int iNumChannels = clInData.GetSize()[1];
  const int iInnerDataNum = clInData.GetSize().Product(2);
  const int iNumTrees = clWeights.GetSize()[0];
  const int iNumDecisionsPerTree = clOrdinals.GetSize()[1];
  const int iNumLeavesPerTree = clWeights.GetSize()[1];
  const int iInnerWeightsNum = clWeights.GetSize().Product(2);

  const dim3 threadsPerBlock(8,8,8);
  const dim3 numBlocks((iOuterNum + threadsPerBlock.x-1)/threadsPerBlock.x, (iNumTrees + threadsPerBlock.y-1)/threadsPerBlock.y, (iInnerDataNum + threadsPerBlock.z-1)/threadsPerBlock.z);

  BackwardDataKernel<TreeTraitsTypeGPU><<<numBlocks, threadsPerBlock>>>(p_inData, p_thresholds, p_ordinals, p_weights, p_outDataGradient, p_inDataGradient, 
    m_iTreeDepth, iNumDecisionsPerTree, iNumLeavesPerTree, iInnerWeightsNum, iNumTrees, iOuterNum, iNumChannels, iInnerDataNum);

  BackwardThresholdsKernel<TreeTraitsTypeGPU><<<numBlocks, threadsPerBlock>>>(p_inData, p_thresholds, p_ordinals, p_weights, p_outDataGradient, p_thresholdsGradient, 
    m_iTreeDepth, iNumDecisionsPerTree, iNumLeavesPerTree, iInnerWeightsNum, iNumTrees, iOuterNum, iNumChannels, iInnerDataNum);

  BackwardWeightsKernel<TreeTraitsTypeGPU><<<numBlocks, threadsPerBlock>>>(p_inData, p_thresholds, p_ordinals, p_outDataGradient, p_weightsGradient, 
    m_iTreeDepth, iNumDecisionsPerTree, iNumLeavesPerTree, iInnerWeightsNum, iNumTrees, iOuterNum, iNumChannels, iInnerDataNum);
}

template class RandomHingeForestTemplate<float, HingeFernCommon<float>>;
template class RandomHingeForestTemplate<double, HingeFernCommon<double>>;

template class RandomHingeForestTemplate<float, HingeTreeCommon<float>>;
template class RandomHingeForestTemplate<double, HingeTreeCommon<double>>;

} // end namespace bleak
