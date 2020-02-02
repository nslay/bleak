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

#include "AdaGrad.h"

namespace bleak {

namespace {

template<typename RealType>
__global__ void GradientHistoryKernel(RealType *d_gradientHistory, const RealType *d_gradient, int iNumElements) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < iNumElements)
    d_gradientHistory[i] += std::hypot(d_gradientHistory[i], d_gradient[i]);
}

template<typename RealType>
__global__ void GradientUpdateKernel(RealType *d_data, const RealType *d_gradientSum, const RealType *d_gradientHistory, int iNumElements) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < iNumElements && d_gradientHistory[i] > RealType(0))
    d_data[i] += d_gradientSum[i] / d_gradientHistory[i];
}

} // end anonymous namespace

template<typename RealType>
void AdaGrad<RealType>::CollectGradientsGPU() {
  constexpr const int iNumThreadsPerBlock = 512;
  const RealType scale = -GetLearningRate() / RealType(GetNumBatchesPerIteration());

  for(auto &clTuple : m_vEdgeAndHistory) {
    const RealType scaleForThisEdge = LearningRateMultiplier(std::get<0>(clTuple)) * scale;
    ArrayType &clGradient = std::get<0>(clTuple)->GetGradient();
    ArrayType &clGradientSum = *std::get<1>(clTuple);
    ArrayType &clGradientHistory = *std::get<2>(clTuple);

    const int iNumElements = clGradient.GetSize().Count();
    const int iNumBlocks = (iNumElements + iNumThreadsPerBlock-1)/iNumThreadsPerBlock;

    GradientHistoryKernel<<<iNumBlocks, iNumThreadsPerBlock>>>(clGradientHistory.data(GPU), clGradient.data(GPU), iNumElements);

    gpu_blas::axpy(clGradient.GetSize().Count(), scaleForThisEdge, clGradient.data(GPU), 1, clGradientSum.data(GPU), 1);
  }
}

template<typename RealType>
void AdaGrad<RealType>::GradientUpdateGPU() {
  constexpr const int iNumThreadsPerBlock = 512;

  for(auto &clTuple : m_vEdgeAndHistory) {
    ArrayType &clData = std::get<0>(clTuple)->GetData();
    ArrayType &clGradientSum = *std::get<1>(clTuple);
    ArrayType &clGradientHistory = *std::get<2>(clTuple);

    const int iNumElements = clData.GetSize().Count();
    const int iNumBlocks = (iNumElements + iNumThreadsPerBlock-1)/iNumThreadsPerBlock;

    GradientUpdateKernel<<<iNumBlocks, iNumThreadsPerBlock>>>(clData.data(GPU), clGradientSum.data(GPU), clGradientHistory.data(GPU), iNumElements);

    clGradientSum.Fill(RealType());
  }
}

template class AdaGrad<float>;
template class AdaGrad<double>;

} // end namespace bleak
