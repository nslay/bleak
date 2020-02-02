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

#include "Adam.h"

namespace bleak {

namespace {

template<typename RealType>
__global__ void Moment1UpdateKernel(RealType *d_gradientMoment1, const RealType *d_gradientSum, RealType beta1, RealType numBatchesPerIteration, int iNumElements) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < iNumElements)
    d_gradientMoment1[i] = (RealType(1) - beta1) * (d_gradientSum[i] / numBatchesPerIteration) + beta1 * d_gradientMoment1[i];
}

template<typename RealType>
__global__ void Moment2UpdateKernel(RealType *d_gradientMoment2, const RealType *d_gradientSum, RealType beta2, RealType numBatchesPerIteration, int iNumElements) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < iNumElements)
    d_gradientMoment2[i] = (RealType(1) - beta2) * std::pow(d_gradientSum[i] / numBatchesPerIteration, 2) + beta2 * d_gradientMoment2[i];
}

template<typename RealType>
__global__ void GradientUpdateKernel(RealType *d_data, const RealType *d_gradientMoment1, const RealType *d_gradientMoment2, RealType scaleForThisEdge, 
  RealType gradientMoment1Divisor, RealType gradientMoment2Divisor, RealType small, int iNumElements) {

  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < iNumElements) {
    const RealType gradientMoment1Hat = d_gradientMoment1[i] / gradientMoment1Divisor;
    const RealType gradientMoment2Hat = std::sqrt(d_gradientMoment2[i] / gradientMoment2Divisor) + small;

    d_data[i] += scaleForThisEdge * gradientMoment1Hat / gradientMoment2Hat;
  }
}

} // end anonymous namespace

template<typename RealType>
void Adam<RealType>::GradientUpdateGPU() {
  constexpr const int iNumThreadsPerBlock = 512;

  const unsigned int uiNumBatchesPerIteration = GetNumBatchesPerIteration();

  for (auto &clTuple : m_vEdgeAndHistory) {
    const RealType scaleForThisEdge = -GetLearningRate() * LearningRateMultiplier(std::get<0>(clTuple));
    ArrayType &clData = std::get<0>(clTuple)->GetData();
    ArrayType &clGradientSum = *std::get<1>(clTuple);
    ArrayType &clGradientMoment1 = *std::get<2>(clTuple);
    ArrayType &clGradientMoment2 = *std::get<3>(clTuple);

    const int iNumElements = clData.GetSize().Count();

    gpu_blas::scal(iNumElements, this->m_beta1, clGradientMoment1.data(GPU), 1);
    gpu_blas::axpy(iNumElements, (RealType(1) - this->m_beta1)/RealType(uiNumBatchesPerIteration), clGradientSum.data(GPU), 1, clGradientMoment1.data(GPU), 1);
    
    //gpu_blas::scal(clGradientMoment2.GetSize().Count(), this->m_beta2, clGradientMoment2.data(GPU), 1);

    const int iNumBlocks = (iNumElements + iNumThreadsPerBlock-1) / iNumThreadsPerBlock;
    Moment2UpdateKernel<<<iNumBlocks, iNumThreadsPerBlock>>>(clGradientMoment2.data(GPU), clGradientSum.data(GPU), this->m_beta2, RealType(uiNumBatchesPerIteration), iNumElements);

    const RealType gradientMoment1Divisor = RealType(1) - RealType(std::pow(m_beta1, GetIteration()+1)); // Indexed from 1
    const RealType gradientMoment2Divisor = RealType(1) - RealType(std::pow(m_beta2, GetIteration()+1)); // Indexed from 1

    GradientUpdateKernel<<<iNumBlocks, iNumThreadsPerBlock>>>(clData.data(GPU), clGradientMoment1.data(GPU), clGradientMoment2.data(GPU), scaleForThisEdge,
      gradientMoment1Divisor, gradientMoment2Divisor, m_small, iNumElements);

    clGradientSum.Fill(RealType());
  }
}

template class Adam<float>;
template class Adam<double>;

} // end namespace bleak
