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
#include "BlasWrapper.h"
#include "BatchNormalization.h"

namespace bleak {

namespace {

template<typename RealType>
__global__ void SquareKernel(RealType *d_data, int iCount) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < iCount)
    d_data[i] *= d_data[i];
}

template<typename RealType>
__global__ void ForwardKernel(const RealType *d_inData, const RealType *d_means, const RealType *d_vars, RealType *d_outData, const int iOuterNum, const int iNumChannels, const int iInnerNum, const RealType small) {
  const int i = blockIdx.z * blockDim.z + threadIdx.z;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int k = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < iOuterNum && j < iNumChannels && k < iInnerNum) {
    const RealType std = std::sqrt(std::abs(d_vars[j]) + small);
    d_outData[(i*iNumChannels + j)*iInnerNum + k] = (d_inData[(i*iNumChannels + j)*iInnerNum + k] - d_means[j])/std;
  }
}

template<typename RealType>
__global__ void BackwardKernel(const RealType *d_outDataGradient, /*const RealType *d_means,*/ const RealType *d_vars, RealType *d_inDataGradient, const int iOuterNum, const int iNumChannels, const int iInnerNum, const RealType small) {
  const int i = blockIdx.z * blockDim.z + threadIdx.z;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int k = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < iOuterNum && j < iNumChannels && k < iInnerNum) {
    const RealType std = std::sqrt(std::abs(d_vars[j]) + small);
    d_inDataGradient[(i*iNumChannels + j)*iInnerNum + k] += d_outDataGradient[(i*iNumChannels + j)*iInnerNum + k]/std;
  }
}

} // end anonymous namespace

template<typename RealType>
void BatchNormalization<RealType>::ForwardGPU() {
  bleakGetAndCheckInput(p_clMeans, "inMeans");
  bleakGetAndCheckInput(p_clVars, "inVariances");
  bleakGetAndCheckInput(p_clInData, "inData");
  bleakGetAndCheckOutput(p_clOutData, "outData");

  const ArrayType &clInData = p_clInData->GetData();
  const ArrayType &clMeans = p_clMeans->GetData();
  const ArrayType &clVars = p_clVars->GetData();
  ArrayType &clOutData = p_clOutData->GetData();

  const int iOuterNum = clInData.GetSize()[0];
  const int iNumChannels = clInData.GetSize()[1];
  const int iInnerNum = clInData.GetSize().Product(2);

  const RealType * const d_inData = clInData.data(GPU);
  RealType * const d_outData = clOutData.data_no_sync(GPU);
  const RealType * const d_means = clMeans.data(GPU);
  const RealType * const d_vars = clVars.data(GPU);

  const dim3 threadsPerBlock(8, 8, 8);
  const dim3 numBlocks((iInnerNum + threadsPerBlock.x-1) / threadsPerBlock.x, (iNumChannels + threadsPerBlock.y-1)/threadsPerBlock.y, (iOuterNum + threadsPerBlock.z-1)/threadsPerBlock.z);

  // No easy way to make BLAS do this... (see the CPU version!)
  ForwardKernel<<<numBlocks, threadsPerBlock>>>(d_inData, d_means, d_vars, d_outData, iOuterNum, iNumChannels, iInnerNum, RealType(m_fSmall));
}

template<typename RealType>
void BatchNormalization<RealType>::BackwardGPU() {
  bleakGetAndCheckInput(p_clMeans, "inMeans");
  bleakGetAndCheckInput(p_clVars, "inVariances");
  bleakGetAndCheckInput(p_clInData, "inData");
  bleakGetAndCheckOutput(p_clOutData, "outData");

  const ArrayType &clInData = p_clInData->GetData();
  ArrayType &clMeans = p_clMeans->GetData();
  ArrayType &clVars = p_clVars->GetData();
  ArrayType &clInDataGradient = p_clInData->GetGradient();
  const ArrayType &clOutData = p_clOutData->GetData();
  const ArrayType &clOutDataGradient = p_clOutData->GetGradient();

  if (!clInDataGradient.Valid())
    return; // Nothing to do

  const int iOuterNum = clInData.GetSize()[0];
  const int iNumChannels = clInData.GetSize()[1];
  const int iInnerNum = clInData.GetSize().Product(2);

  const RealType * const d_inData = clInData.data(GPU);
  RealType * const d_inDataGradient = clInDataGradient.data(GPU);
  //const RealType * const p_outData = clOutData.data();
  const RealType * const d_outDataGradient = clOutDataGradient.data(GPU);
  RealType * const d_means = clMeans.data(GPU); 
  RealType * const d_vars = clVars.data(GPU);

  const dim3 threadsPerBlock(8, 8, 8);
  const dim3 numBlocks((iInnerNum + threadsPerBlock.x-1) / threadsPerBlock.x, (iNumChannels + threadsPerBlock.y-1)/threadsPerBlock.y, (iOuterNum + threadsPerBlock.z-1)/threadsPerBlock.z);

  BackwardKernel<<<numBlocks, threadsPerBlock>>>(d_outDataGradient, d_vars, d_inDataGradient, iOuterNum, iNumChannels, iInnerNum, RealType(m_fSmall));

  const RealType m = RealType(iOuterNum * iInnerNum);

  if (m < 2) {
    std::cerr << GetName() << ": Error: Batch size or number of channels is less than 2" << std::endl;
    return;
  }

  RealType * const d_work = m_clWork.data_no_sync(GPU);
  RealType * const d_meansTmp = m_clMeansTmp.data_no_sync(GPU);
  RealType * const d_varsTmp = m_clVarsTmp.data_no_sync(GPU); // CPU!!!

  m_clMeansTmp.Fill(RealType(0));
  m_clVarsTmp.Fill(RealType(0));

  if (iInnerNum == 1) {
    gpu_blas::gemv('N', iNumChannels, iOuterNum, RealType(1)/m, d_inData, iNumChannels, m_clOnes.data(GPU), 1, RealType(0), d_meansTmp, 1);
    gpu_blas::copy(iOuterNum*iNumChannels, d_inData, 1, d_work, 1);
    gpu_blas::ger(iNumChannels, iOuterNum, RealType(-1), d_meansTmp, 1, m_clOnes.data(GPU), 1, d_work, iNumChannels);

    constexpr const int iThreadsPerBlock = 512;
    const int iCount = iNumChannels*iOuterNum;
    const int iNumBlocks = (iCount + iThreadsPerBlock-1)/iThreadsPerBlock;

    // Square all the elements
    SquareKernel<<<iNumBlocks, iThreadsPerBlock>>>(d_work, iCount);

    gpu_blas::gemv('N', iNumChannels, iOuterNum, RealType(1)/RealType(m-1), d_work, iNumChannels, m_clOnes.data(GPU), 1, RealType(0), d_varsTmp, 1);
  }
  else {
    // As opposed to iOuterNum x C x iInnerNum
    // I want C x iOuterNum x iInnerNum
    for (int i = 0; i < iOuterNum; ++i) {
      for (int j = 0; j < iNumChannels; ++j) {
        gpu_blas::copy(iInnerNum, d_inData + (i*iNumChannels + j)*iInnerNum, 1, d_work + (j*iOuterNum + i)*iInnerNum, 1);
      }
    }

    // Now X = C x (iOuterNum * iInnerNum)
    // So I would want:
    // X * 1

    gpu_blas::gemv('T', iOuterNum*iInnerNum, iNumChannels, RealType(1)/m, d_work, iOuterNum*iInnerNum, m_clOnes.data(GPU), 1, RealType(0), d_meansTmp, 1);
    gpu_blas::ger(iOuterNum*iInnerNum, iNumChannels, RealType(-1), m_clOnes.data(GPU), 1, d_meansTmp, 1, d_work, iOuterNum*iInnerNum);

    constexpr const int iThreadsPerBlock = 512;
    const int iCount = iNumChannels*iOuterNum*iInnerNum;
    const int iNumBlocks = (iCount + iThreadsPerBlock-1)/iThreadsPerBlock;

    // Square all the elements
    SquareKernel<<<iNumBlocks, iThreadsPerBlock>>>(d_work, iCount);

    gpu_blas::gemv('T', iOuterNum*iInnerNum, iNumChannels, RealType(1)/RealType(m-1), d_work, iOuterNum*iInnerNum, m_clOnes.data(GPU), 1, RealType(0), d_varsTmp, 1);
  }

  const RealType alpha = std::min(RealType(m_iIteration)/RealType(m_iIteration+1), RealType(m_fAlphaMax));

  gpu_blas::scal(iNumChannels, alpha, d_means, 1);
  gpu_blas::axpy(iNumChannels, RealType(1)-alpha, d_meansTmp, 1, d_means, 1);

  gpu_blas::scal(iNumChannels, alpha, d_vars, 1);
  gpu_blas::axpy(iNumChannels, RealType(1)-alpha, d_varsTmp, 1, d_vars, 1);

  ++m_iIteration;
}

template class BatchNormalization<float>;
template class BatchNormalization<double>;

} // end namespace bleak
