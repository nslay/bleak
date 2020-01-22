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

#include "Pooling.h"

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

template<typename RealType>
__global__ void ForwardKernel(const RealType *d_matrix, RealType *d_outData, int iRows, int iCols) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;

  if (k < iRows) {
    const RealType * const d_row = d_matrix + k*iCols;
    RealType maxValue = d_row[0];

    for (int i = 1; i < iCols; ++i)
      maxValue = max(maxValue, d_row[i]);

    d_outData[k] = maxValue;
  }
}

template<typename RealType>
__global__ void BackwardKernel(const RealType *d_matrix, const int *d_indexMatrix, const RealType *d_outDataGradient, RealType *d_inDataGradient, int iRows, int iCols) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;

  if (k < iRows) {
    const RealType * const d_row = d_matrix + k*iCols;
    const int * const d_indexRow = d_indexMatrix + k*iCols;
    int iMax = 0;
    RealType maxValue = d_row[0];

    for (int i = 1; i < iCols; ++i) {
      if (maxValue < d_row[i]) {
        maxValue = d_row[i];
        iMax = i;
      }
    }
    
    const int index = d_indexRow[iMax];

    if (index >= 0)
      atomicAdd(d_inDataGradient + index, d_outDataGradient[k]);
  }
}

} // end anonymous namespace

template<typename RealType, unsigned int Dimension>
void MaxPooling<RealType, Dimension>::ForwardGPU() {
  bleakGetAndCheckInput(p_clInData, "inData");
  bleakGetAndCheckOutput(p_clOutData, "outData");

  const ArrayType &clInData = p_clInData->GetData();
  ArrayType &clOutData = p_clOutData->GetData();

  const RealType * const p_inData = clInData.data(GPU);
  RealType * const p_outData = clOutData.data_no_sync(GPU);

  Size clImageSize = clInData.GetSize().SubSize(1);
  clImageSize[0] = 1; // One channel at a time!

  const int iOuterNum = clInData.GetSize()[0];
  const int iNumChannels = clInData.GetSize()[1];
  const int iInChannelSize = clInData.GetSize().Product(2);
  //const int iOutChannelSize = clOutData.GetSize().Product(2); // Same as m_iRows

  const dim3 threadsPerBlock(256);
  const dim3 numBlocks((m_iRows + threadsPerBlock.x-1) / threadsPerBlock.x);

  for (int i = 0; i < iOuterNum; ++i) {
    for (int j = 0; j < iNumChannels; ++j) {
      m_clImageToMatrix.ExtractMatrixGPU(m_clMatrix.data_no_sync(GPU), p_inData + (i*iNumChannels + j)*iInChannelSize, m_clIndexMatrix.data(GPU), clImageSize.data());

      ForwardKernel<<<numBlocks, threadsPerBlock>>>(m_clMatrix.data(GPU), p_outData + (i*iNumChannels + j)*m_iRows, m_iRows, m_iCols);
    }
  }
}

template<typename RealType, unsigned int Dimension>
void MaxPooling<RealType, Dimension>::BackwardGPU() {
  bleakGetAndCheckInput(p_clInData, "inData");
  bleakGetAndCheckOutput(p_clOutData, "outData");

  const ArrayType &clInData = p_clInData->GetData();
  ArrayType &clInDataGradient = p_clInData->GetGradient();
  const ArrayType &clOutDataGradient = p_clOutData->GetGradient();

  if (!clInDataGradient.Valid())
    return; // Nothing to do

  const RealType * const p_inData = clInData.data(GPU);
  RealType * const p_inDataGradient = clInDataGradient.data(GPU);
  const RealType * const p_outDataGradient = clOutDataGradient.data(GPU);

  Size clImageSize = clInData.GetSize().SubSize(1);
  clImageSize[0] = 1; // One channel at a time!

  const int iOuterNum = clInData.GetSize()[0];
  const int iNumChannels = clInData.GetSize()[1];
  const int iInChannelSize = clInData.GetSize().Product(2);
  //const int iOutChannelSize = clOutData.GetSize().Product(2); // Same as m_iRows

  const dim3 threadsPerBlock(256);
  const dim3 numBlocks((m_iRows + threadsPerBlock.x-1) / threadsPerBlock.x);

  for (int i = 0; i < iOuterNum; ++i) {
    for (int j = 0; j < iNumChannels; ++j) {
      m_clImageToMatrix.ExtractMatrixGPU(m_clMatrix.data_no_sync(GPU), p_inData + (i*iNumChannels + j)*iInChannelSize, m_clIndexMatrix.data(GPU), clImageSize.data());

      BackwardKernel<<<numBlocks, threadsPerBlock>>>(m_clMatrix.data(GPU), m_clIndexMatrix.data(GPU), 
        p_outDataGradient + (i*iNumChannels + j)*m_iRows, p_inDataGradient + (i*iNumChannels + j)*iInChannelSize, m_iRows, m_iCols);
    }
  }
}

template class MaxPooling<float, 1>;
template class MaxPooling<float, 2>;
template class MaxPooling<float, 3>;

template class MaxPooling<double, 1>;
template class MaxPooling<double, 2>;
template class MaxPooling<double, 3>;

} // end namespace bleak
