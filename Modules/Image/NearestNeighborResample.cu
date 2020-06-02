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

#include "NearestNeighborResample.h"

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
__global__ void ForwardKernel(const RealType *d_inData, const int *d_iIndexMatrix, RealType *d_outData, int iBatchSize, int iNumChannels, int iInInnerNum, int iOutInnerNum) {
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  const int c = blockIdx.y * blockDim.y + threadIdx.y;
  const int i = blockIdx.z * blockDim.z + threadIdx.z;

  if (b < iBatchSize && c < iNumChannels && i < iOutInnerNum) {
    const int index = d_iIndexMatrix[i];
    d_outData[(b*iNumChannels + c)*iOutInnerNum + i] = d_inData[(b*iNumChannels + c)*iInInnerNum + index]; // There's a more efficient way to do this kernel... TODO: Optimize this
  }
}

template<typename RealType>
__global__ void BackwardKernel(const RealType *d_outGradient, const int *d_iIndexMatrix, RealType *d_inGradient, int iBatchSize, int iNumChannels, int iInInnerNum, int iOutInnerNum) {
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  const int c = blockIdx.y * blockDim.y + threadIdx.y;
  const int i = blockIdx.z * blockDim.z + threadIdx.z;

  if (b < iBatchSize && c < iNumChannels && i < iOutInnerNum) {
    const int index = d_iIndexMatrix[i];
    //d_inGradient[(b*iNumChannels + c)*iInInnerNum + index] += d_outGradient[(b*iNumChannels + c)*iOutInnerNum + i];
    atomicAdd(d_inGradient + ((b*iNumChannels + c)*iInInnerNum + index), d_outGradient[(b*iNumChannels + c)*iOutInnerNum + i]);
  }
}

} // end anonymous namespace

template<typename RealType, unsigned int Dimension>
void NearestNeighborResample<RealType, Dimension>::ForwardGPU() {
  bleakGetAndCheckInput(p_clInData, "inData");
  bleakGetAndCheckOutput(p_clOutData, "outData");

  const ArrayType &clInData = p_clInData->GetData();
  ArrayType &clOutData = p_clOutData->GetData();

  const int iBatchSize = clOutData.GetSize()[0];
  const int iNumChannels = clOutData.GetSize()[1];
  const int iInInnerNum = clInData.GetSize().Count(2);
  const int iOutInnerNum = clOutData.GetSize().Count(2);

  const RealType * const d_inData = clInData.data(GPU);
  RealType * const d_outData = clOutData.data_no_sync(GPU);
  const int * const d_iIndexMatrix = m_clIndexMatrix.data(GPU);

  const dim3 threadsPerBlock(8,8,8);
  const dim3 numBlocks((iBatchSize + threadsPerBlock.x-1) / threadsPerBlock.x, (iNumChannels + threadsPerBlock.y-1) / threadsPerBlock.y, (iOutInnerNum + threadsPerBlock.z-1) / threadsPerBlock.z);

  ForwardKernel<<<numBlocks, threadsPerBlock>>>(d_inData, d_iIndexMatrix, d_outData, iBatchSize, iNumChannels, iInInnerNum, iOutInnerNum);
}

template<typename RealType, unsigned int Dimension>
void NearestNeighborResample<RealType, Dimension>::BackwardGPU() {
  bleakGetAndCheckInput(p_clInData, "inData");
  bleakGetAndCheckOutput(p_clOutData, "outData");

  const ArrayType &clInData = p_clInData->GetData();
  ArrayType &clInGradient = p_clInData->GetGradient();

  const ArrayType &clOutData = p_clOutData->GetData();
  const ArrayType &clOutGradient = p_clOutData->GetGradient();

  if (!clInGradient.Valid()) // Nothing to do...
    return;

  const int iBatchSize = clOutData.GetSize()[0];
  const int iNumChannels = clOutData.GetSize()[1];
  const int iInInnerNum = clInData.GetSize().Count(2);
  const int iOutInnerNum = clOutData.GetSize().Count(2);

  RealType * const d_inGradient = clInGradient.data(GPU);
  const RealType * const d_outGradient = clOutGradient.data(GPU);
  const int * const d_iIndexMatrix = m_clIndexMatrix.data(GPU);

  const dim3 threadsPerBlock(8,8,8);
  const dim3 numBlocks((iBatchSize + threadsPerBlock.x-1) / threadsPerBlock.x, (iNumChannels + threadsPerBlock.y-1) / threadsPerBlock.y, (iOutInnerNum + threadsPerBlock.z-1) / threadsPerBlock.z);

  BackwardKernel<<<numBlocks, threadsPerBlock>>>(d_outGradient, d_iIndexMatrix, d_inGradient, iBatchSize, iNumChannels, iInInnerNum, iOutInnerNum);
}

// Pull these GPU functions in by instantiating
template class NearestNeighborResample<float, 1>;
template class NearestNeighborResample<double, 1>;

template class NearestNeighborResample<float, 2>;
template class NearestNeighborResample<double, 2>;

template class NearestNeighborResample<float, 3>;
template class NearestNeighborResample<double, 3>;

} // end namespace bleak
