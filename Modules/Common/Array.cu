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

#include "Array.h"

namespace bleak {

namespace {

template<typename RealType>
__global__ void FillKernel(RealType *d_buffer, int iCount, RealType value) {
  const int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < iCount)
    d_buffer[i] = value;
}

} // end anonymous namespace

template<typename RealType>
void Array<RealType>::Fill(const RealType &value) {
  if (!Valid())
    return;

  switch (GetLocation()) {
  case CPU:
    std::fill(begin(), end(), value);
    break;
  case GPU:
    {
      const int iCount = GetSize().Count();
      constexpr const int iNumThreadsPerBlock = 512;
      const int iNumBlocks = (iCount + iNumThreadsPerBlock-1) / iNumThreadsPerBlock;
      FillKernel<<<iNumBlocks, iNumThreadsPerBlock>>>(data(GPU), iCount, value);
    }
    break;
  }
}

// Instantiate to pull in the above function
template class Array<float>;
template class Array<double>;

} // end namespace bleak

