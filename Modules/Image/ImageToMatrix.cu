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

#include "ImageToMatrix.h"

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

template<typename RealType>
__global__ void ExtractMatrixHelper(RealType *d_matrix, const RealType *d_image, const int *d_indexMatrix, int iRows, int iCols, RealType padValue) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < iRows && j < iCols) {
    const int index = d_indexMatrix[iCols*i + j];
    d_matrix[iCols*i + j] = (index < 0) ? padValue : d_image[index];
  }
}

template<typename RealType>
__global__ void MapAndAddHelper(RealType *d_diff, int iStride, const RealType *d_matrix, const int *d_indexMatrix, int iRows, int iCols) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < iRows && j < iCols) {
    const int index = d_indexMatrix[iCols*i + j];
    if (index >= 0) {
      atomicAdd(d_diff + index*iStride, d_matrix[iCols*i + j]);
      //d_diff[index*iStride] += d_matrix[iCols*i + j];
    }
  }
}

template<typename RealType, unsigned int Dimension>
void ImageToMatrixBase<RealType, Dimension>::ExtractMatrixGPU(RealType *d_matrix, const RealType *d_image, const int *d_indexMatrix, const int a_iImageSize[Dimension+1]) const {
  int iRows = 0;
  int iCols = 0;
  ComputeMatrixDimensions(iRows, iCols, a_iImageSize);

  const dim3 threadsPerBlock(16,16);
  const dim3 numBlocks((iRows + threadsPerBlock.x-1) / threadsPerBlock.x, (iCols + threadsPerBlock.y-1) / threadsPerBlock.y);
  ExtractMatrixHelper<<<numBlocks, threadsPerBlock>>>(d_matrix, d_image, d_indexMatrix, iRows, iCols, padValue);
}

template<typename RealType, unsigned int Dimension>
void ImageToMatrixBase<RealType, Dimension>::MapAndAddGPU(RealType *d_diff, int iStride, const RealType *d_matrix, const int *d_indexMatrix, const int a_iImageSize[Dimension+1]) const {
  int iRows = 0;
  int iCols = 0;
  ComputeMatrixDimensions(iRows, iCols, a_iImageSize);

  const dim3 threadsPerBlock(16,16);
  const dim3 numBlocks((iRows + threadsPerBlock.x-1) / threadsPerBlock.x, (iCols + threadsPerBlock.y-1) / threadsPerBlock.y);
  MapAndAddHelper<<<numBlocks, threadsPerBlock>>>(d_diff, iStride, d_matrix, d_indexMatrix, iRows, iCols);
}

// Instantiate these functions by instantiating duplicate ImageToMatrixBase
template class ImageToMatrixBase<float, 1>;
template class ImageToMatrixBase<float, 2>;
template class ImageToMatrixBase<float, 3>;

template class ImageToMatrixBase<double, 1>;
template class ImageToMatrixBase<double, 2>;
template class ImageToMatrixBase<double, 3>;

} // end namespace bleak
