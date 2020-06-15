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

#ifndef BLEAK_IMAGETOMATRIX_H
#define BLEAK_IMAGETOMATRIX_H

#include <array>
#include <algorithm>
#include <functional>
#include <iterator>
#include <numeric>

#define bleakNewImageToMatrix(className, superClass) \
  typedef className SelfType; \
  typedef superClass SuperType; \
  typedef typename SuperType::RasterType RasterType; \
  typedef typename SuperType::SizeType SizeType; \
  typedef typename SuperType::CoordType CoordType; \
  using SuperType::kernelSize; \
  using SuperType::padding; \
  using SuperType::stride; \
  using SuperType::dilate; \
  using SuperType::padValue; \
  using SuperType::GetDimension; \
  using SuperType::ComputeMatrixDimensions; \
  using SuperType::ComputeOutputSize; \
  using SuperType::ComputeOutputCount; \
  using SuperType::ComputeKernelCount; \
  using SuperType::ComputeWindowSize; \
  using SuperType::ExtractMatrix; \
  using SuperType::Good

namespace bleak {

template<unsigned int Dimension>
class RasterCurve {
public:
  typedef std::array<int, Dimension> SizeType;
  typedef SizeType CoordType;

  explicit RasterCurve(const SizeType &clSize)
  : m_clSize(clSize) {
    for (unsigned int d = 0; d < Dimension; ++d)
      m_clInvSize[d] = 1.0f / m_clSize[d]; // XXX: Divide by 0?
  }

  explicit RasterCurve(const int a_iSize[Dimension]) {
    std::copy_n(a_iSize, Dimension, m_clSize.begin());
    for (unsigned int d = 0; d < Dimension; ++d)
      m_clInvSize[d] = 1.0f / m_clSize[d]; // XXX: Divide by 0?
  }

  int Count() const { return std::accumulate(m_clSize.begin(), m_clSize.end(), 1, std::multiplies<int>()); }

  const SizeType & GetSize() const { return m_clSize; }

  int Index(const CoordType &clCoord) const {
    int index = clCoord[0];

    for (unsigned int d = 1; d < Dimension; ++d)
      index = m_clSize[d] * index + clCoord[d];

    return index;
  }

  int IndexChecked(const CoordType &clCoord) const {
    if (clCoord[0] < 0 || clCoord[0] >= m_clSize[0])
      return -1;

    int index = clCoord[0];

    for (unsigned int d = 1; d < Dimension; ++d) {
      if (clCoord[d] < 0 || clCoord[d] >= m_clSize[d])
        return -1;

      index = m_clSize[d] * index + clCoord[d];
    }

    return index;
  }

  CoordType Coordinate(int index) const {
    CoordType clCoord;

    for (unsigned int d = Dimension-1; d > 0; --d) {
      //const int q = index / m_clSize[d];
      const int q = (int)(index * m_clInvSize[d]);
      const int r = index - q * m_clSize[d];
      clCoord[d] = r;
      index = q;
    }

    clCoord[0] = index;

    return clCoord;
  }

private:
  SizeType m_clSize;
  std::array<float, Dimension> m_clInvSize;
};

// Don't refer to this base class by reference!
template<typename RealType, unsigned int Dimension>
class ImageToMatrixBase {
public:
  static_assert(Dimension > 0, "Dimension must be larger than 0");

  typedef RasterCurve<Dimension> RasterType;
  typedef typename RasterType::SizeType SizeType;
  typedef typename RasterType::CoordType CoordType;

  // Z x Y x X
  SizeType kernelSize;
  SizeType stride;
  SizeType padding;
  SizeType dilate;

  RealType padValue = RealType();

  static constexpr unsigned int GetDimension() {
    return Dimension;
  }

  ImageToMatrixBase() {
    kernelSize.fill(0);
    padding.fill(0);
    stride.fill(1);
    dilate.fill(1);

    padValue = RealType();
  }

  // Convenience functions...
  void SetKernelSize(const int a_iKernelSize[Dimension]) { std::copy(a_iKernelSize, a_iKernelSize + Dimension, kernelSize.begin()); }
  void SetPadding(const int a_iPadding[Dimension]) { std::copy(a_iPadding, a_iPadding + Dimension, padding.begin()); }
  void SetStride(const int a_iStride[Dimension]) { std::copy(a_iStride, a_iStride + Dimension, stride.begin()); }
  void SetDilate(const int a_iDilate[Dimension]) { std::copy(a_iDilate, a_iDilate + Dimension, dilate.begin()); }

  bool Good() const {
    if (*std::min_element(kernelSize.begin(), kernelSize.end()) <= 0 ||
      *std::min_element(padding.begin(), padding.end()) < 0 ||
      *std::min_element(dilate.begin(), dilate.end()) <= 0 ||
      *std::min_element(stride.begin(), stride.end()) <= 0) {
      return false;
    }

    return true;
  }

  // a_iImageSize: C x Z x Y x X x ...
  bool Good(const int a_iImageSize[Dimension+1]) const {
    if (a_iImageSize[0] < 1 || !Good())
      return false;

    const SizeType winSize = ComputeWindowSize();
    
    for (unsigned int d = 0; d < Dimension; ++d) {
      if (a_iImageSize[1+d] < 1 || a_iImageSize[1+d] + 2*padding[d] < winSize[d])
        return false;
    }

    return true;
  }

  // Window = dilated kernel
  // Kernel = not dilated
  
  // Dilated kernel size
  SizeType ComputeWindowSize() const {
    SizeType winSize;

    for (unsigned int d = 0; d < Dimension; ++d)
      winSize[d] = kernelSize[d] + (kernelSize[d] - 1)*(dilate[d] - 1);

    return winSize;
  }

  int ComputeKernelCount() const { return std::accumulate(kernelSize.begin(), kernelSize.end(), 1, std::multiplies<int>()); }

  // Output size neglecting channels: C x Z x Y x X ...
  SizeType ComputeOutputSize(const int a_iImageSize[Dimension+1]) const {
    const SizeType winSize = ComputeWindowSize();
    SizeType outSize;

    for (unsigned int d = 0; d < Dimension; ++d)
      outSize[d] = (a_iImageSize[1+d] + 2*padding[d] - winSize[d]) / stride[d] + 1;

    return outSize;
  }

  // Number of windows in image neglecting channels: C x Z x Y x X ...
  int ComputeOutputCount(const int a_iImageSize[Dimension+1]) const {
    const SizeType outSize = ComputeOutputSize(a_iImageSize);
    return std::accumulate(outSize.begin(), outSize.end(), 1, std::multiplies<int>());
  }

  // Row major (C/C++)
  void ComputeMatrixDimensions(int &iRows, int &iCols, const int a_iImageSize[Dimension+1]) const {
    iRows = ComputeOutputCount(a_iImageSize);
    iCols = ComputeKernelCount() * a_iImageSize[0];
  }

  void ExtractMatrix(RealType *p_matrix, const RealType *p_image, const int *p_indexMatrix, const int a_iImageSize[Dimension+1]) const {
    int iRows = 0;
    int iCols = 0;
    ComputeMatrixDimensions(iRows, iCols, a_iImageSize);

    for (int j = 0; j < iCols; ++j) {
      for (int i = 0; i < iRows; ++i) {
        // This seems less efficient, but it keeps locality better in p_image (or should in most cases)
        const int index = p_indexMatrix[iCols*i + j];
        p_matrix[iCols*i + j] = (index < 0) ? padValue : p_image[index];
      }
    }
  }

  void MapAndAdd(RealType *p_diff, int iStride, const RealType *p_matrix, const int *p_indexMatrix, const int a_iImageSize[Dimension+1]) const {
    int iRows = 0;
    int iCols = 0;
    ComputeMatrixDimensions(iRows, iCols, a_iImageSize);

    for (int j = 0; j < iCols; ++j) {
      for (int i = 0; i < iRows; ++i) {
        const int index = p_indexMatrix[iCols*i + j];
        if (index >= 0)
          p_diff[index*iStride] += p_matrix[iCols*i + j];
      }
    }
  }

#ifdef BLEAK_USE_CUDA
  void ExtractMatrixGPU(RealType *d_matrix, const RealType *d_image, const int *d_indexMatrix, const int a_iImageSize[Dimension+1]) const;
  void MapAndAddGPU(RealType *d_diff, int iStride, const RealType *d_matrix, const int *d_indexMatrix, const int a_iImageSize[Dimension+1]) const;
#endif // BLEAK_USE_CUDA

};

template<typename RealType, unsigned int Dimension>
class ImageToMatrix : public ImageToMatrixBase<RealType, Dimension> { 
public:
  typedef ImageToMatrixBase<RealType, Dimension> WorkAroundVarArgsType;

  bleakNewImageToMatrix(ImageToMatrix, WorkAroundVarArgsType);

  void ExtractMatrix(RealType *p_matrix, const RealType *p_image, const int a_iImageSize[Dimension+1]) const {
    const RasterType outRaster(ComputeOutputSize(a_iImageSize));
    const RasterType kernRaster(kernelSize);
    const RasterType imageRaster(a_iImageSize+1);

    const int iChannels = a_iImageSize[0];

    int iRows = 0;
    int iCols = 0;
    ComputeMatrixDimensions(iRows, iCols, a_iImageSize);

    const int iKernelCount = kernRaster.Count();
    const int iOutCount = outRaster.Count();
    const int iInCount = imageRaster.Count();

    for (int c = 0; c < iChannels; ++c) {
      const int indexOffset = c*iInCount;
      const int jOffset = c*iKernelCount;

      for (int j = 0; j < iKernelCount; ++j) {
        CoordType winCoord = kernRaster.Coordinate(j);

        for (unsigned int d = 0; d < Dimension; ++d)
          winCoord[d] *= dilate[d];

        for (int i = 0; i < iOutCount; ++i) {
          CoordType coord = outRaster.Coordinate(i);

          for (unsigned int d = 0; d < Dimension; ++d)
            coord[d] = stride[d]*coord[d] + winCoord[d] - padding[d];

          const int index = imageRaster.IndexChecked(coord);
          p_matrix[iCols*i + (j + jOffset)] = (index < 0) ? padValue : p_image[index + indexOffset];
        }
      }
    }
  }

  void ExtractIndexMatrix(int *p_matrix, const int a_iImageSize[Dimension+1]) const {
    const RasterType outRaster(ComputeOutputSize(a_iImageSize));
    const RasterType kernRaster(kernelSize);
    const RasterType imageRaster(a_iImageSize+1);

    const int iChannels = a_iImageSize[0];

    int iRows = 0;
    int iCols = 0;
    ComputeMatrixDimensions(iRows, iCols, a_iImageSize);

    const int iKernelCount = kernRaster.Count();
    const int iOutCount = outRaster.Count();
    const int iInCount = imageRaster.Count();

    for (int c = 0; c < iChannels; ++c) {
      const int indexOffset = c*iInCount;
      const int jOffset = c*iKernelCount;

      for (int j = 0; j < iKernelCount; ++j) {
        CoordType winCoord = kernRaster.Coordinate(j);

        for (unsigned int d = 0; d < Dimension; ++d)
          winCoord[d] *= dilate[d];

        for (int i = 0; i < iOutCount; ++i) {
          CoordType coord = outRaster.Coordinate(i);

          for (unsigned int d = 0; d < Dimension; ++d)
            coord[d] = stride[d]*coord[d] + winCoord[d] - padding[d];

          const int index = imageRaster.IndexChecked(coord);
          p_matrix[iCols*i + (j + jOffset)] = (index < 0) ? index : (index + indexOffset);
        }
      }
    }
  }
};

} // end namespace bleak

#endif // !BLEAK_IMAGETOMATRIX_H
