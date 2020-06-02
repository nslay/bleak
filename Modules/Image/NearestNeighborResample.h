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

#pragma once

#ifndef BLEAK_NEARESTNEIGHBORRESAMPLE_H
#define BLEAK_NEARESTNEIGHBORRESAMPLE_H

#include <cmath>
#include <array>
#include <algorithm>
#include "Vertex.h"
#include "ImageToMatrix.h"

namespace bleak {

template<unsigned int Dimension>
void ComputeNearestNeighborIndexMatrix(int *p_iIndexMatrix, const int a_iInputSize[Dimension], const int a_iOutputSize[Dimension]) {
  typedef typename RasterCurve<Dimension>::CoordType CoordType;

  RasterCurve<Dimension> clInputCurve(a_iInputSize), clOutputCurve(a_iOutputSize);

  std::array<float, Dimension> clInvOutputSize;

  for (unsigned int d = 0; d < Dimension; ++d)
    clInvOutputSize[d] = 1.0f / a_iOutputSize[d];

  const int iOutputCount = clOutputCurve.Count();
  for (int i = 0; i < iOutputCount; ++i) {
    CoordType clCoord = clOutputCurve.Coordinate(i);

    for (unsigned int d = 0; d < Dimension; ++d)
      clCoord[d] = std::min((int)(clCoord[d]*a_iInputSize[d]*clInvOutputSize[d] + 0.5f), a_iInputSize[d]-1);

    p_iIndexMatrix[i] = clInputCurve.Index(clCoord);
  }
}

template<typename RealType, unsigned int Dimension>
class NearestNeighborResample : public Vertex<RealType> {
public:
  static_assert(Dimension > 0, "Dimension must be larger than 0.");

  bleakNewVertex(NearestNeighborResample, Vertex<RealType>,
    bleakAddProperty("size", m_vSize),
    bleakAddInput("inData"),
    bleakAddOutput("outData"));

  bleakForwardVertexTypedefs();

  virtual ~NearestNeighborResample() = default;

  virtual bool SetSizes() override {
    bleakGetAndCheckInput(p_clInData, "inData", false);
    bleakGetAndCheckOutput(p_clOutData, "outData", false);

    if (m_vSize.size() != Dimension) {
      std::cerr << GetName() << ": Error: Expected 'size' of dimension " << Dimension << " (got " << m_vSize.size() << ")." << std::endl;
      return false;
    }

    if (!Size(m_vSize).Valid()) {
      std::cerr << "Error: Invalid 'size' " << Size(m_vSize) << '.' << std::endl;
      return false;
    }

    const Size clInSize = p_clInData->GetData().GetSize();

    if (clInSize.GetDimension() != Dimension+2) {
      std::cerr << GetName() << ": Error: Expected input data to be size " << Dimension+2 << "(got " << clInSize.GetDimension() << "). Size should reflect [ BatchSize, Channels, Z, Y, X, ... ]." << std::endl;
      return false;
    }

    Size clOutSize(Dimension + 2);

    // Batch size, channels
    clOutSize[0] = clInSize[0];
    clOutSize[1] = clInSize[1];

    // Copy the rest over
    std::copy_n(m_vSize.begin(), Dimension, clOutSize.begin()+2);

    p_clOutData->GetData().SetSize(clOutSize);

    if (p_clInData->GetGradient().GetSize().Valid()) // Backpropagate?
      p_clOutData->GetGradient().SetSize(clOutSize);

    return true;
  }

  virtual bool Initialize() override {
    bleakGetAndCheckInput(p_clInData, "inData", false);
    bleakGetAndCheckOutput(p_clOutData, "outData", false);

    const Size clInSize = p_clInData->GetData().GetSize();
    const Size clOutSize = p_clInData->GetData().GetSize();

    if (!clInSize.Valid() || !clOutSize.Valid() || clInSize.GetDimension() != Dimension+2 || clOutSize.GetDimension() != clInSize.GetDimension())
      return false;

    m_clIndexMatrix.SetSize(clOutSize.SubSize(2));
    m_clIndexMatrix.Allocate();

    // Only need to do this once!
    ComputeNearestNeighborIndexMatrix<Dimension>(m_clIndexMatrix.data_no_sync(), clInSize.data()+2, clOutSize.data()+2);

    return true;
  }

  virtual void ForwardCPU() override {
    bleakGetAndCheckInput(p_clInData, "inData");
    bleakGetAndCheckOutput(p_clOutData, "outData");

    const ArrayType &clInData = p_clInData->GetData();
    ArrayType &clOutData = p_clOutData->GetData();

    const int iBatchSize = clOutData.GetSize()[0];
    const int iNumChannels = clOutData.GetSize()[1];
    const int iInInnerNum = clInData.GetSize().Count(2);
    const int iOutInnerNum = clOutData.GetSize().Count(2);

    const RealType * const p_inData = clInData.data();
    RealType * const p_outData = clOutData.data_no_sync();
    const int * const p_iIndexMatrix = m_clIndexMatrix.data();

#pragma omp parallel for
    for (int b = 0; b < iBatchSize; ++b) {
      for (int c = 0; c < iNumChannels; ++c) {
        for (int i = 0; i < iOutInnerNum; ++i) { 
          const int index = p_iIndexMatrix[i];
          p_outData[(b*iNumChannels + c)*iOutInnerNum + i] = p_inData[(b*iNumChannels + c)*iInInnerNum + index];
        }
      }
    }
  }

  virtual void BackwardCPU() override {
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

    RealType * const p_inGradient = clInGradient.data();
    const RealType * const p_outGradient = clOutGradient.data();

    const int * const p_iIndexMatrix = m_clIndexMatrix.data();

#pragma omp parallel for
    for (int b = 0; b < iBatchSize; ++b) {
      for (int c = 0; c < iNumChannels; ++c) {
        for (int i = 0; i < iOutInnerNum; ++i) {
          const int index = p_iIndexMatrix[i];
          p_inGradient[(b*iNumChannels + c)*iInInnerNum + index] += p_outGradient[(b*iNumChannels + c)*iOutInnerNum + i];
        }
      }
    }
  }

  virtual void ForwardGPU() override;
  virtual void BackwardGPU() override;


protected:
  NearestNeighborResample() = default;

private:
  std::vector<int> m_vSize;

  Array<int> m_clIndexMatrix;
};

template<typename RealType>
class NearestNeighborResample1D : public NearestNeighborResample<RealType, 1> {
public:
  typedef NearestNeighborResample<RealType, 1> WorkAroundVarArgsType;

  bleakNewVertex(NearestNeighborResample1D, WorkAroundVarArgsType);

protected:
  NearestNeighborResample1D() = default;
};

template<typename RealType>
class NearestNeighborResample2D : public NearestNeighborResample<RealType, 2> {
public:
  typedef NearestNeighborResample<RealType, 2> WorkAroundVarArgsType;

  bleakNewVertex(NearestNeighborResample2D, WorkAroundVarArgsType);

protected:
  NearestNeighborResample2D() = default;
};

template<typename RealType>
class NearestNeighborResample3D : public NearestNeighborResample<RealType, 3> {
public:
  typedef NearestNeighborResample<RealType, 3> WorkAroundVarArgsType;

  bleakNewVertex(NearestNeighborResample3D, WorkAroundVarArgsType);

protected:
  NearestNeighborResample3D() = default;
};


} // end namespace bleak

#endif // !BLEAK_NEARESTNEIGHBORRESAMPLE_H
