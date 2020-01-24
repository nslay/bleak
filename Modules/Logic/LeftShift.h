/*-
 * Copyright (c) 2019 Nathan Lay (enslay@gmail.com)
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

#ifndef BLEAK_LEFTSHIFT_H
#define BLEAK_LEFTSHIFT_H

#include <algorithm>
#include <functional>
#include "Vertex.h"

namespace bleak {

template<typename RealType>
class LeftShift : public Vertex<RealType> {
public:
  bleakNewVertex(LeftShift, Vertex<RealType>,
    bleakAddProperty("fill", m_fFill),
    bleakAddProperty("shift", m_iShift),
    bleakAddInput("inData"),
    bleakAddOutput("outData"));

  bleakForwardVertexTypedefs();

  virtual ~LeftShift() = default;

  virtual bool SetSizes() override {
    bleakGetAndCheckInput(p_clInData, "inData", false);
    bleakGetAndCheckOutput(p_clOutData, "outData", false);

    if (m_iShift < 0) {
      std::cerr << GetName() << ": Error: Invalid shift (" << m_iShift << "). Can only left shift by a non-negative value." << std::endl;
      return false;
    }

    if (m_iShift == 0)
      std::cerr << GetName() << ": Warning: Shift is 0. This is basically no-op." << std::endl;

    const ArrayType &clInData = p_clInData->GetData();
    ArrayType &clOutData = p_clOutData->GetData();
    ArrayType &clOutGradient = p_clOutData->GetGradient();

    if (!clInData.GetSize().Valid()) {
      std::cerr << GetName() << ": Error: Invalid dimensions for inData." << std::endl;
      return false;
    }

    if (clInData.GetSize().GetDimension() < 2) {
      std::cerr << GetName() << ": Error: inData is expected to be 2D or higher." << std::endl;
      return false;
    }

    if ((size_t)m_iShift >= clInData.GetSize().Back()) {
      std::cerr << "Error: There are too few components to shift (" << clInData.GetSize().Back() << " < " << m_iShift << ")." << std::endl;
      return false;
    }

    clOutData.SetSize(clInData.GetSize());
    clOutGradient.SetSize(clInData.GetSize());

    return true;
  }

  virtual bool Initialize() override {
    bleakGetAndCheckInput(p_clInData, "inData", false);
    bleakGetAndCheckOutput(p_clOutData, "outData", false);

    return true;
  }

  virtual void Forward() override {
    bleakGetAndCheckInput(p_clInData, "inData");
    bleakGetAndCheckOutput(p_clOutData, "outData");

    const ArrayType &clInData = p_clInData->GetData();
    ArrayType &clOutData = p_clOutData->GetData();

    const RealType * const p_inData = clInData.data();
    RealType * const p_outData = clOutData.data();

    const int iInnerNum = clInData.GetSize().Back();
    const int iOuterNum = clInData.GetSize().Product()/iInnerNum;

    clOutData.Fill(RealType(m_fFill));

    for (int i = 0; i < iOuterNum; ++i) {
      const RealType * const p_inDataBegin = p_inData + (i*iInnerNum + 0);
      const RealType * const p_inDataEnd = p_inDataBegin + (iInnerNum - m_iShift);
      RealType * const p_outDataBegin = p_outData + (i*iInnerNum + m_iShift);

      std::copy(p_inDataBegin, p_inDataEnd, p_outDataBegin);
    }
  }

  virtual void Backward() override {
    bleakGetAndCheckInput(p_clInData, "inData");
    bleakGetAndCheckOutput(p_clOutData, "outData");

    ArrayType &clInGradient = p_clInData->GetGradient();
    const ArrayType &clOutGradient = p_clOutData->GetGradient();

    RealType * const p_inGradient = clInGradient.data();
    const RealType * const p_outGradient = clOutGradient.data();

    if (p_inGradient == nullptr) // Nothing to do
      return;

    const int iInnerNum = clInGradient.GetSize().Back();
    const int iOuterNum = clInGradient.GetSize().Product()/iInnerNum;

    for (int i = 0; i < iOuterNum; ++i) {
      RealType * const p_inGradientBegin = p_inGradient + (i*iInnerNum);
      const RealType * const p_outGradientBegin = p_outGradient + (i*iInnerNum + m_iShift);
      const RealType * const p_outGradientEnd = p_outGradient + ((i+1)*iInnerNum);

      std::transform(p_outGradientBegin, p_outGradientEnd, p_inGradientBegin, p_inGradientBegin, std::plus<RealType>());
    }
  }

protected:
  LeftShift() = default;

private:
  float m_fFill = -1e-7f;
  int m_iShift = 0;
};

} // end namespace bleak

#endif // !BLEAK_LEFTSHIFT_H
