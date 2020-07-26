/*-
 * Copyright (c) 2017 Nathan Lay (enslay@gmail.com)
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

#ifndef BLEAK_CONCATENATE_H
#define BLEAK_CONCATENATE_H

#include <algorithm>
#include "Vertex.h"
#include "BlasWrapper.h"

namespace bleak {

template<typename RealType>
class Concatenate : public Vertex<RealType> {
public:
  bleakNewVertex(Concatenate, Vertex<RealType>,
    bleakAddProperty("axis", m_iAxis),
    bleakAddInput("inData0"),
    bleakAddInput("inData1"),
    bleakAddInput("inData2"),
    bleakAddInput("inData3"),
    bleakAddInput("inData4"),
    bleakAddOutput("outData"));

  bleakForwardVertexTypedefs();

  static constexpr int GetMaxInputs() {
    return 5;
  }

  static const char * GetInputName(int i) {
    return i >= 0 && i < GetMaxInputs() ? s_a_cInputNames[i] : "";
  }

  virtual ~Concatenate() = default;

  virtual bool SetSizes() override {
    bleakGetAndCheckOutput(p_clOutData, "outData", false);

    if (m_iAxis < 0) {
      std::cerr << GetName() << ": Error: 'axis' is expected to be non-negative." << std::endl;
      return false;
    }

    const char *p_cRefInputName = nullptr;
    std::shared_ptr<EdgeType> p_clRefInput;

    // Check validity of inputs and sum up concatenation axis
    int iAxisSum = 0;
    bool bHasLearnableInputs = false;

    // XXX: May need to revisit for Squeeze()?
    for (int i = 0; i < GetMaxInputs(); ++i) {
      const char * const p_cInputName = GetInputName(i);
      std::shared_ptr<EdgeType> p_clInput = GetInput(p_cInputName);

      if (p_clInput == nullptr)
        continue;

      const ArrayType &clData = p_clInput->GetData();

      if (!clData.GetSize().Valid()) {
        std::cerr << GetName() << ": Error: Invalid input '" << p_cInputName << "'." << std::endl;
        return false;
      }

      if (clData.GetSize().GetDimension() <= m_iAxis) {
        std::cerr << GetName() << ": Error: 'axis' exceeds the dimension of input '" << p_cInputName << "' (" << clData.GetSize().GetDimension() << ")." << std::endl;
        return false;
      }

      if (p_clRefInput != nullptr) {
        const ArrayType &clRefData = p_clRefInput->GetData();
        if (clRefData.GetSize().SubSize(0, m_iAxis) != clData.GetSize().SubSize(0, m_iAxis) || 
          clRefData.GetSize().SubSize(m_iAxis+1) != clData.GetSize().SubSize(m_iAxis+1)) {
          std::cerr << GetName() << ": Error: Dimension mismatch between inputs '" << p_cRefInputName << "' and '" << p_cInputName << '.' << std::endl;
          return false;
        }
      }
      else {
        p_cRefInputName = p_cInputName;
        p_clRefInput = p_clInput;
      }

      iAxisSum += clData.GetSize()[m_iAxis];

      if (p_clInput->GetGradient().GetSize().Valid())
        bHasLearnableInputs = true;
    }

    if (p_clRefInput == nullptr) {
      std::cerr << GetName() << ": Error: At least one input must be provided." << std::endl;
      return false;
    }
    
    Size clOutSize(p_clRefInput->GetData().GetSize());

    clOutSize[m_iAxis] = iAxisSum;

    p_clOutData->GetData().SetSize(clOutSize);

    if (bHasLearnableInputs)
      p_clOutData->GetGradient().SetSize(clOutSize);
    else
      p_clOutData->GetGradient().Clear();

    return true;
  }

  virtual bool Initialize() override {
    return true; // Nothing to do
  }

  virtual void ForwardCPU() override {
    bleakGetAndCheckOutput(p_clOutData, "outData");

    ArrayType &clOutData = p_clOutData->GetData();
    RealType * const p_outData = clOutData.data_no_sync();

    const int iOuterNum = clOutData.GetSize().Product(0, m_iAxis);
    const int iAxisNum = clOutData.GetSize()[m_iAxis];
    const int iInnerNum = clOutData.GetSize().Product(m_iAxis+1);

    // Collect data pointers
    const RealType * a_inData[GetMaxInputs()] = { nullptr };
    int a_iInputAxisNums[GetMaxInputs()] = { 0 };

    for (int i = 0; i < GetMaxInputs(); ++i) {
      const char * const p_cInputName = GetInputName(i);

      std::shared_ptr<EdgeType> p_clInData = GetInput(p_cInputName);

      a_inData[i] = nullptr;
      a_iInputAxisNums[i] = 0;

      if (p_clInData != nullptr) {
        a_inData[i] = p_clInData->GetData().data();
        a_iInputAxisNums[i] = p_clInData->GetData().GetSize()[m_iAxis];
      }
    }

    for (int i = 0; i < iOuterNum; ++i) {
      int j = 0;

      for (int l = 0; l < GetMaxInputs(); ++l) {
        const RealType * const p_inData = a_inData[l];
        const int iInputAxisNum = a_iInputAxisNums[l];

        const int jBegin = j;
        const int jEnd = jBegin + iInputAxisNum;

        if (p_inData != nullptr) {
          const int iCount = (jEnd - jBegin)*iInnerNum;
          cpu_blas::copy(iCount, p_inData + (i*iInputAxisNum + 0)*iInnerNum, 1, p_outData + (i*iAxisNum + jBegin)*iInnerNum, 1);
        }

        j = jEnd;

        //for ( ; j < jEnd; ++j) {
        //  for (int k = 0; k < iInnerNum; ++k) {
        //    p_outData[(i*iAxisNum + j)*iInnerNum + k] = p_inData[(i*iInputAxisNum + j-jBegin)*iInnerNum + k];
        //  }
        //}
      }
    }
  }

  virtual void BackwardCPU() override {
    bleakGetAndCheckOutput(p_clOutData, "outData");

    const ArrayType &clOutData = p_clOutData->GetData();
    const ArrayType &clOutDataGradient = p_clOutData->GetGradient();

    if (!clOutDataGradient.Valid())
      return; // Nothing to do

    const RealType * const p_outDataGradient = clOutDataGradient.data();

    const int iOuterNum = clOutData.GetSize().Product(0, m_iAxis);
    const int iAxisNum = clOutData.GetSize()[m_iAxis];
    const int iInnerNum = clOutData.GetSize().Product(m_iAxis+1);

    // Collect data pointers
    RealType * a_inDataGradient[GetMaxInputs()] = { nullptr };
    int a_iInputAxisNums[GetMaxInputs()] = { 0 };

    for (int i = 0; i < GetMaxInputs(); ++i) {
      const char * const p_cInputName = GetInputName(i);

      std::shared_ptr<EdgeType> p_clInData = GetInput(p_cInputName);

      a_inDataGradient[i] = nullptr;
      a_iInputAxisNums[i] = 0;

      if (p_clInData != nullptr) {
        a_inDataGradient[i] = p_clInData->GetGradient().data(); // Should be nullptr if not learnable
        a_iInputAxisNums[i] = p_clInData->GetData().GetSize()[m_iAxis];
      }
    }

    for (int i = 0; i < iOuterNum; ++i) {
      int j = 0;

      for (int l = 0; l < GetMaxInputs(); ++l) {
        RealType * const p_inDataGradient = a_inDataGradient[l];
        const int iInputAxisNum = a_iInputAxisNums[l];

        const int jBegin = j;
        const int jEnd = jBegin + iInputAxisNum;

        if (p_inDataGradient != nullptr) {
          const int iCount = (jEnd - jBegin)*iInnerNum;
          cpu_blas::axpy(iCount, RealType(1), p_outDataGradient + (i*iAxisNum + jBegin)*iInnerNum, 1, p_inDataGradient + (i*iInputAxisNum + 0)*iInnerNum, 1);
        }

        j = jEnd;

        //for ( ; j < jEnd; ++j) {
        //  for (int k = 0; k < iInnerNum; ++k) {
        //    p_inDataGradient[(i*iInputAxisNum + (j-jBegin))*iInnerNum + k] += p_outDataGradient[(i*iAxisNum + j)*iInnerNum + k];
        //  }
        //}
      }
    }
  }

#ifdef BLEAK_USE_CUDA
  virtual void ForwardGPU() override {
    bleakGetAndCheckOutput(p_clOutData, "outData");

    ArrayType &clOutData = p_clOutData->GetData();
    RealType * const p_outData = clOutData.data_no_sync(GPU);

    const int iOuterNum = clOutData.GetSize().Product(0, m_iAxis);
    const int iAxisNum = clOutData.GetSize()[m_iAxis];
    const int iInnerNum = clOutData.GetSize().Product(m_iAxis+1);

    // Collect data pointers
    const RealType * a_inData[GetMaxInputs()] = { nullptr };
    int a_iInputAxisNums[GetMaxInputs()] = { 0 };

    for (int i = 0; i < GetMaxInputs(); ++i) {
      const char * const p_cInputName = GetInputName(i);

      std::shared_ptr<EdgeType> p_clInData = GetInput(p_cInputName);

      a_inData[i] = nullptr;
      a_iInputAxisNums[i] = 0;

      if (p_clInData != nullptr) {
        a_inData[i] = p_clInData->GetData().data(GPU);
        a_iInputAxisNums[i] = p_clInData->GetData().GetSize()[m_iAxis];
      }
    }

    for (int i = 0; i < iOuterNum; ++i) {
      int j = 0;

      for (int l = 0; l < GetMaxInputs(); ++l) {
        const RealType * const p_inData = a_inData[l];
        const int iInputAxisNum = a_iInputAxisNums[l];

        const int jBegin = j;
        const int jEnd = jBegin + iInputAxisNum;

        if (p_inData != nullptr) {
          const int iCount = (jEnd - jBegin)*iInnerNum;
          gpu_blas::copy(iCount, p_inData + (i*iInputAxisNum + 0)*iInnerNum, 1, p_outData + (i*iAxisNum + jBegin)*iInnerNum, 1);
        }

        j = jEnd;
      }
    }
  }

  virtual void BackwardGPU() override {
    bleakGetAndCheckOutput(p_clOutData, "outData");

    const ArrayType &clOutData = p_clOutData->GetData();
    const ArrayType &clOutDataGradient = p_clOutData->GetGradient();

    if (!clOutDataGradient.Valid())
      return; // Nothing to do

    const RealType * const p_outDataGradient = clOutDataGradient.data(GPU);

    const int iOuterNum = clOutData.GetSize().Product(0, m_iAxis);
    const int iAxisNum = clOutData.GetSize()[m_iAxis];
    const int iInnerNum = clOutData.GetSize().Product(m_iAxis+1);

    // Collect data pointers
    RealType * a_inDataGradient[GetMaxInputs()] = { nullptr };
    int a_iInputAxisNums[GetMaxInputs()] = { 0 };

    for (int i = 0; i < GetMaxInputs(); ++i) {
      const char * const p_cInputName = GetInputName(i);

      std::shared_ptr<EdgeType> p_clInData = GetInput(p_cInputName);

      a_inDataGradient[i] = nullptr;
      a_iInputAxisNums[i] = 0;

      if (p_clInData != nullptr) {
        a_inDataGradient[i] = p_clInData->GetGradient().data(GPU); // Should be nullptr if not learnable
        a_iInputAxisNums[i] = p_clInData->GetData().GetSize()[m_iAxis];
      }
    }

    for (int i = 0; i < iOuterNum; ++i) {
      int j = 0;

      for (int l = 0; l < GetMaxInputs(); ++l) {
        RealType * const p_inDataGradient = a_inDataGradient[l];
        const int iInputAxisNum = a_iInputAxisNums[l];

        const int jBegin = j;
        const int jEnd = jBegin + iInputAxisNum;

        if (p_inDataGradient != nullptr) {
          const int iCount = (jEnd - jBegin)*iInnerNum;
          gpu_blas::axpy(iCount, RealType(1), p_outDataGradient + (i*iAxisNum + jBegin)*iInnerNum, 1, p_inDataGradient + (i*iInputAxisNum + 0)*iInnerNum, 1);
        }

        j = jEnd;
      }
    }
  }
#endif // BLEAK_USE_CUDA

protected:
  Concatenate() = default;

private:
  static const char * const s_a_cInputNames[GetMaxInputs()]; // Declare this for ordering reasons

  int m_iAxis = 1; // Concatenate channels by default
};

template<typename RealType>
const char * const Concatenate<RealType>::s_a_cInputNames[] = { "inData0", "inData1", "inData2", "inData3", "inData4" };

} // end namespace bleak

#endif // !BLEAK_CONCATENATE_H
