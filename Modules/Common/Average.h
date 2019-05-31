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

#ifndef BLEAK_AVERAGE_H
#define BLEAK_AVERAGE_H

#include <vector>
#include "PrintOutput.h"

namespace bleak {

template<typename RealType>
class Average : public PrintOutput<RealType, std::vector<RealType>> {
public:
  typedef PrintOutput<RealType, std::vector<RealType>> WorkAroundVarArgsType;

  bleakNewVertex(Average, WorkAroundVarArgsType,
    bleakAddInput("inData"));

  bleakForwardVertexTypedefs();

  using SuperType::Push;
  using SuperType::GetQueue;

  virtual ~Average() {
    SelfType::Print();
  }


  virtual bool SetSizes() override {
    if (!SuperType::SetSizes())
      return false;

    bleakGetAndCheckInput(p_clData, "inData", false);

    const ArrayType &clData = p_clData->GetData();

    if (!clData.GetSize().Valid()) {
      std::cerr << GetName() << ": Error: inData is invalid." << std::endl;
      return false;
    }

    if (clData.GetSize().GetDimension() > 2) {
      std::cerr << GetName() << ": Error: inData is expected to be at most 2D." << std::endl;
      return false;
    }

    return true;
  }

  virtual bool Initialize() override {
    bleakGetAndCheckInput(p_clData, "inData", false);

    m_iVectorSize = p_clData->GetData().GetSize().Product(1);

    return SuperType::Initialize();
  }

  virtual void Forward() override {
    bleakGetAndCheckInput(p_clData, "inData");

    const ArrayType &clData = p_clData->GetData();

    std::vector<RealType> vAverageData(m_iVectorSize, RealType());

    const RealType * const p_data = clData.data();

    const int iOuterNum = clData.GetSize()[0];
    const int iInnerNum = clData.GetSize().Product(1);

    for (int i = 0; i < iOuterNum; ++i) {
      for (int j = 0; j < iInnerNum; ++j) {
        vAverageData[j] += p_data[i*iInnerNum + j];
      }
    }

    for (int j = 0; j < iInnerNum; ++j)
      vAverageData[j] /= RealType(iOuterNum);

    Push(vAverageData);

    SuperType::Forward();
  }

protected:
  virtual void Print() override {
    if (GetQueue().empty())
      return;

    std::vector<RealType> vAverageData(m_iVectorSize, RealType());

    for (const std::vector<RealType> &vData : GetQueue()) {
      for (int j = 0; j < m_iVectorSize; ++j)
        vAverageData[j] += vData[j];
    }

    for (int j = 0; j < m_iVectorSize; ++j)
      vAverageData[j] /= RealType(GetQueue().size());

    std::cout << GetName() << ": Info: Current running average (last " << GetQueue().size() << " iterations) = ";

    if (m_iVectorSize == 1) {
      std::cout << vAverageData[0] << std::endl;
    }
    else if (m_iVectorSize > 1) {
      std::cout << "[ ";

      std::cout << vAverageData[0];

      for (int j = 1; j < vAverageData.size(); ++j)
        std::cout << ", " << vAverageData[j];

      std::cout << " ]" << std::endl; 
    }
  }

private:
  int m_iVectorSize = 0;
};

} // end namespace bleak

#endif // !BLEAK_AVERAGE_H
