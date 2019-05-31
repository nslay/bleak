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

#ifndef BLEAK_RESHAPE_H
#define BLEAK_RESHAPE_H

#include <vector>
#include "Vertex.h"

namespace bleak {

template<typename RealType>
class Reshape : public Vertex<RealType> {
public:
  bleakNewVertex(Reshape, Vertex<RealType>,
    bleakAddProperty("size", m_vSize),
    bleakAddInput("inData"),
    bleakAddOutput("outData"));

  bleakForwardVertexTypedefs();

  virtual ~Reshape() = default;

  virtual bool SetSizes() override {
    bleakGetAndCheckInput(p_clInData, "inData", false);
    bleakGetAndCheckOutput(p_clOutData, "outData", false);

    const ArrayType &clInData = p_clInData->GetData();

    if (!clInData.GetSize().Valid()) {
      std::cerr << GetName() << ": Error: Invalid dimensions for inData." << std::endl;
      return false;
    }

    Size clSubSize(m_vSize);

    if (clSubSize.GetDimension() > 0 && !clSubSize.Valid()) {
      std::cerr << GetName() << ": Error: Invalid size " << clSubSize << '.' << std::endl;
      return false;
    }
    
    if (clInData.GetSize().Product(1) != clSubSize.Product()) {
      std::cerr << GetName() << ": Error: Output size " << clSubSize << " is incompatible with input size " << 
        clInData.GetSize().SubSize(1) << '.' << std::endl;

      return false;
    }

    Size clOutSize(clSubSize.GetDimension() + 1);

    clOutSize[0] = clInData.GetSize()[0];
    std::copy(clSubSize.begin(), clSubSize.end(), clOutSize.begin() + 1);

    p_clOutData->GetData().SetSize(clOutSize);

    if (p_clInData->GetGradient().GetSize().Valid())
      p_clOutData->GetGradient().SetSize(clOutSize);
    else
      p_clOutData->GetGradient().Clear();

    return true;
  }

  virtual bool Initialize() override {
    bleakGetAndCheckInput(p_clInData, "inData", false);
    bleakGetAndCheckOutput(p_clOutData, "outData", false);

    p_clInData->GetData().ShareWith(p_clOutData->GetData());
    p_clInData->GetGradient().ShareWith(p_clOutData->GetGradient());

    return true;
  }

  virtual void Forward() override { } // Nothing to do
  virtual void Backward() override { } // Nothing to do

protected:
  Reshape() = default;

private:
  std::vector<int> m_vSize;
};

} // end namespace bleak

#endif // !BLEAK_RESHAPE_H
