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

#ifndef BLEAK_INPUT_H
#define BLEAK_INPUT_H

#include <vector>
#include "Size.h"
#include "Vertex.h"

namespace bleak {

// Placeholder vertex to compel memory to be allocated for inputs

template<typename RealType>
class Input : public Vertex<RealType> {
public:
  bleakNewVertex(Input, Vertex<RealType>,
    bleakAddProperty("size", m_vSize),
    bleakAddOutput("outData"));

  bleakForwardVertexTypedefs();

  virtual ~Input() = default;

  virtual bool SetSizes() override {
    bleakGetAndCheckOutput(p_clOutData, "outData", false);

    p_clOutData->GetGradient().Clear(); // This Vertex is never learnable

    Size clSize(m_vSize);

    if (!clSize.Valid()) {
      std::cerr << GetName() << ": Error: size = " << clSize << " is not valid." << std::endl;
      return false;
    }

    ArrayType &clOutData = p_clOutData->GetData();

    clOutData.SetSize(clSize);

    return true;
  }

  virtual bool Initialize() override {
    return true;
  }

  virtual void Forward() override { }
  virtual void Backward() override { }

protected:
  Input() = default;

private:
  std::vector<int> m_vSize;
};

} // end namespace bleak

#endif // !BLEAK_INPUT_H
