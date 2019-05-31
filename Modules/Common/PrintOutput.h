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

#include <deque>
#include "Vertex.h"

#ifndef BLEAK_PRINTOUTPUT_H
#define BLEAK_PRINTOUTPUT_H

namespace bleak {

template<typename RealType, typename OutputType>
class PrintOutput : public Vertex<RealType> {
public:
  bleakNewAbstractVertex(PrintOutput, Vertex<RealType>,
    bleakAddProperty("displayFrequency", m_iDisplayFrequency));

  virtual ~PrintOutput() = default;

  virtual bool SetSizes() override {
    if(m_iDisplayFrequency < 0) {
      std::cerr << GetName() << ": Error: displayFrequency expected to be non-negative." << std::endl;
      return false;
    }

    return true;
  }

  virtual bool Initialize() override {
    m_iIteration = 0;
    return true;
  }

  virtual void Forward() override {
    ++m_iIteration;

    if (m_iDisplayFrequency > 0 && (m_iIteration % m_iDisplayFrequency) == 0)
      Print();
  }

  virtual void Backward() override { } // Nothing to do

protected:
  PrintOutput() = default;

  virtual void Push(const OutputType &clOutput) {
    if (m_iDisplayFrequency > 0 && (int)m_dqOutputs.size() >= m_iDisplayFrequency)
      m_dqOutputs.pop_front();

    m_dqOutputs.push_back(clOutput);
  }

  const std::deque<OutputType> & GetQueue() const {
    return m_dqOutputs;
  }

  virtual void Print() = 0;

private:
  int m_iDisplayFrequency = 0;
  int m_iIteration = 0;

  std::deque<OutputType> m_dqOutputs;
};

} // end namespace bleak

#endif // !BLEAK_PRINTOUTPUT_H
