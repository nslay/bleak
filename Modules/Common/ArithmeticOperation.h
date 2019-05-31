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

#ifndef BLEAK_ARITHMETICOPERATION_H
#define BLEAK_ARITHMETICOPERATION_H

#include <functional>
#include <algorithm>
#include "Vertex.h"

namespace bleak {

template<typename RealType>
struct PlusOp {
  RealType operator()(const RealType &a, const RealType &b) const { return a + b; }
  RealType DiffA(const RealType &a, const RealType &b) const { return RealType(1); }
  RealType DiffB(const RealType &a, const RealType &b) const { return RealType(1); }
};

template<typename RealType>
struct MinusOp {
  RealType operator()(const RealType &a, const RealType &b) const { return a - b; }
  RealType DiffA(const RealType &a, const RealType &b) const { return RealType(1); }
  RealType DiffB(const RealType &a, const RealType &b) const { return RealType(-1); }
};

template<typename RealType>
struct MultipliesOp {
  RealType operator()(const RealType &a, const RealType &b) const { return a * b; }
  RealType DiffA(const RealType &a, const RealType &b) const { return b; }
  RealType DiffB(const RealType &a, const RealType &b) const { return a; }
};

template<typename RealType>
struct DividesOp {
  RealType operator()(const RealType &a, const RealType &b) const { return a / b; }
  RealType DiffA(const RealType &a, const RealType &b) const { return RealType(1) / b; }
  RealType DiffB(const RealType &a, const RealType &b) const { return RealType(-a)/(b*b); }
};

template<typename RealType, template <typename> class OperationTemplate>
class BinaryOperation : public Vertex<RealType> {
public:
  typedef OperationTemplate<RealType> OperationType;

  bleakNewAbstractVertex(BinaryOperation, Vertex<RealType>,
    bleakAddInput("inData0"),
    bleakAddInput("inData1"),
    bleakAddOutput("outData"));

  bleakForwardVertexTypedefs();

  virtual ~BinaryOperation() = default;

  virtual bool SetSizes() override {
    bleakGetAndCheckInput(p_clInData0, "inData0", false);
    bleakGetAndCheckInput(p_clInData1, "inData1", false);
    bleakGetAndCheckOutput(p_clOutData, "outData", false);

    const ArrayType &clInData0 = p_clInData0->GetData();
    const ArrayType &clInData1 = p_clInData1->GetData();

    if (!clInData0.GetSize().Valid() || !clInData1.GetSize().Valid()) {
      std::cerr << GetName() << ": Error: Invalid dimensions for inData0 and/or inData1." << std::endl;
      return false;
    }

    if (clInData0.GetSize().Squeeze() != clInData1.GetSize().Squeeze()) {
      std::cerr << GetName() << ": Error: Dimension mismatch between inData0 and inData1." << std::endl;
      return false;
    }

    p_clOutData->GetData().SetSize(clInData0.GetSize());
    p_clOutData->GetGradient().SetSize(clInData0.GetSize());

    return true;
  }

  virtual bool Initialize() override {
    return true; // Nothing to do
  }

  virtual void Forward() override {
    bleakGetAndCheckInput(p_clInData0, "inData0");
    bleakGetAndCheckInput(p_clInData1, "inData1");
    bleakGetAndCheckOutput(p_clOutData, "outData");

    const ArrayType &clInData0 = p_clInData0->GetData();
    const ArrayType &clInData1 = p_clInData1->GetData();
    ArrayType &clOutData = p_clOutData->GetData();

    std::transform(clInData0.begin(), clInData0.end(), clInData1.begin(), clOutData.begin(), OperationType());
  }

  virtual void Backward() override {
    bleakGetAndCheckInput(p_clInData0, "inData0");
    bleakGetAndCheckInput(p_clInData1, "inData1");
    bleakGetAndCheckOutput(p_clOutData, "outData");

    const OperationType clOp;

    const ArrayType &clInData0 = p_clInData0->GetData();
    ArrayType &clInData0Gradient = p_clInData0->GetGradient();
    const ArrayType &clInData1 = p_clInData1->GetData();
    ArrayType &clInData1Gradient = p_clInData1->GetGradient();
    const ArrayType &clOutData = p_clOutData->GetData();
    const ArrayType &clOutDataGradient = p_clOutData->GetGradient();

    const RealType * const p_inData0 = clInData0.data();
    RealType * const p_inData0Gradient = clInData0Gradient.data();
    const RealType * const p_inData1 = clInData1.data();
    RealType * const p_inData1Gradient = clInData1Gradient.data();
    const RealType * const p_outDataGradient = clOutDataGradient.data();
    
    const size_t length = clOutData.GetSize().Product();

    if (p_inData0Gradient != nullptr) {
      for (size_t i = 0; i < length; ++i)
        p_inData0Gradient[i] += clOp.DiffA(p_inData0[i], p_inData1[i]) * p_outDataGradient[i];
    }

    if (p_inData1Gradient != nullptr) {
      for (size_t i = 0; i < length; ++i)
        p_inData1Gradient[i] += clOp.DiffB(p_inData0[i], p_inData1[i]) * p_outDataGradient[i];
    }
  }

protected:
  BinaryOperation() = default;
};

#define DECLARE_BINARY_OPERATION(vertexName, opType) \
template<typename RealType> \
class vertexName : public BinaryOperation<RealType, opType > { \
public: \
  typedef BinaryOperation<RealType, opType > WorkAroundVarArgsType; \
\
  bleakNewVertex( vertexName , WorkAroundVarArgsType); \
  virtual ~ vertexName () = default; \
\
protected: \
  vertexName () = default; \
}

DECLARE_BINARY_OPERATION(Plus, PlusOp);
DECLARE_BINARY_OPERATION(Minus, MinusOp);
DECLARE_BINARY_OPERATION(Multiplies, MultipliesOp);

#undef DECLARE_BINARY_OPERATION

} // end namespace bleak

#endif // !BLEAK_ARITHMETICOPERATION_H
