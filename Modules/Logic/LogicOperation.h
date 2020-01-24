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

#ifndef BLEAK_LOGICOPERATION_H
#define BLEAK_LOGICOPERATION_H

#include <cmath>
#include <algorithm>
#include "ArithmeticOperation.h"

namespace bleak {

// Reuse ArithmeticOperation for these!

// Let a, b be real numbers and sgn(a), sgn(b) are the ternary logic values of a and b (T = 1, U = 0, F = -1).
//
// Define:
// a or b = max(a + b, sgn(a)*sgn(b)*(a + b))
// a and b = min(a + b, sgn(a)*sgn(b)*(a + b))
// a xor b = min(a or b, -(a and b))
//
// These logic operators behave like binary logic operators in the sign of the number while also mimicking addition.
// As such, the magnitudes of and/or/xor are all |a+b|. That said, they are invalid on the line a + b = 0, producing 0 as the result of the operation!
// They were crafted to be computationally efficient while having a useful gradient!
//
// And/or (but not xor) have a pseudo inverse (subtraction):
// a' = (a or b) - b
// (a' or b) = (a or b)
//
// a' = (a and b) - b
// (a' and b) = (a and b)
//
// In both cases, it may happen that a' != a. The result for a' will be unique when or is false or when and is true.
//
// Some care is needed when mixing these operations. They can cancel each other out in portions of the domain.
// For example: a xor b = (a or b) and -(a and b) will result in a prediction of 0 in two of xor's sqaure partitions! (it's related to a + b = 0 problem)
//
// You can use max(x,y) and min(x,y) as a substitute for or and and respectively.

template<typename RealType>
struct  PlusOrOp {
  RealType Sign(const RealType &a) const { return RealType(a > RealType(0)) - RealType(a < RealType(0)); }
  RealType Identity() const { return RealType(-1e-7); }

  RealType operator()(const RealType &a, const RealType &b) const { return std::max(a + b, Sign(a)*Sign(b)*(a + b)); }

  RealType DiffA(const RealType &a, const RealType &b) const {
    const RealType sgnab = Sign(a)*Sign(b);

    if (a + b < sgnab*(a + b))
      return sgnab;

    return RealType(1);
  }

  RealType DiffB(const RealType &a, const RealType &b) const { return DiffA(a,b); }
};

template<typename RealType>
struct  PlusAndOp {
  RealType Sign(const RealType &a) const { return RealType(a > RealType(0)) - RealType(a < RealType(0)); }
  RealType Identity() const { return RealType(1e-7); }

  RealType operator()(const RealType &a, const RealType &b) const { return std::min(a + b, Sign(a)*Sign(b)*(a + b)); }

  RealType DiffA(const RealType &a, const RealType &b) const {
    const RealType sgnab = Sign(a)*Sign(b);

    if (sgnab*(a + b) < a + b)
      return sgnab;
      
    return RealType(1);
  }

  RealType DiffB(const RealType &a, const RealType &b) const { return DiffA(a,b); }
};

template<typename RealType>
struct  PlusXorOp {
  RealType operator()(const RealType &a, const RealType &b) const { 
    const PlusOrOp<RealType> clOr;
    const PlusAndOp<RealType> clAnd;

    return std::min(clOr(a,b), -clAnd(a,b)); 
  }

  RealType DiffA(const RealType &a, const RealType &b) const {
    const PlusOrOp<RealType> clOr;
    const PlusAndOp<RealType> clAnd;

    if (clOr(a,b) < -clAnd(a,b))
      return clOr.DiffA(a, b);

    return -clAnd.DiffA(a, b);
  }

  RealType DiffB(const RealType &a, const RealType &b) const {
    const PlusOrOp<RealType> clOr;
    const PlusAndOp<RealType> clAnd;

    if (clOr(a,b) < -clAnd(a,b))
      return clOr.DiffB(a, b);

    return -clAnd.DiffB(a, b);
  }
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

DECLARE_BINARY_OPERATION(PlusOr, PlusOrOp);
DECLARE_BINARY_OPERATION(PlusAnd, PlusAndOp);
DECLARE_BINARY_OPERATION(PlusXor, PlusXorOp);

#undef DECLARE_BINARY_OPERATION

} // end namespace bleak

#endif // !BLEAK_LOGICOPERATION_H
