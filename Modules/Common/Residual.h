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

#ifndef BLEAK_RESIDUAL_H
#define BLEAK_RESIDUAL_H

#include <cmath>
#include <algorithm>
#include <iostream>
#include <utility>
#include "PrintOutput.h"

namespace bleak {

template<typename RealType>
class Residual : public PrintOutput<RealType, std::tuple<RealType, RealType, RealType>> {
public:
  typedef std::tuple<RealType, RealType, RealType> TupleType;
  typedef PrintOutput<RealType, TupleType> WorkAroundVarArgsType; // Comma in template declaration

  bleakNewVertex(Residual, WorkAroundVarArgsType,
    bleakAddInput("inData"),
    bleakAddInput("inLabels"));

  bleakForwardVertexTypedefs();

  using SuperType::Push;
  using SuperType::GetQueue;

  virtual ~Residual() {
    Print();
  }

  virtual bool SetSizes() override {
    if (!SuperType::SetSizes())
      return false;

    bleakGetAndCheckInput(p_clData, "inData", false);
    bleakGetAndCheckInput(p_clLabels, "inLabels", false);

    const ArrayType &clData = p_clData->GetData();
    const ArrayType &clLabels = p_clLabels->GetData();

    if (!clData.GetSize().Valid() || !clLabels.GetSize().Valid()) {
      std::cerr << GetName() << ": Error: inData and/or inLabels are invalid." << std::endl;
      return false;
    }

    if (clData.GetSize().GetDimension() > 2 || clLabels.GetSize().GetDimension() > 2) {
      std::cerr << GetName() << ": Error: inData and inLabels expected to be no larger than 2D." << std::endl;
      return false;
    }

    if (clData.GetSize().Squeeze().GetDimension() != clLabels.GetSize().Squeeze().GetDimension()) {
      std::cerr << GetName() << ": Error: Dimension mismatch between inData and inLabels." << std::endl;
      return false;
    }

    return true;
  }

  virtual void Forward() override {
    bleakGetAndCheckInput(p_clData, "inData");
    bleakGetAndCheckInput(p_clLabels,"inLabels");

    const ArrayType &clData = p_clData->GetData();
    const ArrayType &clLabels = p_clLabels->GetData();

    const RealType * const p_data = clData.data();
    const RealType * const p_labels = clLabels.data();

    const int iOuterNum = clData.GetSize()[0];
    const int iNumLabels = clData.GetSize().Product(1);

    RealType totalResidualMean = RealType();
    RealType totalLabelMean = RealType();
    RealType totalLabelMean2 = RealType();

    for (int j = 0; j < iNumLabels; ++j) {
      RealType residualMean = RealType();
      RealType labelMean = RealType();
      RealType labelMean2 = RealType();

      for (int i = 0; i < iOuterNum; ++i) {
        const RealType &label = p_labels[i*iNumLabels + j];

        const RealType residual = std::pow(p_data[i*iNumLabels + j] - label, 2);

        residualMean += residual;
        labelMean += label;
        labelMean2 += label*label;
      }

      residualMean /= RealType(iOuterNum);
      labelMean /= RealType(iOuterNum);
      labelMean2 /= RealType(iOuterNum);

      totalResidualMean += residualMean;
      totalLabelMean += labelMean;
      totalLabelMean2 += labelMean2;
    }

    Push(std::make_tuple(totalResidualMean, totalLabelMean, totalLabelMean2));

    SuperType::Forward();
  }

protected:
  Residual() = default;

  virtual void Print() override {
    if (GetQueue().empty())
      return;

    RealType totalResidualMean = RealType();
    RealType totalLabelMean = RealType();
    RealType totalLabelMean2 = RealType();

    for (const auto &clTuple : GetQueue()) {
      totalResidualMean += std::get<0>(clTuple);
      totalLabelMean += std::get<1>(clTuple);
      totalLabelMean2 += std::get<2>(clTuple);
    }

    totalResidualMean /= GetQueue().size();
    totalLabelMean /= GetQueue().size();
    totalLabelMean2 /= GetQueue().size();

    const RealType totalLabelVar = std::max(RealType(0), totalLabelMean2 - totalLabelMean*totalLabelMean);

    const RealType R2 = RealType(1) - totalResidualMean/(RealType(1e-7) + totalLabelVar);

    std::cout << GetName() << ": Info: Current running mean residual and R^2 (last " << GetQueue().size() << " iterations) = " << totalResidualMean << ", " << R2 << std::endl;
  }
};

} // end namespace bleak

#endif // !BLEAK_RESIDUAL_H
