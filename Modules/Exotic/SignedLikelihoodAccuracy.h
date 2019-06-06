/*-
 * Copyright (c) 2018 Nathan Lay (enslay@gmail.com)
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

#ifndef BLEAK_SIGNEDLIKELIHOODACCURACY_H
#define BLEAK_SIGNEDLIKELIHOODACCURACY_H

#include <cmath>
#include "Vertex.h"
#include "Accuracy.h"

namespace bleak {

template<typename RealType>
class SignedLikelihoodAccuracy : public PrintOutput<RealType, ConfusionMatrix<RealType>> {
public:
  typedef PrintOutput<RealType, ConfusionMatrix<RealType>> WorkAroundVarArgsType; // Comma in template declaration

  bleakNewVertex(SignedLikelihoodAccuracy, WorkAroundVarArgsType,
    bleakAddProperty("printConfusionMatrix", m_bPrintConfusionMatrix),
    bleakAddProperty("ignoreUncertain", m_bIgnoreUncertain),
    bleakAddInput("inLikelihoods"),
    bleakAddInput("inLabels"));

  bleakForwardVertexTypedefs();

  using SuperType::Push;
  using SuperType::GetQueue;

  virtual ~SignedLikelihoodAccuracy() {
    SelfType::Print();
  }

  virtual bool SetSizes() override {
    if (!SuperType::SetSizes())
      return false;

    bleakGetAndCheckInput(p_clLikelihoods, "inLikelihoods", false);
    bleakGetAndCheckInput(p_clLabels, "inLabels", false);

    const ArrayType &clLikelihoods = p_clLikelihoods->GetData();
    const ArrayType &clLabels = p_clLabels->GetData();

    if (!clLikelihoods.GetSize().Valid() || !clLabels.GetSize().Valid()) {
      std::cerr << GetName() << ": Error: inLikelihoods and/or inLabels are invalid." << std::endl;
      return false;
    }

    if (clLikelihoods.GetSize().GetDimension() < 2) {
      std::cerr << GetName() << ": Error: inLikelihoods is expected to be at least 2D." << std::endl;
      return false;
    }

    if (clLikelihoods.GetSize()[0] != clLabels.GetSize()[0] || clLikelihoods.GetSize().SubSize(2) != clLabels.GetSize().SubSize(1)) {
      std::cerr << GetName() << ": Error: Dimension mismatch between inLikelihoods and inLabels." << std::endl;
      return false;
    }

    if (clLikelihoods.GetSize()[1] < 2) {
      std::cerr << GetName() << ": Error: inLikelihoods expected to have at least 2 channels." << std::endl;
      return false;
    }

    return true;
  }

  virtual bool Initialize() override {
    bleakGetAndCheckInput(p_clLikelihoods, "inLikelihoods", false);

    m_iNumClasses = p_clLikelihoods->GetData().GetSize()[1];

    return SuperType::Initialize();
  }

  virtual void Forward() override {
    bleakGetAndCheckInput(p_clLikelihoods, "inLikelihoods");
    bleakGetAndCheckInput(p_clLabels, "inLabels");

    const ArrayType &clLikelihoods = p_clLikelihoods->GetData();
    const ArrayType &clLabels = p_clLabels->GetData();

    const RealType * const p_likelihoods = clLikelihoods.data();
    const RealType * const p_labels = clLabels.data();

    const int iOuterNum = clLikelihoods.GetSize()[0];
    const int iNumClasses = clLikelihoods.GetSize()[1];
    const int iInnerNum = clLikelihoods.GetSize().Product(2);

    MatrixType confusionMatrix;

    for (int i = 0; i < iOuterNum; ++i) {
      for (int k = 0; k < iInnerNum; ++k) {
        const int iGTLabel = (int)(p_labels[i*iInnerNum + k]);

        int iMinLabel = -1;
        RealType minCertaintyFor = RealType(1000000);

        for (int j = 0; j < iNumClasses; ++j) {
          const RealType likelihood = p_likelihoods[(i*iNumClasses + j)*iInnerNum + k];

          if (likelihood <= RealType(0))
            continue;

          const RealType certaintyFor = std::abs(std::log(likelihood));

          if (certaintyFor < minCertaintyFor) {
            minCertaintyFor = certaintyFor;
            iMinLabel = j;
          }
        }

        // TODO: Average certainty?

        // TODO: Not predicting is not the same as predicting incorrectly!
        if (iMinLabel < 0) {
          if (m_bIgnoreUncertain)
            continue;

          iMinLabel = !iGTLabel; // Pick anything but GT when we fail to predict anything!
        }

        confusionMatrix(iMinLabel, iGTLabel) += RealType(1);
      }
    }

    Push(confusionMatrix);

    SuperType::Forward();
  }

protected:
  SignedLikelihoodAccuracy() = default;

  virtual void Print() override {
    PrintAccuracy();

    if (m_bPrintConfusionMatrix)
      PrintConfusionMatrix();
  }

private:
  typedef ConfusionMatrix<RealType> MatrixType;

  bool m_bPrintConfusionMatrix = false;
  bool m_bIgnoreUncertain = false;
  int m_iNumClasses = 0;

  void PrintAccuracy() {
    if (GetQueue().empty())
      return;

    MatrixType sumConfusionMatrix;

    for (const MatrixType &confusionMatrix : GetQueue())
      sumConfusionMatrix += confusionMatrix;

    std::cout << GetName() << ": Info: Current running accuracy (last " << GetQueue().size() << " iterations) = " << sumConfusionMatrix.Accuracy() << std::endl;
  }

  void PrintConfusionMatrix() {
    if (GetQueue().empty() || m_iNumClasses <= 0)
      return;

    MatrixType sumConfusionMatrix;

    for (const MatrixType &confusionMatrix : GetQueue())
      sumConfusionMatrix += confusionMatrix;

    std::cout << GetName() << ": Info: Current running confusion matrix (last " << GetQueue().size() << " iterations) = " << std::endl;

    std::cout << '[';

    for (int i = 0; i < m_iNumClasses; ++i) {
      for (int j = 0; j < m_iNumClasses; ++j) {
        const RealType value = sumConfusionMatrix.HasElement(i,j) ? sumConfusionMatrix(i,j) : RealType();

        std::cout << ' ' << value;
      }

      if (i+1 < m_iNumClasses)
        std::cout << ";\n";
      else
        std::cout << "; ];\n";
    }

    std::cout << std::flush;
  }
};

} // end namespace bleak

#endif // !BLEAK_SIGNEDLIKELIHOODACCURACY_H
