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

#ifndef BLEAK_ACCURACY_H
#define BLEAK_ACCURACY_H

#include <cstdint>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include "PrintOutput.h"

namespace bleak {

// (predictedClass, actualClass)
template<typename RealType>
class ConfusionMatrix {
public:
  typedef uint32_t IndexType;
  typedef uint64_t KeyType;
  typedef std::unordered_map<KeyType, RealType> MapType;
  typedef std::unordered_set<IndexType> SetType;

  static KeyType MakeKey(IndexType i, IndexType j) {
    return (KeyType(i) << 32) | KeyType(j);
  }

  bool HasElement(IndexType i, IndexType j) const {
    return m_mMatrix.find(MakeKey(i,j)) != m_mMatrix.end();
  }

  RealType & operator()(KeyType key) {
    const IndexType i = IndexType(key);
    const IndexType j = IndexType(key >> 32);

    m_sRowIndices.insert(i);
    m_sColumnIndices.insert(j);

    return m_mMatrix[key];
  }

  RealType operator()(KeyType key) const {
    auto itr = m_mMatrix.find(key);
    return itr != m_mMatrix.end() ? itr->second : RealType();
  }

  RealType & operator()(IndexType i, IndexType j) {
    return operator()(MakeKey(i,j));
  }

  RealType operator()(IndexType i, IndexType j) const {
    return operator()(MakeKey(i,j));
  }

  void Clear() {
    m_mMatrix.clear();
    m_sRowIndices.clear();
    m_sColumnIndices.clear();
  }

  RealType Count(IndexType j) const {
    if (m_sColumnIndices.find(j) == m_sColumnIndices.end())
      return RealType();

    RealType sum = RealType();

    for (const auto &i : m_sRowIndices)
      sum += operator()(i, j);

    return sum;
  }

  RealType Count() const {
    RealType sum = RealType();

    for (const auto &clPair : m_mMatrix)
      sum += clPair.second;

    return sum;
  }

  RealType Accuracy() const {
    const RealType count = Count();
    RealType trace = RealType();

    for(const auto &j : m_sColumnIndices)
      trace += operator()(j, j);

    return count != RealType() ? trace/count : RealType();
  }

  void Normalize() {
    for (const auto &j : m_sColumnIndices) {
      const RealType count = Count(j);

      if (count != RealType()) {
        for (const auto &i : m_sRowIndices)
          operator()(i,j) /= count;
      }
    }
  }

  ConfusionMatrix operator+=(const ConfusionMatrix &clOther) {
    for (const auto &clPair : clOther.m_mMatrix)
      operator()(clPair.first) += clPair.second;

    return *this;
  }

private:
  MapType m_mMatrix;
  SetType m_sRowIndices;
  SetType m_sColumnIndices;
};

template<typename RealType>
class Accuracy : public PrintOutput<RealType, ConfusionMatrix<RealType>> {
public:
  typedef PrintOutput<RealType, ConfusionMatrix<RealType>> WorkAroundVarArgsType; // Comma in template declaration

  bleakNewVertex(Accuracy, WorkAroundVarArgsType,
    bleakAddProperty("printConfusionMatrix", m_bPrintConfusionMatrix),
    bleakAddInput("inProbabilities"),
    bleakAddInput("inLabels"));

  bleakForwardVertexTypedefs();

  using SuperType::Push;
  using SuperType::GetQueue;

  virtual ~Accuracy() {
    SelfType::Print();
  }

  virtual bool SetSizes() override {
    if (!SuperType::SetSizes())
      return false;

    bleakGetAndCheckInput(p_clProbs, "inProbabilities", false);
    bleakGetAndCheckInput(p_clLabels, "inLabels", false);

    const ArrayType &clProbs = p_clProbs->GetData();
    const ArrayType &clLabels = p_clLabels->GetData();

    if (!clProbs.GetSize().Valid() || !clLabels.GetSize().Valid()) {
      std::cerr << GetName() << ": Error: inProbabilities and/or inLabels are invalid." << std::endl;
      return false;
    }

    if (clProbs.GetSize().GetDimension() < 2) {
      std::cerr << GetName() << ": Error: inProbabilities is expected to be at least 2D." << std::endl;
      return false;
    }

    if (clProbs.GetSize()[0] != clLabels.GetSize()[0] || clProbs.GetSize().SubSize(2) != clLabels.GetSize().SubSize(1)) {
      std::cerr << GetName() << ": Error: Dimension mismatch between inProbabilities and inLabels." << std::endl;
      return false;
    }

    if (clProbs.GetSize()[1] < 2) {
      std::cerr << GetName() << ": Error: inProbabilities expected to have at least 2 channels." << std::endl;
      return false;
    }

    return true;
  }

  virtual bool Initialize() override {
    bleakGetAndCheckInput(p_clProbs, "inProbabilities", false);

    m_iNumClasses = p_clProbs->GetData().GetSize()[1];

    return SuperType::Initialize();
  }

  virtual void Forward() override {
    bleakGetAndCheckInput(p_clProbs,"inProbabilities");
    bleakGetAndCheckInput(p_clLabels,"inLabels");

    const ArrayType &clProbs = p_clProbs->GetData();
    const ArrayType &clLabels = p_clLabels->GetData();

    const RealType * const p_probs = clProbs.data();
    const RealType * const p_labels = clLabels.data();

    const int iOuterNum = clProbs.GetSize()[0];
    const int iNumClasses = clProbs.GetSize()[1];
    const int iInnerNum = clProbs.GetSize().Product(2);

    MatrixType confusionMatrix;

    for (int i = 0; i < iOuterNum; ++i) {
      for (int k = 0; k < iInnerNum; ++k) {
        const int iGTLabel = (int)(p_labels[i*iInnerNum + k]);

        if (iGTLabel < 0 || iGTLabel >= iNumClasses)
          continue;

        int iMaxLabel = 0;
        RealType maxProb = p_probs[(i*iNumClasses + 0)*iInnerNum + k];

        for (int j = 1; j < iNumClasses; ++j) {
          const RealType prob = p_probs[(i*iNumClasses + j)*iInnerNum + k];

          if (prob > maxProb) {
            iMaxLabel = j;
            maxProb = prob;
          }
        }

        //std::cout << GetName() << ": Info: prob = " << maxProb << ", label = " << iMaxLabel << ", GT label = " << iGTLabel << std::endl;

        confusionMatrix(iMaxLabel, iGTLabel) += RealType(1);
      }
    }

    Push(confusionMatrix);

    SuperType::Forward();
  }

protected:
  Accuracy() = default;

  virtual void Print() override {
    PrintAccuracy();

    if(m_bPrintConfusionMatrix)
      PrintConfusionMatrix();
  }

private:
  typedef ConfusionMatrix<RealType> MatrixType;

  bool m_bPrintConfusionMatrix = false;
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

#endif // !BLEAK_ACCURACY_H
