/*-
 * Copyright (c) 2020 Nathan Lay (enslay@gmail.com)
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

#ifndef BLEAK_ROC_H
#define BLEAK_ROC_H

#include <algorithm>
#include <tuple>
#include <utility>
#include <vector>
#include "PrintOutput.h"

namespace bleak {

template<typename RealType>
class ROC : public PrintOutput<RealType, std::vector<std::pair<RealType, int>>> {
public:
  typedef PrintOutput<RealType, std::vector<std::pair<RealType, int>>> WorkAroundVarArgsType;

  bleakNewVertex(ROC, WorkAroundVarArgsType,
    bleakAddProperty("positiveLabel", m_iPositiveLabel),
    bleakAddProperty("printAUC", m_bPrintAUC),
    bleakAddProperty("printROC", m_bPrintROC),
    bleakAddInput("inProbabilities"),
    bleakAddInput("inLabels"));

  typedef std::tuple<std::vector<RealType>, std::vector<RealType>, std::vector<RealType>> ROCCurveType;

  using SuperType::Push;
  using SuperType::GetQueue;

  static RealType ComputeAUC(const ROCCurveType &tplROC) {
    constexpr RealType small = RealType(1e-5);

    const std::vector<RealType> &vThresholds = std::get<0>(tplROC);
    const std::vector<RealType> &vTruePositiveRates = std::get<1>(tplROC);
    const std::vector<RealType> &vFalsePositiveRates = std::get<2>(tplROC);

    if (vThresholds.empty() || vThresholds.size() != vTruePositiveRates.size() || vThresholds.size() != vFalsePositiveRates.size())
      return RealType(-1);

    RealType sum = RealType(0);

     // Trapezoid rule
    for (size_t i = 1; i < vFalsePositiveRates.size(); ++i) {
      const RealType delta = vFalsePositiveRates[i-1] - vFalsePositiveRates[i];
      sum += delta * (vTruePositiveRates[i-1] + vTruePositiveRates[i]);
    }

    return sum / RealType(2);
  }

  ROCCurveType ComputeROC(std::vector<std::pair<RealType, int>> &vScoresAndLabels) {
    constexpr RealType small = RealType(1e-5);
  
    if (vScoresAndLabels.empty())
      return ROCCurveType();
  
    // Sort over scores in ascending order
    std::sort(vScoresAndLabels.begin(), vScoresAndLabels.end(),
      [](const auto &a, const auto &b) -> bool {
        return a.first < b.first;
      });
  
    size_t a_totalCounts[2] = { 0, 0 };
  
    for (const auto &stPair : vScoresAndLabels)
      ++a_totalCounts[stPair.second != 0];
  
    // Print a warning but try to proceed anyway...
    if (a_totalCounts[0] == 0 || a_totalCounts[1] == 0)
      std::cerr << GetName() << ": Warning: Positive or negative counts are 0: N- = " << a_totalCounts[0] << ", N+ = " << a_totalCounts[1] << std::endl;
  
    auto itr = vScoresAndLabels.begin();
    auto prevItr = itr++;
  
    size_t a_counts[2] = { 0, 0 }; // Counts less up to itr (excluding itr) ... counts for score <= threshold
  
    ROCCurveType tplROC;

    std::vector<RealType> &vThresholds = std::get<0>(tplROC);
    std::vector<RealType> &vTruePositiveRates = std::get<1>(tplROC);
    std::vector<RealType> &vFalsePositiveRates = std::get<2>(tplROC);
  
    // Edge case (> smallest score - small value)
    vThresholds.push_back(vScoresAndLabels.front().first - small);
    vTruePositiveRates.push_back(a_totalCounts[1] != 0 ? RealType(1) : RealType(0));
    vFalsePositiveRates.push_back(a_totalCounts[0] != 0 ? RealType(1) : RealType(0));
  
    while (itr != vScoresAndLabels.end()) {
      ++a_counts[prevItr->second != 0];
  
      if (itr->first - prevItr->first > small) {
        const RealType threshold = (prevItr->first + itr->first)/RealType(2);
  
        // Compute counts for score > threshold
        const size_t truePositiveCount = (a_totalCounts[1] - a_counts[1]);
        const size_t falsePositiveCount = (a_totalCounts[0] - a_counts[0]);
  
        const RealType falsePositiveRate = a_totalCounts[0] > 0 ? falsePositiveCount / RealType(a_totalCounts[0]) : RealType(0);
        const RealType truePositiveRate = a_totalCounts[1] > 0 ? truePositiveCount / RealType(a_totalCounts[1]) : RealType(0);
  
        vThresholds.push_back(threshold);
        vTruePositiveRates.push_back(truePositiveRate);
        vFalsePositiveRates.push_back(falsePositiveRate);
      }
  
      prevItr = itr++;
    }
  
    // Edge case (> largest score ... which is none of them)
    vThresholds.push_back(vScoresAndLabels.back().first);
    vTruePositiveRates.push_back(RealType(0));
    vFalsePositiveRates.push_back(RealType(0));
  
    return tplROC;
  }

  ROCCurveType SubsampleROC(const ROCCurveType &tplROC, unsigned int uiNumEntries) {
    if (uiNumEntries <= 1)
      return ROCCurveType();

    const std::vector<RealType> &vThresholds = std::get<0>(tplROC);
    const std::vector<RealType> &vTruePositiveRates = std::get<1>(tplROC);
    const std::vector<RealType> &vFalsePositiveRates = std::get<2>(tplROC);

    if (vThresholds.empty() || vThresholds.size() != vTruePositiveRates.size() || vThresholds.size() != vFalsePositiveRates.size())
      return ROCCurveType();

    if (vThresholds.size() < uiNumEntries)
      return tplROC; // Nothing to do

    ROCCurveType tplNewROC;

    std::vector<RealType> &vNewThresholds = std::get<0>(tplNewROC);
    std::vector<RealType> &vNewTruePositiveRates = std::get<1>(tplNewROC);
    std::vector<RealType> &vNewFalsePositiveRates = std::get<2>(tplNewROC);

    vNewThresholds.reserve(uiNumEntries);
    vNewTruePositiveRates.reserve(uiNumEntries);
    vNewFalsePositiveRates.reserve(uiNumEntries);

    for (unsigned int i = 0; i < uiNumEntries-1; ++i) {
      const size_t j = i*vThresholds.size() / (uiNumEntries-1);
      vNewThresholds.push_back(vThresholds[j]);
      vNewTruePositiveRates.push_back(vTruePositiveRates[j]);
      vNewFalsePositiveRates.push_back(vFalsePositiveRates[j]);
    }

    vNewThresholds.push_back(vThresholds.back());
    vNewTruePositiveRates.push_back(vTruePositiveRates.back());
    vNewFalsePositiveRates.push_back(vFalsePositiveRates.back());

    return tplNewROC;
  }

  virtual ~ROC() {
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

    if (m_iPositiveLabel < 0 || m_iPositiveLabel >= clProbs.GetSize()[1]) {
      std::cerr << GetName() << ": Error: 'positiveLabel' is invalid (" << m_iPositiveLabel << "). Expected 0 <= positiveLabel < " << clProbs.GetSize()[1] << "." << std::endl;
      return false;
    }

    return true;
  }

  virtual bool Initialize() override {
    bleakGetAndCheckInput(p_clProbs, "inProbabilities", false);

    m_iVectorSize = p_clProbs->GetData().GetSize()[0] * p_clProbs->GetData().GetSize().Product(2);

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

    std::vector<std::pair<RealType, int>> vScoresAndLabels;
    vScoresAndLabels.reserve(m_iVectorSize);

    for (int i = 0; i < iOuterNum; ++i) {
      for (int k = 0; k < iInnerNum; ++k) {
        const int iGTLabel = (int)(p_labels[i*iInnerNum + k]);

        if (iGTLabel < 0 || iGTLabel >= iNumClasses)
          continue;

        const RealType prob = p_probs[(i*iNumClasses + m_iPositiveLabel)*iInnerNum + k];
        
        vScoresAndLabels.emplace_back(prob, (iGTLabel == m_iPositiveLabel));
      }
    }

    Push(vScoresAndLabels);

    SuperType::Forward();
  }

protected:
  ROC() = default;

  virtual void Print() override {
    if (GetQueue().empty())
      return;

    std::vector<std::pair<RealType, int>> vAllScoresAndLabels;

    vAllScoresAndLabels.reserve(GetQueue().size() * m_iVectorSize);

    for (const std::vector<std::pair<RealType, int>> &vScoresAndLabels : GetQueue())
      vAllScoresAndLabels.insert(vAllScoresAndLabels.end(), vScoresAndLabels.begin(), vScoresAndLabels.end());

    ROCCurveType tplROC = ComputeROC(vAllScoresAndLabels);

    if (m_bPrintROC) {
      ROCCurveType tplSubROC = SubsampleROC(tplROC, 20);

      std::vector<RealType> &vThresholds = std::get<0>(tplSubROC);
      std::vector<RealType> &vTruePositiveRates = std::get<1>(tplSubROC);
      std::vector<RealType> &vFalsePositiveRates = std::get<2>(tplSubROC);

      std::cout << GetName() << ": Info: Current running ROC (last " << GetQueue().size() << " iterations):" << std::endl;
      std::cout << "# threshold, false positive rate, true positive rate" << std::endl;

      for (size_t i = 0; i < vThresholds.size(); ++i)
        std::cout << vThresholds[i] << ", " << vFalsePositiveRates[i] << ", " << vTruePositiveRates[i] << std::endl;
    }

    if (m_bPrintAUC)
      std::cout << GetName() << ": Info: Current running AUC (last " << GetQueue().size() << " iterations) = " << ComputeAUC(tplROC) << std::endl;
  }

private:
  bool m_bPrintAUC = true;
  bool m_bPrintROC = false;

  int m_iPositiveLabel = 1;
  int m_iVectorSize = 0;
};

} // end namespace bleak

#endif // !BLEAK_ROC_H
