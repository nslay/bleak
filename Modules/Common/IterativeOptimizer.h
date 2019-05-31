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

#ifndef BLEAK_ITERATIVEOPTIMIZER_H
#define BLEAK_ITERATIVEOPTIMIZER_H

#include <algorithm>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include "ParameterContainer.h"
#include "Optimizer.h"
#include "Common.h"
#include "DatabaseFactory.h"

namespace bleak {

template<typename RealType>
class IterativeOptimizer : public Optimizer<RealType> {
public:
  bleakNewAbstractOptimizer(IterativeOptimizer,Optimizer<RealType>);
  bleakForwardOptimizerTypedefs();

  typedef std::pair<unsigned int, RealType> LearningRatePair;

  static bool ParseLearningRate(const std::string &strToken, LearningRatePair &stLearningRatePair) {
    if (strToken.empty())
      return false;

    size_t p = strToken.find(':');

    if (p == std::string::npos) {
      stLearningRatePair.first = 0;
      stLearningRatePair.second = RealType(std::stof(strToken, &p));

      return p >= strToken.size();
    }

    if (p+1 >= strToken.size())
      return false;

    const std::string strIteration = strToken.substr(0, p);
    const std::string strLearningRate = strToken.substr(p+1);

    stLearningRatePair.first = std::stoul(strIteration, &p);

    if (p < strIteration.size())
      return false;

    stLearningRatePair.second = RealType(std::stof(strLearningRate, &p));

    return p >= strLearningRate.size();
  }

  IterativeOptimizer(const std::shared_ptr<GraphType> &p_clGraph)
  : SuperType(p_clGraph) {
    m_strSnapshotPath = "bleak_";
    m_uiIterationsPerSnapshot = 0; // Disabled
    m_uiNumBatchesPerIteration = 1;
    m_uiMaxIterations = 1000;
    m_uiIteration = 0;

    m_vLearningRateSchedule.emplace_back(0, RealType(0.001));
  }

  virtual ~IterativeOptimizer() = default;

  virtual bool SetParameters(const ParameterContainer &clParams) override {
    m_vLearningRateSchedule.clear();

    if (!SuperType::SetParameters(clParams))
      return false;

    m_strSnapshotPath = clParams.GetValue<std::string>("snapshotPath", "bleak_");
    m_uiIterationsPerSnapshot = clParams.GetValue<unsigned int>("iterationsPerSnapshot", 0);

    m_uiNumBatchesPerIteration = clParams.GetValue<unsigned int>("numBatchesPerIteration", 1);
    m_uiMaxIterations = clParams.GetValue<unsigned int>("maxIterations", 1000);

    if (m_uiNumBatchesPerIteration == 0) {
      std::cerr << "Error: numBatchesPerIteration is expected to be positive." << std::endl;
      return false;
    }

    if (m_uiMaxIterations == 0) {
      std::cerr << "Error: maxIterations is expected to be positive." << std::endl;
      return false;
    }

    std::string strLearningRateSchedule = clParams.GetValue<std::string>("learningRate", "0.001");

    std::vector<std::string> vLearningRateScheduleTokens = SplitString<std::string>(strLearningRateSchedule, ",");

    if (vLearningRateScheduleTokens.empty()) {
      std::cerr << "Error: No learning rate specified." << std::endl;
      return false;
    }

    std::vector<LearningRatePair> vLearningRateSchedule;

    for (size_t i = 0; i < vLearningRateScheduleTokens.size(); ++i) {
      std::string &strToken = vLearningRateScheduleTokens[i];

      Trim(strToken);

      vLearningRateSchedule.emplace_back();

      if (!ParseLearningRate(strToken, vLearningRateSchedule.back())) {
        std::cerr << "Error: Could not parse learning rate token '" << strToken << "'." << std::endl;
        return false;
      }
    }

    std::sort(vLearningRateSchedule.begin(), vLearningRateSchedule.end(),
      [](const LearningRatePair &a, const LearningRatePair &b) -> bool {
        return a.first < b.first;
      });

    if (vLearningRateSchedule[0].first != 0) {
      std::cerr << "Error: Learning rate for iteration 0 must be specified." << std::endl;
      return false;
    }

    for (size_t i = 1; i < vLearningRateSchedule.size(); ++i) {
      if (vLearningRateSchedule[i-1].first == vLearningRateSchedule[i].first) {
        std::cerr << "Error: Duplicate learning rates specified: itr = " << vLearningRateSchedule[i].first << " could be " << 
          vLearningRateSchedule[i-1].second << " or " << vLearningRateSchedule[i].second << std::endl;

        return false;
      }
    }

    m_vLearningRateSchedule.swap(vLearningRateSchedule);

    std::cout << "Info: Learning rate schedule:" << std::endl;
    for (size_t i = 0; i < m_vLearningRateSchedule.size(); ++i)
      std::cout << "Info: " << (i+1) << ": itr >= " << m_vLearningRateSchedule[i].first << " --> " << m_vLearningRateSchedule[i].second << std::endl;

    return true;
  }

  virtual bool Good() const override {
    return SuperType::Good() && m_uiNumBatchesPerIteration > 0 && m_uiMaxIterations > 0 && 
      m_vLearningRateSchedule.size() > 0 && 
      std::any_of(m_vLearningRateSchedule.begin(), m_vLearningRateSchedule.end(),
        [](const LearningRatePair &stPair) -> bool {
          return stPair.second > RealType(0);
        });
  }

  virtual bool Minimize() override {
    if (!Good())
      return false;

    std::shared_ptr<GraphType> p_clGraph = GetGraph();

    m_uiIteration = 0;

    for (unsigned int &e = m_uiIteration; e < m_uiMaxIterations; ++e) {
      SaveSnapshot(e, false);

      IterationStart();

      RealType loss = RealType();

      for (unsigned int b = 0; b < m_uiNumBatchesPerIteration; ++b) {
        p_clGraph->Forward();

        loss += ComputeLoss();

        p_clGraph->Backward();

        if (b+1 == m_uiNumBatchesPerIteration)
          ApplyWeightDecay();

        CollectGradients();
      }

      GradientUpdate();

      loss /= RealType(m_uiNumBatchesPerIteration);

      // TODO: More intelligent display with running loss average
      std::cout << "Info: itr = " << e << ", loss = " << loss << std::endl;
    }

    if (!SaveSnapshot(m_uiMaxIterations, true))
      std::cerr << "Warning: Failed to save model snapshot on last iteration!" << std::endl;

    // XXX: Convergence criteria? Who cares!

    return true;
  }

  unsigned int GetNumBatchesPerIteration() const {
    return m_uiNumBatchesPerIteration;
  }

  // Indexed from 0
  unsigned int GetIteration() const {
    return m_uiIteration;
  }

  RealType GetLearningRate(unsigned int uiIteration) const {
    auto itr = std::find_if(m_vLearningRateSchedule.rbegin(), m_vLearningRateSchedule.rend(),
      [&uiIteration](const LearningRatePair &stLearningRatePair) -> bool {
        return uiIteration >= stLearningRatePair.first;
      });

    return itr != m_vLearningRateSchedule.rend() ? itr->second : RealType(0);
  }

  RealType GetLearningRate() const {
    return GetLearningRate(m_uiIteration);
  }

  unsigned int GetMaxIterations() const {
    return m_uiMaxIterations;
  }

protected:
  // This is called when an iteration begins
  virtual void IterationStart() { }

  // This is called to collect gradients (but not perform the parameter update)
  virtual void CollectGradients() = 0;

  // This is called to perform the gradient update
  virtual void GradientUpdate() = 0;

  virtual bool SaveSnapshot(unsigned int uiIteration, bool bForce) const {
    if (!Good())
      return false;

    if (!bForce && (m_uiIterationsPerSnapshot == 0 || (uiIteration % m_uiIterationsPerSnapshot) != 0))
      return false;

    std::shared_ptr<Database> p_clDatabase = DatabaseFactory::GetInstance().Create("LMDB");
    if (!p_clDatabase) {
      std::cerr << "Error: Failed to create database." << std::endl;
      return false;
    }

    const std::string strPath = m_strSnapshotPath + std::to_string(uiIteration);

    std::cout << "Info: Writing snapshot '" << strPath << "' ..." << std::endl;

    if (!p_clDatabase->Open(strPath, Database::WRITE)) {
      std::cerr << "Error: Failed to open database for writing." << std::endl;
      return false;
    }

    return GetGraph()->SaveToDatabase(p_clDatabase);
  }

private:
  std::string m_strSnapshotPath;
  unsigned int m_uiIterationsPerSnapshot;
  unsigned int m_uiNumBatchesPerIteration;
  unsigned int m_uiMaxIterations;
  unsigned int m_uiIteration;

  std::vector<LearningRatePair> m_vLearningRateSchedule;
};

} // end namespace bleak

#endif // !BLEAK_ITERATIVEOPTIMIZER_H
