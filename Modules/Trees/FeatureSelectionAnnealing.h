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

#ifndef BLEAK_FEATURESELECTIONANNEALING_H
#define BLEAK_FEATURESELECTIONANNEALING_H

#include <cmath>
#include <algorithm>
#include <vector>
#include <unordered_set>
#include <functional>
#include "Vertex.h"

namespace bleak {

template<typename RealType>
class FeatureSelectionAnnealing : public Vertex<RealType> {
public:
  bleakNewVertex(FeatureSelectionAnnealing, Vertex<RealType>,
    bleakAddInput("inData"),
    bleakAddInput("inWeights"),
    bleakAddInput("inBias"),
    bleakAddInput("inIndicators"),
    bleakAddOutput("outData"),
    bleakAddOutput("outSelectedData"),
    bleakAddProperty("ignoreValue", m_fIgnoreValue),
    bleakAddProperty("scheduleType", m_strScheduleType),
    bleakAddProperty("beginIteration", m_iBeginIteration),
    bleakAddProperty("endIteration", m_iEndIteration),
    bleakAddProperty("numFeatures", m_iNumFeatures),
    bleakAddProperty("mu", m_fMu));

  bleakForwardVertexTypedefs();

  virtual ~FeatureSelectionAnnealing() = default;

  virtual bool SetSizes() override {
    bleakGetAndCheckInput(p_clInData, "inData", false);
    bleakGetAndCheckInput(p_clWeights, "inWeights", false);
    bleakGetAndCheckInput(p_clIndicators, "inIndicators", false);
    bleakGetAndCheckOutput(p_clOutData, "outData", false);
    bleakGetAndCheckOutput(p_clOutSelectedData, "outSelectedData", false);

    std::shared_ptr<EdgeType> p_clBias = GetInput("inBias"); // if this is nullptr, we will treat bias as though it were 0.

    if (m_iNumFeatures <= 0) {
      std::cerr << GetName() << ": Error: numFeatures is expected to be positive." << std::endl;
      return false;
    }

    m_funSchedule = GetScheduleFunction(m_strScheduleType);

    if (m_funSchedule == nullptr) {
      std::cerr << GetName() << ": Error: Unknown schedule type '" << m_strScheduleType << "'." << std::endl;
      return false;
    }

    const ArrayType &clInData = p_clInData->GetData();
    const ArrayType &clWeights = p_clWeights->GetData();
    const ArrayType &clIndicators = p_clIndicators->GetData();
    ArrayType &clOutData = p_clOutData->GetData();
    ArrayType &clOutDataGradient = p_clOutData->GetGradient();
    ArrayType &clOutSelectedData = p_clOutSelectedData->GetData();
    ArrayType &clOutSelectedDataGradient = p_clOutSelectedData->GetGradient();

    if (!clInData.GetSize().Valid() || !clWeights.GetSize().Valid() || !clIndicators.GetSize().Valid()) {
      std::cerr << GetName() << ": Error: Invalid dimensions for inData, inWeights and/or inIndicators." << std::endl;
      return false;
    }

    if (p_clIndicators->GetGradient().GetSize().Valid()) {
      std::cerr << GetName() << ": Error: inIndicators appears learnable. However, FSA has its own custom updates for this input." << std::endl;
      return false;
    }

    if (clInData.GetSize().GetDimension() < 2) {
      std::cerr << GetName() << ": Error: inData is expected to be 2D or higher." << std::endl;
      return false;
    }

    // Weights are [ $numOutputs, $numChannels ]
    // InnerProduct is computed along axis 1
    if (clWeights.GetSize().GetDimension() != 2) {
      std::cerr << GetName() << ": Error: inWeights is expected to be 2D." << std::endl;
      return false;
    }

    if (clInData.GetSize()[1] != clWeights.GetSize()[1]) {
      std::cerr << GetName() << ": Error: inData and inWeights are expected to have the same number of channels." << std::endl;
      return false;
    }

    if (clWeights.GetSize() != clIndicators.GetSize()) {
      std::cerr << GetName() << ": Error: Dimension mismatch between inWeights and inIndicators." << std::endl;
      return false;
    }

    if (p_clBias == nullptr) {
      std::cout << GetName() << ": Info: inBias is not connected. Bias will be assumed to be fixed as 0." << std::endl;
    }
    else {
      const ArrayType &clBias = p_clBias->GetData();
      if (clBias.GetSize().GetDimension() != 1) {
        std::cerr << GetName() << ": Error: inBias is expected to be 1D." << std::endl;
        return false;
      }

      if (clBias.GetSize()[0] != clWeights.GetSize()[0]) {
        std::cerr << GetName() << ": Error: inWeights and inBias are expected to have the same number of outputs." << std::endl;
        return false;
      }
    }

    Size clOutSize(clInData.GetSize());
    clOutSize[1] = clWeights.GetSize()[0];

    clOutData.SetSize(clOutSize);
    clOutDataGradient.SetSize(clOutSize);

    clOutSelectedData.SetSize(clInData.GetSize());
    clOutSelectedDataGradient.SetSize(clInData.GetSize());

    return true;
  }

  virtual bool Initialize() override {
    m_iIteration = 0;

    ComputeSelectedIndices();

    if (m_vIndices.empty()) {
      std::cerr << GetName() << "Error: No selected features." << std::endl;
      return false;
    }

    return true;
  }

  // Catch this load event. It probably means indicators were also loaded
  virtual bool LoadFromDatabase(const std::unique_ptr<Cursor> &p_clCursor) override {
    ComputeSelectedIndices();

    return SuperType::LoadFromDatabase(p_clCursor);
  }

  virtual void Forward() override {
    bleakGetAndCheckInput(p_clInData, "inData");
    bleakGetAndCheckInput(p_clWeights, "inWeights");
    bleakGetAndCheckOutput(p_clOutSelectedData, "outSelectedData");
    bleakGetAndCheckOutput(p_clOutData, "outData");

    std::shared_ptr<EdgeType> p_clBias = GetInput("inBias");

    ArrayType &clOutData = p_clOutData->GetData();
    ArrayType &clOutSelectedData = p_clOutSelectedData->GetData();
    const ArrayType &clWeights = p_clWeights->GetData();
    const ArrayType &clInData = p_clInData->GetData();

    // Initially fill output data with ignore value (if needed)
    FillSelectedOutput();

    const int iOuterNum = clInData.GetSize()[0];
    const int iNumChannels = clInData.GetSize()[1];
    const int iNumOutputs = clWeights.GetSize()[0];
    const int iInnerNum = clInData.GetSize().Product(2);

    const RealType * const p_inData = clInData.data();
    const RealType * const p_weights = clWeights.data();
    const RealType * const p_bias = p_clBias != nullptr ? p_clBias->GetData().data() : nullptr;

    RealType * const p_outSelectedData = clOutSelectedData.data();
    RealType * const p_outData = clOutData.data();

    if (p_bias == nullptr) {
      clOutData.Fill(RealType());
    }
    else {
      for (int i = 0; i < iOuterNum; ++i) {
        for (int k = 0; k < iInnerNum; ++k) {
          for (int j = 0; j < iNumOutputs; ++j) {
            p_outData[(i*iNumOutputs + j)*iInnerNum + k] = p_bias[j];
          }
        }
      }
    }

    for (int i = 0; i < iOuterNum; ++i) {
      // Assign selected data
      for(int k = 0; k < iInnerNum; ++k) {
        for (int l : m_vIndices) {
          p_outSelectedData[(i*iNumChannels + l)*iInnerNum + k] = p_inData[(i*iNumChannels + l)*iInnerNum + k];
        }
      }

      for (int j = 0; j < iNumOutputs; ++j) {
        for (int k = 0; k < iInnerNum; ++k) {
          for (int l : m_vIndicesByOutput[j]) {
            p_outData[(i*iNumOutputs + j)*iInnerNum + k] += p_weights[j*iNumChannels + l]*p_inData[(i*iNumChannels + l)*iInnerNum + k];
          }
        }
      }
    }
  }

  virtual void Backward() override {
    bleakGetAndCheckInput(p_clInData, "inData");
    bleakGetAndCheckInput(p_clWeights, "inWeights");
    bleakGetAndCheckOutput(p_clOutSelectedData, "outSelectedData");
    bleakGetAndCheckOutput(p_clOutData, "outData");

    std::shared_ptr<EdgeType> p_clBias = GetInput("inBias");

    ArrayType &clInData = p_clInData->GetData();
    ArrayType &clInDataGradient = p_clInData->GetGradient();
    ArrayType &clWeights = p_clWeights->GetData();
    ArrayType &clWeightsGradient = p_clWeights->GetGradient();
    const ArrayType &clOutData = p_clOutData->GetData();
    const ArrayType &clOutDataGradient = p_clOutData->GetGradient();
    const ArrayType &clOutSelectedData = p_clOutSelectedData->GetData();
    const ArrayType &clOutSelectedDataGradient = p_clOutSelectedData->GetGradient();

    RealType * const p_inData = clInData.data();
    RealType * const p_inDataGradient = clInDataGradient.data();
    RealType * const p_weights = clWeights.data();
    RealType * const p_weightsGradient = clWeightsGradient.data();
    const RealType * const p_outData = clOutData.data();
    const RealType * const p_outDataGradient = clOutDataGradient.data();
    const RealType * const p_outSelectedData = clOutSelectedData.data();
    const RealType * const p_outSelectedDataGradient = clOutSelectedDataGradient.data();

    RealType * const p_biasGradient = p_clBias != nullptr && p_clBias->GetGradient().Valid() ? p_clBias->GetGradient().data() : nullptr;

    const int iOuterNum = clInData.GetSize()[0];
    const int iNumChannels = clInData.GetSize()[1];
    const int iNumOutputs = clWeights.GetSize()[0];
    const int iInnerNum = clInData.GetSize().Product(2);

    if (p_biasGradient != nullptr) {
      for (int i = 0; i < iOuterNum; ++i) {
        for (int k = 0; k < iInnerNum; ++k) { 
          for (int j = 0; j < iNumOutputs; ++j) {
            p_biasGradient[j] += p_outDataGradient[(i*iNumOutputs + j)*iInnerNum + k];
          }
        }
      }
    }

    if (p_weightsGradient != nullptr) {
      for (int i = 0; i < iOuterNum; ++i) {
        for (int j = 0; j < iNumOutputs; ++j) {
          for (int k = 0; k < iInnerNum; ++k) {
            for (int l : m_vIndicesByOutput[j]) {
              p_weightsGradient[j*iNumChannels + l] += p_outDataGradient[(i*iNumOutputs + j)*iInnerNum + k] * p_inData[(i*iNumChannels + l)*iInnerNum + k];
            }
          }
        }
      }
    }

    if (p_inDataGradient != nullptr) {
      for(int i = 0; i < iOuterNum; ++i) {

        // outSelectedData gradient
        for (int k = 0; k < iInnerNum; ++k) {
          for (int l : m_vIndices) {
            p_inDataGradient[(i*iNumChannels + l)*iInnerNum + k] += p_outSelectedDataGradient[(i*iNumChannels + l)*iInnerNum + k];
          }
        }

        // outData gradient
        for (int j = 0; j < iNumOutputs; ++j) {
          for (int k = 0; k < iInnerNum; ++k) {
            for (int l : m_vIndicesByOutput[j]) {
              p_inDataGradient[(i*iNumChannels + l)*iInnerNum + k] += p_outDataGradient[(i*iNumOutputs + j)*iInnerNum + k] * p_weights[j*iNumChannels + l];
            }
          }
        }
      }
    }

    // This selection is actually delayed (or early) compared to the original formulation...
    // It should actually happen just after a gradient update! This is just before an update.
    //   Or it's after an update, depending on how you look at it.
    SelectFeatures();

    ++m_iIteration;
  }

protected:
  typedef std::function<int()> ScheduleFunctionType;

  int m_iBeginIteration,m_iEndIteration;
  int m_iNumFeatures;
  int m_iIteration;

  FeatureSelectionAnnealing() { 
    m_fIgnoreValue = 0.0f;
    m_strScheduleType = "inverse";

    m_iIteration = 0;
    m_iBeginIteration = 0;
    m_iEndIteration = 10000;
    m_iNumFeatures = 0;

    // Specific to "inverse"
    m_fMu = 20.0f;

    m_bOutputNeedsFill = true; // Weights may be loaded
  }

  // Allows a deriving class to define more schedule types
  virtual ScheduleFunctionType GetScheduleFunction(const std::string &strScheduleType) {
    if (strScheduleType == "inverse")
      return std::bind(&SelfType::ComputeNumberToKeepInverse, this);

    return ScheduleFunctionType();
  }

  // Convenience function
  int ComputeNumberToKeep() {
    return m_funSchedule != nullptr ? m_funSchedule() : -1;
  }

  void FillSelectedOutput() {
    if(!m_bOutputNeedsFill)
      return;

    bleakGetAndCheckOutput(p_clOutSelectedData, "outSelectedData");

    ArrayType &clOutSelectedData = p_clOutSelectedData->GetData();

    clOutSelectedData.Fill(RealType(m_fIgnoreValue));

    m_bOutputNeedsFill = false;
  }

private:
  ScheduleFunctionType m_funSchedule;

  std::string m_strScheduleType;
  float m_fMu;

  // Since the input is probably very high dimensional, this will be used to flag when 
  //   outSelectedData should be completely filled with ignoreValue (e.g. when feature suppression happens).
  //   the fill happens in the next Forward() call.
  bool m_bOutputNeedsFill;

  float m_fIgnoreValue; // Value to mark outputs that were not selected

  // For speed, store selected indices
  std::vector<std::vector<int>> m_vIndicesByOutput;
  std::vector<int> m_vIndices;

  int ComputeNumberToKeepInverse() {
    bleakGetAndCheckInput(p_clInData, "inData", -1);

    const int iNumChannels = p_clInData->GetData().GetSize()[1];

    const int iIteration = m_iIteration - m_iBeginIteration;

    if (iIteration < 1)
      return iNumChannels;

    const int iNumIterations = m_iEndIteration - m_iBeginIteration + 1;

    return (int)(m_iNumFeatures + (iNumChannels - m_iNumFeatures)*std::max(0.0f, (iNumIterations - 2.0f*iIteration)/(2.0f*iIteration*m_fMu + iNumIterations)));
  }

  void SelectFeatures() {
    bleakGetAndCheckInput(p_clWeights, "inWeights");
    bleakGetAndCheckInput(p_clIndicators, "inIndicators");

    const int iNumToKeep = ComputeNumberToKeep();

    if (iNumToKeep < 0 || iNumToKeep >= (int)m_vIndicesByOutput[0].size()) // Should all be the same size
      return;

    ArrayType &clIndicators = p_clIndicators->GetData();
    ArrayType &clWeights = p_clWeights->GetData();

    RealType * const p_indicators = clIndicators.data();
    RealType * const p_weights = clWeights.data();

    const int iNumOutputs = clWeights.GetSize()[0];
    const int iNumChannels = clWeights.GetSize()[1];

    std::unordered_set<int> sIndices;
    std::vector<RealType *> vWeightPointers;
    vWeightPointers.reserve(iNumChannels);

    for (int i = 0; i < iNumOutputs; ++i) {
      std::vector<int> &vOutputIndices = m_vIndicesByOutput[i];

      RealType * const p_weightsBegin = p_weights + (i*iNumChannels);

      vWeightPointers.clear();

      for (int j : vOutputIndices)
        vWeightPointers.push_back(p_weightsBegin + j);

      // Sort in descending order
      std::sort(vWeightPointers.begin(), vWeightPointers.end(),
        [](const RealType *a, const RealType *b) -> bool {
          return std::abs(*a) > std::abs(*b);
        });

      for (size_t k = iNumToKeep; k < vWeightPointers.size(); ++k) {
        const int j = (vWeightPointers[k] - p_weightsBegin);
        p_indicators[i*iNumChannels + j] = RealType(0);
      }

      vOutputIndices.resize(iNumToKeep);

      for (int j = 0; j < iNumToKeep; ++j)
        vOutputIndices[j] = (vWeightPointers[j] - p_weightsBegin);

      std::sort(vOutputIndices.begin(), vOutputIndices.end());

      sIndices.insert(vOutputIndices.begin(), vOutputIndices.end());
    }

    m_bOutputNeedsFill = true;
    m_vIndices.assign(sIndices.begin(), sIndices.end());
    std::sort(m_vIndices.begin(), m_vIndices.end());
  }

  void ComputeSelectedIndices() {
    m_vIndices.clear();

    for (std::vector<int> &vOutputIndices : m_vIndicesByOutput)
      vOutputIndices.clear();

    bleakGetAndCheckInput(p_clIndicators, "inIndicators");
    const ArrayType &clIndicators = p_clIndicators->GetData();

    if (!clIndicators.Valid())
      return;

    const RealType * const p_indicators = clIndicators.data();
    const int iNumOutputs = clIndicators.GetSize()[0];
    const int iNumChannels = clIndicators.GetSize()[1];

    m_vIndicesByOutput.resize(iNumOutputs);

    std::unordered_set<int> sIndices;

    for (int i = 0; i < iNumOutputs; ++i) {
      m_vIndicesByOutput[i].reserve(iNumChannels);

      for (int j = 0; j < iNumChannels; ++j) {
        if (p_indicators[i*iNumChannels + j] != RealType(0)) {
          sIndices.insert(j);
          m_vIndicesByOutput[i].push_back(j);
        }
      }
    }

    m_bOutputNeedsFill = true;
    m_vIndices.assign(sIndices.begin(), sIndices.end());
    std::sort(m_vIndices.begin(), m_vIndices.end());
  }
};

} // end namespace bleak

#endif // !BLEAK_FEATURESELECTIONANNEALING_H
