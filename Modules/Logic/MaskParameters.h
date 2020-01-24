#pragma once

#ifndef BLEAK_MASKPARAMETERS_H
#define BLEAK_MASKPARAMETERS_H

#include <cctype>
#include <algorithm>
#include <string>
#include "LogicCommon.h"
#include "Vertex.h"

namespace bleak {

template<typename RealType>
class MaskParameters : public Vertex<RealType> {
public:
  bleakNewVertex(MaskParameters, Vertex<RealType>, 
    bleakAddOutput("outData"),
    bleakAddProperty("size", m_vSize),
    bleakAddProperty("value", m_strValue),
    bleakAddGetterSetter("learnable",  &MaskParameters::GetLearnable, &MaskParameters::SetLearnable),
    bleakAddProperty("initType", m_strInitType),
    bleakAddProperty("mu", m_fMu), bleakAddProperty("sigma", m_fSigma),
    bleakAddProperty("on", m_fOn), bleakAddProperty("off", m_fOff),
    bleakAddProperty("threshold", m_fThreshold),
    bleakAddProperty("learningRateMultiplier", m_fLearningRateMultiplier),
    bleakAddProperty("applyWeightDecay", m_bApplyWeightDecay));

  bleakForwardVertexTypedefs();

  virtual ~MaskParameters() = default;

  // Returns arbitrary length bitmask from least significant bit to most significant bit
  std::vector<RealType> FromString() const {
    return ConvertIntegerStringToBits(m_strValue, RealType(m_fOff), RealType(m_fOn));
  }

  // Returns string representation of a mask integer
  std::string ToString(int iBase = 10) const {
    bleakGetAndCheckOutput(p_clOutData, "outData", std::string());

    const ArrayType &clOutData = p_clOutData->GetData();

    if (!clOutData.Valid())
      return std::string();

    return ConvertBitsToIntegerString(clOutData.begin(), clOutData.end(), iBase, RealType(m_fThreshold), RealType(m_fOn));
  }

  bool SetLearnable(const bool &bLearnable) {
    m_bLearnable = bLearnable;
    // Learnable parameters are probably ones you want to save (and vice versa)
    SetProperty("saveOutputs", bLearnable); 
    return true;
  }

  bool GetLearnable(bool &bLearnable) const {
    bLearnable = m_bLearnable;
    return true;
  }

  virtual bool SetSizes() override {
    bleakGetAndCheckOutput(p_clOutput, "outData", false);

    ArrayType &clOutput = p_clOutput->GetData();
    ArrayType &clGradient = p_clOutput->GetGradient();

    Size clSize(m_vSize);

    if (!clSize.Valid()) {
      std::cerr << GetName() << ": Error: size = " << clSize << " is not valid." << std::endl;
      return false;
    }    

    clOutput.SetSize(clSize);

    if (m_bLearnable)
      clGradient.SetSize(clSize);
    else
      clGradient.Clear(); // Leaves marker not to backpropagate on this edge

    return true;
  }

  virtual bool Initialize() override {
    bleakGetAndCheckOutput(p_clOutput,"outData", false);

    ArrayType &clOutput = p_clOutput->GetData();

    if (!clOutput.Valid())
      return false;

    std::vector<RealType> vBits = FromString();

    if (vBits.empty()) {
      std::cerr << GetName() << ": Error: Could not parse integer string '" << m_strValue << "'." << std::endl;
      return false;
    }

    if (vBits.size() > clOutput.GetSize().Count()) {
      std::cerr << GetName() << ": Error: Integer exceeds total output size (" << vBits.size() << " > " << clOutput.GetSize().Count() << ")." << std::endl;
      return false;
    }

    if (m_strInitType == "fill") {
      clOutput.Fill(RealType(m_fOff));
      std::copy(vBits.begin(), vBits.end(), clOutput.begin());
    } 
    else if (m_strInitType == "gaussian") {
      if (m_fMu != 0.0f)
        std::cerr << GetName() << ": Warning: mu is not 0 (mu = " << m_fMu << ")." << std::endl;

      clOutput.Fill(RealType(m_fOff));
      std::copy(vBits.begin(), vBits.end(), clOutput.begin());

      GeneratorType &clGenerator = GetGenerator();

      std::normal_distribution<RealType> clGaussian((RealType)m_fMu,(RealType)m_fSigma);

      std::transform(clOutput.begin(), clOutput.end(), clOutput.begin(),
        [&clGenerator, &clGaussian](const RealType &x) -> RealType {
          return x + clGaussian(clGenerator);
        });
    }
    else {
      std::cerr << GetName() << ": Error: Unrecognized initType '" << m_strInitType << "'." << std::endl;
      return false;
    }

    return true;
  }

  // Nothing to do
  virtual void Forward() override { }
  virtual void Backward() override { }

protected:
  MaskParameters() = default;

private:
  std::vector<int> m_vSize;
  std::string m_strValue;
  float m_fLearningRateMultiplier = 1.0f;
  float m_fMu = 0.0f;
  float m_fSigma = 1.0f;
  float m_fOn = 1.0f;
  float m_fOff = -1.0f;
  float m_fThreshold = 0.0f;
  std::string m_strInitType = std::string("fill");
  bool m_bApplyWeightDecay = true;
  bool m_bLearnable = false;
};

} // end namespace bleak

#endif // BLEAK_MASKPARAMETERS_H
