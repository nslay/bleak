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

#ifndef BLEAK_PARAMETERS_H
#define BLEAK_PARAMETERS_H

#include <random>
#include <string>
#include <vector>
#include "Common.h"
#include "Vertex.h"
#include "Size.h"

namespace bleak {

template<typename RealType>
class Parameters : public Vertex<RealType> {
public:
  bleakNewVertex(Parameters, Vertex<RealType>,
    bleakAddOutput("outData"),
    bleakAddProperty("size", m_vSize),
    bleakAddGetterSetter("learnable",  &Parameters::GetLearnable, &Parameters::SetLearnable),
    bleakAddProperty("initType", m_strInitType),
    bleakAddProperty("mu", m_fMu), bleakAddProperty("sigma", m_fSigma),
    bleakAddProperty("a", m_fA), bleakAddProperty("b", m_fB),
    bleakAddProperty("fill", m_fFill),
    bleakAddProperty("learningRateMultiplier", m_fLearningRateMultiplier),
    bleakAddProperty("applyWeightDecay", m_bApplyWeightDecay));

  bleakForwardVertexTypedefs();

  virtual ~Parameters() = default;

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

    if (m_strInitType == "fill") {
      clOutput.Fill(m_fFill);
    } 
    else if (m_strInitType == "gaussian") {
      GeneratorType &clGenerator = GetGenerator();

      std::normal_distribution<RealType> clGaussian((RealType)m_fMu,(RealType)m_fSigma);

      std::generate(clOutput.begin(),clOutput.end(),
        [&clGenerator,&clGaussian]() -> RealType {
        return clGaussian(clGenerator);
      });
    } 
    else if (m_strInitType == "uniform") {
      GeneratorType &clGenerator = GetGenerator();

      std::uniform_real_distribution<RealType> clUniform((RealType)m_fA,(RealType)m_fB);

      std::generate(clOutput.begin(),clOutput.end(),
        [&clGenerator,&clUniform]() -> RealType {
        return clUniform(clGenerator);
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
  Parameters() { 
    m_bLearnable = false;
    m_strInitType = "fill";
    m_fMu = 0.0f;
    m_fSigma = 1.0f;
    m_fA = 0.0f;
    m_fB = 1.0f;
    m_fFill = 0.0f;
    m_fLearningRateMultiplier = 1.0f;
    m_bApplyWeightDecay = true;
  }

private:
  std::vector<int> m_vSize;
  bool m_bLearnable;

  std::string m_strInitType;

  float m_fMu, m_fSigma; // Normal
  float m_fA, m_fB; // Uniform
  float m_fFill; // Fill

  float m_fLearningRateMultiplier; // Manipulate learning rate
  bool m_bApplyWeightDecay; // Apply weight decay on these parameters?
};

} // end namespace bleak

#endif // !BLEAK_PARAMETERS_H
