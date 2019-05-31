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

#include <iostream>
#include <random>
#include <memory>
#include <vector>
#include <fstream>
#include "Common.h"
#include "Graph.h"
#include "InitializeModules.h"
#include "SadLoader.h"
#include "OptimizerFactory.h"

typedef float RealType;
typedef bleak::Graph<RealType> GraphType;
typedef GraphType::VertexType VertexType;
typedef bleak::Optimizer<RealType> OptimizerType;

const char * const p_cCsvFileName = "trainingData.csv";
const unsigned int uiNumClasses = 10;
const unsigned int uiNumFeatures = 10000;
const unsigned int uiNumUsefulFeatures = 100;
const unsigned int uiNumExamples = 10000;

std::shared_ptr<GraphType> LoadGraph();
bool MakeToyData();

int main(int argc, char **argv) {
  bleak::InitializeModules();

  std::shared_ptr<GraphType> p_clGraph = LoadGraph();

  if (!p_clGraph) {
    std::cerr << "Error: Failed to load graph." << std::endl;
    return -1;
  }

#if 0
  std::cout << "Info: Making toy data... " << std::flush;

  if (!MakeToyData()) {
    std::cerr << "Error: Failed to make toy data." << std::endl;
    return -1;
  }

  std::cout << "Done." << std::endl;
#endif

  if(!p_clGraph->Initialize(true)) {
    std::cerr << "Error: Failed to initialize graph." << std::endl;
    return -1;
  }

  //std::shared_ptr<OptimizerType> p_clSGD = bleak::OptimizerFactory<RealType>::GetInstance().Create("SGD", p_clGraph);
  std::shared_ptr<OptimizerType> p_clSGD = bleak::OptimizerFactory<RealType>::GetInstance().Create("AdaGrad",p_clGraph);

  if (!p_clSGD) {
    std::cerr << "Error: Failed to create SGD optimizer." << std::endl;
    return -1;
  }

  // Prepare parameters for optimizer
  bleak::ParameterMap clParams;

  clParams.SetValue("numBatchesPerIteration",1);
  clParams.SetValue("maxIterations",7000);
  clParams.SetValue("momentum",0.9);
  clParams.SetValue("learningRate",1.0);

  if (!p_clSGD->SetParameters(clParams)) {
    std::cerr << "Error: Failed to set parameters for SGD." << std::endl;
    return -1;
  }

  if (!p_clSGD->Minimize()) {
    std::cerr << "Error: Failed to minimize graph." << std::endl;
    return -1;
  }

  return 0;
}

std::shared_ptr<GraphType> LoadGraph() {
  return bleak::LoadGraph<RealType>("FSA.sad");
}

bool MakeToyData() {
  std::ofstream csvStream(p_cCsvFileName);

  if(!csvStream) {
    std::cerr << "Error: Could not open file '" << p_cCsvFileName << "'." << std::endl;
    return false;
  }

  std::vector<std::vector<RealType>> vAllMu(uiNumClasses),vAllSigma(uiNumClasses);

  // Let's sample multiple Gaussians with diagonal covariance
  // Then a linear decision boundary can be found between the two (if they don't overlap much)

  std::uniform_real_distribution<RealType> clMuDist(RealType(-10),RealType(10)),clSigmaDist(RealType(0.5),RealType(5));
  std::uniform_real_distribution<RealType> clNoise(RealType(-20), RealType(20));

  typedef bleak::GeneratorType GeneratorType;

  GeneratorType &clGenerator = bleak::GetGenerator();

  for(unsigned int i = 0; i < uiNumClasses; ++i) {
    std::vector<RealType> &vMu = vAllMu[i];
    std::vector<RealType> &vSigma = vAllSigma[i];

    vMu.resize(uiNumUsefulFeatures);
    vSigma.resize(uiNumUsefulFeatures);

    std::generate(vMu.begin(),vMu.end(),
      [&clMuDist,&clGenerator]() -> RealType {
      return clMuDist(clGenerator);
    });

    std::generate(vSigma.begin(),vSigma.end(),
      [&clSigmaDist,&clGenerator]() -> RealType {
      return clSigmaDist(clGenerator);
    });
  }

  std::uniform_int_distribution<int> clLabelDist(0,(int)uiNumClasses - 1);

  for(unsigned int i = 0; i < uiNumExamples; ++i) {
    const int iLabel = clLabelDist(clGenerator);

    const std::vector<RealType> &vMu = vAllMu[iLabel];
    const std::vector<RealType> &vSigma = vAllSigma[iLabel];

    for(unsigned int j = 0; j < uiNumUsefulFeatures; ++j) {
      std::normal_distribution<RealType> clDist(vMu[j],vSigma[j]);

      if(j > 0)
        csvStream << ',';

      csvStream << clDist(clGenerator);
    }

    for (unsigned int j = uiNumUsefulFeatures; j < uiNumFeatures; ++j) {
      csvStream << ',' << clNoise(clGenerator);
    }

    csvStream << ',' << iLabel << '\n';
  }

  return true;
}
