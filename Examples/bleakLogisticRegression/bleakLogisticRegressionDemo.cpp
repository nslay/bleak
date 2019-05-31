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
#include <fstream>
#include <random>
#include <string>
#include "Graph.h"
#include "Common.h"
#include "CsvReader.h"
#include "Parameters.h"
#include "InnerProduct.h"
#include "SoftmaxLoss.h"
#include "StochasticGradientDescent.h"
#include "SadLoader.h"
#include "LMDBDatabase.h"
#include "InitializeModules.h"

#include "OptimizerFactory.h"

typedef float RealType;
typedef bleak::Graph<RealType> GraphType;
typedef GraphType::VertexType VertexType;
typedef bleak::StochasticGradientDescent<RealType> OptimizerType;
typedef bleak::LMDBDatabase DatabaseType;

// NOTE: These values would need to change in the .sad graph if using LoadGraph().
const char * const p_cCsvFileName = "trainingData.csv";
const unsigned int uiNumClasses = 10;
const unsigned int uiNumFeatures = 100;
const unsigned int uiNumExamples = 10000;

std::shared_ptr<GraphType> MakeGraph();
std::shared_ptr<GraphType> LoadGraph();
bool MakeToyData();

int main(int argc, char **argv) {
  bleak::InitializeModules();

  if (!MakeToyData()) {
    std::cerr << "Error: Failed to create toy data." << std::endl;
    return -1;
  }

  std::shared_ptr<GraphType> p_clGraph = MakeGraph();
  //std::shared_ptr<GraphType> p_clGraph = LoadGraph();

  if (p_clGraph == nullptr) {
    std::cerr << "Error: Failed to make/load graph." << std::endl;
    return -1;
  }

  if (!p_clGraph->Initialize(true)) {
    std::cerr << "Error: Failed to initialize graph." << std::endl;
    return -1;
  }

  // Print out input/output sizes
  for (const auto &p_clVertex : p_clGraph->GetPlan()) {
    const auto &mInputs = p_clVertex->GetAllInputs();
    const auto &mOutputs = p_clVertex->GetAllOutputs();

    if (mInputs.size() > 0) {
      std::cout << '\n' << p_clVertex->GetName() << " inputs:" << std::endl;
      for (const auto &clPair : mInputs) {
        const std::string &strName = clPair.first;
        const auto &p_clEdge = clPair.second.lock();

        std::cout << strName << " size = ";

        if (p_clEdge != nullptr)
          std::cout << p_clEdge->GetData().GetSize() << std::endl;
        else
          std::cout << "null" << std::endl;
      }
    }

    if (mOutputs.size() > 0) {
      std::cout << '\n' << p_clVertex->GetName() << " outputs:" << std::endl;

      for(const auto &clPair : mOutputs) {
        const std::string &strName = clPair.first;
        const auto &p_clEdge = clPair.second;

        std::cout << strName << " size = " << p_clEdge->GetData().GetSize() << std::endl;
      }
    }
  }

  std::cout << std::endl;

  // Prepare parameters for optimizer
  bleak::ParameterMap clParams;

  clParams.SetValue("numBatchesPerIteration", 2);
  clParams.SetValue("maxIterations", 1000);
  clParams.SetValue("momentum", 0.9);
  clParams.SetValue("learningRate", 0.01);

  OptimizerType clOptimizer(p_clGraph);

  if (!clOptimizer.SetParameters(clParams)) {
    std::cerr << "Error: Failed to set optimizer parameters." << std::endl;
    return -1;
  }

  if (!clOptimizer.Minimize()) {
    std::cerr << "Error: Failed to minimize graph." << std::endl;
    return -1;
  }

  std::cout << "Info: Testing IO ..." << std::endl;

  const std::string &strWeightsFile = "logisticRegressor.lmdb";

  std::shared_ptr<DatabaseType> p_clDatabase = std::make_shared<DatabaseType>();

  if (!p_clDatabase->Open(strWeightsFile, DatabaseType::WRITE)) {
    std::cerr << "Error: Failed to open database to save weights." << std::endl;
    return -1;
  }

  if (!p_clGraph->SaveToDatabase(p_clDatabase)) {
    std::cerr << "Error: Failed to save to database." << std::endl;
    return -1;
  }

  p_clDatabase->Close();

  if (!p_clDatabase->Open(strWeightsFile, DatabaseType::READ)) {
    std::cerr << "Error: Failed to open database to load weights." << std::endl;
    return -1;
  }

  if(!p_clGraph->LoadFromDatabase(p_clDatabase)) {
    std::cerr << "Error: Failed to load from database." << std::endl;
    return -1;
  }

  return 0;
}

bool MakeToyData() {
  std::ofstream csvStream(p_cCsvFileName);

  if(!csvStream) {
    std::cerr << "Error: Could not open file '" << p_cCsvFileName << "'." << std::endl;
    return false;
  }

  std::vector<std::vector<RealType>> vAllMu(uiNumClasses), vAllSigma(uiNumClasses);

  // Let's sample multiple Gaussians with diagonal covariance
  // Then a linear decision boundary can be found between the two (if they don't overlap much)

  std::uniform_real_distribution<RealType> clMuDist(RealType(-10), RealType(10)), clSigmaDist(RealType(0.5), RealType(5));

  typedef bleak::GeneratorType GeneratorType;

  GeneratorType &clGenerator = bleak::GetGenerator();

  for (unsigned int i = 0; i < uiNumClasses; ++i) {
    std::vector<RealType> &vMu = vAllMu[i];
    std::vector<RealType> &vSigma = vAllSigma[i];

    vMu.resize(uiNumFeatures);
    vSigma.resize(uiNumFeatures);

    std::generate(vMu.begin(), vMu.end(),
      [&clMuDist, &clGenerator]() -> RealType {
        return clMuDist(clGenerator);
      });

    std::generate(vSigma.begin(), vSigma.end(),
      [&clSigmaDist, &clGenerator]() -> RealType {
        return clSigmaDist(clGenerator);
      });
  }

  std::uniform_int_distribution<int> clLabelDist(0, (int)uiNumClasses - 1);

  for (unsigned int i = 0; i < uiNumExamples; ++i) {
    const int iLabel = clLabelDist(clGenerator);

    const std::vector<RealType> &vMu = vAllMu[iLabel];
    const std::vector<RealType> &vSigma = vAllSigma[iLabel];

    for (unsigned int j = 0; j < uiNumFeatures; ++j) {
      std::normal_distribution<RealType> clDist(vMu[j], vSigma[j]);

      if (j > 0)
        csvStream << ',';

      csvStream << clDist(clGenerator);
    }

    csvStream << ',' << iLabel << '\n';
  }

  return true;
}

std::shared_ptr<GraphType> LoadGraph() {
  return bleak::LoadGraph<RealType>("logisticRegression.sad");
}

std::shared_ptr<GraphType> MakeGraph() {
  std::shared_ptr<GraphType> p_clGraph = std::make_shared<GraphType>();

  std::shared_ptr<VertexType> p_clDataVertex = bleak::CsvReader<RealType>::New();
  std::shared_ptr<VertexType> p_clWeights = bleak::Parameters<RealType>::New();
  std::shared_ptr<VertexType> p_clBias = bleak::Parameters<RealType>::New();
  std::shared_ptr<VertexType> p_clInner = bleak::InnerProduct<RealType>::New();
  std::shared_ptr<VertexType> p_clLoss = bleak::SoftmaxLoss<RealType>::New();

  // Data
  p_clDataVertex->SetName("trainData");
  p_clDataVertex->SetProperty("shuffle",true);
  p_clDataVertex->SetProperty("batchSize",50);
  p_clDataVertex->SetProperty("csvFileName",p_cCsvFileName);
  p_clDataVertex->SetProperty("labelColumn",(int)uiNumFeatures);

  // Weights
  p_clWeights->SetName("weights");
  p_clWeights->SetProperty("learnable",true);

  {
    std::vector<int> vSize ={(int)uiNumClasses, (int)uiNumFeatures};
    p_clWeights->SetProperty("size",vSize);
  }

  p_clWeights->SetProperty("initType","gaussian");
  p_clWeights->SetProperty("mu",0.0f);
  p_clWeights->SetProperty("sigma",3.0f);

  // Bias
  p_clBias->SetName("bias");
  p_clBias->SetProperty("learnable",true);

  {
    std::vector<int> vSize ={(int)uiNumClasses};
    p_clBias->SetProperty("size",vSize);
  }

  // Inner
  p_clInner->SetName("inner");

  // Loss
  p_clLoss->SetName("loss");

  // Connections

  p_clInner->SetInput("inData",p_clDataVertex->GetOutput("outData"));
  p_clInner->SetInput("inWeights",p_clWeights->GetOutput("outData"));
  p_clInner->SetInput("inBias",p_clBias->GetOutput("outData"));

  p_clLoss->SetInput("inData",p_clInner->GetOutput("outData"));
  p_clLoss->SetInput("inLabels",p_clDataVertex->GetOutput("outLabels"));

  // Add to graph (this would not be the order a parser would do it!)

  p_clGraph->AddVertex(p_clDataVertex);
  p_clGraph->AddVertex(p_clWeights);
  p_clGraph->AddVertex(p_clBias);
  p_clGraph->AddVertex(p_clInner);
  p_clGraph->AddVertex(p_clLoss);

  return p_clGraph;
}