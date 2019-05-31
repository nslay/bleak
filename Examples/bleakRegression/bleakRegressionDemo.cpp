#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <random>
#include "ParameterContainer.h"
#include "Common.h"
#include "Graph.h"
#include "InitializeModules.h"
#include "OptimizerFactory.h"
#include "DatabaseFactory.h"
#include "SadLoader.h"

typedef float RealType;
typedef bleak::Graph<RealType> GraphType;
typedef GraphType::VertexType VertexType;
typedef GraphType::EdgeType EdgeType;
typedef bleak::Array<RealType> ArrayType;
typedef bleak::Optimizer<RealType> OptimizerType;

constexpr const RealType g_a = RealType(0);
constexpr const RealType g_b = RealType(4*M_PI);
constexpr const int g_iNumValues = 2000;

RealType FunctionToFit(const RealType &x) {
  return std::sin(x);
}

bool WriteData(const std::string &strFileName);
std::string TrainModel(const std::string &strGraphFile);
bool TestModel(const std::string &strGraphFile, const std::string &strWeightsFile);

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " modelPrefix" << std::endl;
    return -1;
  }

  bleak::InitializeModules();

  const std::string strSeedString = "bleak";
  std::seed_seq clSeed(strSeedString.begin(), strSeedString.end());

  bleak::GetGenerator().seed(clSeed);

  const std::string strModelPrefix = argv[1];
  const std::string strTrainGraph = strModelPrefix + "_train.sad";
  const std::string strTestGraph = strModelPrefix + "_test.sad";

  if (!WriteData("trainingData.csv")) {
    std::cerr << "Error: Failed to write trainingData.csv." << std::endl;
    return -1;
  }

  const std::string strWeightsFile = TrainModel(strTrainGraph);

  if (strWeightsFile.empty())
    return -1;

  if (!TestModel(strTestGraph, strWeightsFile)) {
    std::cerr << "Error: Failed to test model." << std::endl;
    return -1;
  }

  return 0;
}

bool WriteData(const std::string &strFileName) {
  std::ofstream csvStream(strFileName.c_str());

  if(!csvStream)
    return false;

  std::uniform_real_distribution<RealType> clXRand(g_a,g_b);

  bleak::GeneratorType &clGenerator = bleak::GetGenerator();

  for(int i = 0; i < g_iNumValues; ++i) {
    const RealType x = clXRand(clGenerator);
    const RealType y = FunctionToFit(x);

    csvStream << x << ',' << y << '\n';
  }

  return true;
}

std::string TrainModel(const std::string &strGraphFile) {
  std::shared_ptr<GraphType> p_clGraph = bleak::LoadGraph<RealType>(strGraphFile);

  if (!p_clGraph) {
    std::cerr << "Error: Could not load graph file '" << strGraphFile << "'." << std::endl;
    return std::string();
  }

  if (!p_clGraph->Initialize(true)) {
    std::cerr << "Error: Failed to initialize graph." << std::endl;
    return std::string();
  }

  std::shared_ptr<OptimizerType> p_clSGD = bleak::OptimizerFactory<RealType>::GetInstance().Create("AdaGrad",p_clGraph);

  if(!p_clSGD) {
    std::cerr << "Error: Could not create optimizer." << std::endl;
    return std::string();
  }

  bleak::ParameterMap clParams;

  const int iMaxIterations = 1000;
  std::string strWeightsFile = "funcFit_";

  clParams.SetValue("maxIterations",iMaxIterations);
  clParams.SetValue("snapshotPath",strWeightsFile);
  clParams.SetValue("learningRate", 0.1);
  clParams.SetValue("momentum",0.9f);


  if(!p_clSGD->SetParameters(clParams)) {
    std::cerr << "Error: Failed to set parameters." << std::endl;
    return std::string();
  }

  if(!p_clSGD->Minimize()) {
    std::cerr << "Error: Failed to minimize graph." << std::endl;
    return std::string();
  }

  return strWeightsFile + std::to_string(iMaxIterations);
}

bool TestModel(const std::string &strGraphFile,const std::string &strWeightsFile) {
  std::shared_ptr<GraphType> p_clGraph = bleak::LoadGraph<RealType>(strGraphFile);

  if (!p_clGraph) {
    std::cerr << "Error: Could not load graph file '" << strGraphFile << "'." << std::endl;
    return false;
  }

  if (!p_clGraph->Initialize(false)) {
    std::cerr << "Error: Failed to initialize graph." << std::endl;
    return false;
  }

  std::shared_ptr<bleak::Database> p_clDB = bleak::DatabaseFactory::GetInstance().Create("LMDB");

  if (!p_clDB) {
    std::cerr << "Error: Could not create database instance." << std::endl;
    return false;
  }

  if (!p_clDB->Open(strWeightsFile, bleak::Database::READ)) {
    std::cerr << "Error: Could not open database '" << strWeightsFile << "'." << std::endl;
    return false;
  }

  if (!p_clGraph->LoadFromDatabase(p_clDB)) {
    std::cerr << "Error: Failed to load weights." << std::endl;
    return false;
  }

  std::shared_ptr<VertexType> p_clInputVertex = p_clGraph->FindVertex("input");
  std::shared_ptr<VertexType> p_clOutputVertex = p_clGraph->FindVertex("output");

  if (!p_clInputVertex || !p_clOutputVertex) {
    std::cerr << "Error: Could not find input/output vertex." << std::endl;
    return false;
  }

  std::shared_ptr<EdgeType> p_clInput = p_clInputVertex->GetOutput("outData");
  std::shared_ptr<EdgeType> p_clOutput = p_clOutputVertex->GetOutput("outData");

  if (!p_clInput) {
    std::cerr << "Error: Could not get 'outData' from input/output vertex." << std::endl;
    return false;
  }

  ArrayType &clInput = p_clInput->GetData();
  const ArrayType &clOutput = p_clOutput->GetData();

  if (!clInput.Valid() || !clOutput.Valid() || clInput.GetSize().GetDimension() < 2 || clOutput.GetSize().GetDimension() < 2) {
    std::cerr << "Error: Input/output is not valid?" << std::endl;
    return false;
  }

  const int iOuterNum = clInput.GetSize()[0];

  if (clInput.GetSize()[1] != 1) {
    std::cerr << "Error: Only 1 input per instance is expected!" << std::endl;
    return false;
  }

  if (clOutput.GetSize()[1] != 1) {
    std::cerr << "Error: Only 1 output per instance is expected!" << std::endl;
    return false;
  }

  RealType * const p_input = clInput.data();
  const RealType * const p_output = clOutput.data();

  for (int i = 0; i < g_iNumValues; i += iOuterNum) {
    const int jBegin = i;
    const int jEnd = std::min(g_iNumValues, jBegin + iOuterNum);

    for (int j = jBegin; j < jEnd; ++j) {
      const RealType x = g_a + (g_b - g_a)*(j + RealType(0.5))/RealType(g_iNumValues);
      p_input[j - jBegin] = x;
    }

    p_clGraph->Forward();

    for (int j = jBegin; j < jEnd; ++j) {
      const RealType x = p_input[j - jBegin];
      const RealType y = p_output[j - jBegin];

      std::cout << x << '\t' << y << std::endl;
    }
  }

  return true;
}
