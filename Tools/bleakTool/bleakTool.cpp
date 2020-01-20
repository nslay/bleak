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

#include <cstdlib>
#include <iostream>
#include <string>
#include "VertexFactory.h"
#include "DatabaseFactory.h"
#include "OptimizerFactory.h"
#include "ParameterContainer.h"
#include "Graph.h"
#include "SadLoader.h"
#include "InitializeModules.h"
#include "bsdgetopt.h"

#ifdef BLEAK_USE_CUDA
#include "cuda_runtime.h"
#endif // BLEAK_USE_CUDA

std::string g_strGraphFile;
std::string g_strParameterFile;
std::string g_strWeightsDatabasePath;
unsigned int g_uiMaxIterations = 0;
std::vector<std::string> g_vIncludeDirs;

void Usage(const char *p_cArg0) {
  std::cerr << "Usage: " << p_cArg0 << " train|test [-h] [-t float|double] -c configFile [-d gpuDeviceIndex] [-n maxIterations] [-s seedString] [-I includeDir] -g graphFile -w weightsDatabasePath" << std::endl;
  exit(1);
}

template<typename RealType>
int Train();

template<typename RealType>
int Test();

template<typename RealType>
bool LoadWeights(const std::shared_ptr<bleak::Graph<RealType>> &p_clGraph, const std::string &strDatabaseFile);

template<typename RealType>
bool SaveWeights(const std::shared_ptr<bleak::Graph<RealType>> &p_clGraph,const std::string &strDatabaseFile);

int main(int argc, char **argv) {
  const char * const p_cArg0 = argv[0];

  if (argc < 2)
    Usage(p_cArg0);

  int iDevice = -1;
  std::string strType = "float";
  std::string strMode = argv[1];
  std::string strSeedString = "bleak";

  if (strMode != "train" && strMode != "test")
    Usage(p_cArg0);

  ++argv;
  --argc;

  int c = 0;
  while ((c = getopt(argc, argv, "c:d:g:hn:s:t:w:I:")) != -1) {
    switch (c) {
    case 'c':
      g_strParameterFile = optarg;
      break;
    case 'd':
      {
        char *p = nullptr;
        iDevice = strtol(optarg, &p, 10);
        if (*p != '\0')
          Usage(p_cArg0);
      }
      break;
    case 'g':
      g_strGraphFile = optarg;
      break;
    case 'h':
      Usage(p_cArg0);
      break;
    case 'n':
      {
        char *p = nullptr;
        g_uiMaxIterations = strtoul(optarg, &p, 10);

        if (*p != '\0')
          Usage(p_cArg0);
      }
      break;
    case 's':
      strSeedString = optarg;
      break;
    case 't':
      strType = optarg;
      break;
    case 'w':
      g_strWeightsDatabasePath = optarg;
      break;
    case 'I':
      g_vIncludeDirs.emplace_back(optarg);
      break;
    case '?':
    default:
      Usage(p_cArg0);
    }
  }

  if (g_strGraphFile.empty()) {
    std::cerr << "Error: Graph file not specified." << std::endl;
    Usage(p_cArg0);
  }

  if (iDevice < 0) {
    bleak::SetUseGPU(false);
    std::cout << "Info: Using CPU." << std::endl;
  }
  else {
#ifdef BLEAK_USE_CUDA
    // TODO: Multi-GPU setup
    if (cudaSetDevice(iDevice) != cudaSuccess) {
      std::cerr << "Error: Could not use GPU " << iDevice << '.' << std::endl;
      return -1;
    }

    bleak::SetUseGPU(true);
    std::cout << "Info: Using GPU " << iDevice << '.' << std::endl;

#else // !BLEAK_USE_CUDA
    std::cerr << "Error: GPU support is not compiled in." << std::endl;
    return -1;
#endif // BLEAK_USE_CUDA
  }

  // Now initialize...
  bleak::InitializeModules();

  std::cout << "Info: Using seed string '" << strSeedString << "'." << std::endl;

  std::seed_seq clSeed(strSeedString.begin(), strSeedString.end());

  bleak::GetGenerator().seed(clSeed);

  if (strType == "float") {
    if (strMode == "train")
      return Train<float>();
    if (strMode == "test")
      return Test<float>();
  }
  else if (strType == "double") {
    if(strMode == "train")
      return Train<double>();
    if(strMode == "test")
      return Test<double>();
  }
  else {
    std::cerr << "Error: Unknown type '" << strType << "'." << std::endl;
    Usage(p_cArg0);
  }

  return -1;
}

template<typename RealType>
int Train() {
  typedef bleak::Graph<RealType> GraphType;
  typedef bleak::Optimizer<RealType> OptimizerType;

  if (g_strParameterFile.empty()) {
    std::cerr << "Error: No parameter file provided." << std::endl;
    return -1;
  }

  std::shared_ptr<GraphType> p_clGraph = bleak::LoadGraph<RealType>(g_strGraphFile, g_vIncludeDirs);

  if (p_clGraph == nullptr) {
    std::cerr << "Error: Failed to load graph '" << g_strGraphFile << "'." << std::endl;
    return -1;
  }

  if (!p_clGraph->Initialize(true)) {
    std::cerr << "Error: Failed to intialize graph." << std::endl;
    return -1;
  }

  if (g_strWeightsDatabasePath.size() > 0 && !LoadWeights<RealType>(p_clGraph, g_strWeightsDatabasePath)) {
    std::cerr << "Error: Failed to load weights." << std::endl;
    return -1;
  }

  bleak::ParameterFile clParamsFile(g_strParameterFile, "bleak");
  bleak::ParameterMap clParams(clParamsFile);

  // Let the command line value take priority over the config file
  if (g_uiMaxIterations > 0)
    clParams.SetValue("maxIterations", g_uiMaxIterations);

  std::string strOptimizerType = clParams.GetValue<std::string>("optimizerType", std::string());

  if (strOptimizerType.empty()) {
    std::cerr << "Error: 'optimizerType' not specified in configuration file." << std::endl;
    return -1;
  }

  std::shared_ptr<OptimizerType> p_clOptimizer = bleak::OptimizerFactory<RealType>::GetInstance().Create(strOptimizerType, p_clGraph);

  if (!p_clOptimizer) {
    std::cerr << "Error: Could not create optimizer of type '" << strOptimizerType << "'." << std::endl;
    return -1;
  }

  if (!p_clOptimizer->SetParameters(clParams)) {
    std::cerr << "Error: Failed to set optimizer parameters." << std::endl;
    return -1;
  }

  if (!p_clOptimizer->Minimize()) {
    std::cerr << "Error: Optimizer failed." << std::endl;
    return -1;
  }

  return 0;
}

template<typename RealType>
int Test() {
  typedef bleak::Graph<RealType> GraphType;
  typedef bleak::Optimizer<RealType> OptimizerType;

  if (g_strWeightsDatabasePath.empty()) {
    std::cerr << "Error: No weights database path provided." << std::endl;
    return -1;
  }

  if (g_uiMaxIterations == 0) {
    std::cerr << "Error: Number of iterations is 0." << std::endl;
    return -1;
  }

  std::shared_ptr<GraphType> p_clGraph = bleak::LoadGraph<RealType>(g_strGraphFile, g_vIncludeDirs);

  if (p_clGraph == nullptr) {
    std::cerr << "Error: Failed to load graph '" << g_strGraphFile << "'." << std::endl;
    return -1;
  }

  if (!p_clGraph->Initialize(false)) {
    std::cerr << "Error: Failed to intialize graph." << std::endl;
    return -1;
  }

  if (!LoadWeights<RealType>(p_clGraph, g_strWeightsDatabasePath)) {
    std::cerr << "Error: Failed to load weights." << std::endl;
    return -1;
  }

  for (unsigned int e = 0; e < g_uiMaxIterations; ++e)
    p_clGraph->Forward();

  return 0;
}

template<typename RealType>
bool LoadWeights(const std::shared_ptr<bleak::Graph<RealType>> &p_clGraph,const std::string &strDatabasePath) {
  std::shared_ptr<bleak::Database> p_clDatabase = bleak::DatabaseFactory::GetInstance().Create("LMDB");

  if (p_clDatabase == nullptr) {
    std::cerr << "Error: Could not create database." << std::endl;
    return false;
  }

  if (!p_clDatabase->Open(strDatabasePath, bleak::Database::READ)) {
    std::cerr << "Error: Failed to open database '" << strDatabasePath << "'." << std::endl;
    return false;
  }

  return p_clGraph->LoadFromDatabase(p_clDatabase);
}

template<typename RealType>
bool SaveWeights(const std::shared_ptr<bleak::Graph<RealType>> &p_clGraph,const std::string &strDatabasePath) {
  std::shared_ptr<bleak::Database> p_clDatabase = bleak::DatabaseFactory::GetInstance().Create("LMDB");

  if (p_clDatabase == nullptr) {
    std::cerr << "Error: Could not create database." << std::endl;
    return false;
  }

  if (!p_clDatabase->Open(strDatabasePath, bleak::Database::WRITE)) {
    std::cerr << "Error: Failed to open database '" << strDatabasePath << "'." << std::endl;
    return false;
  }

  return p_clGraph->SaveToDatabase(p_clDatabase);
}
