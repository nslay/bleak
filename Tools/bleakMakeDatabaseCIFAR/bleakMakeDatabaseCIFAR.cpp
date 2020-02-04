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

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <utility>
#include <unordered_map>
#include <fstream>
#include "InitializeModules.h"
#include "DatabaseFactory.h"
#include "ParameterContainer.h"
#include "Common.h"
#include "bsdgetopt.h"

void Usage(const char *p_cArg0) {
  std::cerr << "Usage: " << p_cArg0 << " [-a] [-h] [-f] [-t databaseType] [-k numClasses] -o trainOutputPath [-v validationOutputPath] [-V validationRatio] batchFile [batchFile2 ...]" << std::endl;
  exit(1);
}

std::shared_ptr<bleak::Database> OpenDatabase(const std::string &strOutputPath, const std::string &strDatabaseType) {
  std::shared_ptr<bleak::Database> p_clDB = bleak::DatabaseFactory::GetInstance().Create(strDatabaseType);

  if (p_clDB == nullptr) {
    std::cerr << "Error: Could not create database of type '" << strDatabaseType << "'." << std::endl;
    return std::shared_ptr<bleak::Database>();
  }

  if (!p_clDB->Open(strOutputPath, bleak::Database::WRITE)) {
    std::cerr << "Error: Failed to open '" << strOutputPath << "' for writing." << std::endl;
    return std::shared_ptr<bleak::Database>();
  }

  return p_clDB;
}

bool LoadCIFAR(std::vector<std::vector<double>> &vData, const std::string &strImagesFile, unsigned int uiNumClasses);
void FlipImages(std::vector<std::vector<double>> &vData); // Horizontal flip

int main(int argc, char **argv) {
  bleak::InitializeModules();

  const char * const p_cArg0 = argv[0];

  std::string strDatabaseType = "LMDB";
  std::string strTrainOutputPath;
  std::string strValidationOutputPath;
  bool bShuffle = false;
  bool bAugment = false;
  unsigned int uiNumClasses = 0;
  double dValidationRatio = 0.0;

  int c = 0;
  while ((c = getopt(argc, argv, "afhk:o:t:v:V:")) != -1) {
    switch (c) {
    case 'a':
      bAugment = true;
      break;
    case 'f':
      bShuffle = true;
      break;
    case 'h':
      Usage(p_cArg0);
      break;
    case 'k':
      {
        char *p = nullptr;
        uiNumClasses = strtoul(optarg, &p, 10);
        if (*p != '\0')
          Usage(p_cArg0);
      }
      break;
    case 'o':
      strTrainOutputPath = optarg;
      break;
    case 't':
      strDatabaseType = optarg;
      break;
    case 'v':
      strValidationOutputPath = optarg;
      break;
    case 'V':
      {
        char *p = nullptr;
        dValidationRatio = strtod(optarg, &p);
        if (*p != '\0' || dValidationRatio < 0.0 || dValidationRatio > 1.0)
          Usage(p_cArg0);
      }
      break;
    case '?':
    default:
      Usage(p_cArg0);
    }
  }

  argc -= optind;
  argv += optind;

  if (argc < 1 || strTrainOutputPath.empty() || strDatabaseType.empty())
    Usage(p_cArg0);

  if (uiNumClasses != 10 && uiNumClasses != 100) {
    std::cerr << "Error: Number of classes must be either 10 or 100.\n" << std::endl;
    Usage(p_cArg0);
  }

  std::shared_ptr<bleak::Database> p_clTrainDB = OpenDatabase(strTrainOutputPath, strDatabaseType);

  if (!p_clTrainDB)
    return -1;

  std::shared_ptr<bleak::Database> p_clValidationDB;
  std::unique_ptr<bleak::Transaction> p_clValidationTransaction;

  if (strValidationOutputPath.size() > 0) {
    p_clValidationDB = OpenDatabase(strValidationOutputPath, strDatabaseType);

    if (!p_clValidationDB)
      return -1;

    p_clValidationTransaction = p_clValidationDB->NewTransaction();
  }

  std::unique_ptr<bleak::Transaction> p_clTrainTransaction = p_clTrainDB->NewTransaction();

  std::vector<std::vector<double>> vData;

  size_t count = 0;
  char a_cKey[32] = "";

  for (int i = 0; i < argc; ++i) {
    const char * const p_cBatchFile = argv[i];
    std::cout << "Info: Loading '" << p_cBatchFile << "' ..." << std::endl;
    if (!LoadCIFAR(vData, p_cBatchFile, uiNumClasses)) {
      std::cerr << "Error: Failed to load binary file." << std::endl;
      return -1;
    }
  }

  if (vData.empty()) // Uhh?
    return -1;

  if (bShuffle)
    std::shuffle(vData.begin(), vData.end(), bleak::GetGenerator());

  size_t validationSize = (size_t)(dValidationRatio * vData.size());
  size_t trainBegin = 0;
  size_t trainEnd = vData.size() - validationSize;
  size_t validationBegin = trainEnd;
  size_t validationEnd = vData.size();

  if (bAugment) {
    std::cout << "Info: Augmenting with flipped images ..." << std::endl;

    FlipImages(vData);

    // vData is now double the size with original0, flipped0, original1, flipped1, etc...
    trainBegin *= 2;
    trainEnd *= 2;
    validationBegin *= 2;
    validationEnd *= 2;

    // Shuffle again within training/validation sets (can't pollute training with validation and vice versa!)
    if (bShuffle) {
      std::shuffle(vData.begin() + trainBegin, vData.begin() + trainEnd, bleak::GetGenerator());
      std::shuffle(vData.begin() + validationBegin, vData.begin() + validationEnd, bleak::GetGenerator());
    }
  }

  std::cout << "Info: Writing training entries..." << std::endl;

  for (size_t i = trainBegin; i < trainEnd; ++i) {
    const std::vector<double> &vRow = vData[i];

    snprintf(a_cKey, sizeof(a_cKey), "%09u", (unsigned int)count);
    p_clTrainTransaction->Put(a_cKey, (uint8_t *)vRow.data(), sizeof(vRow[0])*vRow.size());
    ++count;

    if ((count % 1000) == 0)
      p_clTrainTransaction->Commit();
  }

  if ((count % 1000) != 0)
    p_clTrainTransaction->Commit();

  std::cout << "Info: Wrote " << count << " training entries to the database." << std::endl;

  p_clTrainTransaction.reset();
  p_clTrainDB->Close();
  p_clTrainDB.reset();

  if (p_clValidationTransaction != nullptr) {
    count = 0;
    std::cout << "Info: Writing validation entries..." << std::endl;

    for (size_t i = validationBegin; i < validationEnd; ++i) {
      const std::vector<double> &vRow = vData[i];

      snprintf(a_cKey, sizeof(a_cKey), "%09u", (unsigned int)count);
      p_clValidationTransaction->Put(a_cKey, (uint8_t *)vRow.data(), sizeof(vRow[0])*vRow.size());
      ++count;

      if ((count % 1000) == 0)
        p_clValidationTransaction->Commit();
    }

    if ((count % 1000) != 0)
      p_clValidationTransaction->Commit();

    std::cout << "Info: Wrote " << count << " validation entries to the database." << std::endl;

    p_clValidationTransaction.reset();
    p_clValidationDB->Close();
    p_clValidationDB.reset();
  }

  std::cout << "\n# Data vertex definition for training. Add this to your training sad graph and adjust accordingly." << std::endl;
  std::cout << "DatabaseReader {" << std::endl;
  std::cout << "\tdatabaseType = \"" << strDatabaseType << "\";" << std::endl;
  std::cout << "\tdatabasePath = \"" << strTrainOutputPath << "\";" << std::endl;
  std::cout << "\tsize = [ $batchSize, " << vData[0].size()-1 << " ];" << std::endl;
  std::cout << "\tlabelIndex = 0;" << std::endl;
  std::cout << "} trainData;" << std::endl;

  if (strValidationOutputPath.size() > 0) {
    std::cout << "\n# Data vertex definition for validation. Add this to your validation sad graph and adjust accordingly." << std::endl;
    std::cout << "DatabaseReader {" << std::endl;
    std::cout << "\tdatabaseType = \"" << strDatabaseType << "\";" << std::endl;
    std::cout << "\tdatabasePath = \"" << strValidationOutputPath << "\";" << std::endl;
    std::cout << "\tsize = [ $batchSize, " << vData[0].size()-1 << " ];" << std::endl;
    std::cout << "\tlabelIndex = 0;" << std::endl;
    std::cout << "} validData;" << std::endl;
  }

  return 0;
}

bool LoadCIFAR(std::vector<std::vector<double>> &vData, const std::string &strImagesFile, unsigned int uiNumClasses) {
  size_t rowSize = 0;
  size_t labelOffset = 0;

  switch (uiNumClasses) {
  case 10:
    rowSize = 3073;
    labelOffset = 0;
    break;
  case 100:
    rowSize = 3074;
    labelOffset = 1;
    break;
  default:
    std::cerr << "Error: Invalid number of classes." << std::endl;
    return false;
  }

  std::ifstream imagesStream(strImagesFile.c_str(), std::ifstream::binary);

  if (!imagesStream) {
    std::cerr << "Error: Could not open '" << strImagesFile << "'." << std::endl;
    return false;
  }

  imagesStream.seekg(0, std::ifstream::end);
  const size_t totalSize = imagesStream.tellg();
  imagesStream.seekg(0, std::ifstream::beg);

  const size_t numRows = totalSize / rowSize;

  if (numRows * rowSize != totalSize) {
    std::cerr << "Error: Unexpected file size " << rowSize << '.' << std::endl;
    return false;
  }

  std::vector<uint8_t> vRow(rowSize);

  for (size_t i = 0; i < numRows; ++i) {
    if (!imagesStream.read((char *)vRow.data(), rowSize)) {
      std::cerr << "Error: Failed to read row." << std::endl;
      return false;
    }

    vData.emplace_back(vRow.begin() + labelOffset, vRow.end());
  }

  return true;
}

void FlipImages(std::vector<std::vector<double>> &vData) {
  if (vData.empty() || vData[0].size() != 3073)
    return;

  std::vector<std::vector<double>> vCombined;
  vCombined.reserve(2*vData.size());

  std::vector<double> vNewRow(3073);

  for (auto &vRow : vData) {
    vNewRow[0] = vRow[0];

    const double * const p_dImage = vRow.data()+1;
    double * const p_dNewImage = vNewRow.data()+1;

    for (int c = 0; c < 3; ++c) {
      for (int y = 0; y < 32; ++y) {
        for (int x = 0; x < 32; ++x) {
          p_dNewImage[(c*32 + y)*32 + 31 - x] = p_dImage[(c*32 + y)*32 + x];
        }
      }
    }

    vCombined.push_back(std::move(vRow));
    vCombined.push_back(vNewRow);
  }

  vData.swap(vCombined);
}
