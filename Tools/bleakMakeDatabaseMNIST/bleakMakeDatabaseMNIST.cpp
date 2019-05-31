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

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <fstream>
#include "InitializeModules.h"
#include "DatabaseFactory.h"
#include "ParameterContainer.h"
#include "Common.h"
#include "bsdgetopt.h"

void Usage(const char *p_cArg0) {
  std::cerr << "Usage: " << p_cArg0 << " [-h] [-f] [-t databaseType] -o outputPath imagesFile labelsFile" << std::endl;
  exit(1);
}

uint32_t FromBigEndianU32(uint32_t &x) {
  // TODO: Make no-op on big endian machines
  std::reverse((uint8_t *)&x, ((uint8_t *)&x) + sizeof(x));
  return x;
}

bool LoadMNIST(std::vector<std::vector<double>> &vData, const std::string &strImagesFile, const std::string &strLabelsFile);

int main(int argc, char **argv) {
  bleak::InitializeModules();

  const char * const p_cArg0 = argv[0];

  std::string strDatabaseType = "LMDB";
  std::string strOutputPath;
  bool bShuffle = false;

  int c = 0;
  while ((c = getopt(argc, argv, "fho:t:")) != -1) {
    switch (c) {
    case 'f':
      bShuffle = true;
      break;
    case 'h':
      Usage(p_cArg0);
      break;
    case 'o':
      strOutputPath = optarg;
      break;
    case 't':
      strDatabaseType = optarg;
      break;
    case '?':
    default:
      Usage(p_cArg0);
    }
  }

  argc -= optind;
  argv += optind;

  if (argc != 2 || strOutputPath.empty() || strDatabaseType.empty())
    Usage(p_cArg0);

  std::shared_ptr<bleak::Database> p_clDB = bleak::DatabaseFactory::GetInstance().Create(strDatabaseType);

  if (p_clDB == nullptr) {
    std::cerr << "Error: Could not create database of type '" << strDatabaseType << "'." << std::endl;
    return -1;
  }

  if (!p_clDB->Open(strOutputPath, bleak::Database::WRITE)) {
    std::cerr << "Error: Failed to open '" << strOutputPath << "' for writing." << std::endl;
    return -1;
  }

  std::unique_ptr<bleak::Transaction> p_clTransaction = p_clDB->NewTransaction();

  std::vector<std::vector<double>> vData;

  size_t count = 0;
  char a_cKey[32] = "";

  const char * const p_cImagesFile = argv[0];
  const char * const p_cLabelsFile = argv[1];

  std::cout << "Info: Processing '" << p_cImagesFile << "' and '" << p_cLabelsFile << "' ..." << std::endl;

  if (!LoadMNIST(vData, p_cImagesFile, p_cLabelsFile)) {
    std::cerr << "Error: Failed to read CSV." << std::endl;
    return -1;
  }

  if (vData.empty()) // Uhh?
    return -1;

  if (bShuffle)
    std::shuffle(vData.begin(), vData.end(), bleak::GetGenerator());

  for (const std::vector<double> &vRow : vData) {
    snprintf(a_cKey, sizeof(a_cKey), "%09u", (unsigned int)count);
    p_clTransaction->Put(a_cKey, (uint8_t *)vRow.data(), sizeof(vRow[0])*vRow.size());
    ++count;

    if ((count % 1000) == 0)
      p_clTransaction->Commit();
  }

  if ((count % 1000) != 0)
    p_clTransaction->Commit();

  std::cout << "Info: Wrote " << count << " entries to the database." << std::endl;

  p_clTransaction.reset();

  p_clDB->Close();

  p_clDB.reset();

  std::cout << "\n# Data vertex definition. Add this to your sad graph and adjust accordingly." << std::endl;
  std::cout << "DatabaseReader {" << std::endl;
  std::cout << "\tdatabaseType = \"" << strDatabaseType << "\";" << std::endl;
  std::cout << "\tdatabasePath = \"" << strOutputPath << "\";" << std::endl;
  std::cout << "\tsize = [ $batchSize, " << vData[0].size()-1 << " ];" << std::endl;
  std::cout << "\tlabelIndex = 0;" << std::endl;
  std::cout << "} data;" << std::endl;

  return 0;
}

bool LoadMNIST(std::vector<std::vector<double>> &vData, const std::string &strImagesFile, const std::string &strLabelsFile) {
  vData.clear();

  std::ifstream imagesStream(strImagesFile.c_str(), std::ifstream::binary);
  std::ifstream labelsStream(strLabelsFile.c_str(), std::ifstream::binary);

  if (!imagesStream) {
    std::cerr << "Error: Could not open '" << strImagesFile << "'." << std::endl;
    return false;
  }

  if (!labelsStream) {
    std::cerr << "Error: Could not open '" << strLabelsFile << "'." << std::endl;
    return false;
  }

  uint32_t ui32Magic = 0;

  if (!imagesStream.read((char *)&ui32Magic, sizeof(ui32Magic)) || FromBigEndianU32(ui32Magic) != 0x00000803) {
    std::cerr << "Error: Could not read magic number or wrong magic number for image file '" << strImagesFile << "'." << std::endl;
    return false;
  }

  ui32Magic = 0;

  if (!labelsStream.read((char *)&ui32Magic, sizeof(ui32Magic)) || FromBigEndianU32(ui32Magic) != 0x00000801) {
    std::cerr << "Error: Could not read magic number or wrong magic number for label file '" << strImagesFile << "'." << std::endl;
    return false;
  }

  uint32_t ui32NumberOfLabels = 0;
  uint32_t ui32NumberOfImages = 0;
  uint32_t ui32Rows = 0;
  uint32_t ui32Columns = 0;

  if (!labelsStream.read((char *)&ui32NumberOfLabels, sizeof(ui32NumberOfLabels))) {
    std::cerr << "Error: Could not read number of labels." << std::endl;
    return false;
  }

  if (!imagesStream.read((char *)&ui32NumberOfImages, sizeof(ui32NumberOfImages))) {
    std::cerr << "Error: Could not read number of images." << std::endl;
    return false;
  }

  FromBigEndianU32(ui32NumberOfImages);
  FromBigEndianU32(ui32NumberOfLabels);

  if (ui32NumberOfImages != ui32NumberOfLabels) {
    std::cerr << "Error: Mismatch between number of images and number of labels (" << ui32NumberOfImages << " != " << ui32NumberOfLabels << ")." << std::endl;
    return false;
  }

  if (!imagesStream.read((char *)&ui32Rows, sizeof(ui32Rows)) || !imagesStream.read((char *)&ui32Columns, sizeof(ui32Columns))) {
    std::cerr << "Error: Could not read rows/columns from image file '" << strImagesFile << "'." << std::endl;
    return false;
  }

  FromBigEndianU32(ui32Rows);
  FromBigEndianU32(ui32Columns);

  if (ui32Rows == 0 || ui32Columns == 0) {
    std::cerr << "Error: Invalid rows/columns: rows = " << ui32Rows << ", columns = " << ui32Columns << std::endl;
    return false;
  }

  uint8_t ui8Label = 0;
  std::vector<uint8_t> vPixels(ui32Rows*ui32Columns);

  for (uint32_t i = 0; i < ui32NumberOfImages; ++i) {
    ui8Label = 0;

    if (!labelsStream.read((char *)&ui8Label, sizeof(ui8Label)) || !imagesStream.read((char *)vPixels.data(), vPixels.size()*sizeof(uint8_t))) {
      std::cerr << "Error: Failed to read label or image." << std::endl;
      return false;
    }

    vData.emplace_back(vPixels.size()+1);

    std::vector<double> &vRow = vData.back();

    vRow[0] = ui8Label;
    std::copy(vPixels.begin(), vPixels.end(), vRow.begin()+1);
  }

  return true;
}
