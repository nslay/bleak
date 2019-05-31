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

typedef std::unordered_map<int, std::unordered_map<std::string, double>> ColumnMapperType;

void Usage(const char *p_cArg0) {
  std::cerr << "Usage: " << p_cArg0 << " [-c configFile] [-d delimeter] [-h] [-f] [-s skipLines] [-t databaseType] -o outputPath file.csv [file2.csv ...]" << std::endl;
  exit(1);
}

bool LoadMappings(ColumnMapperType &mColumnMappings, const std::string &strConfigFile);
bool LoadCsv(std::vector<std::vector<double>> &vData, const std::string &strFileName, const std::string &strDelim = ",", 
  unsigned int uiSkipLines = 0, const ColumnMapperType &mColumnMappings = ColumnMapperType());

int main(int argc, char **argv) {
  bleak::InitializeModules();

  const char * const p_cArg0 = argv[0];

  unsigned int uiSkipLines = 0;
  std::string strDatabaseType = "LMDB";
  std::string strConfigFile;
  std::string strDelim = ",";
  std::string strOutputPath;
  bool bShuffle = false;

  int c = 0;
  while ((c = getopt(argc, argv, "c:d:fho:s:t:")) != -1) {
    switch (c) {
    case 'c':
      strConfigFile = optarg;
      break;
    case 'd':
      strDelim = optarg;
      break;
    case 'f':
      bShuffle = true;
      break;
    case 'h':
      Usage(p_cArg0);
      break;
    case 'o':
      strOutputPath = optarg;
      break;
    case 's':
      {
        char *p = nullptr;
        uiSkipLines = strtoul(optarg, &p, 10);
        if (*p != '\0')
          Usage(p_cArg0);
      }
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

  if (argc < 1 || strOutputPath.empty() || strDelim.empty() || strDatabaseType.empty())
    Usage(p_cArg0);

  ColumnMapperType mColumnMappings;

  if (strConfigFile.size() > 0 && !LoadMappings(mColumnMappings, strConfigFile)) {
    std::cerr << "Error: Failed to load column mappings." << std::endl;
    return -1;
  }

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

  size_t rowSize = 0, count = 0;
  char a_cKey[32] = "";

  for (int i = 0; i < argc; ++i) {
    const char * const p_cCsvFile = argv[i];

    std::cout << "Info: Processing '" << p_cCsvFile << "' ..." << std::endl;

    vData.clear();

    if (!LoadCsv(vData, p_cCsvFile, strDelim, uiSkipLines, mColumnMappings)) {
      std::cerr << "Error: Failed to read CSV." << std::endl;
      return -1;
    }

    if (vData.empty()) // Uhh?
      continue;

    if (rowSize == 0)
      rowSize = vData[0].size();

    if (rowSize != vData[0].size()) {
      std::cerr << "Error: Dimension mismatch between CSVs: Got " << vData[0].size() << " columns but expected " << rowSize << '.' << std::endl;
      return -1;
    }

    if (bShuffle)
      std::shuffle(vData.begin(), vData.end(), bleak::GetGenerator());

    for (const std::vector<double> &vRow : vData) {
      snprintf(a_cKey, sizeof(a_cKey), "%09u", (unsigned int)count);
      p_clTransaction->Put(a_cKey, (uint8_t *)vRow.data(), sizeof(vRow[0])*vRow.size());
      ++count;

      if ((count % 1000) == 0)
        p_clTransaction->Commit();
    }
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
  std::cout << "\tsize = [ $batchSize, " << rowSize-1 << " ];" << std::endl;
  std::cout << "\tlabelIndex = $labelIndex;" << std::endl;
  std::cout << "} data;" << std::endl;

  return 0;
}

bool LoadMappings(ColumnMapperType &mColumnMappings, const std::string &strConfigFile) {
  mColumnMappings.clear();

  bleak::ParameterFile clParams(strConfigFile, "Mappings");

  std::string strColumns = clParams.GetValue<std::string>("columns", std::string());

  if (strColumns.empty()) {
    std::cerr << "Error: Mappings:columns not defined." << std::endl;
    return false;
  }

  std::vector<std::string> vColumns = bleak::SplitString<std::string>(strColumns, ",");

  for (std::string &strColumn : vColumns) {
    bleak::Trim(strColumn);

    if (strColumn.empty())
      continue;

    std::cout << "Info: Loading mappings for column '" << strColumn << "' ..." << std::endl;

    clParams.SetSection(strColumn);

    const int iColumn = clParams.GetValue<int>("columnIndex", -1);

    if (iColumn < 0) {
      std::cerr << "Error: Invalid column index " << iColumn << '.' << std::endl;
      return false;
    }

    std::string strMappings = clParams.GetValue<std::string>("mappings", std::string());
    std::vector<std::string> vMappings = bleak::SplitString<std::string>(strMappings, ",");

    for (size_t i = 0; i < vMappings.size(); ++i) {
      std::string &strMapping = vMappings[i];
      bleak::Trim(strMapping);

      if (strMapping.empty())
        continue;

      const double dValue = clParams.GetValue<double>(vMappings[i], (double)i);

      std::cout << "Info: " << iColumn << ": " << strMapping << " --> " << dValue << std::endl;

      mColumnMappings[iColumn][strMapping] = dValue;
    }
  }

  return true;
}

bool LoadCsv(std::vector<std::vector<double>> &vData, const std::string &strFileName, const std::string &strDelim, unsigned int uiSkipLines, const ColumnMapperType &mColumnMappings) {
  vData.clear();

  std::ifstream csvStream(strFileName.c_str());

  if (!csvStream) {
    std::cerr << "Error: Could not open '" << strFileName << "'." << std::endl;
    return false;
  }

  std::string strLine;
  for (unsigned int i = 0; i < uiSkipLines; ++i)
    std::getline(csvStream, strLine);

  std::vector<std::string> vTokens;

  while (std::getline(csvStream, strLine)) {
    bleak::Trim(strLine);

    if (strLine.empty())
      continue;

    vTokens = bleak::SplitString<std::string>(strLine, strDelim);

    if (vData.size() > 0 && vData[0].size() != vTokens.size()) {
      std::cerr << "Error: Invalid number of tokens: Got " << vTokens.size() << " tokens but expected " << vData[0].size() << '.' << std::endl;
      return false;
    }

    vData.emplace_back(vTokens.size());
    std::vector<double> &vRow = vData.back();

    for (size_t i = 0; i < vTokens.size(); ++i) {
      auto colItr = mColumnMappings.find((int)i);
      std::string &strToken = vTokens[i];

      bleak::Trim(strToken);

      if (strToken.empty()) {
        std::cerr << "Error: Encountered empty token." << std::endl;
        return false;
      }

      if (colItr == mColumnMappings.end()) {
        // Just read as double
        char *p = NULL;
        const double dValue = strtod(strToken.c_str(), &p);
        if (*p != '\0') {
          std::cerr << "Error: Could not parse token '" << strToken << "' as double." << std::endl;
          return false;
        }

        vRow[i] = dValue;
      }
      else {
        const auto &mMapper = colItr->second;

        auto mapItr = mMapper.find(strToken);

        if (mapItr == mMapper.end()) {
          std::cerr << "Error: Could not map token '" << strToken << "'. Mapping was not defined." << std::endl;
          return false;
        }

        vRow[i] = mapItr->second;
      }
    }
  }

  return true;
}
