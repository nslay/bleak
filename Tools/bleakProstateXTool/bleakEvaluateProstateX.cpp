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

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include "Common.h"
#include "ProstateXCommon.h"
#include "bsdgetopt.h"

void Usage(const char *p_cArg0) {
  std::cerr << "Usage: " << p_cArg0 << " -f findings.csv -d probMapFolder [-l listFile] [-o outCsvFile]" << std::endl;
  exit(1);
}

int main(int argc, char **argv) {
  constexpr double dMaxDistance = 3.0; // 3mm radius ball
  constexpr double dProbPerc = 0.9; // Probability percentile inside of ball
  const char * const p_cArg0 = argv[0];

  std::string strProbMapRoot;
  std::string strFindingsCsvFile;
  std::string strListFile;
  std::string strOutCsvFile;

  int c = 0;
  while ((c = getopt(argc, argv, "d:f:hl:o:")) != -1) {
    switch (c) {
    case 'd':
      strProbMapRoot = optarg;
      break;
    case 'f':
      strFindingsCsvFile = optarg;
      break;
    case 'h':
      Usage(p_cArg0);
      break;
    case 'l':
      strListFile = optarg;
      break;
    case 'o':
      strOutCsvFile = optarg;
      break;
    case '?':
    default:
      Usage(p_cArg0);
    }
  }

  if (strProbMapRoot.empty() || strFindingsCsvFile.empty())
    Usage(p_cArg0);

  bleak::RegisterITKFactories();

  std::vector<bleak::Finding> vFindings = bleak::LoadFindings(strFindingsCsvFile);

  if (vFindings.empty()) {
    std::cerr << "Error: Could not load findings CSV file '" << strFindingsCsvFile << "'." << std::endl;
    return -1;
  }

  std::cout << "Info: Loaded " << vFindings.size() << " findings." << std::endl;

  std::unordered_map<std::string, std::vector<const bleak::Finding *>> mFindingsMap;

  for (const bleak::Finding &stFinding : vFindings)
    mFindingsMap[stFinding.strPatientId].push_back(&stFinding);
  
  std::vector<std::string> vPatientIds;

  if (strListFile.empty()) {
    for (const auto &stPair : mFindingsMap)
      vPatientIds.push_back(stPair.first);
  }
  else {
    std::ifstream listStream(strListFile);

    if (!listStream) {
      std::cerr << "Error: Could not open '" << strListFile << "'." << std::endl;
      return -1;
    }

    std::string strLine;
    while (std::getline(listStream, strLine)) {
      bleak::Trim(strLine);

      if (strLine.size() > 0)
        vPatientIds.push_back(strLine);
    }

    if (vPatientIds.empty()) {
      std::cerr << "Error: No patient IDs listed?" << std::endl;
      return -1;
    }
  }

  std::ofstream outCsvStream;
  if (strOutCsvFile.size() > 0) {
    outCsvStream.open(strOutCsvFile, std::ofstream::out | std::iostream::trunc);

    if (!outCsvStream) {
      std::cerr << "Error: Could not open output CSV file '" << strOutCsvFile << "'." << std::endl;
      return -1;
    }

    outCsvStream << "ProxID,fid,ClinSig\n";
  }

  typedef float PixelType;
  typedef itk::Image<PixelType, 3> ImageType;

  std::unordered_map<const bleak::Finding *, std::vector<PixelType>> mProbsByFinding;
  std::vector<std::pair<double, int>> vScoresAndLabels;

  for (const std::string &strPatientId : vPatientIds) {
    auto patientItr = mFindingsMap.find(strPatientId);

    std::cout << "Info: Processing '" << strPatientId << "' ..." << std::endl;

    if (patientItr == mFindingsMap.end()) {
      std::cerr << "Error: Could not find patient ID '" << strPatientId << "'. Ignoring ..." << std::endl;
      continue;
    }

    std::string strProbMapPath = strProbMapRoot + '/' + strPatientId + ".mha";

    ImageType::Pointer p_clProbMap = bleak::LoadImg<PixelType, 3>(strProbMapPath);

    if (!p_clProbMap) {
      std::cerr << "Error: Could not load '" << strProbMapPath << "'." << std::endl;
      return -1;
    }

    mProbsByFinding.clear();

    const itk::Size<3> clSize = p_clProbMap->GetBufferedRegion().GetSize();

    for (itk::IndexValueType z = 0; itk::SizeValueType(z) < clSize[2]; ++z) {
      for (itk::IndexValueType y = 0; itk::SizeValueType(y) < clSize[1]; ++y) {
        for (itk::IndexValueType x = 0; itk::SizeValueType(x) < clSize[0]; ++x) {
          const itk::Index<3> clIndex = {{ x, y, z }};
          ImageType::PointType clImagePosition;
          p_clProbMap->TransformIndexToPhysicalPoint(clIndex, clImagePosition);

          auto minItr = std::min_element(patientItr->second.begin(), patientItr->second.end(), 
            [&clImagePosition](const bleak::Finding *p_a, const bleak::Finding *p_b) -> bool {
              return p_a->Distance2(clImagePosition) < p_b->Distance2(clImagePosition);
            });

          // TODO: Foreground prostate segmentation?
          // TODO: Zones?
          if ((*minItr)->Distance(clImagePosition) <= dMaxDistance)
            mProbsByFinding[*minItr].push_back(p_clProbMap->GetPixel(clIndex));
        }
      }
    }

    for (const bleak::Finding *p_stFinding : patientItr->second) {
      std::vector<PixelType> &vProbs = mProbsByFinding[p_stFinding];

      PixelType prob = PixelType(0);

      if (vProbs.size() > 0) {
        std::sort(vProbs.begin(), vProbs.end());
        prob = vProbs[(size_t)(dProbPerc*vProbs.size())];
      }

      outCsvStream << p_stFinding->strPatientId << ',' << p_stFinding->iFindingId << ',' << prob << '\n';

      // If we're evaluating on unseen data... all of these should unknown labels!
      if (p_stFinding->eLabel != bleak::Finding::UnknownLabel)
        vScoresAndLabels.emplace_back(prob, (p_stFinding->eLabel == bleak::Finding::True));
    }
  }

  bleak::ROCCurve stROC = bleak::ROCCurve::Compute(vScoresAndLabels);

  std::cout << "Info: Overall performance ROC: " << std::endl;
  std::cout << stROC << std::endl;

  return 0;
}
