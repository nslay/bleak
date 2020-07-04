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

#include <algorithm>
#include "Common.h"
#include "ProstateXCommon.h"

// ITK stuff
#include "itkMetaImageIOFactory.h"
#include "itkNiftiImageIOFactory.h"

namespace bleak {

ROCCurve ROCCurve::Compute(std::vector<std::pair<double, int>> &vScoresAndLabels) {
  constexpr double dSmall = 1e-5;

  if (vScoresAndLabels.empty())
    return ROCCurve();

  // Sort over scores in ascending order
  std::sort(vScoresAndLabels.begin(), vScoresAndLabels.end(),
    [](const auto &a, const auto &b) -> bool {
      return a.first < b.first;
    });

  size_t a_totalCounts[2] = { 0, 0 };

  for (const auto &stPair : vScoresAndLabels)
    ++a_totalCounts[stPair.second != 0];

  // Print a warning but try to proceed anyway...
  if (a_totalCounts[0] == 0 || a_totalCounts[1] == 0)
    std::cerr << "Warning: Positive or negative counts are 0: N- = " << a_totalCounts[0] << ", N+ = " << a_totalCounts[1] << std::endl;

  auto itr = vScoresAndLabels.begin();
  auto prevItr = itr++;

  size_t a_counts[2] = { 0, 0 }; // Counts less up to itr (excluding itr) ... counts for score <= threshold

  ROCCurve stROC;

  // Edge case (> smallest score - small value)
  stROC.vThresholds.push_back(vScoresAndLabels.front().first - dSmall);
  stROC.vTruePositiveRates.push_back(a_totalCounts[1] != 0 ? 1.0 : 0.0);
  stROC.vFalsePositiveRates.push_back(a_totalCounts[0] != 0 ? 1.0 : 0.0);

  while (itr != vScoresAndLabels.end()) {
    ++a_counts[prevItr->second != 0];

    if (itr->first - prevItr->first > dSmall) {
      const double dThreshold = 0.5*(prevItr->first + itr->first);

      // Compute counts for score > threshold
      const size_t truePositiveCount = (a_totalCounts[1] - a_counts[1]);
      const size_t falsePositiveCount = (a_totalCounts[0] - a_counts[0]);

      const double dFalsePositiveRate = a_totalCounts[0] > 0 ? falsePositiveCount / (double)a_totalCounts[0] : 0.0;
      const double dTruePositiveRate = a_totalCounts[1] > 0 ? truePositiveCount / (double)a_totalCounts[1] : 0.0;

      stROC.vThresholds.push_back(dThreshold);
      stROC.vTruePositiveRates.push_back(dTruePositiveRate);
      stROC.vFalsePositiveRates.push_back(dFalsePositiveRate);
    }

    prevItr = itr++;
  }

  // Edge case (> largest score ... which is none of them)
  stROC.vThresholds.push_back(vScoresAndLabels.back().first);
  stROC.vTruePositiveRates.push_back(0.0);
  stROC.vFalsePositiveRates.push_back(0.0);

  return stROC;
}

double ROCCurve::AUC() const {
  if (!Good())
    return -1.0;

  double dSum = 0.0;

   // Trapezoid rule
  for (size_t i = 1; i < vFalsePositiveRates.size(); ++i) {
    const double dDelta = vFalsePositiveRates[i-1] - vFalsePositiveRates[i];
    dSum += dDelta * (vTruePositiveRates[i-1] + vTruePositiveRates[i]);
  }

  return 0.5*dSum;
}

std::ostream & operator<<(std::ostream &os, const ROCCurve &stROC) {
  if (!stROC.Good())
    return os << "# Invalid ROCCurve";

  os << "# threshold, false positive rate, true positive rate\n";

  for (size_t i = 0; i < stROC.vThresholds.size(); ++i)
    os << stROC.vThresholds[i] << ", " << stROC.vFalsePositiveRates[i] << ", " << stROC.vTruePositiveRates[i] << '\n';

  return os << "# AUC = " << stROC.AUC();
}

void RegisterITKFactories() {
  itk::MetaImageIOFactory::RegisterOneFactory();
  itk::NiftiImageIOFactory::RegisterOneFactory();
}

std::vector<Finding> LoadFindings(const std::string &strFileName) {
  std::ifstream csvStream(strFileName);

  if (!csvStream) {
    std::cerr << "Error: Failed to open '" << strFileName << "'." << std::endl;
    return std::vector<Finding>();
  }

  // First line is a header
  std::string strLine;
  if (!std::getline(csvStream, strLine))
    return std::vector<Finding>();

  Trim(strLine);
  std::vector<std::string> vFields = SplitString<std::string>(strLine, ",");

  const size_t numFields = vFields.size();

  if (numFields != 4 && numFields != 5) {
    std::cerr << "Error: Expected 4 or 5 comma (',') delimited fields (got " << numFields << " fields)." << std::endl;
    return std::vector<Finding>();
  }

  Finding stFinding;
  std::vector<Finding> vFindings;
  std::vector<Finding::PointValueType> vPoint;

  while (std::getline(csvStream, strLine)) {
    Trim(strLine);

    if (strLine.empty()) // Empty line?
      continue;

    vFields = SplitString<std::string>(strLine, ",");

    if (vFields.size() != numFields) {
      std::cerr << "Error: Unexpected number of fields in '" << strLine << "' (expected " << numFields << " but got " << vFields.size() << ")." << std::endl;
      return std::vector<Finding>();
    }

    stFinding.strPatientId = vFields[0];

    {
      char *p = nullptr;
      stFinding.iFindingId = strtol(vFields[1].c_str(), &p, 10);

      if (*p != '\0') {
        std::cerr << "Error: Failed to parse finding ID '" << vFields[1] << "'." << std::endl;
        return std::vector<Finding>();
      }
    }

    Trim(vFields[2]);

    vPoint = SplitString<Finding::PointValueType>(vFields[2], " \t");

    if (vPoint.size() != 3) {
      std::cerr << "Error: Failed to parse 3D position '" << vFields[2] << "'." << std::endl;
      return std::vector<Finding>();
    }

    std::copy(vPoint.begin(), vPoint.end(), stFinding.clPosition.Begin());
    
    if (!stFinding.SetZone(vFields[3])) {
      std::cerr << "Error: Failed to set zone '" << vFields[3] << "'." << std::endl;
      return std::vector<Finding>();
    }

    if (numFields > 4 && !stFinding.SetLabel(vFields[4])) {
      std::cerr << "Error: Failed to set label '" << vFields[4] << "'." << std::endl;
      return std::vector<Finding>();
    }
    
    vFindings.emplace_back(std::move(stFinding));
  }

  return vFindings;
}

std::map<std::string, std::vector<Finding>> LoadFindingsMap(const std::string &strFileName) {
  std::vector<Finding> vFindings = LoadFindings(strFileName);

  std::map<std::string, std::vector<Finding>> mFindings;

  for (Finding &stFinding : vFindings)
    mFindings[stFinding.strPatientId].emplace_back(std::move(stFinding));

  return mFindings;
}

std::unordered_map<std::string, std::vector<Finding>> LoadFindingsUnorderedMap(const std::string &strFileName) {
  std::vector<Finding> vFindings = LoadFindings(strFileName);

  std::unordered_map<std::string, std::vector<Finding>> mFindings;

  for (Finding &stFinding : vFindings)
    mFindings[stFinding.strPatientId].emplace_back(std::move(stFinding));

  return mFindings;
}

void FindDicomFiles(const char *p_cDir, const char *p_cPattern, std::vector<std::string> &vFiles, bool bRecursive) {
  typedef itk::GDCMImageIO ImageIOType;

  ImageIOType::Pointer p_clImageIO = ImageIOType::New();

  std::vector<std::string> vTmpFiles;

  FindFiles(p_cDir, p_cPattern, vTmpFiles, bRecursive);

  vFiles.reserve(vTmpFiles.size());

  for (std::string &strFile : vTmpFiles) {
    if (p_clImageIO->CanReadFile(strFile.c_str()))
      vFiles.emplace_back(std::move(strFile));
  }
}

void FindDicomFolders(const char *p_cDir, const char *p_cPattern, std::vector<std::string> &vFolders, bool bRecursive) {
  typedef itk::GDCMImageIO ImageIOType;

  ImageIOType::Pointer p_clImageIO = ImageIOType::New();

  std::vector<std::string> vTmpFolders, vTmpFiles;

  vTmpFolders.push_back(p_cDir); // Check base folder too

  FindFolders(p_cDir, p_cPattern, vTmpFolders, bRecursive);

  for (size_t i = 0; i < vTmpFolders.size(); ++i) {
    const std::string &strFolder = vTmpFolders[i];

    vTmpFiles.clear();

    FindFiles(strFolder.c_str(), "*", vTmpFiles, false);

    for (size_t j = 0; j < vTmpFiles.size(); ++j) {
      const std::string &strFile = vTmpFiles[j];

      if (p_clImageIO->CanReadFile(strFile.c_str())) {
        vFolders.push_back(strFolder);
        break;
      }
    }
  }
}


} // end namespace bleak
