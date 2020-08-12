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
#include <cstring>
#include <iostream>
#include <string>
#include <utility>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include "Common.h"
#include "bsdgetopt.h"

// ITK stuff
#include "itkImage.h"
#include "itkMetaImageIOFactory.h"
#include "itkNiftiImageIOFactory.h"
#include "itkImageFileWriter.h"
#include "itkMetaDataDictionary.h"
#include "itkMetaDataObject.h"

// This tool prepares data from: https://kdd.ics.uci.edu/databases/eeg/eeg.html
// It constructs 2D float images with Y axis = channel and X axis = time
// This is probably a good data set for 1D convolution

namespace {

const char * const g_a_cChannelNames[] = {
  "FP1", "FP2", "F7", "F8",
  "AF1", "AF2", "FZ", "F4",
  "F3", "FC6", "FC5", "FC2",
  "FC1", "T8", "T7", "CZ",
  "C3", "C4", "CP5", "CP6",
  "CP1", "CP2", "P3", "P4",
  "PZ", "P8", "P7", "PO2",
  "PO1", "O2", "O1", "X",
  "AF7", "AF8", "F5", "F6",
  "FT7", "FT8", "FPZ", "FC4",
  "FC3", "C6", "C5", "F2",
  "F1", "TP8", "TP7", "AFZ",
  "CP3", "CP4", "P5", "P6",
  "C1", "C2", "PO7", "PO8",
  "FCZ", "POZ", "OZ", "P2",
  "P1", "CPZ", "nd", "Y",
  nullptr
};

std::unordered_map<std::string, int> g_mChannelMappings;
std::unordered_set<std::string> g_sExcludedChannels;

} // end anonymous namespace

void Usage(const char *p_cArg0) {
  std::cerr << "Usage: " << p_cArg0 << " [-h] [-e excludedChannelName,excludedChannelName2,...] eegReadFile eegOutputImage.mha" << std::endl;
  exit(1);
}

std::ostream & WarnOnce(const std::string &strChannelName) {
  static std::unordered_set<std::string> sWarnedChannels;
  static std::ostream blackhole(nullptr);

  return sWarnedChannels.emplace(strChannelName).second ? std::cerr : blackhole;
}

void RegisterITKFactories();
int GetChannelIndex(const std::string &strChannelName);

// From AlignVolumes
template<typename PixelType, unsigned int Dimension>
bool SaveImg(const itk::Image<PixelType, Dimension> *p_clImage, const std::string &strPath, bool bCompress);

itk::Image<float, 2>::Pointer ReadEeg(const char *p_cEegFile);

int main(int argc, char **argv) {
  const char * const p_cArg0 = argv[0];

  RegisterITKFactories();

  int c = 0;
  while ((c = getopt(argc, argv, "e:h")) != -1) {
    switch (c) {
    case 'e':
      {
        std::vector<std::string> vTokens = bleak::SplitString<std::string>(optarg, ",");
        g_sExcludedChannels.insert(vTokens.begin(), vTokens.end());
      }
      break;
    case 'h':
      Usage(p_cArg0);
      break;
    case '?':
    default:
      Usage(p_cArg0);
    }
  }

  argc -= optind;
  argv += optind;

  if (argc != 2)
    Usage(p_cArg0);

  const char *p_cEegFile = argv[0];
  const char *p_cEegImage = argv[1];

  itk::Image<float, 2>::Pointer p_clImage = ReadEeg(p_cEegFile);

  if (!p_clImage) {
    std::cerr << "Error: Could not read EEG file '" << p_cEegFile << "'." << std::endl;
    return -1;
  }

  std::cout << "Info: Saving '" << p_cEegImage << "' ..." << std::endl;

  if (!SaveImg<float, 2>(p_clImage,  p_cEegImage, true)) {
    std::cerr << "Error: Failed to save image." << std::endl;
    return -1;
  }

  return 0;
}

void RegisterITKFactories() {
  itk::MetaImageIOFactory::RegisterOneFactory();
  itk::NiftiImageIOFactory::RegisterOneFactory();
}

int GetChannelIndex(const std::string &strChannelName) {
  if (g_mChannelMappings.empty()) {
    // Construct mapping
    for (int i = 0; g_a_cChannelNames[i] != nullptr; ++i) {
      std::string strChannelName = g_a_cChannelNames[i];

      if (g_sExcludedChannels.find(strChannelName) == g_sExcludedChannels.end()) {
        const int iChannelIndex = (int)g_mChannelMappings.size();
        g_mChannelMappings.emplace(std::move(strChannelName), iChannelIndex);
      }
    }
  }

  auto itr = g_mChannelMappings.find(strChannelName);

  return itr != g_mChannelMappings.end() ? itr->second : -1;
}

template<typename PixelType, unsigned int Dimension>
bool SaveImg(const itk::Image<PixelType, Dimension> *p_clImage, const std::string &strPath, bool bCompress) {
  typedef itk::Image<PixelType, Dimension> ImageType;
  typedef itk::ImageFileWriter<ImageType> WriterType;

  typename WriterType::Pointer p_clWriter = WriterType::New();

  p_clWriter->SetFileName(strPath);
  p_clWriter->SetUseCompression(bCompress);
  p_clWriter->SetInput(p_clImage);

  try {
    p_clWriter->Update();
  }
  catch (itk::ExceptionObject &e) {
    std::cerr << "Error: " << e << std::endl;
    return false;
  }

  return true;
}

itk::Image<float, 2>::Pointer ReadEeg(const char *p_cEegFile) {
  (void)GetChannelIndex("FP1"); // Forcibly build g_mChannelMappings

  typedef itk::Image<float, 2> ImageType;

  if (p_cEegFile == nullptr || g_mChannelMappings.empty())
    return nullptr;

  std::ifstream eegStream(p_cEegFile);

  if (!eegStream) {
    std::cerr << "Error: Could not open '" << p_cEegFile << "'." << std::endl;
    return nullptr;
  }

  itk::Size<2> clSize;
  clSize[0] = 256;
  clSize[1] = g_mChannelMappings.size();

  ImageType::Pointer p_clImage = ImageType::New();
  ImageType::SpacingType clSpacing;

  //clSpacing[0] = 1.0/256; // In seconds
  //clSpacing[1] = 1.0;

  p_clImage->SetRegions(clSize);
  //p_clImage->SetSpacing(clSpacing);

  try {
    p_clImage->Allocate();
  }
  catch (itk::ExceptionObject &e) {
    std::cerr << "Error: " << e << std::endl;
    return nullptr;
  }

  p_clImage->FillBuffer(0.0f);

  std::vector<std::string> vTokens;

  std::string strLine;
  while (std::getline(eegStream, strLine)) {
    {
      size_t p = strLine.find('#');
      if (p != std::string::npos)
        strLine.erase(p);
    }

    bleak::Trim(strLine);

    if (strLine.empty())
      continue;

    vTokens = bleak::SplitString<std::string>(strLine, " ");

    if (vTokens.size() != 4)
      continue;

    const int iChannelNumber = GetChannelIndex(vTokens[1]);

    if (iChannelNumber < 0) {
      WarnOnce(vTokens[1]) << "Warning: Could not determine channel index for channel name '" << vTokens[1] << "'." << std::endl;  
      continue;
    }

    int iTimeIndex = 0;

    {
      char *p = nullptr;
      iTimeIndex = strtol(vTokens[2].c_str(), &p, 10);
      if (*p != '\0' || iTimeIndex < 0 || iTimeIndex >= clSize[0]) {
        std::cerr << "Error: Could not parse time index '" << vTokens[2] << "'." << std::endl;
        return nullptr;
      }
    }

    float fVoltage = 0.0f;

    {
      char *p = nullptr;
      fVoltage = strtof(vTokens[3].c_str(), &p);

      if (*p != '\0') {
        std::cerr << "Error: Could not parse voltage '" << vTokens[3] << "'." << std::endl;
        return nullptr;
      }
    }

    const itk::Index<2> clIndex = {{ iTimeIndex, iChannelNumber }};

    p_clImage->SetPixel(clIndex, fVoltage);
  }

  // Embed channel names to metadata
  {
    itk::MetaDataDictionary &clMetaData = p_clImage->GetMetaDataDictionary();

    std::vector<std::string> vChannelNames(g_mChannelMappings.size());

    for (const auto &stPair : g_mChannelMappings)
      vChannelNames[stPair.second] = stPair.first;

    std::string strChannelNames = vChannelNames[0];
    strChannelNames += "=0";

    for (size_t i = 1; i < vChannelNames.size(); ++i) {
      strChannelNames += ' ';
      strChannelNames += vChannelNames[i];
      strChannelNames += '=';
      strChannelNames += std::to_string(i);
    }

    itk::EncapsulateMetaData(clMetaData, "channelNames", strChannelNames);
  }



  return p_clImage;
}
