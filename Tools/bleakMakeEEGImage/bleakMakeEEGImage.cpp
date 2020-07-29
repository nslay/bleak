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
#include <fstream>
#include "Common.h"

// ITK stuff
#include "itkImage.h"
#include "itkMetaImageIOFactory.h"
#include "itkNiftiImageIOFactory.h"
#include "itkImageFileWriter.h"

// This tool prepares data from: https://kdd.ics.uci.edu/databases/eeg/eeg.html
// It constructs 2D float images with Y axis = channel and X axis = time
// This is probably a good data set for 1D convolution

void Usage(const char *p_cArg0) {
  std::cerr << "Usage: " << p_cArg0 << " eegCaseFile eegOutputImage.mha" << std::endl;
  exit(1);
}

void RegisterITKFactories();
int GetChannelIndex(const char *p_cChannelName);

// From AlignVolumes
template<typename PixelType, unsigned int Dimension>
bool SaveImg(const itk::Image<PixelType, Dimension> *p_clImage, const std::string &strPath, bool bCompress);

itk::Image<float, 2>::Pointer ReadEeg(const char *p_cEegFile);

int main(int argc, char **argv) {
  const char * const p_cArg0 = argv[0];

  RegisterITKFactories();

  if (argc != 3)
    Usage(p_cArg0);

  const char *p_cEegFile = argv[1];
  const char *p_cEegImage = argv[2];

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

int GetChannelIndex(const char *p_cChannelName) {
  static const char * const a_cChannelNames[] = {
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

  if (p_cChannelName == nullptr)
    return -1;

  for (int iChannelIndex = 0; a_cChannelNames[iChannelIndex] != nullptr; ++iChannelIndex) {
    if (strcmp(p_cChannelName, a_cChannelNames[iChannelIndex]) == 0)
      return iChannelIndex;
  }

  return -1;
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
  typedef itk::Image<float, 2> ImageType;

  if (p_cEegFile == nullptr)
    return nullptr;

  std::ifstream eegStream(p_cEegFile);

  if (!eegStream) {
    std::cerr << "Error: Could not open '" << p_cEegFile << "'." << std::endl;
    return nullptr;
  }

  itk::Size<2> clSize;
  clSize[0] = 256;
  clSize[1] = 64;

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

    const int iChannelNumber = GetChannelIndex(vTokens[1].c_str());

    if (iChannelNumber < 0) {
      std::cerr << "Error: Could not determine channel index for channel name '" << vTokens[1] << "'." << std::endl;  
      return nullptr;
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

  return p_clImage;
}
