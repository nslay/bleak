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

#ifdef WITH_ITK
#include "itkRGBPixel.h"
#include "itkImage.h"
#include "itkResampleImageFilter.h"
#include "itkImageFileWriter.h"
#include "itkMetaImageIOFactory.h"
#endif // WITH_ITK

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

#ifdef WITH_ITK
void RotateImages(std::vector<std::vector<double>> &vData); // +/-15 degree rotations
#endif // WITH_ITK

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

#ifdef WITH_ITK
    std::cout << "Info: Augmenting with rotated images ..." << std::endl;

    RotateImages(vData);

    // vData is now triple the size of the original0, rotated0, rotated2, original1, rotated1, rotated2, etc...
    trainBegin *= 3;
    trainEnd *= 3;
    validationBegin *= 3;
    validationEnd *= 3;
#endif // WITH_ITK

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

#ifdef WITH_ITK
void RegisterITKFactories() {
  // Images
  itk::MetaImageIOFactory::RegisterOneFactory();
  //itk::NiftiImageIOFactory::RegisterOneFactory();

  // Meshes
  //itk::OBJMeshIOFactory::RegisterOneFactory();
  //itk::STLMeshIOFactory::RegisterOneFactory();
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

void RotateImages(std::vector<std::vector<double>> &vData) {
  //RegisterITKFactories();

  typedef itk::RGBPixel<double> PixelType;
  typedef itk::Image<PixelType, 2> ImageType;

  typedef itk::ResampleImageFilter<ImageType, ImageType> ResamplerType;

  std::vector<std::vector<double>> vCombined;
  vCombined.reserve(3*vData.size());

  ResamplerType::Pointer p_clResampler = ResamplerType::New();
  ImageType::Pointer p_clImage = ImageType::New();

  itk::Size<2> clSize;
  clSize[0] = clSize[1] = 32;

  const itk::SizeValueType imageSize = clSize[0]*clSize[1];

  p_clImage->SetRegions(clSize);
  p_clImage->Allocate();

  p_clResampler->SetOutputOrigin(p_clImage->GetOrigin());
  p_clResampler->SetSize(clSize);
  p_clResampler->SetOutputSpacing(p_clImage->GetSpacing());

  std::vector<double> vNewRow(3073);

  constexpr int iNumRotations = 2;
  ImageType::DirectionType a_clR[iNumRotations];

  for (int i = 0; i < iNumRotations; ++i) {
    const double dAngle = -15 + 30*i;
    a_clR[i](0,0) = a_clR[i](1,1) = std::cos(dAngle*M_PI/180.0);
    a_clR[i](1,0) = std::sin(dAngle*M_PI/180.0);
    a_clR[i](0,1) = -a_clR[i](1,0);
  }

  for (auto &vRow : vData) {
    vNewRow[0] = vRow[0]; // Class label

    const double *p_dImage = vRow.data()+1;
    double *p_dNewImage = vNewRow.data()+1;

    for (int c = 0; c < 3; ++c) {
      PixelType * const p_clBuffer = p_clImage->GetBufferPointer();

      for (itk::SizeValueType i = 0; i < imageSize; ++i)
        p_clBuffer[i][c] = p_dImage[c*imageSize + i];
    }

    vCombined.push_back(std::move(vRow)); // Already copied

    //SaveImg(p_clImage.GetPointer(), "original.mha", true);

    for (int i = 0; i < iNumRotations; ++i) {
      p_clResampler->SetOutputDirection(a_clR[i]);

      // ITK rotates with respect to the origin (top left corner). 
      // We need to calculate a new origin so that the center point maps to itself after the resampling.
      ImageType::PointType clCenter;
      clCenter[0] = clCenter[1] = 16.0;

      ImageType::PointType clOutputOrigin = clCenter - a_clR[i] * clCenter;

      p_clResampler->SetOutputOrigin(clOutputOrigin);

      p_clResampler->SetInput(p_clImage);
      p_clResampler->Update();
  
      ImageType::Pointer p_clNewImage = p_clResampler->GetOutput();

      //SaveImg(p_clNewImage.GetPointer(), std::to_string(i) + ".mha", true);

      const PixelType * const p_clBegin = p_clNewImage->GetBufferPointer();
      const PixelType * const p_clEnd = p_clBegin + imageSize;

      for (int c = 0; c < 3; ++c) {
        std::transform(p_clBegin, p_clEnd, p_dNewImage + c*imageSize, 
          [&c](const PixelType &clPixel) -> double {
            return clPixel[c];
          });
      }

      vCombined.push_back(vNewRow);
    }

    //exit(0);
  }

  vData.swap(vCombined);
}
#endif // WITH_ITK