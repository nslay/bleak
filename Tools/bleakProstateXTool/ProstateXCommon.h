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

#pragma once

#ifndef BLEAK_PROSTATEXCOMMON_H
#define BLEAK_PROSTATEXCOMMON_H

#include <vector>
#include <string>
#include <unordered_map>
#include <map>

// ITK stuff
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageSeriesReader.h"
#include "itkResampleImageFilter.h"
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"

namespace bleak {

// Point, finding ID, Zone, Label
struct Finding {
  typedef itk::ImageBase<3>::PointValueType PointValueType;
  typedef itk::ImageBase<3>::PointType PointType;

  // Peripherap Zone
  // Transition Zone
  // Anterior fibromuscular Stroma
  // Seminal Vesicles
  enum ZoneType { UnknownZone = -1, PZ, TZ, AS, SV };
  enum LabelType { UnknownLabel = -1, False, True };

  std::string strPatientId;
  int iFindingId = -1;
  PointType clPosition;
  ZoneType eZone = UnknownZone;
  LabelType eLabel = UnknownLabel;

  bool SetZone(const std::string &strZone) {
    if (strZone == "PZ")
      eZone = PZ;
    else if (strZone == "TZ")
      eZone = TZ;
    else if (strZone == "AS")
      eZone = AS;
    else if (strZone == "SV")
      eZone = SV;
    else
      eZone = UnknownZone;

    return eZone != UnknownZone;
  }

  bool SetLabel(const std::string &strLabel) {
    if (strLabel == "TRUE")
      eLabel = True;
    else if (strLabel == "FALSE")
      eLabel = False;
    else
      eLabel = UnknownLabel;

    return eLabel != UnknownLabel;
  }

  double Distance2(const PointType &clOther) const { return (double)clPosition.SquaredEuclideanDistanceTo(clOther); }
  double Distance(const PointType &clOther) const { return (double)clPosition.EuclideanDistanceTo(clOther); }
};

struct ROCCurve {
  std::vector<double> vThresholds;
  std::vector<double> vTruePositiveRates;
  std::vector<double> vFalsePositiveRates;

  static ROCCurve Compute(std::vector<std::pair<double, int>> &vScoresAndLabels);

  bool Good() const { return vThresholds.size() > 0 && vThresholds.size() == vTruePositiveRates.size() && vThresholds.size() == vFalsePositiveRates.size(); }
  double AUC() const;
};

std::ostream & operator<<(std::ostream &os, const ROCCurve &stROC);

void RegisterITKFactories();

std::vector<Finding> LoadFindings(const std::string &strFileName);
std::map<std::string, std::vector<Finding>> LoadFindingsMap(const std::string &strFileName);
std::unordered_map<std::string, std::vector<Finding>> LoadFindingsUnorderedMap(const std::string &strFileName);

void FindDicomFiles(const char *p_cDir, const char *p_cPattern, std::vector<std::string> &vFiles, bool bRecursive = false);
void FindDicomFolders(const char *p_cDir, const char *p_cPattern, std::vector<std::string> &vFolders, bool bRecursive = false);

// From AlignVolumes
template<typename PixelType, unsigned int Dimension>
typename itk::Image<PixelType, Dimension>::Pointer LoadImg(const std::string &strPath);

// From AlignVolumes
template<typename PixelType, unsigned int Dimension>
bool SaveImg(const itk::Image<PixelType, Dimension> *p_clImage, const std::string &strPath, bool bCompress);

// From AlignVolumes
template<typename PixelType, unsigned int Dimension>
typename itk::Image<PixelType, Dimension>::Pointer LoadDicomImage(const std::string &strPath, const std::string &strSeriesUID = std::string());

////////////// Template implementations below //////////////

template<typename PixelType, unsigned int Dimension>
typename itk::Image<PixelType, Dimension>::Pointer LoadImg(const std::string &strPath) {
  typedef itk::Image<PixelType, Dimension> ImageType;
  typedef itk::ImageFileReader<ImageType> ReaderType;

  typename ReaderType::Pointer p_clReader = ReaderType::New();

  p_clReader->SetFileName(strPath);

  try {
    p_clReader->Update();
  }
  catch (itk::ExceptionObject &e) {
    std::cerr << "Error: " << e << std::endl;
    return typename ImageType::Pointer();
  }

  return p_clReader->GetOutput();
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

template<typename PixelType, unsigned int Dimension>
typename itk::Image<PixelType, Dimension>::Pointer LoadDicomImage(const std::string &strPath, const std::string &strSeriesUID) {
  typedef itk::Image<PixelType, Dimension> ImageType;
  typedef itk::GDCMImageIO ImageIOType;
  typedef itk::GDCMSeriesFileNames FileNameGeneratorType;

  if (!FileExists(strPath)) // File or folder must exist
    return typename ImageType::Pointer();

  ImageIOType::Pointer p_clImageIO = ImageIOType::New();

  p_clImageIO->LoadPrivateTagsOn();
  p_clImageIO->KeepOriginalUIDOn();

  if (Dimension == 2) {
    // Read a 2D image
    typedef itk::ImageFileReader<ImageType> ReaderType;

    if (IsFolder(strPath)) // Must be a file
      return typename ImageType::Pointer();
    
    if (!p_clImageIO->CanReadFile(strPath.c_str()))
      return typename ImageType::Pointer();

    typename ReaderType::Pointer p_clReader = ReaderType::New();

    p_clReader->SetImageIO(p_clImageIO);
    p_clReader->SetFileName(strPath);

    try {
      p_clReader->Update();
    }
    catch (itk::ExceptionObject &e) {
      std::cerr << "Error: " << e << std::endl;
      return typename ImageType::Pointer();
    }

    typename itk::Image<PixelType, Dimension>::Pointer p_clImage = p_clReader->GetOutput();
    p_clImage->SetMetaDataDictionary(p_clImageIO->GetMetaDataDictionary());

    return p_clImage;
  }

  // Passed a file, read the series UID (ignore the one provided, if any)
  if (!IsFolder(strPath)) {

    if (!p_clImageIO->CanReadFile(strPath.c_str()))
      return typename ImageType::Pointer();

    p_clImageIO->SetFileName(strPath.c_str());

    try {
      p_clImageIO->ReadImageInformation();
    }
    catch (itk::ExceptionObject &e) {
      std::cerr << "Error: " << e << std::endl;
      return typename ImageType::Pointer();
    }

    const itk::MetaDataDictionary &clDicomTags = p_clImageIO->GetMetaDataDictionary();

    std::string strTmpSeriesUID;
    if (!itk::ExposeMetaData(clDicomTags, "0020|000e", strTmpSeriesUID))
      return typename ImageType::Pointer();

    Trim(strTmpSeriesUID);

    return LoadDicomImage<PixelType, Dimension>(DirName(strPath), strTmpSeriesUID); // Call this function again
  }

  FileNameGeneratorType::Pointer p_clFileNameGenerator = FileNameGeneratorType::New();

  // Use the ACTUAL series UID ... not some custom ITK concatenations of lots of junk.
  p_clFileNameGenerator->SetUseSeriesDetails(false); 
  p_clFileNameGenerator->SetDirectory(strPath);

  if (strSeriesUID.empty()) {
    // Passed a folder but no series UID ... pick the first series UID
    const FileNameGeneratorType::SeriesUIDContainerType &vSeriesUIDs = p_clFileNameGenerator->GetSeriesUIDs();

    if (vSeriesUIDs.empty())
      return typename ImageType::Pointer();

    // Use first series UID
    return LoadDicomImage<PixelType, Dimension>(strPath, vSeriesUIDs[0]);
  }

  const FileNameGeneratorType::FileNamesContainerType &vDicomFiles = p_clFileNameGenerator->GetFileNames(strSeriesUID);

  if (vDicomFiles.empty())
    return typename ImageType::Pointer();

  // Read 3D or higher (but 4D probably doesn't work correctly)
  typedef itk::ImageSeriesReader<ImageType> ReaderType;

  typename ReaderType::Pointer p_clReader = ReaderType::New();

  p_clReader->SetImageIO(p_clImageIO);
  p_clReader->SetFileNames(vDicomFiles);

  try {
    p_clReader->Update();
  }
  catch (itk::ExceptionObject &e) {
    std::cerr << "Error: " << e << std::endl;
    return typename ImageType::Pointer();
  }

  typename itk::Image<PixelType, Dimension>::Pointer p_clImage = p_clReader->GetOutput();
  p_clImage->SetMetaDataDictionary(p_clImageIO->GetMetaDataDictionary());

  return p_clImage;
  //return p_clReader->GetOutput();
}

} // end namespace bleak

#endif // !BLEAK_PROSTATEXCOMMON_H
