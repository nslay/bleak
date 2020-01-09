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

#ifndef BLEAK_ITKIMAGELOADER_H
#define BLEAK_ITKIMAGELOADER_H

#include <algorithm>
#include <vector>
#include <fstream>
#include <string>
#include "Common.h"
#include "Vertex.h"

// ITK stuff
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageSeriesReader.h"
#include "itkRGBPixel.h"
#include "itkRGBAPixel.h"
#include "itkVector.h"
#include "itkImageIOBase.h"
#include "itkImageIOFactory.h"
#include "itkResampleImageFilter.h"

// DICOM images
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"

namespace bleak {

template<typename RealType, unsigned int NumComponents>
struct PixelTraitsByComponents { 
  static_assert(NumComponents > 0, "Number of components must be larger than 0.");
  typedef itk::Vector<RealType, NumComponents> PixelType; 

  static RealType * Copy(const PixelType *p_begin, const PixelType *p_end, RealType *p_outData) {
    for (unsigned int c = 0; c < NumComponents; ++c) {
      p_outData = std::transform(p_begin, p_end, p_outData,
        [&c](const PixelType &pixel) -> RealType {
          return pixel[c];
        });
    }
    return p_outData;
  }
};

// Try to use specializations so scalar <-> RGB <-> RGBA automatically happen
template<typename RealType>
struct PixelTraitsByComponents<RealType, 1> { 
  typedef RealType PixelType; 

  static RealType * Copy(const PixelType *p_begin, const PixelType *p_end, RealType *p_outData) {
    return std::copy(p_begin, p_end, p_outData);
  }
};

template<typename RealType>
struct PixelTraitsByComponents<RealType, 3> { 
  typedef itk::RGBPixel<RealType> PixelType; 

  static RealType * Copy(const PixelType *p_begin, const PixelType *p_end, RealType *p_outData) {
    for (unsigned int c = 0; c < 3; ++c) {
      p_outData = std::transform(p_begin, p_end, p_outData,
        [&c](const PixelType &pixel) -> RealType {
          return pixel[c];
        });
    }
    return p_outData;
  }
};

template<typename RealType>
struct PixelTraitsByComponents<RealType, 4> { 
  typedef itk::RGBAPixel<RealType> PixelType; 

  static RealType * Copy(const PixelType *p_begin, const PixelType *p_end, RealType *p_outData) {
    for (unsigned int c = 0; c < 4; ++c) {
      p_outData = std::transform(p_begin, p_end, p_outData,
        [&c](const PixelType &pixel) -> RealType {
          return pixel[c];
        });
    }
    return p_outData;
  }
};

template<typename RealType, unsigned int Dimension>
class ITKImageLoader : public Vertex<RealType> {
public:
  bleakNewAbstractVertex(ITKImageLoader, Vertex<RealType>,
    bleakAddProperty("imageFile", m_strImageFile), // For one image
    bleakAddProperty("listFile", m_strListFile), // For a list of files relative to "directory"
    bleakAddProperty("directory", m_strDirectory), // Or just search the directory for images...
    bleakAddProperty("size", m_vSize), // BatchSize x C x Z x Y x X ...
    bleakAddOutput("outData")
  );

  bleakForwardVertexTypedefs();

  static constexpr unsigned int GetDimension() { return Dimension; }

  virtual ~ITKImageLoader() = default;

  virtual bool SetSizes() override {
    bleakGetAndCheckOutput(p_clOutData, "outData", false);

    if (2+GetDimension() != m_vSize.size()) {
      std::cerr << GetName() << ": Error: 'size' field expected to be [ BatchSize, Channels, Z, Y, X, ... ]. Sizes do not match (expected " <<
        GetDimension()+2 << " components, but got " << m_vSize.size() << " components)." << std::endl;
      return false;
    }

    const Size outSize(m_vSize);

    if (!outSize.Valid()) {
      std::cerr << GetName() << ": Error: 'size' is not valid (" << outSize << "). Components must be positive." << std::endl;
      return false;
    }

    p_clOutData->GetData().SetSize(outSize);
    p_clOutData->GetGradient().Clear(); // This is not learnable

    return true;
  }

  virtual bool Initialize() override {
    bleakGetAndCheckOutput(p_clOutData, "outData", false);

    m_iItr = 0;
    m_vPaths.clear();

    if (m_strImageFile.empty() && m_strListFile.empty() && m_strDirectory.empty()) {
      std::cerr << GetName() << ": Error: At least one of 'imageFile', 'listFile' or 'directory' must be specified." << std::endl;
      return false;
    }

    if (m_strListFile.size() > 0 && m_strImageFile.size() > 0) {
      std::cerr << GetName() << ": Error: Only one of 'imageFile' or 'listFile' should be specified." << std::endl;
      return false;
    }

    if (m_strImageFile.size() > 0) {
      m_vPaths.emplace_back(GetPath(m_strImageFile)); // One file
    }
    else if (m_strListFile.size() > 0) {
      // Load files from list
      std::ifstream listStream(m_strListFile);

      if (!listStream) {
        std::cerr << GetName() << "Error: Failed to load list file '" << m_strListFile << "'." << std::endl;
        return false;
      }

      std::string strLine;
      while (std::getline(listStream, strLine)) {
        Trim(strLine);
        m_vPaths.emplace_back(GetPath(strLine));
      }

      return false;
    }
    else {
      FindFiles(m_strDirectory.c_str(), "*", m_vPaths);
      FindFolders(m_strDirectory.c_str(), "*", m_vPaths); // There could be DICOM folders instead of files...

      if (m_vPaths.empty()) {
        std::cerr << GetName() << ": Error: Did not find any files/folders in '" << m_strDirectory << "'." << std::endl;
        return false;
      }
    }

    return m_vPaths.size() > 0;
  }

  virtual void Forward() override {
    bleakGetAndCheckOutput(p_clOutData, "outData");

    if (m_vPaths.empty())
      return;

    ArrayType &clOutData = p_clOutData->GetData();

    const int iBatchSize = clOutData.GetSize()[0];

    RealType *p_outData = clOutData.data();

    for (int i = 0; i < iBatchSize; ++i) {
      const std::string &strPath = m_vPaths[m_iItr++];

      if (m_iItr >= (int)m_vSize.size())
        m_iItr = 0;

      p_outData = LoadImg(strPath, p_outData);

      if (p_outData == nullptr) {
        std::cerr << GetName() << ": Error: Failed to load image '" << strPath << "'. Giving up ..." << std::endl;
        return;
      }
    }
  }

  virtual void Backward() override { } // Nothing to do

protected:
  ITKImageLoader() = default;

private:
  std::string m_strDirectory;
  std::string m_strListFile;
  std::string m_strImageFile;
  std::vector<int> m_vSize;

  std::vector<std::string> m_vPaths;
  int m_iItr = 0;

  std::string GetPath(const std::string &strFile) const {
    if (m_strDirectory.size() > 0)
      return m_strDirectory + '/' + strFile;

    return strFile;
  }

  template<unsigned int NumComponents>
  RealType * LoadImgHelper(const std::string &strPath, RealType *p_outData) const {
    if (m_vSize.size() != 2 + Dimension || m_vSize[1] != (int)NumComponents)
      return nullptr;

    typedef PixelTraitsByComponents<RealType, NumComponents> PixelTraitsType;
    typedef typename PixelTraitsType::PixelType PixelType;
    typedef itk::Image<PixelType, Dimension> ImageType;
    typedef itk::ImageFileReader<ImageType> ImageReaderType;
    typedef itk::ResampleImageFilter<ImageType, ImageType, RealType> ResampleImageType;

    itk::ImageIOBase::Pointer p_clImageIO = itk::ImageIOFactory::CreateImageIO(strPath.c_str(), itk::ImageIOFactory::ReadMode);

    if (!p_clImageIO)
      return nullptr;

    if (p_clImageIO->GetNumberOfDimensions() != GetDimension())
      return nullptr;

    itk::Size<Dimension> outSize;
    for (unsigned int d = 0; d < Dimension; ++d)
      outSize[Dimension-1-d] = itk::SizeValueType(m_vSize[2+d]);

    const int iImageSize = Size(m_vSize).Product(1);

    typename ImageReaderType::Pointer p_clReader = ImageReaderType::New();
    typename ResampleImageType::Pointer p_clResampleImage = ResampleImageType::New();

    p_clReader->SetImageIO(p_clImageIO);
    p_clReader->SetFileName(strPath);
    
    p_clResampleImage->SetInput(p_clReader->GetOutput());
    p_clResampleImage->SetSize(outSize);

    try {
      p_clResampleImage->Update();
    }
    catch (itk::ExceptionObject &e) {
      std::cerr << GetName() << ": Error: Failed to load image '" << strPath << "': " << e << std::endl;
      return nullptr;
    }

    typename ImageType::Pointer p_clImage = p_clResampleImage->GetOutput();

    if (!p_clImage)
      return nullptr;

    const itk::SizeValueType imageSize = p_clImage->GetBufferedRegion().GetNumberOfPixels();

    if (NumComponents*imageSize != itk::SizeValueType(iImageSize))
      return nullptr; // What?

    const PixelType * const p_buffer = p_clImage->GetBufferPointer();

    return PixelTraitsType::Copy(p_buffer, p_buffer + imageSize, p_outData);
  }

  template<unsigned int NumComponents>
  RealType * LoadDicomHelper(const std::string &strPath, RealType *p_outData) const {
    if (m_vSize.size() != 2 + Dimension || m_vSize[1] != (int)NumComponents || GetDimension() != 3 || !IsFolder(strPath))
      return nullptr;

    typedef PixelTraitsByComponents<RealType, NumComponents> PixelTraitsType;
    typedef typename PixelTraitsType::PixelType PixelType;
    typedef itk::Image<PixelType, Dimension> ImageType;
    typedef itk::ImageSeriesReader<ImageType> ReaderType;
    typedef itk::ResampleImageFilter<ImageType, ImageType, RealType> ResampleImageType;
    typedef itk::GDCMImageIO ImageIOType;
    typedef itk::GDCMSeriesFileNames FileNameGeneratorType;

    ImageIOType::Pointer p_clImageIO = ImageIOType::New();
    FileNameGeneratorType::Pointer p_clFileNameGenerator = FileNameGeneratorType::New();

    // Use the ACTUAL series UID ... not some custom ITK concatenations of lots of junk.
    p_clFileNameGenerator->SetUseSeriesDetails(false);
    p_clFileNameGenerator->SetDirectory(strPath);

    // Get a series UID
    const FileNameGeneratorType::SeriesUIDContainerType &vSeriesUIDs = p_clFileNameGenerator->GetSeriesUIDs();

    if (vSeriesUIDs.empty())
      return nullptr;

    const std::string strSeriesUID = vSeriesUIDs.front();

    const FileNameGeneratorType::FileNamesContainerType &vDicomFiles = p_clFileNameGenerator->GetFileNames(strSeriesUID);

    if (vDicomFiles.empty())
      return nullptr;

    itk::Size<Dimension> outSize;
    for (unsigned int d = 0; d < Dimension; ++d)
      outSize[Dimension-1-d] = itk::SizeValueType(m_vSize[2+d]);

    const int iImageSize = Size(m_vSize).Product(1);

    typename ReaderType::Pointer p_clReader = ReaderType::New();
    typename ResampleImageType::Pointer p_clResampleImage = ResampleImageType::New();

    p_clReader->SetImageIO(p_clImageIO);
    p_clReader->SetFileNames(vDicomFiles);

    p_clResampleImage->SetInput(p_clReader->GetOutput());
    p_clResampleImage->SetSize(outSize);

    try {
      p_clResampleImage->Update();
    }
    catch (itk::ExceptionObject &e) {
      std::cerr << GetName() << ": Error: Failed to load DICOM image '" << strPath << "': " << e << std::endl;
      return nullptr;
    }

    typename ImageType::Pointer p_clImage = p_clResampleImage->GetOutput();

    if (!p_clImage)
      return nullptr;

    const itk::SizeValueType imageSize = p_clImage->GetBufferedRegion().GetNumberOfPixels();

    if (NumComponents*imageSize != itk::SizeValueType(iImageSize))
      return nullptr; // What?

    const PixelType * const p_buffer = p_clImage->GetBufferPointer();

    return PixelTraitsType::Copy(p_buffer, p_buffer + imageSize, p_outData);    
  }

  RealType * LoadImg(const std::string &strPath, RealType *p_outData) const {
    if (m_vSize.size() != 2 + Dimension)
      return nullptr;

    if (!FileExists(strPath))
      return nullptr;

    if (IsFolder(strPath)) {
      switch (m_vSize[1]) {
      case 1:
        return LoadDicomHelper<1>(strPath, p_outData);
      case 2:
        return LoadDicomHelper<2>(strPath, p_outData);
      case 3:
        return LoadDicomHelper<3>(strPath, p_outData);
      case 4:
        return LoadDicomHelper<4>(strPath, p_outData);
      }

      return nullptr;
    }

    switch (m_vSize[1]) {
    case 1:
      return LoadImgHelper<1>(strPath, p_outData);
    case 2:
      return LoadImgHelper<2>(strPath, p_outData);
    case 3:
      return LoadImgHelper<3>(strPath, p_outData);
    case 4:
      return LoadImgHelper<4>(strPath, p_outData);
    }

    return nullptr;
  }

};

template<typename RealType>
class ITKImageLoader2D : public ITKImageLoader<RealType, 2> {
public:
  typedef ITKImageLoader<RealType, 2> WorkAroundVarArgsType;
  bleakNewVertex(ITKImageLoader2D, WorkAroundVarArgsType);
};

template<typename RealType>
class ITKImageLoader3D : public ITKImageLoader<RealType, 3> {
public:
  typedef ITKImageLoader<RealType, 3> WorkAroundVarArgsType;
  bleakNewVertex(ITKImageLoader3D, WorkAroundVarArgsType);
};

} // end namespace bleak

#endif // !BLEAK_ITKIMAGELOADER_H
