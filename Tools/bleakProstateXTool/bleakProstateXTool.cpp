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
#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include "bsdgetopt.h"

#include "Common.h"
#include "Graph.h"
#include "SadLoader.h"
#include "BlasWrapper.h"
#include "DatabaseFactory.h"
#include "InitializeModules.h"
#include "ProstateXCommon.h"

#ifdef BLEAK_USE_CUDA
#include "cuda_runtime.h"
#endif // BLEAK_USE_CUDA

// ITK stuff
#include "itkResampleImageFilter.h"

void Usage(const char *p_cArg0) {
  std::cerr << "Usage: " << p_cArg0 << " [-h] [-d gpuDeviceIndex] [-I includeDir] [-t float|double] -g graphFile -w weightsDatabasePath -T t2wPath -A adcPath -B bValuePath [-o outputPath]" << std::endl;
  exit(1);
}

class ProstateXToolBase {
public:
  virtual ~ProstateXToolBase() = default;

  virtual bool LoadGraph(const std::string &strGraphFile, const std::string &strWeightsFile, const std::vector<std::string> &vIncludeDirs) = 0;

  virtual bool LoadT2W(const std::string &strPath) = 0;
  virtual bool LoadADC(const std::string &strPath) = 0;
  virtual bool LoadBValue(const std::string &strPath) = 0;
  virtual bool SaveScoreMap(const std::string &strPath, bool bCompress = true) const = 0;

  virtual bool Run() = 0;
};

template<typename RealType>
class ProstateXTool : public ProstateXToolBase {
public:
  typedef itk::Image<RealType, 3> ImageType;
  typedef bleak::Graph<RealType> GraphType;

  virtual ~ProstateXTool() = default;

  // Everything with BLAS and hardware is assumed to be initialized BEFORE this call
  virtual bool LoadGraph(const std::string &strGraphFile, const std::string &strWeightsFile, const std::vector<std::string> &vIncludeDirs) override;

  // These are assumed pre-processed. If you pass these functions raw/unaligned volumes, that is your mistake!
  virtual bool LoadT2W(const std::string &strPath) override;
  virtual bool LoadADC(const std::string &strPath) override;
  virtual bool LoadBValue(const std::string &strPath) override;
  virtual bool SaveScoreMap(const std::string &strPath, bool bCompress = true) const override;

  ImageType * GetScoreMap() const { return m_p_clScoreMap.GetPointer(); }

  virtual bool Run() override;

private:
  std::shared_ptr<GraphType> m_p_clGraph;

  typename ImageType::Pointer m_p_clT2WImage;
  typename ImageType::Pointer m_p_clADCImage;
  typename ImageType::Pointer m_p_clBValueImage;
  typename ImageType::Pointer m_p_clScoreMap;

  // Load and resample image to fit vertex's input size
  typename ImageType::Pointer LoadImg(const std::string &strPath, const std::string &strVertex);
};

template<typename PixelType, unsigned int Dimension>
typename itk::Image<PixelType, Dimension>::Pointer ResampleBySize(typename itk::Image<PixelType, Dimension>::Pointer p_clImage, const itk::Size<Dimension> clNewSize);

int main(int argc, char **argv) {
  const char * const p_cArg0 = argv[0];

  int iDevice = -1;
  std::string strType = "float";
  std::string strGraphFile;
  std::string strWeightsFile;
  std::string strOutputPath = "output.mha";
  std::string strT2WPath;
  std::string strADCPath;
  std::string strBValuePath;
  std::vector<std::string> vIncludeDirs;

  int c = 0;
  while ((c = getopt(argc, argv, "d:g:ho:t:w:A:B:I:T:")) != -1) {
    switch (c) {
    case 'd': 
      {
        char *p = nullptr;
        iDevice = strtol(optarg, &p, 10);
        if (*p != '\0')
          Usage(p_cArg0);
      }
      break;
    case 'g':
      strGraphFile = optarg;
      break;
    case 'h':
      Usage(p_cArg0);
      break;
    case 'o':
      strOutputPath = optarg;
      break;
    case 't':
      strType = optarg;
      break;
    case 'w':
      strWeightsFile = optarg;
      break;
    case 'A':
      strADCPath = optarg;
      break;
    case 'B':
      strBValuePath = optarg;
      break;
    case 'I':
      vIncludeDirs.emplace_back(optarg);
      break;
    case 'T':
      strT2WPath = optarg;
      break;
    case '?':
    default:
      Usage(p_cArg0);
    }
  }

  if (strGraphFile.empty() || strWeightsFile.empty() || strT2WPath.empty() || strADCPath.empty() || strBValuePath.empty())
    Usage(p_cArg0);

  bleak::RegisterITKFactories();

  if (iDevice < 0) {
    bleak::SetUseGPU(false);
    std::cout << "Info: Using CPU." << std::endl;
  }
  else {
#ifdef BLEAK_USE_CUDA
    // TODO: Multi-GPU setup
    if (cudaSetDevice(iDevice) != cudaSuccess) {
      std::cerr << "Error: Could not use GPU " << iDevice << '.' << std::endl;
      return -1;
    }

    bleak::SetUseGPU(true);
    std::cout << "Info: Using GPU " << iDevice << '.' << std::endl;

#else // !BLEAK_USE_CUDA
    std::cerr << "Error: GPU support is not compiled in." << std::endl;
    return -1;
#endif // BLEAK_USE_CUDA
  }

  // Now initialize...
  bleak::InitializeModules();

  std::unique_ptr<ProstateXToolBase> p_clTool;

  if (strType == "float")
    p_clTool = std::make_unique<ProstateXTool<float>>();
  else if (strType == "double")
    p_clTool = std::make_unique<ProstateXTool<double>>();
  else
    Usage(p_cArg0);

  if (!p_clTool->LoadGraph(strGraphFile, strWeightsFile, vIncludeDirs)) {
    std::cerr << "Error: Failed to load graph." << std::endl;
    return -1;
  }

  if (!p_clTool->LoadT2W(strT2WPath)) {
    std::cerr << "Error: Failed to load T2W image." << std::endl;
    return -1;
  }

  if (!p_clTool->LoadADC(strADCPath)) {
    std::cerr << "Error: Failed to load ADC image." << std::endl;
    return -1;
  }

  if (!p_clTool->LoadBValue(strBValuePath)) {
    std::cerr << "Error: Failed to load BValue image." << std::endl;
    return -1;
  }

  if (!p_clTool->Run()) {
    std::cerr << "Error: Failed to calculate score map." << std::endl;
    return -1;
  }

  std::cout << "Info: Saving score map to '" << strOutputPath << "' ..." << std::endl;

  if (!p_clTool->SaveScoreMap(strOutputPath)) {
    std::cerr << "Error: Failed to save score map." << std::endl;
    return -1;
  }

  return 0;
}

template<typename PixelType, unsigned int Dimension>
typename itk::Image<PixelType, Dimension>::Pointer ResampleBySize(typename itk::Image<PixelType, Dimension>::Pointer p_clImage, const itk::Size<Dimension> clNewSize) {
  typedef itk::Image<PixelType, Dimension> ImageType;
  typedef itk::ResampleImageFilter<ImageType, ImageType> ResamplerType;

  if (!p_clImage)
    return nullptr;

  const itk::Size<Dimension> clSize = p_clImage->GetBufferedRegion().GetSize();
  typename ImageType::SpacingType clResampleSpacing = p_clImage->GetSpacing();

  for (unsigned int d = 0; d < Dimension; ++d)
    clResampleSpacing[d] = clSize[d] * clResampleSpacing[d] / clNewSize[d];

  typename ResamplerType::Pointer p_clResampler = ResamplerType::New();

  p_clResampler->SetSize(clNewSize);
  p_clResampler->SetOutputSpacing(clResampleSpacing);
  p_clResampler->SetOutputOrigin(p_clImage->GetOrigin());
  p_clResampler->SetOutputDirection(p_clImage->GetDirection());
  p_clResampler->SetInput(p_clImage);

  try {
    p_clResampler->Update();
  }
  catch (itk::ExceptionObject &e) {
    std::cerr << "Error: Failed to resample: " << e << std::endl;
    return nullptr;
  }

  return p_clResampler->GetOutput();
}

////////////////////////// ProstateXTool //////////////////////////

// Everything is assumed to be initialized BEFORE this call
template<typename RealType>
bool ProstateXTool<RealType>::LoadGraph(const std::string &strGraphFile, const std::string &strWeightsFile, const std::vector<std::string> &vIncludeDirs) {
  m_p_clGraph = nullptr;

  std::shared_ptr<GraphType> p_clGraph = bleak::LoadGraph<RealType>(strGraphFile, vIncludeDirs);
  if (!p_clGraph) {
    std::cerr << "Error: Failed to load SAD graph '" << strGraphFile << "'." << std::endl;
    return false;
  }

  if (!p_clGraph->Initialize(false)) {
    std::cerr << "Error: Failed to initialize graph '" << strGraphFile << "'." << std::endl;
    return false;
  }

  std::shared_ptr<bleak::Database> p_clDatabase = bleak::DatabaseFactory::GetInstance().Create("LMDB");

  if (!p_clDatabase || !p_clDatabase->Open(strWeightsFile, bleak::Database::READ)) {
    std::cerr << "Error: Failed to load weights database '" << strWeightsFile << "'." << std::endl;
    return false;
  }

  if (!p_clGraph->LoadFromDatabase(p_clDatabase)) {
    std::cerr << "Error: Failed to load weights." << std::endl;
    return false;
  }

  if (!p_clGraph->FindVertex("t2w") || !p_clGraph->FindVertex("adc") || !p_clGraph->FindVertex("bvalue") || !p_clGraph->FindVertex("scores")) {
    std::cerr << "Error: Missing 't2w', 'adc', 'bvalue' or 'scores' vertices." << std::endl;
    return false;
  }

  m_p_clGraph = p_clGraph;

  return true;
}

// These are assumed pre-processed. If you pass these functions raw/unaligned volumes, that is your mistake!
template<typename RealType>
bool ProstateXTool<RealType>::LoadT2W(const std::string &strPath) {
  m_p_clT2WImage = LoadImg(strPath, "t2w");
  return m_p_clT2WImage.IsNotNull();
}

template<typename RealType>
bool ProstateXTool<RealType>::LoadADC(const std::string &strPath) {
  m_p_clADCImage = LoadImg(strPath, "adc");
  return m_p_clADCImage.IsNotNull();
}

template<typename RealType>
bool ProstateXTool<RealType>::LoadBValue(const std::string &strPath) {
  m_p_clBValueImage = LoadImg(strPath, "bvalue");
  return m_p_clBValueImage.IsNotNull();
}

template<typename RealType>
bool ProstateXTool<RealType>::SaveScoreMap(const std::string &strPath, bool bCompress) const {
  if (!m_p_clScoreMap)
    return false;

  return bleak::SaveImg<RealType, 3>(m_p_clScoreMap, strPath, bCompress);
}

template<typename RealType>
bool ProstateXTool<RealType>::Run() {
  typedef typename GraphType::EdgeType EdgeType;
  typedef typename GraphType::VertexType VertexType;
  typedef typename VertexType::ArrayType ArrayType;

  if (!m_p_clGraph || !m_p_clT2WImage || !m_p_clADCImage || !m_p_clBValueImage)
    return false;

  if (m_p_clT2WImage->GetBufferedRegion() != m_p_clADCImage->GetBufferedRegion() || m_p_clT2WImage->GetBufferedRegion() != m_p_clBValueImage->GetBufferedRegion()) {
    std::cerr << "Error: Images are different sizes." << std::endl;
    return false;
  }

  std::shared_ptr<VertexType> p_clScoresVertex = m_p_clGraph->FindVertex("scores");

  if (!p_clScoresVertex) {
    std::cerr << "Error: Could not find output vertex 'scores'." << std::endl;
    return false;
  }

  std::shared_ptr<EdgeType> p_clScoresEdge = p_clScoresVertex->GetInput("inData");

  if (!p_clScoresEdge) {
    std::cerr << "Error: Could not get input 'scores.inData'." << std::endl;
    return false;
  }

  const bleak::Size clOutputVertexSize = p_clScoresEdge->GetData().GetSize();

  std::cout << "Info: 'scores.inData' has size " << clOutputVertexSize << std::endl;

  // B x C x H x W
  // C = number of classes
  if (clOutputVertexSize.GetDimension() != 4 || !clOutputVertexSize.Valid() || clOutputVertexSize[0] != 1 || clOutputVertexSize[1] < 2) {
    std::cerr << "Error: Unexpected output size: " << clOutputVertexSize << std::endl;
    return false;
  }

  // These should not fail
  std::shared_ptr<VertexType> p_clT2WVertex = m_p_clGraph->FindVertex("t2w");
  std::shared_ptr<VertexType> p_clADCVertex = m_p_clGraph->FindVertex("adc");
  std::shared_ptr<VertexType> p_clBValueVertex = m_p_clGraph->FindVertex("bvalue");

  std::shared_ptr<EdgeType> p_clT2WEdge = p_clT2WVertex->GetOutput("outData");
  std::shared_ptr<EdgeType> p_clADCEdge = p_clADCVertex->GetOutput("outData");
  std::shared_ptr<EdgeType> p_clBValueEdge = p_clBValueVertex->GetOutput("outData");

  itk::Size<3> clT2WSize = m_p_clT2WImage->GetBufferedRegion().GetSize();

  typename ImageType::Pointer p_clScoreMap = ImageType::New();

  itk::Size<3> clScoreMapSize;
  clScoreMapSize[0] = clOutputVertexSize[3]; // Width
  clScoreMapSize[1] = clOutputVertexSize[2]; // Height;
  clScoreMapSize[2] = clT2WSize[2];

  typename ImageType::SpacingType clScoreMapSpacing = m_p_clT2WImage->GetSpacing();

  for (unsigned int d = 0; d < 3; ++d)
    clScoreMapSpacing[d] = clT2WSize[d] * clScoreMapSpacing[d] / clScoreMapSize[d];

  p_clScoreMap->SetOrigin(m_p_clT2WImage->GetOrigin());
  p_clScoreMap->SetDirection(m_p_clT2WImage->GetDirection());
  p_clScoreMap->SetSpacing(clScoreMapSpacing);
  p_clScoreMap->SetRegions(clScoreMapSize);

  try {
    p_clScoreMap->Allocate();
    p_clScoreMap->FillBuffer(RealType(0));
  }
  catch (itk::ExceptionObject &) {
    return false;
  }

  ArrayType &clT2WData = p_clT2WEdge->GetData();
  ArrayType &clADCData = p_clADCEdge->GetData();
  ArrayType &clBValueData = p_clBValueEdge->GetData();
  ArrayType &clScoresData = p_clScoresEdge->GetData();

  const int iT2WSize = clT2WData.GetSize().Count(2);
  const int iADCSize = clADCData.GetSize().Count(2);
  const int iBValueSize = clBValueData.GetSize().Count(2);

  //const int iNumClasses = clScoresData.GetSize()[1];
  const int iScoreSize = clScoresData.GetSize().Count(2);

  for (itk::SizeValueType z = 0; z < clT2WSize[2]; ++z) {
    RealType * const p_t2wBuffer = clT2WData.data_no_sync();
    RealType * const p_adcBuffer = clADCData.data_no_sync();
    RealType * const p_bValueBuffer = clBValueData.data_no_sync();

    std::copy_n(m_p_clT2WImage->GetBufferPointer() + z*iT2WSize, iT2WSize, p_t2wBuffer);
    std::copy_n(m_p_clADCImage->GetBufferPointer() + z*iADCSize, iADCSize, p_adcBuffer);
    std::copy_n(m_p_clBValueImage->GetBufferPointer() + z*iBValueSize, iBValueSize, p_bValueBuffer);

    std::cout << "Info: z = " << z << std::endl;

    m_p_clGraph->Forward();

    const RealType * const p_scoresBuffer = clScoresData.data();

    // Copy class 1 values over
    std::copy_n(p_scoresBuffer + 1*iScoreSize, iScoreSize, p_clScoreMap->GetBufferPointer() + z*iScoreSize);
  }

  std::cout << "Info: Done." << std::endl;

  m_p_clScoreMap = ResampleBySize<RealType, 3>(p_clScoreMap, clT2WSize);

  return m_p_clScoreMap.IsNotNull();
}

// Load and resample image to fit vertex's input size
template<typename RealType>
auto ProstateXTool<RealType>::LoadImg(const std::string &strPath, const std::string &strVertex) -> typename ImageType::Pointer {
  if (!m_p_clGraph)
    return nullptr;

  typedef typename GraphType::VertexType VertexType;
  typedef typename GraphType::EdgeType EdgeType;

  std::shared_ptr<VertexType> p_clVertex = m_p_clGraph->FindVertex(strVertex);

  if (!p_clVertex)
    return nullptr;

  std::shared_ptr<EdgeType> p_clEdge = p_clVertex->GetOutput("outData");

  if (!p_clEdge) {
    std::cerr << "Error: Could not get edge '" << strVertex << ".outData'." << std::endl;
    return nullptr;
  }

  if (!p_clEdge->GetData().Valid()) {
    std::cerr << "Error: Graph not initialized?" << std::endl;
    return nullptr;
  }

  const bleak::Size clVertexSize = p_clEdge->GetData().GetSize();

  // BatchSize x Channels x Height x Width
  if (clVertexSize.GetDimension() != 4 || clVertexSize[0] != 1 || clVertexSize[1] != 1) {
    std::cerr << "Error: Unexpected input size for '" << strVertex << ".outData': " << clVertexSize << std::endl;
    return nullptr;
  }

  itk::Size<3> clResampleSize; // Z to be determined after we load image

  clResampleSize[0] = clVertexSize[3]; // Width
  clResampleSize[1] = clVertexSize[2]; // Height

  typename ImageType::Pointer p_clImage;

  if (bleak::IsFolder(strPath)) // DICOM
    p_clImage = bleak::LoadDicomImage<RealType, 3>(strPath);
  else
    p_clImage = bleak::LoadImg<RealType, 3>(strPath);

  if (!p_clImage) {
    std::cerr << "Error: Failed to load image '" << strPath << "'." << std::endl;
    return nullptr;
  }

  const itk::Size<3> clSize = p_clImage->GetBufferedRegion().GetSize();

  clResampleSize[2] = clSize[2]; // Z axis remains the same

  return ResampleBySize<RealType, 3>(p_clImage, clResampleSize);
}