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

#ifndef BLEAK_CUDNNCOMMON_H
#define BLEAK_CUDNNCOMMON_H

#include <cstdint>
#include <algorithm>
#include <memory>
#include <vector>
#include "Size.h"
#include "cuda_runtime.h"
#include "cudnn.h"

namespace bleak {

bool InitializeCudnn();
cudnnHandle_t GetCudnnHandle();

// Convenience classes below

template<typename RealType>
struct CudnnDataType { };

template<>
struct CudnnDataType<float> {
  static constexpr cudnnDataType_t Value() { return CUDNN_DATA_FLOAT; }
};

template<>
struct CudnnDataType<double> {
  static constexpr cudnnDataType_t Value() { return CUDNN_DATA_DOUBLE; }
};

class CudnnWorkspace {
public:
  ~CudnnWorkspace() { Reset(); }

  static CudnnWorkspace & GetInstance();

  // For Initialize() pass
  void SetSize(size_t workspaceSize) { m_workspaceSize = std::max(m_workspaceSize, workspaceSize); }

  // For lazy allocation when computing... Only does anything if m_workspaceSize changes...
  bool Allocate();

  void Reset() {
    cudaFree(m_p_workspace);
    m_p_workspace = nullptr;
    m_capacity = m_workspaceSize = 0;
  }

  uint8_t * GetWorkspace() { 
    Allocate(); // Lazy allocate!
    return m_p_workspace; 
  }

  size_t GetWorkspaceSize() const { return m_workspaceSize; } // Technically, this should return m_capacity... but it may not be set when this is called! Lazy allocate will adjust m_capacity anyway.

private:
  uint8_t *m_p_workspace = nullptr;
  size_t m_workspaceSize = 0;
  size_t m_capacity = 0; // Current m_p_workspace size

  CudnnWorkspace() = default;
  CudnnWorkspace(const CudnnWorkspace &) = delete;
  CudnnWorkspace(CudnnWorkspace &&) = delete;

  CudnnWorkspace & operator=(const CudnnWorkspace &) = delete;
  CudnnWorkspace & operator=(CudnnWorkspace &&) = delete;
};

// RAII descriptors...
template<typename RealType>
class CudnnTensorDescriptor {
public:
  CudnnTensorDescriptor() = default;
  ~CudnnTensorDescriptor() { Destroy(); }

  bool SetFullyPacked(const Size &clSize) {
    Destroy();

    if (cudnnCreateTensorDescriptor(&m_desc) != CUDNN_STATUS_SUCCESS)
      return false;

    std::vector<int> vStride(clSize.GetDimension());

    vStride.back() = 1;

    for (int d = clSize.GetDimension()-1; d > 0; --d)
      vStride[d-1] = clSize[d]*vStride[d];

    if (cudnnSetTensorNdDescriptor(m_desc, CudnnDataType<RealType>::Value(), clSize.GetDimension(), clSize.data(), vStride.data()) != CUDNN_STATUS_SUCCESS) {
      Destroy();  
      return false;
    }

    return true;
  }

  operator const cudnnTensorDescriptor_t () const { return m_desc; }

  void Destroy() {
    cudnnDestroyTensorDescriptor(m_desc);
    m_desc = cudnnTensorDescriptor_t();
  }

private:
  cudnnTensorDescriptor_t m_desc = cudnnTensorDescriptor_t();

  CudnnTensorDescriptor(const CudnnTensorDescriptor &) = delete;
  CudnnTensorDescriptor & operator=(const CudnnTensorDescriptor &) = delete;
};

template<typename RealType>
class CudnnFilterDescriptor {
public:
  CudnnFilterDescriptor() = default;
  ~CudnnFilterDescriptor() { Destroy(); }

  bool Set(const Size &clSize) {
    Destroy();

    if (cudnnCreateFilterDescriptor(&m_desc) != CUDNN_STATUS_SUCCESS)
      return false;

    if (cudnnSetFilterNdDescriptor(m_desc, CudnnDataType<RealType>::Value(), CUDNN_TENSOR_NCHW, clSize.GetDimension(), clSize.data()) != CUDNN_STATUS_SUCCESS) {
      Destroy();
      return false;
    }

    return true;
  }

  void Destroy() {
    cudnnDestroyFilterDescriptor(m_desc);
    m_desc = cudnnFilterDescriptor_t();
  }

  operator const cudnnFilterDescriptor_t () const { return m_desc; }

private:
  cudnnFilterDescriptor_t m_desc = cudnnFilterDescriptor_t();

  CudnnFilterDescriptor(const CudnnFilterDescriptor &) = delete;
  CudnnFilterDescriptor & operator=(const CudnnFilterDescriptor &) = delete;
};

template<typename RealType>
class CudnnConvolutionDescriptor {
public:
  CudnnConvolutionDescriptor() = default;
  ~CudnnConvolutionDescriptor() { Destroy(); }

  bool Set(int iDimension, const int a_iPad[], const int a_iStride[], const int a_iDilate[]) {
    Destroy();

    if (cudnnCreateConvolutionDescriptor(&m_desc) != CUDNN_STATUS_SUCCESS)
      return false;

    if (cudnnSetConvolutionNdDescriptor(m_desc, iDimension, a_iPad, a_iStride, a_iDilate, CUDNN_CONVOLUTION, CudnnDataType<RealType>::Value()) != CUDNN_STATUS_SUCCESS) {
      Destroy();
      return false;
    }

    return true;
  }

  bool GetOutputSize(const cudnnTensorDescriptor_t inputDesc, const cudnnFilterDescriptor_t filterDesc, Size &clSize) const {
    return cudnnGetConvolutionNdForwardOutputDim(m_desc, inputDesc, filterDesc, clSize.GetDimension(), clSize.data()) == CUDNN_STATUS_SUCCESS;
  }

  void Destroy() {
    cudnnDestroyConvolutionDescriptor(m_desc);
    m_desc = cudnnConvolutionDescriptor_t();
  }

  operator const cudnnConvolutionDescriptor_t () const { return m_desc; }

private:
  cudnnConvolutionDescriptor_t m_desc = cudnnConvolutionDescriptor_t();

  CudnnConvolutionDescriptor(const CudnnConvolutionDescriptor &) = delete;
  CudnnConvolutionDescriptor & operator=(const CudnnConvolutionDescriptor &) = delete;
};

class CudnnActivationDescriptor {
public:
  CudnnActivationDescriptor() = default;
  ~CudnnActivationDescriptor() { Destroy(); }

  bool Set(cudnnActivationMode_t mode, double coef = 0.0) {
    Destroy();

    if (cudnnCreateActivationDescriptor(&m_desc) != CUDNN_STATUS_SUCCESS)
      return false;

    if (cudnnSetActivationDescriptor(m_desc, mode, CUDNN_NOT_PROPAGATE_NAN, coef) != CUDNN_STATUS_SUCCESS) {
      Destroy();
      return false;
    }

    return true;
  }

  void Destroy() {
    cudnnDestroyActivationDescriptor(m_desc);
    m_desc = cudnnActivationDescriptor_t();
  }

  operator const cudnnActivationDescriptor_t () const { return m_desc; }

private:
  cudnnActivationDescriptor_t m_desc = cudnnActivationDescriptor_t();

  CudnnActivationDescriptor(const CudnnActivationDescriptor &) = delete;
  CudnnActivationDescriptor & operator=(const CudnnActivationDescriptor &) = delete;
};

class CudnnConvolutionFwdAlgorithm {
public:
  bool Set(const cudnnTensorDescriptor_t inDataDesc, const cudnnFilterDescriptor_t filterDesc, const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t outDataDesc, cudnnConvolutionFwdAlgo_t algo) {
    size_t size = 0;
    if (cudnnGetConvolutionForwardWorkspaceSize(GetCudnnHandle(), inDataDesc, filterDesc, convDesc, outDataDesc, algo, &size) != CUDNN_STATUS_SUCCESS)
      return false;

    m_algo = algo;

    CudnnWorkspace &clWorkspace = CudnnWorkspace::GetInstance();
    clWorkspace.SetSize(size);

    return true;
  }

  bool Set(const cudnnTensorDescriptor_t inDataDesc, const cudnnFilterDescriptor_t filterDesc, const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t outDataDesc) {
    int iReturnedCount = 0;
    cudnnConvolutionFwdAlgoPerf_t algoPerf = {};

    if (cudnnGetConvolutionForwardAlgorithm_v7(GetCudnnHandle(), inDataDesc, filterDesc, convDesc, outDataDesc, 1, &iReturnedCount, &algoPerf) != CUDNN_STATUS_SUCCESS || iReturnedCount <= 0)
      return false;

    return Set(inDataDesc, filterDesc, convDesc, outDataDesc, algoPerf.algo);
  }

  operator cudnnConvolutionFwdAlgo_t () const { return m_algo; } // For consistency

private:
  cudnnConvolutionFwdAlgo_t m_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
};

class CudnnConvolutionBwdDataAlgorithm {
public:
  bool Set(const cudnnFilterDescriptor_t filterDesc, const cudnnTensorDescriptor_t outDataGradDesc, const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t inDataGradDesc, cudnnConvolutionBwdDataAlgo_t algo) {
    size_t size = 0;
    if (cudnnGetConvolutionBackwardDataWorkspaceSize(GetCudnnHandle(), filterDesc, outDataGradDesc, convDesc, inDataGradDesc, algo, &size) != CUDNN_STATUS_SUCCESS)
      return false;

    m_algo = algo;

    CudnnWorkspace &clWorkspace = CudnnWorkspace::GetInstance();
    clWorkspace.SetSize(size);

    return true;
  }

  bool Set(const cudnnFilterDescriptor_t filterDesc, const cudnnTensorDescriptor_t outDataGradDesc, const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t inDataGradDesc) {
    int iReturnedCount = 0;
    cudnnConvolutionBwdDataAlgoPerf_t algoPerf = {};

    if (cudnnGetConvolutionBackwardDataAlgorithm_v7(GetCudnnHandle(), filterDesc, outDataGradDesc, convDesc, inDataGradDesc, 1, &iReturnedCount, &algoPerf) != CUDNN_STATUS_SUCCESS || iReturnedCount <= 0)
      return false;

    return Set(filterDesc, outDataGradDesc, convDesc, inDataGradDesc, algoPerf.algo);
  }

  operator cudnnConvolutionBwdDataAlgo_t () const { return m_algo; } // For consistency

private:
  cudnnConvolutionBwdDataAlgo_t m_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
};

class CudnnConvolutionBwdFilterAlgorithm {
public:
  bool Set(const cudnnTensorDescriptor_t inDataDesc, const cudnnTensorDescriptor_t outDataGradDesc, const cudnnConvolutionDescriptor_t convDesc, const cudnnFilterDescriptor_t filterGradDesc, cudnnConvolutionBwdFilterAlgo_t algo) {
    size_t size = 0;
    if (cudnnGetConvolutionBackwardFilterWorkspaceSize(GetCudnnHandle(), inDataDesc, outDataGradDesc, convDesc, filterGradDesc, algo, &size) != CUDNN_STATUS_SUCCESS)
      return false;

    m_algo = algo;

    CudnnWorkspace &clWorkspace = CudnnWorkspace::GetInstance();
    clWorkspace.SetSize(size);

    return true;
  }

  bool Set(const cudnnTensorDescriptor_t inDataDesc, const cudnnTensorDescriptor_t outDataGradDesc, const cudnnConvolutionDescriptor_t convDesc, const cudnnFilterDescriptor_t filterGradDesc) {
    int iReturnedCount = 0;
    cudnnConvolutionBwdFilterAlgoPerf_t algoPerf = {};

    if (cudnnGetConvolutionBackwardFilterAlgorithm_v7(GetCudnnHandle(), inDataDesc, outDataGradDesc, convDesc, filterGradDesc, 1, &iReturnedCount, &algoPerf) != CUDNN_STATUS_SUCCESS || iReturnedCount <= 0)
      return false;

    return Set(inDataDesc, outDataGradDesc, convDesc, filterGradDesc, algoPerf.algo);
  }

  operator cudnnConvolutionBwdFilterAlgo_t () const { return m_algo; } // For consistency

private:
  cudnnConvolutionBwdFilterAlgo_t m_algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
};

// Convenience (very) functional-like classes! Try to shorten those ridiculously long cudnn calls!
template<typename RealType>
class CudnnConvolutionForward {
public:
  CudnnConvolutionForward(const CudnnTensorDescriptor<RealType> &clInDataDesc, const CudnnFilterDescriptor<RealType> &clFilterDesc, const CudnnConvolutionDescriptor<RealType> &clConvolutionDesc, 
    const CudnnConvolutionFwdAlgorithm &clAlgorithmDesc, const CudnnTensorDescriptor<RealType> &clOutDataDesc)
  : m_clInDataDesc(clInDataDesc), m_clFilterDesc(clFilterDesc), m_clConvolutionDesc(clConvolutionDesc), m_clAlgorithmDesc(clAlgorithmDesc), m_clOutDataDesc(clOutDataDesc) { }

  bool operator()(const RealType &alpha, const RealType *p_inData, const RealType *p_filter, const RealType &beta, RealType *p_outData) const {
    CudnnWorkspace &clWorkspace = CudnnWorkspace::GetInstance();
    return cudnnConvolutionForward(GetCudnnHandle(), &alpha, m_clInDataDesc, p_inData, m_clFilterDesc, p_filter, m_clConvolutionDesc, m_clAlgorithmDesc, 
      clWorkspace.GetWorkspace(), clWorkspace.GetWorkspaceSize(), &beta, m_clOutDataDesc, p_outData) == CUDNN_STATUS_SUCCESS;
  }

private:
  const CudnnTensorDescriptor<RealType> &m_clInDataDesc;
  const CudnnFilterDescriptor<RealType> &m_clFilterDesc;
  const CudnnConvolutionDescriptor<RealType> &m_clConvolutionDesc;
  const CudnnConvolutionFwdAlgorithm &m_clAlgorithmDesc;
  const CudnnTensorDescriptor<RealType> &m_clOutDataDesc;
};

template<typename RealType>
class CudnnConvolutionBiasActivationForward {
public:
  CudnnConvolutionBiasActivationForward(const CudnnTensorDescriptor<RealType> &clInDataDesc, const CudnnFilterDescriptor<RealType> &clFilterDesc, const CudnnConvolutionDescriptor<RealType> &clConvolutionDesc, 
    const CudnnConvolutionFwdAlgorithm &clAlgorithmDesc, const CudnnTensorDescriptor<RealType> &clInterDataDesc, const CudnnTensorDescriptor<RealType> &clBiasDesc, 
    const CudnnActivationDescriptor &clActivationDesc, const CudnnTensorDescriptor<RealType> &clOutDataDesc)
  : m_clInDataDesc(clInDataDesc), m_clFilterDesc(clFilterDesc), m_clConvolutionDesc(clConvolutionDesc), m_clAlgorithmDesc(clAlgorithmDesc), m_clInterDataDesc(clInterDataDesc), 
    m_clBiasDesc(clBiasDesc), m_clActivationDesc(clActivationDesc), m_clOutDataDesc(clOutDataDesc) { }

  bool operator()(const RealType &alpha, const RealType *p_inData, const RealType *p_filter, const RealType &beta, const RealType *p_interData, const RealType *p_bias, RealType *p_outData) const {
    CudnnWorkspace &clWorkspace = CudnnWorkspace::GetInstance();
    return cudnnConvolutionBiasActivationForward(GetCudnnHandle(), &alpha, m_clInDataDesc, p_inData, m_clFilterDesc, p_filter, m_clConvolutionDesc, m_clAlgorithmDesc, 
      clWorkspace.GetWorkspace(), clWorkspace.GetWorkspaceSize(), &beta, m_clInterDataDesc, p_interData, m_clBiasDesc, p_bias, m_clActivationDesc, m_clOutDataDesc, p_outData) == CUDNN_STATUS_SUCCESS;
  }

private:
  const CudnnTensorDescriptor<RealType> &m_clInDataDesc;
  const CudnnFilterDescriptor<RealType> &m_clFilterDesc;
  const CudnnConvolutionDescriptor<RealType> &m_clConvolutionDesc;
  const CudnnConvolutionFwdAlgorithm &m_clAlgorithmDesc;
  const CudnnTensorDescriptor<RealType> &m_clInterDataDesc;
  const CudnnTensorDescriptor<RealType> &m_clBiasDesc;
  const CudnnActivationDescriptor &m_clActivationDesc;
  const CudnnTensorDescriptor<RealType> &m_clOutDataDesc;
};

template<typename RealType>
class CudnnConvolutionBackwardData {
public:
  CudnnConvolutionBackwardData(const CudnnFilterDescriptor<RealType> &clFilterDesc, const CudnnTensorDescriptor<RealType> &clOutDataGradDesc, const CudnnConvolutionDescriptor<RealType> &clConvolutionDesc,
    const CudnnConvolutionBwdDataAlgorithm &clAlgorithmDesc, const CudnnTensorDescriptor<RealType> &clInDataGradDesc)
  : m_clFilterDesc(clFilterDesc), m_clOutDataGradDesc(clOutDataGradDesc), m_clConvolutionDesc(clConvolutionDesc), m_clAlgorithmDesc(clAlgorithmDesc), m_clInDataGradDesc(clInDataGradDesc) { }

  bool operator()(const RealType &alpha, const RealType *p_filter, const RealType *p_outDataGrad, const RealType &beta, RealType *p_inDataGrad) const {
    CudnnWorkspace &clWorkspace = CudnnWorkspace::GetInstance();
    return cudnnConvolutionBackwardData(GetCudnnHandle(), &alpha, m_clFilterDesc, p_filter, m_clOutDataGradDesc, p_outDataGrad, m_clConvolutionDesc, m_clAlgorithmDesc,
      clWorkspace.GetWorkspace(), clWorkspace.GetWorkspaceSize(), &beta, m_clInDataGradDesc, p_inDataGrad) == CUDNN_STATUS_SUCCESS;
  }

private:
  const CudnnFilterDescriptor<RealType> &m_clFilterDesc;
  const CudnnTensorDescriptor<RealType> &m_clOutDataGradDesc;
  const CudnnConvolutionDescriptor<RealType> &m_clConvolutionDesc;
  const CudnnConvolutionBwdDataAlgorithm &m_clAlgorithmDesc;
  const CudnnTensorDescriptor<RealType> &m_clInDataGradDesc;
};

template<typename RealType>
class CudnnConvolutionBackwardFilter {
public:
  CudnnConvolutionBackwardFilter(const CudnnTensorDescriptor<RealType> &clInDataDesc, const CudnnTensorDescriptor<RealType> &clOutDataGradDesc, const CudnnConvolutionDescriptor<RealType> &clConvolutionDesc,
    const CudnnConvolutionBwdFilterAlgorithm &clAlgorithmDesc, const CudnnFilterDescriptor<RealType> &clFilterGradDesc)
  : m_clInDataDesc(clInDataDesc), m_clOutDataGradDesc(clOutDataGradDesc), m_clConvolutionDesc(clConvolutionDesc), m_clAlgorithmDesc(clAlgorithmDesc), m_clFilterGradDesc(clFilterGradDesc) { }

  bool operator()(const RealType &alpha, const RealType *p_inData, const RealType *p_outDataGrad, const RealType &beta, RealType *p_filterGrad) const {
    CudnnWorkspace &clWorkspace = CudnnWorkspace::GetInstance();
    return cudnnConvolutionBackwardFilter(GetCudnnHandle(), &alpha, m_clInDataDesc, p_inData, m_clOutDataGradDesc, p_outDataGrad, m_clConvolutionDesc, m_clAlgorithmDesc,
      clWorkspace.GetWorkspace(), clWorkspace.GetWorkspaceSize(), &beta, m_clFilterGradDesc, p_filterGrad) == CUDNN_STATUS_SUCCESS;
  }

private:
  const CudnnTensorDescriptor<RealType> &m_clInDataDesc;
  const CudnnTensorDescriptor<RealType> &m_clOutDataGradDesc;
  const CudnnConvolutionDescriptor<RealType> &m_clConvolutionDesc;
  const CudnnConvolutionBwdFilterAlgorithm &m_clAlgorithmDesc;
  const CudnnFilterDescriptor<RealType> &m_clFilterGradDesc;
};

template<typename RealType>
class CudnnConvolutionBackwardBias {
public:
  CudnnConvolutionBackwardBias(const CudnnTensorDescriptor<RealType> &clOutDataGradDesc, const CudnnTensorDescriptor<RealType> &clBiasGradDesc)
  : m_clOutDataGradDesc(clOutDataGradDesc), m_clBiasGradDesc(clBiasGradDesc) { }

  bool operator()(const RealType &alpha, const RealType *p_outDataGrad, const RealType &beta, RealType *p_biasGrad) const {
    return cudnnConvolutionBackwardBias(GetCudnnHandle(), &alpha, m_clOutDataGradDesc, p_outDataGrad, &beta, m_clBiasGradDesc, p_biasGrad) == CUDNN_STATUS_SUCCESS;
  }

private:
  const CudnnTensorDescriptor<RealType> &m_clOutDataGradDesc;
  const CudnnTensorDescriptor<RealType> &m_clBiasGradDesc;
};

} // end namespace bleak

#endif // !BLEAK_CUDNNCOMMON_H
