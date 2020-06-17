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

#include <iostream>
#include "CudnnCommon.h"

namespace bleak {

namespace {

#ifdef _OPENMP
bool g_bInitialized = false;
cudnnHandle_t g_handle = cudnnHandle_t();
#pragma omp threadprivate(g_handle, g_bInitialized)
#else // !_OPENMP
thread_local bool g_bInitialized = false;
thread_local cudnnHandle_t g_handle = cudnnHandle_t();
#endif // _OPENMP

} // end anonymous namespace

bool InitializeCudnn() {
  if (g_bInitialized)
    return true;

  if (cudnnCreate(&g_handle) != CUDNN_STATUS_SUCCESS) {
    std::cerr << "Error: Failed to initialize cudnn." << std::endl;
    return false;
  }

  g_bInitialized = true;

  return true;
}

cudnnHandle_t GetCudnnHandle() { 
  return g_handle; 
}

CudnnWorkspace & CudnnWorkspace::GetInstance() {
#ifdef _OPENMP
// I guess threadprivate doesn't work for C++ objects
  static CudnnWorkspace *p_clWorkspace = nullptr;
#pragma omp threadprivate(p_clWorkspace)

  if (p_clWorkspace == nullptr)
    p_clWorkspace = new CudnnWorkspace();

  return *p_clWorkspace;
#else // !_OPENMP
  thread_local CudnnWorkspace clWorkspace;
  return clWorkspace;
#endif // _OPENMP
}

bool CudnnWorkspace::Allocate() {
  if (m_p_workspace != nullptr && m_capacity == m_workspaceSize) // Nothing to do...
    return true;

  cudaFree(m_p_workspace);
  m_p_workspace = nullptr;

  if (m_workspaceSize == 0) {
    m_capacity = 0; // Probably not needed...
    return true;
  }

  if (cudaMalloc(&m_p_workspace, m_workspaceSize) != cudaSuccess)
    return false;

  m_capacity = m_workspaceSize;
  return true;
}

} // end namespace bleak
