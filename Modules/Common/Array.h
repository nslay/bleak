/*-
 * Copyright (c) 2017 Nathan Lay (enslay@gmail.com)
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

#ifndef BLEAK_ARRAY_H
#define BLEAK_ARRAY_H

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <memory>
#include "Size.h"

#ifdef BLEAK_USE_CUDA
#include "cuda_runtime.h"
#endif // BLEAK_USE_CUDA

namespace bleak {

enum MemoryLocation { CPU, GPU };

template<typename RealType>
class Array {
public:
  typedef RealType * iterator;
  typedef const RealType * const_iterator;

  // Allows for the full context to be shareable (e.g. in a Reshape, p_gpuBuffer may be allocated which will be reflected in input/output no matter where it happens).
  struct MemoryTuple {
    MemoryLocation eMemoryLocation = CPU;
    std::shared_ptr<RealType> p_cpuBuffer;
    std::shared_ptr<RealType> p_gpuBuffer;
  };

  Array() { 
    m_p_stMem = std::make_shared<MemoryTuple>();
  }

  MemoryLocation GetLocation() const { return m_p_stMem->eMemoryLocation; } // You shouldn't need this!

  // Horrible name, but necessary since you can really screw yourself up with this... You must be deliberate with this.
  RealType * data_no_sync(MemoryLocation eMemoryLocation = CPU) {
    switch (eMemoryLocation) {
    case CPU:
      m_p_stMem->eMemoryLocation = CPU;
      return m_p_stMem->p_cpuBuffer.get();
    case GPU:
      AllocateGPU();
      if (m_p_stMem->p_gpuBuffer != nullptr)
        m_p_stMem->eMemoryLocation = GPU;
      return m_p_stMem->p_gpuBuffer.get();
    }
    return nullptr; // Not reached
  }

  RealType * data(MemoryLocation eMemoryLocation = CPU) { 
    CopyTo(eMemoryLocation);
    switch (eMemoryLocation) {
    case CPU:
      return m_p_stMem->p_cpuBuffer.get();
    case GPU:
      return m_p_stMem->p_gpuBuffer.get();
    }
    return nullptr; // Not reached
  }

  const RealType * data(MemoryLocation eMemoryLocation = CPU) const {
    CopyTo(eMemoryLocation);
    switch (eMemoryLocation) {
    case CPU:
      return m_p_stMem->p_cpuBuffer.get();
    case GPU:
      return m_p_stMem->p_gpuBuffer.get();
    }
    return nullptr; // Not reached
  }

  // CPU memory!
  iterator begin() { return data(); }
  const_iterator begin() const { return data(); }

  // CPU memory!
  iterator end() { return data() + GetSize().Count(); }
  const_iterator end() const { return data() + GetSize().Count(); }

  // CPU memory!
  // XXX: This is very hackish
  void SetData(RealType *p_buffer) {
    if (p_buffer == nullptr) {
      Free();
    }
    else if (p_buffer != m_p_stMem->p_cpuBuffer.get()) {
      m_p_stMem->p_cpuBuffer.reset(p_buffer, [](RealType *) { });
      m_p_stMem->p_gpuBuffer.reset();
      m_p_stMem->eMemoryLocation = CPU;
    }
  }

  // CPU memory!
  void SetData(const std::shared_ptr<RealType> &p_buffer) {
    m_p_stMem->p_cpuBuffer = p_buffer;
    m_p_stMem->p_gpuBuffer.reset();
    m_p_stMem->eMemoryLocation = CPU;
  }

  void CopyFrom(const RealType *p_buffer, MemoryLocation eMemoryLocation = CPU) {
#ifdef BLEAK_USE_CUDA
    switch (eMemoryLocation) {
    case CPU:
      switch (GetLocation()) {
      case CPU:
        std::copy(p_buffer, p_buffer + GetSize().Count(), begin());
        break;
      case GPU:
        if (cudaMemcpy(m_p_stMem->p_gpuBuffer.get(), p_buffer, sizeof(RealType)*GetSize().Count(), cudaMemcpyHostToDevice) != cudaSuccess) {
          std::cerr << "Error: Failed to copy host memory to GPU memory." << std::endl;
          throw std::runtime_error("Error: Failed to copy host memory to GPU memory.");
        }
        break;
      }
      break;
    case GPU:
      switch (GetLocation()) {
      case CPU:
        if (cudaMemcpy(m_p_stMem->p_cpuBuffer.get(), p_buffer, sizeof(RealType)*GetSize().Count(), cudaMemcpyDeviceToHost) != cudaSuccess) {
          std::cerr << "Error: Failed to copy GPU memory to host memory." << std::endl;
          throw std::runtime_error("Error: Failed to copy GPU memory to host memory.");
        }
        break;
      case GPU:
        if (cudaMemcpy(m_p_stMem->p_gpuBuffer.get(), p_buffer, sizeof(RealType)*GetSize().Count(), cudaMemcpyDeviceToDevice) != cudaSuccess) {
          std::cerr << "Error: Failed to copy GPU memory to GPU memory." << std::endl;
          throw std::runtime_error("Error: Failed to copy GPU memory to GPU memory.");
        }
        break;
      }
      break;
    }
#else // !BLEAK_USE_CUDA
    std::copy(p_buffer, p_buffer + GetSize().Count(), begin());
#endif // BLEAK_USE_CUDA
  }

  void CopyTo(RealType *p_buffer, MemoryLocation eMemoryLocation = CPU) const {
#ifdef BLEAK_USE_CUDA
    switch (eMemoryLocation) {
    case CPU:
      switch (GetLocation()) {
      case CPU:
        std::copy(begin(), end(), p_buffer);
        break;
      case GPU:
        if (cudaMemcpy(p_buffer, m_p_stMem->p_gpuBuffer.get(), sizeof(RealType)*GetSize().Count(), cudaMemcpyDeviceToHost) != cudaSuccess) {
          std::cerr << "Error: Failed to copy GPU memory to host memory." << std::endl;
          throw std::runtime_error("Error: Failed to copy GPU memory to host memory.");
        }
        break;
      }
      break;
    case GPU:
      switch (GetLocation()) {
      case CPU:
        if (cudaMemcpy(p_buffer, m_p_stMem->p_cpuBuffer.get(), sizeof(RealType)*GetSize().Count(), cudaMemcpyHostToDevice) != cudaSuccess) {
          std::cerr << "Error: Failed to copy host memory to GPU memory." << std::endl;
          throw std::runtime_error("Error: Failed to copy host memory to GPU memory.");
        }
        break;
      case GPU:
        if (cudaMemcpy(p_buffer, m_p_stMem->p_gpuBuffer.get(), sizeof(RealType)*GetSize().Count(), cudaMemcpyDeviceToDevice) != cudaSuccess) {
          std::cerr << "Error: Failed to copy GPU memory to GPU memory." << std::endl;
          throw std::runtime_error("Error: Failed to copy GPU memory to GPU memory.");
        }
        break;
      }
      break;
    }
#else // !BLEAK_USE_CUDA
    std::copy(begin(), end(), p_buffer);
#endif // BLEAK_USE_CUDA
  }

  void CopyTo(Array &clOther) const {
    if (this == &clOther || !Valid())
      return;

    if (!clOther.Valid() || clOther.GetSize() != GetSize()) {
      clOther.SetSize(GetSize());
      clOther.Allocate();
    }
    
    CopyTo(clOther.data(clOther.GetLocation()), clOther.GetLocation());
  }

  // XXX: This is very hackish
  void ShareWith(Array &clOther) {
    clOther.m_p_stMem = m_p_stMem;
  }

  void SetSize(const Size &clSize) {
    m_clSize = clSize;
  }

  const Size & GetSize() const {
    return m_clSize;
  }

  bool Valid() const {
    return m_p_stMem->p_cpuBuffer != nullptr && GetSize().Valid();
  }

  bool Allocate() {
    if (GetSize().Valid()) {
      m_p_stMem->p_cpuBuffer.reset(new RealType[GetSize().Count()], [](RealType *p_mem) { delete [] p_mem; });
      m_p_stMem->p_gpuBuffer.reset();
      m_p_stMem->eMemoryLocation = CPU;
      return m_p_stMem->p_cpuBuffer != nullptr;
    }

    return false;
  }

  void Free() {
    m_p_stMem->p_cpuBuffer.reset();
    m_p_stMem->p_gpuBuffer.reset();
    m_p_stMem->eMemoryLocation = CPU;
  }

  void Fill(const RealType &value) {
    const MemoryLocation eLocationBefore = m_p_stMem->eMemoryLocation;
    m_p_stMem->eMemoryLocation = CPU;
    std::fill(begin(), end(), value);
    CopyTo(eLocationBefore);
  }

  void Clear() {
    m_clSize.Clear();
    Free();
  }

  void Swap(Array<RealType> &clOther) {
    m_p_stMem.swap(clOther.m_p_stMem);
    m_clSize.Swap(clOther.m_clSize);
  }

private:
  Size m_clSize;
  std::shared_ptr<MemoryTuple> m_p_stMem;

  // cudaMallocManaged looks attractive... but not all vertices will need to keep data on the GPU. So let's not use cudaMallocManaged. We'll allocate memory in a lazy fashion.
  // This design is inspired by Caffe.
  void CopyTo(MemoryLocation eMemoryLocation) const {
#ifdef BLEAK_USE_CUDA
    if (m_p_stMem->eMemoryLocation == eMemoryLocation || !Valid())
      return;

    switch (eMemoryLocation) {
    case CPU:
      if (cudaMemcpy(m_p_stMem->p_cpuBuffer.get(), m_p_stMem->p_gpuBuffer.get(), sizeof(RealType)*GetSize().Count(), cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::cerr << "Error: Failed to copy GPU memory to host memory." << std::endl;
        throw std::runtime_error("Error: Failed to copy GPU memory to host memory.");
      }

      m_p_stMem->eMemoryLocation = CPU;
      break;
    case GPU:
      AllocateGPU();

      if (cudaMemcpy(m_p_stMem->p_gpuBuffer.get(), m_p_stMem->p_cpuBuffer.get(), sizeof(RealType)*GetSize().Count(), cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cerr << "Error: Failed to copy host memory to GPU." << std::endl;
        throw std::runtime_error("Error: Failed to copy host memory to GPU.");
      }

      m_p_stMem->eMemoryLocation = GPU;
      break;
    }
#endif // BLEAK_USE_CUDA
  }

  void AllocateGPU() const {
#ifdef BLEAK_USE_CUDA
    if (m_p_stMem->p_gpuBuffer != nullptr || !Valid())
      return;

    RealType *p_buffer = nullptr;
    if (cudaMalloc(&p_buffer, sizeof(RealType)*GetSize().Count()) != cudaSuccess) {
      std::cerr << "Error: cudaMalloc failed. Out of memory?" << std::endl;
      throw std::runtime_error("Error: cudaMalloc failed. Out of memory?");
    }

    m_p_stMem->p_gpuBuffer.reset(p_buffer, &cudaFree);
#endif // BLEAK_USE_CUDA
  }

  Array(const Array &) = delete;
  Array & operator=(const Array &) = delete;
};

} // end namespace bleak

#endif // !BLEAK_ARRAY_H
