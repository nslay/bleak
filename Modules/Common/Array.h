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

#include <algorithm>
#include <memory>
#include "Size.h"

namespace bleak {

template<typename RealType>
class Array {
public:
  typedef RealType * iterator;
  typedef const RealType * const_iterator;

  Array() { }

  RealType * data() {
    return m_p_buffer.get();
  }

  const RealType * data() const {
    return m_p_buffer.get();
  }

  iterator begin() {
    return m_p_buffer.get();
  }

  const_iterator begin() const {
    return m_p_buffer.get();
  }

  iterator end() {
    if (!GetSize().Valid()) // Product() returns 1 even when empty
      return m_p_buffer.get();

    return m_p_buffer.get() + GetSize().Product();
  }

  const_iterator end() const {
    if(!GetSize().Valid()) // Product() returns 1 even when empty
      return m_p_buffer.get();

    return m_p_buffer.get() + GetSize().Product();
  }

  // XXX: This is very hackish
  void SetData(RealType *p_buffer) {
    if (p_buffer == nullptr)
      m_p_buffer.reset();
    else if (p_buffer != m_p_buffer.get())
      m_p_buffer.reset(p_buffer, [](RealType *) { });
  }

  void SetData(const std::shared_ptr<RealType> &p_buffer) {
    m_p_buffer = p_buffer;
  }

  void CopyFrom(const RealType *p_buffer) {
    if (Valid())
      std::copy(p_buffer, p_buffer + GetSize().Product(), begin());
  }

  void CopyTo(RealType *p_buffer) const {
    if (Valid())
      std::copy(begin(), end(), p_buffer);
  }

  void CopyTo(Array &clOther) const {
    if (this == &clOther || !Valid())
      return;

    if (clOther.data() == nullptr || clOther.GetSize() != GetSize()) {
      clOther.SetSize(GetSize());
      clOther.Allocate();
    }
    
    CopyTo(clOther.data());
  }

  // XXX: This is very hackish
  void ShareWith(Array &clOther) {
    clOther.m_p_buffer = m_p_buffer;
  }

  void SetSize(const Size &clSize) {
    m_clSize = clSize;
  }

  const Size & GetSize() const {
    return m_clSize;
  }

  bool Valid() const {
    return m_p_buffer != nullptr && GetSize().Valid();
  }

  bool Allocate() {
    if (GetSize().Valid()) {
      m_p_buffer.reset(new RealType[GetSize().Product()], [](RealType *p_mem) { delete [] p_mem; });
      return m_p_buffer != nullptr;
    }

    return false;
  }

  void Free() {
    m_p_buffer.reset();
  }

  void Fill(const RealType &value) {
    if (Valid())
      std::fill(begin(), end(), value);
  }

  void Clear() {
    m_clSize.Clear();
    Free();
  }

  void Swap(Array<RealType> &clOther) {
    m_p_buffer.swap(clOther.m_p_buffer);
    m_clSize.Swap(clOther.m_clSize);
  }

private:
  Size m_clSize;
  std::shared_ptr<RealType> m_p_buffer;

  Array(const Array &) = delete;
  Array & operator=(const Array &) = delete;
};

} // end namespace bleak

#endif // !BLEAK_ARRAY_H
