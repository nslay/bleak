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

#ifndef BLEAK_SIZE_H
#define BLEAK_SIZE_H

#include <iostream>
#include <algorithm>
#include <limits>
#include <functional>
#include <initializer_list>
#include <vector>

namespace bleak {

// XXX: Use int instead of size_t since BLAS routines use int... Lame
class Size {
public:
  typedef std::vector<int>::iterator iterator;
  typedef std::vector<int>::const_iterator const_iterator;

  Size() { }

  explicit Size(int iDim, int iValue = 1) 
  : m_vSize(iDim, iValue) { }

  explicit Size(const std::vector<int> &vSize) 
  : m_vSize(vSize) { }

  explicit Size(std::vector<int> &&vSize)
    : m_vSize(vSize) { }

  Size(std::initializer_list<int> ilSize) 
  : m_vSize(ilSize) { }

  iterator begin() {
    return m_vSize.begin();
  }

  const_iterator begin() const {
    return m_vSize.begin();
  }

  iterator end() {
    return m_vSize.end();
  }

  const_iterator end() const {
    return m_vSize.end();
  }

  int * data() {
    return m_vSize.data();
  }

  const int * data() const {
    return m_vSize.data();
  }

  void SetDimension(int iDim, int iValue = 1) { 
    m_vSize.resize(iDim, iValue); 
  }

  void Clear() {
    m_vSize.clear();
  }

  void Fill(int iValue) {
    std::fill(m_vSize.begin(), m_vSize.end(), iValue);
  }

  int GetDimension() const { 
    return (int)m_vSize.size(); 
  }

  // Count returns 0 when size range is empty
  int Count(int iAxisBegin = 0, int iAxisEnd = std::numeric_limits<int>::max()) const {
    iAxisEnd = std::min(iAxisEnd, (int)m_vSize.size());
    iAxisBegin = std::max(0, iAxisBegin);

    int iProduct = (iAxisBegin < iAxisEnd);

    for (int i = iAxisBegin; i < iAxisEnd; ++i)
      iProduct *= m_vSize[i];

    return iProduct;
  }

  // Product returns 1 when size range is empty
  int Product(int iAxisBegin = 0, int iAxisEnd = std::numeric_limits<int>::max()) const {
    iAxisEnd = std::min(iAxisEnd, (int)m_vSize.size());
    iAxisBegin = std::max(0, iAxisBegin);

    int iProduct = 1; // Not intuitive, but follows from mathematical convention (for begin == end)

    for (int i = iAxisBegin; i < iAxisEnd; ++i)
      iProduct *= m_vSize[i];

    return iProduct;
  }

  bool Empty() const {
    return m_vSize.empty() || std::find(m_vSize.begin(), m_vSize.end(), 0) != m_vSize.end();
  }

  bool Valid() const {
    return m_vSize.size() > 0 && *std::min_element(m_vSize.begin(), m_vSize.end()) > 0;
  }

  void Swap(Size &clOther) {
    m_vSize.swap(clOther.m_vSize);
  }

  int & Front() {
    return m_vSize.front();
  }

  const int & Front() const {
    return m_vSize.front();
  }

  int & Back() {
    return m_vSize.back();
  }

  const int & Back() const {
    return m_vSize.back();
  }

  int & operator[](int iAxis) {
    return m_vSize[iAxis];
  }

  const int & operator[](int iAxis) const {
    return m_vSize[iAxis];
  }

  Size SubSize(int iAxisBegin, int iAxisEnd = std::numeric_limits<int>::max()) const {
    iAxisEnd = std::min(iAxisEnd, (int)m_vSize.size());
    iAxisBegin = std::max(0, iAxisBegin);

    std::vector<int> vSubSize(m_vSize.begin()+iAxisBegin, m_vSize.begin()+iAxisEnd);

    return Size(vSubSize);
  }

  // Remove trailing singletons (not all singletons like Matlab)
  // TODO: May need to revisit this function...
  Size Squeeze() const {
    std::vector<int> vNewSize(m_vSize);

    while (vNewSize.size() > 1 && vNewSize.back() == 1)
      vNewSize.pop_back();

    return Size(vNewSize);
  }

  Size & operator=(const std::vector<int> &vSize) {
    m_vSize = vSize;
    return *this;
  }

  Size & operator=(std::vector<int> &&vSize) {
    m_vSize = vSize;
    return *this;
  }

  bool operator==(const Size &clOther) const {
    return m_vSize.size() == clOther.m_vSize.size() &&
      std::equal(m_vSize.begin(), m_vSize.end(), clOther.m_vSize.begin());
  }

  bool operator!=(const Size &clOther) const {
    return !(*this == clOther);
  }

private:
  std::vector<int> m_vSize;
};

inline std::ostream & operator<<(std::ostream &os, const Size &clSize) {
  os << '[';

  for (int d : clSize)
    os << ' ' << d;

  return os << " ]";
}

} // end namespace bleak

#endif // !BLEAK_SIZE_H
