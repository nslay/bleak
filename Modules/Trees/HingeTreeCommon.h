/*-
 * Copyright (c) 2019 Nathan Lay (enslay@gmail.com)
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

#ifndef BLEAK_HINGETREECOMMON_H
#define BLEAK_HINGETREECOMMON_H

#include <cmath>
#include <cstdint>
#include <climits>
#include <algorithm>
#include <utility>
#include <tuple>
#include <type_traits>

namespace bleak {

template<typename RealTypeT, typename KeyTypeT = uint32_t>
class HingeFernCommon {
public:
  typedef RealTypeT RealType;
  typedef KeyTypeT KeyType;
  typedef std::tuple<KeyType, RealType, KeyType> KeyMarginTupleType; // leaf key, margin, and threshold/ordinal index

  static_assert(std::is_integral<KeyType>::value && std::is_unsigned<KeyType>::value, "Unsigned integral type is required for leaf keys.");

  static constexpr int GetMaxDepth() { return CHAR_BIT*sizeof(KeyType) - 1; }

  // -1 if an error (not power of 2)
  // This can be determined from threshold/ordinals size though!
  static int ComputeDepth(int iLeafCount) {
    if (iLeafCount <= 0 || (iLeafCount & (iLeafCount-1)) != 0)
      return -1; // Not a power of 2

    int iTreeDepth = 0;

    for ( ; iLeafCount != 0; iLeafCount >>= 1)
      ++iTreeDepth;

    return iTreeDepth-1;
  }

  // For sanity checks!
  static int GetThresholdCount(int iTreeDepth) { return iTreeDepth; } // Internal tree vertices
  static int GetLeafCount(int iTreeDepth) { return 1 << iTreeDepth; }

  // Returns leaf key, signed margin and threshold/ordinal index
  static KeyMarginTupleType ComputeKeyAndSignedMargin(const RealType *p_data, const RealType *p_thresholds, const RealType *p_ordinals, int iTreeDepth, int iStride = 1) {
    KeyType leafKey = KeyType();
    RealType minMargin = p_data[iStride*(int)p_ordinals[0]] - p_thresholds[0];
    KeyType minFernIndex = 0;

    for (int i = 0; i < iTreeDepth; ++i) {
      const int j = (int)p_ordinals[i];
      const RealType margin = p_data[iStride*j] - p_thresholds[i];
      const KeyType bit = (margin > RealType(0));

      leafKey |= (bit << i);

      if (std::abs(margin) < std::abs(minMargin)) {
        minMargin = margin;
        minFernIndex = i;
      }
    }

    return std::make_tuple(leafKey, minMargin, minFernIndex);
  }
};

template<typename RealTypeT, typename KeyTypeT = uint32_t>
class HingeTreeCommon {
public:
  typedef RealTypeT RealType;
  typedef KeyTypeT KeyType;
  typedef std::tuple<KeyType, RealType, KeyType> KeyMarginTupleType; // leaf key, margin, and threshold/ordinal index

  static_assert(std::is_integral<KeyType>::value && std::is_unsigned<KeyType>::value, "Unsigned integral type is required for leaf keys.");

  static constexpr int GetMaxDepth() { return CHAR_BIT*sizeof(KeyType) - 1; }

  // -1 if an error (not power of 2)
  static int ComputeDepth(int iLeafCount) {
    if (iLeafCount <= 0 || (iLeafCount & (iLeafCount-1)) != 0)
      return -1; // Not a power of 2

    int iTreeDepth = 0;

    for ( ; iLeafCount != 0; iLeafCount >>= 1)
      ++iTreeDepth;

    return iTreeDepth-1;
  }

  static int GetThresholdCount(int iTreeDepth) { return (1 << iTreeDepth) - 1; } // Internal tree vertices
  static int GetLeafCount(int iTreeDepth) { return 1 << iTreeDepth; }

  // Returns leaf key, signed margin and threshold/ordinal index
  static KeyMarginTupleType ComputeKeyAndSignedMargin(const RealType *p_data, const RealType *p_thresholds, const RealType *p_ordinals, int iTreeDepth, int iStride = 1) {
    KeyType leafKey = KeyType();
    KeyType treeIndex = KeyType();
    RealType minMargin = p_data[iStride * (int)p_ordinals[0]] - p_thresholds[0];
    KeyType minTreeIndex = KeyType();

    for (int i = 0; i < iTreeDepth; ++i) {
      const int j = (int)p_ordinals[treeIndex];
      const RealType margin = p_data[j*iStride] - p_thresholds[treeIndex];
      const KeyType bit = (margin > RealType(0));

      if (std::abs(margin) < std::abs(minMargin)) {
        minMargin = margin;
        minTreeIndex = treeIndex;
      }

      leafKey |= (bit << i);
      treeIndex = 2*treeIndex + 1 + bit;
    }

    return std::make_tuple(leafKey, minMargin, minTreeIndex);
  }
};

} // end namespace bleak

#endif // !BLEAK_HINGETREECOMMON_H
