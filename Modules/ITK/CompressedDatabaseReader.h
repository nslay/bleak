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

#ifndef BLEAK_COMPRESSEDDATABASEREADER_H
#define BLEAK_COMPRESSEDDATABASEREADER_H

#include <algorithm>
#include <string>
#include <vector>
#include <memory>
#include "Vertex.h"
#include "DatabaseFactory.h"

// ITK stuff
#include "itk_zlib.h"

// NOTE: This is mostly a copy of DatabaseReader from bleakCommon and exists here since ITK can be relied upon to have zlib (independent of operating system!).

namespace bleak {

template<typename RealType>
class CompressedDatabaseReader : public Vertex<RealType> {
public:
  bleakNewVertex(CompressedDatabaseReader, Vertex<RealType>,
    bleakAddOutput("outData"),
    bleakAddOutput("outLabels"),
    bleakAddProperty("size", m_vSize),
    bleakAddProperty("rowCount", m_iRowCount), // The uncompressed size of each row
    bleakAddProperty("labelIndices", m_vLabelIndices),
    bleakAddGetterSetter("labelIndex", &CompressedDatabaseReader::GetLabelIndex, &CompressedDatabaseReader::SetLabelIndex),
    bleakAddProperty("databaseType", m_strDatabaseType),
    bleakAddProperty("databasePath", m_strDatabasePath));

  bleakForwardVertexTypedefs();

  virtual ~CompressedDatabaseReader() {
    CloseDatabase(); // Make sure this is done in proper order
  }

  // English language convenience (to reduce confusion!)
  // These functions just define "labelIndex" to be a 1D synonym of "labelIndices" while accessing the same member!
  bool GetLabelIndex(int &iIndex) {
    if (m_vLabelIndices.size() != 1)
      return false;

    iIndex = m_vLabelIndices[0];

    return true;
  }

  bool SetLabelIndex(const int &iIndex) {
    m_vLabelIndices.resize(1);
    m_vLabelIndices[0] = iIndex;
    return true;
  }

  virtual bool SetSizes() override {
    bleakGetAndCheckOutput(p_clOutData, "outData", false);
    bleakGetAndCheckOutput(p_clOutLabels, "outLabels", false);

    Size clOutDataSize(m_vSize);

    if (!clOutDataSize.Valid()) {
      std::cerr << GetName() << ": Error: Size was not specified or is invalid." << std::endl;
      return false;
    }

    if (clOutDataSize.GetDimension() < 2) {
      std::cerr << GetName() << ": Error: Size is expected to be at least 2D." << std::endl;
      return false;
    }

    if (!OpenDatabase())
      return false;

    if (m_vRowBuf.size() < m_vLabelIndices.size()) {
      CloseDatabase();
      std::cerr << GetName() << ": Error: There are more label indices than data." << std::endl;
      return false;
    }

    // XXX: Integer overflow in the future?
    for (int iIndex : m_vLabelIndices) {
      if (iIndex < 0 || iIndex >= (int)m_vRowBuf.size()) {
        CloseDatabase();
        std::cerr << GetName() << ": Invalid label index " << iIndex << ": " << iIndex << " vs. " << (int)m_vRowBuf.size() << std::endl;
        return false;
      }
    }

    if (clOutDataSize.SubSize(1).Product() != (int)(m_vRowBuf.size() - m_vLabelIndices.size())) {
      CloseDatabase();
      std::cerr << GetName() << ": Dimension mismatch between data size and requested size " << clOutDataSize.SubSize(1) << '.' << std::endl;
      return false;
    }

    // Make sure these are in an order that makes sense!
    std::sort(m_vLabelIndices.begin(), m_vLabelIndices.end());

    Size clOutLabelsSize;

    if (m_vLabelIndices.size() > 1) {
      clOutLabelsSize.SetDimension(2);  
      clOutLabelsSize[0] = clOutDataSize[0];
      clOutLabelsSize[1] = (int)m_vLabelIndices.size();
    }
    else {
      clOutLabelsSize.SetDimension(1);
      clOutLabelsSize[0] = clOutDataSize[0];
    }

    p_clOutData->GetData().SetSize(clOutDataSize);
    p_clOutLabels->GetData().SetSize(clOutLabelsSize);

    p_clOutData->GetGradient().Clear(); // No back propagation on this edge
    p_clOutLabels->GetGradient().Clear(); // No back propagation on this edge

    return true;
  }

  virtual bool Initialize() override {
    if (m_p_clCursor == nullptr)
      return false;

    m_p_clCursor->Rewind();

    return m_p_clCursor->Good();
  }

  virtual void Forward() override {
    bleakGetAndCheckOutput(p_clOutData, "outData");
    bleakGetAndCheckOutput(p_clOutLabels, "outLabels");

    if (m_p_clCursor == nullptr)
      return;

    if (!m_p_clCursor->Good())
      m_p_clCursor->Rewind();

    ArrayType &clOutData = p_clOutData->GetData();
    ArrayType &clOutLabels = p_clOutLabels->GetData();

    RealType * const p_outData = clOutData.data();
    RealType * const p_outLabels = clOutLabels.data();

    const int iOuterNum = clOutData.GetSize()[0];
    const int iInnerNum = clOutData.GetSize().Product(1);

    const int iNumLabels = clOutLabels.GetSize().Product(1);

    for (int i = 0; i < iOuterNum; ++i) {
      size_t dataSize = 0;
      const uint8_t * const p_ui8Data = m_p_clCursor->Value(dataSize);

      if (p_ui8Data == nullptr) {
        std::cerr << GetName() << ": Error: Invalid data entry at key '" << m_p_clCursor->Key() << "' with byte size = " << dataSize << '.' << std::endl;
        return;
      }

      uLongf ulRowBufSize = uLongf(m_vRowBuf.size()*sizeof(double));

      if (uncompress((uint8_t *)m_vRowBuf.data(), &ulRowBufSize, p_ui8Data, uLong(dataSize)) != Z_OK) {
        std::cerr << GetName() << ": Error: Failed to uncompress data entry at key '" << m_p_clCursor->Key() << "'." << std::endl;
        return;
      }

      if (ulRowBufSize != m_vRowBuf.size()*sizeof(double)) {
        std::cerr << GetName() << ": Error: Unexpected data size. Got " << ulRowBufSize << " bytes but expected " << m_vRowBuf.size()*sizeof(double) << " bytes." << std::endl;
        return;
      }

      const double * const p_dData = m_vRowBuf.data();

      RealType *p_outDataBegin = p_outData + (i*iInnerNum + 0);
      const double *p_dDataBegin = p_dData;
      const double *p_dDataEnd = nullptr;

      for (size_t j = 0; j < m_vLabelIndices.size(); ++j) {
        const int k = m_vLabelIndices[j];
        p_dDataEnd = p_dData + k;

        p_outDataBegin = std::transform(p_dDataBegin, p_dDataEnd, p_outDataBegin,
          [](const double &x) -> RealType {
            return RealType(x);
          });

        p_dDataBegin = p_dDataEnd;

        // Assign the label
        p_outLabels[i*iNumLabels + j] = RealType(*p_dDataBegin);
        ++p_dDataBegin;
      }

      p_dDataEnd = p_dData + m_vRowBuf.size();

      std::transform(p_dDataBegin, p_dDataEnd, p_outDataBegin, 
        [](const double &x) -> RealType {
          return RealType(x);
        });

      if (!m_p_clCursor->Next())
        m_p_clCursor->Rewind();
    }
  }

  virtual void Backward() override { } // Nothing to do

protected:
  CompressedDatabaseReader() = default;

private:
  std::vector<int> m_vSize;
  std::vector<int> m_vLabelIndices;
  std::string m_strDatabaseType;
  std::string m_strDatabasePath;

  // Expected uncompressed data size
  int m_iRowCount = 0;

  mutable std::vector<double> m_vRowBuf;

  std::shared_ptr<Database> m_p_clDatabase;
  std::unique_ptr<Cursor> m_p_clCursor;

  void CloseDatabase() {
    m_vRowBuf.clear();
    m_p_clCursor.reset();
    m_p_clDatabase.reset();
  }

  bool OpenDatabase() {
    CloseDatabase();

    if (m_iRowCount <= 0) {
      std::cerr << GetName() << ": Error: Invalid rowCount specified (" << m_iRowCount << ")." << std::endl;
      return false;
    }

    m_p_clDatabase = DatabaseFactory::GetInstance().Create(m_strDatabaseType);

    if (!m_p_clDatabase) {
      std::cerr << GetName() << ": Error: Could not create database of type '" << m_strDatabaseType << "'." << std::endl;
      return false;
    }

    if (!m_p_clDatabase->Open(m_strDatabasePath, Database::READ)) {
      std::cerr << GetName() << ": Error: Could not open database '" << m_strDatabasePath << "' for reading." << std::endl;
      return false;
    }

    m_p_clCursor = m_p_clDatabase->NewCursor();

    if (!m_p_clCursor || !m_p_clCursor->Good()) {
      std::cerr << GetName() << ": Error: Failed to create cursor (or cursor is not good)." << std::endl;
      CloseDatabase();
      return false;
    }

    // Check that the data is actually compressed
    size_t dataSize = 0;
    const uint8_t * const p_ui8Data = m_p_clCursor->Value(dataSize);

    m_vRowBuf.resize(m_iRowCount);

    uLongf ulRowBufSize = uLongf(m_vRowBuf.size()*sizeof(double));

    if (uncompress((uint8_t *)m_vRowBuf.data(), &ulRowBufSize, p_ui8Data, uLong(dataSize)) != Z_OK) {
      std::cerr << GetName() << ": Error: Database is not compressed or rowCount is incorrect." << std::endl;
      CloseDatabase();
      return false;
    }

    if (ulRowBufSize != m_vRowBuf.size()*sizeof(double)) {
      std::cerr << GetName() << ": Error: Unexpected uncompressed size. Got " << ulRowBufSize << " bytes but expected " << m_vRowBuf.size()*sizeof(double) << " bytes." << std::endl;
      CloseDatabase();
      return false;
    }

    return true;
  }
};

} // end namespace bleak

#endif // !BLEAK_COMPRESSEDDATABASEREADER_H
