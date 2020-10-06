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

#ifndef BLEAK_DATABASEREADER_H
#define BLEAK_DATABASEREADER_H

#include <algorithm>
#include <string>
#include <vector>
#include <memory>
#include <random>
#include <set>
#include "Vertex.h"
#include "DatabaseFactory.h"

namespace bleak {

template<typename RealType>
class DatabaseReader : public Vertex<RealType> {
public:
  typedef std::mt19937_64 GeneratorType;

  bleakNewVertex(DatabaseReader, Vertex<RealType>,
    bleakAddOutput("outData"),
    bleakAddOutput("outLabels"),
    bleakAddProperty("size", m_vSize),
    bleakAddProperty("labelIndices", m_vLabelIndices),
    bleakAddGetterSetter("labelIndex", &DatabaseReader::GetLabelIndex, &DatabaseReader::SetLabelIndex),
    bleakAddProperty("databaseType", m_strDatabaseType),
    bleakAddProperty("databasePath", m_strDatabasePath),
    bleakAddProperty("seed", m_strSeed),
    bleakAddProperty("shuffle", m_bShuffle));

  bleakForwardVertexTypedefs();

  virtual ~DatabaseReader() {
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

    if (m_dataSize < m_vLabelIndices.size()) {
      CloseDatabase();
      std::cerr << GetName() << ": Error: There are more label indices than data." << std::endl;
      return false;
    }

    // XXX: Integer overflow in the future?
    for (int iIndex : m_vLabelIndices) {
      if (iIndex < 0 || iIndex >= (int)m_dataSize) {
        CloseDatabase();
        std::cerr << GetName() << ": Invalid label index " << iIndex << ": " << iIndex << " vs. " << (int)m_dataSize << std::endl;
        return false;
      }
    }

    if (clOutDataSize.Product(1) != (int)(m_dataSize - m_vLabelIndices.size())) {
      CloseDatabase();
      std::cerr << GetName() << ": Dimension mismatch between data size and requested size " << clOutDataSize.SubSize(1) << '.' << std::endl;
      return false;
    }

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
    m_vKeys.clear();
    m_iItr = 0;

    if (m_p_clCursor == nullptr)
      return false;

    m_p_clCursor->Rewind();
    if (!m_p_clCursor->Good())
      return false;

    // Collect and sort unique label partitions
    std::set<int> m_sLabelIndices(m_vLabelIndices.begin(), m_vLabelIndices.end());
    m_vPartitions.assign(m_sLabelIndices.begin(), m_sLabelIndices.end());

    std::cout << GetName() << ": Info: Shuffling " << (m_bShuffle ? "enabled" : "disabled") << '.' << std::endl;

    if (m_strSeed.size() > 0) {
      std::cout << GetName() << ": Info: Using seed string '" << m_strSeed << "' for shuffling." << std::endl;
      std::seed_seq clSeed(m_strSeed.begin(), m_strSeed.end());
      this->GetGenerator().seed(clSeed);
    }
    else {
      this->GetGenerator().seed();
    }

    if (m_bShuffle) {
      do {
        m_vKeys.push_back(m_p_clCursor->Key());
      } while (m_p_clCursor->Next());

      m_p_clCursor->Rewind();

      std::cout << GetName() << ": Info: Collected " << m_vKeys.size() << " database keys (for shuffling)." << std::endl;
    }

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
      const double * const p_dData = Next();

      if (p_dData == nullptr) {
        std::cerr << GetName() << ": Next() failed. Giving up." << std::endl;
        return;
      }

      RealType *p_outDataBegin = p_outData + (i*iInnerNum + 0);
      
      const double *p_dDataBegin = p_dData;
      const double *p_dDataEnd = nullptr;

      for (const int k : m_vPartitions) {
        p_dDataEnd = p_dData + k;

        // XXX: This kind of std::transform miscompiles on GCC 7.5? Crashes on Ubuntu 18... but a for loop does not
        p_outDataBegin = std::transform(p_dDataBegin, p_dDataEnd, p_outDataBegin,
          [](const double &x) -> RealType {
            return RealType(x);
          });

        p_dDataBegin = p_dDataEnd+1;
      }

      p_dDataEnd = p_dData + m_dataSize;

      // XXX: Ditto
      std::transform(p_dDataBegin, p_dDataEnd, p_outDataBegin,
        [](const double &x) -> RealType {
          return RealType(x);
        });

      // Now assign labels...
      for (size_t j = 0; j < m_vLabelIndices.size(); ++j) {
        const int k = m_vLabelIndices[j];
        p_outLabels[i*iNumLabels + j] = RealType(p_dData[k]);
      }
    }
  }

  virtual void Backward() override { } // Nothing to do

protected:
  DatabaseReader() = default;

private:
  std::vector<int> m_vSize;
  std::vector<int> m_vLabelIndices;
  std::vector<int> m_vPartitions; // Partition the range
  std::string m_strDatabaseType;
  std::string m_strDatabasePath;

  GeneratorType m_clGenerator;
  bool m_bShuffle = false;
  std::string m_strSeed;

  // Expected data size
  size_t m_dataSize = 0;

  // Keys if we're shuffling
  std::vector<std::string> m_vKeys;
  int m_iItr = 0;

  std::shared_ptr<Database> m_p_clDatabase;
  std::unique_ptr<Cursor> m_p_clCursor;

  DatabaseReader(const DatabaseReader &) = delete;
  DatabaseReader(DatabaseReader &&) = delete;
  DatabaseReader & operator=(const DatabaseReader &) = delete;
  DatabaseReader & operator=(DatabaseReader &&) = delete;

  GeneratorType & GetGenerator() { return m_clGenerator; }

  const double * Next() {
    if (m_p_clCursor == nullptr)
      return nullptr;

    if (m_vKeys.empty()) {
      if (!m_p_clCursor->Next())
        m_p_clCursor->Rewind();
    }
    else {
      if (m_iItr >= (int)m_vKeys.size()) {
        std::shuffle(m_vKeys.begin(), m_vKeys.end(), this->GetGenerator()); 
        m_iItr = 0;
      }

      const std::string &strKey = m_vKeys[m_iItr++];

      if (!m_p_clCursor->Find(strKey)) {
        std::cerr << GetName() << ": Error: Could not find key '" << strKey << "' in database?" << std::endl;
        return nullptr;
      }
    }

    size_t dataSize = 0;
    const uint8_t * const p_ui8Data = m_p_clCursor->Value(dataSize);

    if (p_ui8Data == nullptr) {
      std::cerr << GetName() << ": Error: nullptr data value at key '" << m_p_clCursor->Key() << "'?" << std::endl;
      return nullptr;
    }

    if ((dataSize % sizeof(double)) != 0) {
      std::cerr << GetName() << ": Error: Data at key '" << m_p_clCursor->Key() << "' is not a multiple of " << sizeof(double) << " (got " << dataSize << " bytes)." << std::endl;
      return nullptr;
    }

    dataSize /= sizeof(double);

    if (m_dataSize != dataSize) {
      std::cerr << GetName() << ": Error: Data at key '" << m_p_clCursor->Key() << "' has an unexpected number of elements (expected " << m_dataSize << " but got " << dataSize << ")." << std::endl;
      return nullptr;
    }

    return (double *)p_ui8Data;
  }

  void CloseDatabase() {
    m_dataSize = 0;
    m_p_clCursor.reset();
    m_p_clDatabase.reset();
  }

  bool OpenDatabase() {
    CloseDatabase();

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

    m_p_clCursor->Value(m_dataSize); // All data must now be this size

    if (m_dataSize == 0 || (m_dataSize % sizeof(double)) != 0) {
      std::cerr << GetName() << ": Error: Invalid data. Either empty or not a multiple of sizeof(double): " << m_dataSize << " vs. " << sizeof(double) << '.' << std::endl;
      CloseDatabase();
      return false;
    }

    m_dataSize /= sizeof(double); // They are assumed to be all stored as double

    return true;
  }
};

} // end namespace bleak

#endif // !BLEAK_DATABASEREADER_H
