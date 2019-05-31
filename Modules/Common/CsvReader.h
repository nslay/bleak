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

#ifndef BLEAK_CSVREADER_H
#define BLEAK_CSVREADER_H

#include <algorithm>
#include <fstream>
#include <utility>
#include <vector>
#include "Common.h"
#include "Vertex.h"

namespace bleak {

template<typename RealType>
class CsvReader : public Vertex<RealType> {
public:
  bleakNewVertex(CsvReader, Vertex<RealType>,
    bleakAddOutput("outData"),
    bleakAddOutput("outLabels"),
    bleakAddProperty("batchSize", m_iBatchSize),
    bleakAddProperty("csvFileName", m_strCsvFileName),
    bleakAddProperty("skipLines", m_iSkipLines),
    bleakAddProperty("labelColumn", m_iLabelColumn),
    bleakAddProperty("shuffle", m_bShuffle));

  bleakForwardVertexTypedefs();

  virtual bool SetSizes() override {
    if (!LoadData()) {
      std::cerr << GetName() << ": Error: Failed to load data to infer output sizes." << std::endl;
      return false;
    }

    bleakGetAndCheckOutput(p_clData, "outData", false);
    bleakGetAndCheckOutput(p_clLabels, "outLabels", false);

    if (m_iBatchSize <= 0 || m_vData.empty() || m_vData[0].first.empty()) {
      std::cerr << GetName() << ": Error: Invalid batch size or empty data." << std::endl;
      return false;
    }

    const Size clOutDataSize = { m_iBatchSize, (int)m_vData[0].first.size() };
    const Size clOutLabelsSize = { m_iBatchSize };

    //std::cout << GetName() << ": Info: outData size = " << clOutDataSize << ", outLabels size = " << clOutLabelsSize << std::endl;

    p_clData->GetData().SetSize(clOutDataSize);
    p_clData->GetGradient().Clear(); // We do not backpropagate on this vertex

    p_clLabels->GetData().SetSize(clOutLabelsSize);
    p_clLabels->GetGradient().Clear(); // We do not backpropagate on this vertex

    return true;
  }

  virtual bool Initialize() override {
    m_currentIndex = 0;

    if (m_bShuffle)
      std::shuffle(m_vData.begin(), m_vData.end(), GetGenerator());

    return true;
  }

  virtual void Forward() override {
    if (m_vData.empty())
      return;

    bleakGetAndCheckOutput(p_clData, "outData");
    bleakGetAndCheckOutput(p_clLabels, "outLabels");

    ArrayType &clData = p_clData->GetData();
    ArrayType &clLabels = p_clLabels->GetData();

    const int iOuterNum = clData.GetSize()[0];
    const int iInnerNum = clData.GetSize()[1];

    bool bShouldShuffle = false;

    RealType * const p_data = clData.data();
    RealType * const p_labels = clLabels.data();

    if (m_currentIndex + iOuterNum >= m_vData.size())
      bShouldShuffle = m_bShuffle;

    for (int i = 0; i < iOuterNum; ++i) {
      const std::vector<RealType> &vRow = m_vData[m_currentIndex].first;
      const RealType label = m_vData[m_currentIndex].second;

      if((int)vRow.size() != iInnerNum) // TODO: Warning?
        continue;

      std::copy(vRow.begin(), vRow.end(), p_data + (i * iInnerNum));
      p_labels[i] = label;

      m_currentIndex = ((m_currentIndex+1) % m_vData.size());
    }

    if (bShouldShuffle) {
      std::shuffle(m_vData.begin(), m_vData.end(), GetGenerator());
      m_currentIndex = 0;
    }
  }

  virtual void Backward() override { } // Nothing to do

protected:
  CsvReader() {
    m_iBatchSize = 10;
    m_iSkipLines = 0;
    m_iLabelColumn = -1; // There may not be labels available
    m_bShuffle = true;
    m_currentIndex = 0;
  }

private:
  int m_iBatchSize;
  std::string m_strCsvFileName;
  int m_iSkipLines;
  int m_iLabelColumn;
  bool m_bShuffle;

  size_t m_currentIndex;

  typedef std::pair<std::vector<RealType>, RealType> ExamplePairType;

  std::vector<ExamplePairType> m_vData;

  bool LoadData() {
    m_vData.clear();

    std::ifstream csvStream(m_strCsvFileName.c_str());

    if (!csvStream) {
      std::cerr << GetName() << ": Error: Failed to open file '" << m_strCsvFileName << "'." << std::endl;
      return false;
    }

    std::string strLine;

    for (int i = 0; i < m_iSkipLines; ++i)
      std::getline(csvStream, strLine);

    std::vector<RealType> vTmp, vRow;
    RealType label = RealType();

    while (std::getline(csvStream, strLine)) {
      Trim(strLine);

      if (strLine.empty())
        continue;

      vTmp = SplitString<RealType>(strLine, ",");

      if (m_iLabelColumn >= 0) {
        if (m_iLabelColumn >= (int)vTmp.size()) {
          std::cerr << GetName() << "Error: Invalid labelColumn (" << m_iLabelColumn << " >= " << vTmp.size() << ")." << std::endl;
          m_vData.clear();
          return false;
        }

        label = vTmp[m_iLabelColumn];

        vRow.assign(vTmp.begin(), vTmp.begin() + m_iLabelColumn);
        vRow.insert(vRow.end(), vTmp.begin() + (m_iLabelColumn + 1), vTmp.end());
      }
      else {
        vRow.swap(vTmp);
      }

      if (m_vData.size() > 0 && m_vData[0].first.size() != vRow.size()) {
        std::cerr << GetName() << ": Error: Row size mismatch in CSV file." << std::endl; // TODO: Improve error message
        m_vData.clear();
        return false;
      }

      m_vData.emplace_back(vRow, label);
    }

    return m_vData.size() > 0;
  }
};

} // end namespace bleak

#endif // !BLEAK_CSVREADER_H
