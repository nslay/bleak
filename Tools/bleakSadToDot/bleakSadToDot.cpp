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

#include <cstdlib>
#include <iostream>
#include <utility>
#include <memory>
#include <fstream>
#include <memory>
#include "Graph.h"
#include "SadLoader.h"
#include "InitializeModules.h"
#include "bsdgetopt.h"

typedef float RealType;
typedef bleak::Graph<RealType> GraphType;
typedef bleak::Vertex<RealType> VertexType;

void Usage(const char *p_cArg0) {
  std::cerr << "Usage: " << p_cArg0 << " [-h] [-I includeDir] graph.sad output.dot" << std::endl;
  exit(1);
}

std::pair<std::string, std::string> SplitFirstToken(const std::string &strString, const std::string &strDelim);
std::string GetTailToken(const std::string &strString,const std::string &strDelim);

// A node can be one or more vertices
class DotNode {
public:
  typedef std::unordered_map<std::string, size_t> NodeMapType;

  const std::string & GetLabel() const {
    return m_strLabel;
  }

  bool AddVertex(const std::string &strName, const std::shared_ptr<VertexType> &p_clVertex);

  void DeclareNodes(std::ostream &os, NodeMapType &mIndexMap, const std::string &strIndent = std::string()) const;
  void DeclareSubgraphs(std::ostream &os, size_t &subGraphIndex, const NodeMapType &mVertexIndexMap, const std::string &strIndent = std::string()) const;
  void DeclareOutputConnections(std::ostream &os, const NodeMapType &mVertexIndexMap, const std::string &strIndent = std::string()) const;

private:
  std::string m_strLabel;
  std::vector<DotNode> m_vSubgraph;
  std::unordered_set<std::shared_ptr<VertexType>> m_sVertexSet;
};

class DotGraph {
public:
  DotGraph(const std::shared_ptr<GraphType> &p_clGraph);

  void SaveToStream(std::ostream &os) const;

  bool SaveToFile(const std::string &strFileName) const {
    std::ofstream dotStream(strFileName);

    if (!dotStream)
      return false;

    SaveToStream(dotStream);

    return true;
  }

private:
  std::vector<DotNode> m_vNodes;

  void AddVertex(const std::shared_ptr<VertexType> &p_clVertex);
};

int main(int argc, char **argv) {
  const char * const p_cArg0 = argv[0];

  std::vector<std::string> vIncludeDirs;

  int c = 0;
  while ((c = getopt(argc, argv, "hI:")) != -1) {
    switch (c) {
    case 'h':
      Usage(p_cArg0);
      break;
    case 'I':
      vIncludeDirs.emplace_back(optarg);
      break;
    case '?':
    default:
      Usage(p_cArg0);
      break;
    }
  }

  argc -= optind;
  argv += optind;

  if (argc != 2)
    Usage(p_cArg0);

  const char * const p_cGraphFile = argv[0];
  const char * const p_cDotFile = argv[1];

  bleak::InitializeModules();

  std::shared_ptr<GraphType> p_clGraph = bleak::LoadGraph<RealType>(p_cGraphFile, vIncludeDirs);

  if (!p_clGraph) {
    std::cerr << "Error: Could not load graph file '" << p_cGraphFile << "'." << std::endl;
    return -1;
  }

  DotGraph clDotGraph(p_clGraph);

  if (!clDotGraph.SaveToFile(p_cDotFile)) {
    std::cerr << "Error: Failed to save dot file '" << p_cDotFile << "'." << std::endl;
    return -1;
  }

  return 0;
}

std::pair<std::string, std::string> SplitFirstToken(const std::string &strString, const std::string &strDelim) {
  size_t p = strString.find_first_of(strDelim);

  if (p != std::string::npos) {
    if (p+1 < strString.size())
      return { strString.substr(0, p), strString.substr(p+1) };
    else
      return { strString.substr(0, p), std::string() };
  }

  return { strString, std::string() };
}

std::string GetTailToken(const std::string &strString,const std::string &strDelim) {
  size_t p = strString.find_last_of(strDelim);

  if (p != std::string::npos) {
    if (p+1 < strString.size())
      return strString.substr(p+1);
    else
      return std::string();
  }

  return strString;
}

////////// DotNode //////////

bool DotNode::AddVertex(const std::string &strName, const std::shared_ptr<VertexType> &p_clVertex) {
  if (strName.empty())
    return false; // Must have a name

  auto clPair = SplitFirstToken(strName, "#");

  if (m_strLabel.size() > 0 && clPair.first != m_strLabel)
    return false;

  m_strLabel = clPair.first;

  size_t p = clPair.second.find('#');

  if (p != std::string::npos) {
    for (auto &clSubgraph : m_vSubgraph) {
      if (clSubgraph.AddVertex(clPair.second, p_clVertex))
        return true;
    }

    m_vSubgraph.emplace_back();

    return m_vSubgraph.back().AddVertex(clPair.second, p_clVertex);
  }
  else {
    m_sVertexSet.insert(p_clVertex);
  }

  return true;
}

void DotNode::DeclareNodes(std::ostream &os, NodeMapType &mVertexIndexMap, const std::string &strIndent) const {
  for (const auto &p_clVertex : m_sVertexSet) {
    const size_t index = mVertexIndexMap.size() + 1;

    if (mVertexIndexMap.emplace(p_clVertex->GetName(), index).second) {
      //os << strIndent << index << ";\n";
      os << strIndent << index << " [ label=\"{";

      if (p_clVertex->HasInputs()) {
        os << '{';

        size_t i = 0;
        for (const auto &clPair : p_clVertex->GetAllInputs()) {
          os << '<' << clPair.first << "> " << clPair.first;

          if (++i < p_clVertex->GetAllInputs().size())
            os << '|';
        }

        os << '}';
      }
      else
        os << "No Inputs";

      os << '|';

      os << GetTailToken(p_clVertex->GetName(), "#");

      os << '|';

      if (p_clVertex->HasOutputs()) {
        os << '{';

        size_t i = 0;
        for (const auto &clPair : p_clVertex->GetAllOutputs()) {
          os << '<' << clPair.first << "> " << clPair.first;

          if (++i < p_clVertex->GetAllOutputs().size())
            os << '|';
        }

        os << '}';
      }
      else
        os << "No Outputs";

      os << "}\" ];\n";
    }
  }

  for (auto &clSubgraph : m_vSubgraph)
    clSubgraph.DeclareNodes(os, mVertexIndexMap, strIndent);
}

void DotNode::DeclareSubgraphs(std::ostream &os, size_t &subGraphIndex, const NodeMapType &mVertexIndexMap, const std::string &strIndent) const {
  if (m_vSubgraph.empty() && m_sVertexSet.size() < 2)
    return;

  os << strIndent << "subgraph cluster" << subGraphIndex++ << " {\n";

  // Should always declare this even if it's empty (inherits parent's label too)
  os << strIndent << "  " << "label = \"" << m_strLabel << "\";\n";

  for (const auto &p_clVertex : m_sVertexSet) {
    auto itr = mVertexIndexMap.find(p_clVertex->GetName());

    if (itr == mVertexIndexMap.end())
      continue;

    os << strIndent << "  " << itr->second << ";\n";
  }

  for (auto &clSubgraph : m_vSubgraph)
    clSubgraph.DeclareSubgraphs(os, subGraphIndex, mVertexIndexMap, strIndent + "  ");

  os << strIndent << "}\n";
}

void DotNode::DeclareOutputConnections(std::ostream &os, const NodeMapType &mVertexIndexMap, const std::string &strIndent) const {
  for (const auto &p_clVertex : m_sVertexSet) {
    auto itr = mVertexIndexMap.find(p_clVertex->GetName());

    if (itr == mVertexIndexMap.end()) { // Should not happen
      std::cout << "Error: Could not find index mapping for vertex '" << p_clVertex->GetName() << "'." << std::endl;
      continue;
    }

    const size_t sourceIndex = itr->second;

    for (const auto &clOutputPair : p_clVertex->GetAllOutputs()) {
      if (clOutputPair.second == nullptr)
        continue;

      const std::string &strOutputName = clOutputPair.first;

      const auto &vTargets = clOutputPair.second->GetAllTargets();

      for (const auto &clTargetPair : vTargets) {
        const auto p_clTarget = clTargetPair.first.lock();

        itr = mVertexIndexMap.find(p_clTarget->GetName());

        if (itr == mVertexIndexMap.end()) {
          std::cout << "Error: Could not find index mapping for vertex '" << p_clTarget->GetName() << "'." << std::endl;
          continue;
        }

        const size_t targetIndex = itr->second;

        for (const auto &clInputPair : p_clTarget->GetAllInputs()) {
          const std::string &strInputName = clInputPair.first;
          if (clInputPair.second.lock() == clOutputPair.second)
            os << strIndent << sourceIndex << ':' << strOutputName << " -> " << targetIndex << ':' << strInputName << ";\n";
        }
      }
    }
  }

  for (auto &clSubgraph : m_vSubgraph)
    clSubgraph.DeclareOutputConnections(os, mVertexIndexMap, strIndent);
}

////////// End DotNode //////////

////////// DotGraph //////////

DotGraph::DotGraph(const std::shared_ptr<GraphType> &p_clGraph) {
  for (const auto &p_clVertex : p_clGraph->GetAllVertices()) {
    AddVertex(p_clVertex);
  }
}

void DotGraph::AddVertex(const std::shared_ptr<VertexType> &p_clVertex) {
  if (p_clVertex->GetName().empty())
    return;

  for (DotNode &clNode : m_vNodes) {
    if (clNode.AddVertex(p_clVertex->GetName(), p_clVertex))
      return;
  }

  m_vNodes.emplace_back();
  m_vNodes.back().AddVertex(p_clVertex->GetName(), p_clVertex);
}

void DotGraph::SaveToStream(std::ostream &os) const {
  os << "digraph SadGraph {\n";

  const std::string strIndent = "  ";

  os << strIndent << "node [shape=record];\n";

  DotNode::NodeMapType mVertexIndexMap;
  size_t subGraphIndex = 1;

  for (const DotNode &clNode : m_vNodes)
    clNode.DeclareNodes(os, mVertexIndexMap, strIndent);

  for (const DotNode &clNode : m_vNodes)
    clNode.DeclareSubgraphs(os, subGraphIndex, mVertexIndexMap, strIndent);

  for (const DotNode &clNode : m_vNodes)
    clNode.DeclareOutputConnections(os, mVertexIndexMap, strIndent);

  os << "}" << std::endl;
}

////////// End DotGraph //////////
