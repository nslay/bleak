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

#ifndef BLEAK_SADLOADER_H
#define BLEAK_SADLOADER_H

#include <cstring>
#include <iostream>
#include <memory>
#include <unordered_map>
#include "Parser/bleak_graph_ast.h"
#include "Parser/bleak_expression.h"
#include "VertexFactory.h"
#include "Graph.h"
#include "Subgraph.h"

namespace bleak {

std::shared_ptr<bleak_graph> LoadGraphAst(const std::string &strFileName, const std::vector<std::string> &vSearchDirs = std::vector<std::string>());

// Special factory that will create subgraph placeholders
template<typename RealType>
class VertexAstFactory {
public:
  typedef Vertex<RealType> VertexType;
  typedef Subgraph<RealType> SubgraphType;
  typedef VertexFactory<RealType> VertexFactoryType;

  VertexAstFactory(const std::shared_ptr<bleak_graph> &p_stGraph)
  : m_p_stGraph(p_stGraph) { }

  bool Good() const {
    if (m_p_stGraph == nullptr)
      return false;

    VertexFactoryType &clFactory = VertexFactoryType::GetInstance();

    for (bleak_graph *p_stSubgraph : *m_p_stGraph->p_stSubgraphs) {
      if (clFactory.CanCreate(p_stSubgraph->p_cName)) {
        std::cerr << "Error: Subgraph has the same name '" << p_stSubgraph->p_cName << "' as a built-in Vertex." << std::endl;
        return false;
      }
    }

    for(bleak_graph *p_stSubgraph : *m_p_stGraph->p_stSubgraphs) {
      for(bleak_graph *p_stSubgraphOther : *m_p_stGraph->p_stSubgraphs) {
        if (p_stSubgraph != p_stSubgraphOther && strcmp(p_stSubgraph->p_cName, p_stSubgraphOther->p_cName) == 0) {
          std::cerr << "Error: Two subgraphs have the same name '" << p_stSubgraph->p_cName << "'." << std::endl;
          return false;
        }
      }
    }

    return true;
  }

  std::shared_ptr<VertexType> Create(const std::string &strTypeName) const {
    {
      VertexFactoryType &clFactory = VertexFactoryType::GetInstance();

      std::shared_ptr<VertexType> p_clVertex = clFactory.Create(strTypeName);

      if (p_clVertex != nullptr)
        return p_clVertex;
    }

    for (bleak_graph *p_stSubgraph : *m_p_stGraph->p_stSubgraphs) {
      if (strTypeName == p_stSubgraph->p_cName) {
        std::shared_ptr<bleak_graph> p_stSubgraphAst(bleak_expr_graph_resolve_variables(p_stSubgraph, 0), &bleak_graph_free);
        std::shared_ptr<SubgraphType> p_clSubgraph = SubgraphType::New();

        if (!p_clSubgraph->SetGraphAst(p_stSubgraphAst))
          return std::shared_ptr<VertexType>();

        return p_clSubgraph;
      }
    }

    return std::shared_ptr<VertexType>();
  }

private:
  const std::shared_ptr<bleak_graph> m_p_stGraph;
};

inline std::string ProperVertexName(const std::string &strName, const std::string &strPrefix = std::string()) {
  if (strName == "this" || strPrefix.empty())
    return strName;

  return strPrefix + '#' + strName;
}

template<typename RealType>
bool AssignProperties(const std::shared_ptr<Vertex<RealType>> &p_clVertex, bleak_vertex *p_stVertexAst) {
  typedef Vertex<RealType> VertexType;

  if (p_clVertex == nullptr || p_stVertexAst == nullptr)
    return false;

  for (bleak_key_value_pair *p_stKvp : *p_stVertexAst->p_stProperties) {
    if (strcmp(p_stKvp->p_cKey, "name") == 0) {
      std::cerr << "Warning: Property 'name' for vertex '" << p_clVertex->GetName() << "' declared. Ignoring property..." << std::endl;
      continue;
    }

    switch (p_stKvp->p_stValue->eType) {
    case BLEAK_INTEGER:
      if (!p_clVertex->SetProperty(p_stKvp->p_cKey, p_stKvp->p_stValue->iValue)) {
        std::cerr << "Error: Failed to set integer property for vertex '" << p_clVertex->GetName() << "': " << 
          p_stKvp->p_cKey << " = " << p_stKvp->p_stValue->iValue << std::endl;
        return false;
      }
      break;
    case BLEAK_FLOAT:
      if (!p_clVertex->SetProperty(p_stKvp->p_cKey, p_stKvp->p_stValue->fValue)) {
        std::cerr << "Error: Failed to set float property for vertex '" << p_clVertex->GetName() << "': " << 
          p_stKvp->p_cKey << " = " << p_stKvp->p_stValue->fValue << std::endl;
        return false;
      }
      break;
    case BLEAK_BOOL:
      if (!p_clVertex->SetProperty(p_stKvp->p_cKey, p_stKvp->p_stValue->bValue)) {
        std::cerr << "Error: Failed to set bool property for vertex '" << p_clVertex->GetName() << "': " << 
          p_stKvp->p_cKey << " = " << p_stKvp->p_stValue->bValue << std::endl;
        return false;
      }
      break;
    case BLEAK_STRING:
      if (!p_clVertex->SetProperty(p_stKvp->p_cKey, p_stKvp->p_stValue->p_cValue)) {
        std::cerr << "Error: Failed to set string property for vertex '" << p_clVertex->GetName() << "': " << 
          p_stKvp->p_cKey << " = '" << p_stKvp->p_stValue->p_cValue << "'" << std::endl;
        return false;
      }
      break;
    case BLEAK_INTEGER_VECTOR:
      if (!p_clVertex->SetProperty(p_stKvp->p_cKey, *p_stKvp->p_stValue->p_stIVValue)) {
        std::cerr << "Error: Failed to set integer vector property for vertex '" << p_clVertex->GetName() << "'." << std::endl;
        return false;
      }
      break;
    case BLEAK_FLOAT_VECTOR:
      if (!p_clVertex->SetProperty(p_stKvp->p_cKey, *p_stKvp->p_stValue->p_stFVValue)) {
        std::cerr << "Error: Failed to set float vector property for vertex '" << p_clVertex->GetName() << "'." << std::endl;
        return false;
      }
      break;
    default:
      std::cerr << "Error: Unrecognized property type (" << p_stKvp->p_stValue->eType << ") for vertex '" << p_clVertex->GetName() << "'. Graph not instantiated?" << std::endl;
      return false;
    }
  }

  return true;
}

template<typename RealType>
bool InstantiateGraph(const std::shared_ptr<Graph<RealType>> &p_clGraph, const std::shared_ptr<bleak_graph> &p_stGraphAst, const std::string &strPrefix = std::string()) {
  typedef Graph<RealType> GraphType;
  typedef Vertex<RealType> VertexType;

  if(p_clGraph == nullptr || p_stGraphAst == nullptr)
    return false;

  VertexAstFactory<RealType> clFactory(p_stGraphAst);

  if (!clFactory.Good())
    return false;

  std::unordered_map<std::string,std::shared_ptr<VertexType>> mVertexMap;

  for (bleak_vertex *p_stVertex : *p_stGraphAst->p_stVertices) {
    std::shared_ptr<VertexType> p_clVertex = clFactory.Create(p_stVertex->p_cType);

    if (!p_clVertex) {
      std::cerr << "Error: Failed to create vertex of type '" << p_stVertex->p_cType << "'." << std::endl;
      return false;
    }

    const std::string strName = ProperVertexName(p_stVertex->p_cName, strPrefix);

    p_clVertex->SetName(strName);

    if (!mVertexMap.emplace(p_clVertex->GetName(), p_clVertex).second) {
      std::cerr << "Error: Found duplicate vertex with name '" << p_clVertex->GetName() << "'." << std::endl;
      return false;
    }

    if (!AssignProperties(p_clVertex, p_stVertex))
      return false;
  }

  // Setup inner connections
  for (bleak_connection *p_stConnection : *p_stGraphAst->p_stConnections) {
    const std::string strSourceName = ProperVertexName(p_stConnection->p_cSourceName, strPrefix);
    const std::string strTargetName = ProperVertexName(p_stConnection->p_cTargetName, strPrefix);

    if (strSourceName == "this" || strTargetName == "this") // Handle this separately
      continue;

    std::shared_ptr<VertexType> p_clSource;
    std::shared_ptr<VertexType> p_clTarget;

    auto sourceItr = mVertexMap.find(strSourceName);
    auto targetItr = mVertexMap.find(strTargetName);

    if (sourceItr != mVertexMap.end())
      p_clSource = sourceItr->second;

    if (targetItr != mVertexMap.end())
      p_clTarget = targetItr->second;

    if (p_clSource == nullptr || p_clTarget == nullptr || !p_clSource->HasOutput(p_stConnection->p_cOutputName) || !p_clTarget->HasInput(p_stConnection->p_cInputName)) {
      std::cerr << "Error: Invalid input or output." << std::endl;
      std::cerr << "Error: Connection is: " << p_stConnection->p_cSourceName << '.' << p_stConnection->p_cOutputName << " -> " <<
        p_stConnection->p_cTargetName << '.' << p_stConnection->p_cInputName << std::endl;

      return false;
    }

    p_clTarget->SetInput(p_stConnection->p_cInputName,p_clSource->GetOutput(p_stConnection->p_cOutputName));
  }

  for (const auto &clPairType : mVertexMap)
    p_clGraph->AddVertex(clPairType.second);

  return true;
}

template<typename RealType>
bool ExpandSubgraph(const std::shared_ptr<Graph<RealType>> &p_clGraph, const std::shared_ptr<Subgraph<RealType>> &p_clSubgraph) {
  typedef Graph<RealType> GraphType;
  typedef Vertex<RealType> VertexType;
  typedef typename VertexType::EdgeType EdgeType;

  if (p_clGraph == nullptr || p_clSubgraph == nullptr || !p_clGraph->HasVertex(p_clSubgraph))
    return false;

  const std::shared_ptr<bleak_graph> &p_stSubgraphAst = p_clSubgraph->GetGraphAst();

  // Resolve subvertex variables!
  // XXX: Let Subgraph instantiate and own the bleak_graph to prevent stale pointers.
  const std::shared_ptr<bleak_graph> &p_stNewSubgraph = p_clSubgraph->InstantiateGraph();

  if (p_stNewSubgraph == nullptr)
    return false;

  const std::string strPrefix = p_clSubgraph->GetName();

  if (!InstantiateGraph(p_clGraph, p_stNewSubgraph, strPrefix)) {
    std::cerr << "Error: Could not instantiate graph." << std::endl;
    return false;
  }

  VertexAstFactory<RealType> clFactory(p_stSubgraphAst);

  // Create vertex map
  std::unordered_map<std::string,std::shared_ptr<VertexType>> mVertexMap;

  for (const auto &p_clVertex : p_clGraph->GetAllVertices()) {
    if (!mVertexMap.emplace(p_clVertex->GetName(), p_clVertex).second) {
      std::cerr << "Error: Duplicate vertex '" << p_clVertex->GetName() << "' found." << std::endl;
      return false;
    }
  }

  // Map outer connections to subgraph
  for (bleak_connection *p_stConnection : *p_stNewSubgraph->p_stConnections) {
    const std::string strSourceName = ProperVertexName(p_stConnection->p_cSourceName, strPrefix);
    const std::string strTargetName = ProperVertexName(p_stConnection->p_cTargetName, strPrefix);

    if (strSourceName != "this" && strTargetName != "this")
      continue;

    // XXX: Recurrence might be affected by this!

    // At least one has to be "this"
    if (strTargetName != "this") {
      auto itr = mVertexMap.find(strTargetName);
      std::shared_ptr<VertexType> p_clTarget;

      if (itr != mVertexMap.end())
        p_clTarget = itr->second;

      if (p_clTarget == nullptr || !p_clSubgraph->HasInput(p_stConnection->p_cOutputName) || !p_clTarget->HasInput(p_stConnection->p_cInputName)) {
        std::cerr << "Error: Failed to find source/target vertex." << std::endl;
        std::cerr << "Error: Connection is: " << strSourceName << '.' << p_stConnection->p_cOutputName << " -> " <<
          strTargetName << '.' << p_stConnection->p_cInputName << std::endl;
        return false;
      }

      // Totally weird!
      p_clTarget->SetInput(p_stConnection->p_cInputName, p_clSubgraph->GetInput(p_stConnection->p_cOutputName));
    } 
    else if (strSourceName != "this") {
      auto itr = mVertexMap.find(strSourceName);
      std::shared_ptr<VertexType> p_clSource;

      if (itr != mVertexMap.end())
        p_clSource = itr->second;

      if (p_clSource == nullptr || !p_clSubgraph->HasOutput(p_stConnection->p_cInputName) || !p_clSource->HasOutput(p_stConnection->p_cOutputName)) {
        std::cerr << "Error: Failed to find source/target vertex." << std::endl;
        std::cerr << "Error: Connection is: " << strSourceName << '.' << p_stConnection->p_cOutputName << " -> " <<
          strTargetName << '.' << p_stConnection->p_cInputName << std::endl;
        return false;
      }

      std::shared_ptr<EdgeType> p_clEdge = p_clSubgraph->GetOutput(p_stConnection->p_cInputName);

      // Collect targets first since replacing inputs can change p_clEdge
      std::vector<std::shared_ptr<VertexType>> vTargets;

      for (const auto &clTargetPair : p_clEdge->GetAllTargets()) {
        std::shared_ptr<VertexType> p_clTarget = clTargetPair.first.lock();

        if (p_clTarget != nullptr)
          vTargets.push_back(p_clTarget);
      }

      // Now process targets...

      for (const auto &p_clTarget : vTargets) {
        // Could be multiple inputs with this edge
        std::string strInputName;
        while ((strInputName = p_clTarget->FindInputName(p_clEdge)).size() > 0) {
          p_clTarget->SetInput(strInputName, p_clSource->GetOutput(p_stConnection->p_cOutputName));
        }
      }
    }
    else {
      // Pass through (this -> this)
      if (!p_clSubgraph->HasOutput(p_stConnection->p_cInputName) || !p_clSubgraph->HasInput(p_stConnection->p_cOutputName)) {
        std::cerr << "Error: Failed to find source/target vertex." << std::endl;
        std::cerr << "Error: Connection is: " << strSourceName << '.' << p_stConnection->p_cOutputName << " -> " <<
          strTargetName << '.' << p_stConnection->p_cInputName << std::endl;
        return false;
      }

      std::shared_ptr<EdgeType> p_clEdge = p_clSubgraph->GetOutput(p_stConnection->p_cInputName);

      // Collect targets first since replacing inputs can change p_clEdge
      std::vector<std::shared_ptr<VertexType>> vTargets;

      for (const auto &clTargetPair : p_clEdge->GetAllTargets()) {
        std::shared_ptr<VertexType> p_clTarget = clTargetPair.first.lock();

        if (p_clTarget != nullptr)
          vTargets.push_back(p_clTarget);
      }

      // Now process targets...

      for (const auto &p_clTarget : vTargets) {
        // Could be multiple inputs with this edge
        std::string strInputName;
        while((strInputName = p_clTarget->FindInputName(p_clEdge)).size() > 0) {
          p_clTarget->SetInput(strInputName, p_clSubgraph->GetInput(p_stConnection->p_cOutputName));
        }
      }
    }
  }

  p_clGraph->RemoveVertex(p_clSubgraph);

  return true;
}

template<typename RealType>
std::shared_ptr<Graph<RealType>> LoadGraph(const std::string &strFileName, const std::vector<std::string> &vSearchDirs = std::vector<std::string>()) {
  typedef Graph<RealType> GraphType;
  typedef Vertex<RealType> VertexType;

  std::shared_ptr<bleak_graph> p_stGraphAst = LoadGraphAst(strFileName, vSearchDirs);

  if (p_stGraphAst == nullptr)
    return std::shared_ptr<GraphType>();

  std::shared_ptr<GraphType> p_clGraph = std::make_shared<GraphType>();

  if (!InstantiateGraph(p_clGraph, p_stGraphAst)) {
    std::cerr << "Error: Could not instantiate graph." << std::endl;
    return std::shared_ptr<GraphType>();
  }

  // Expand subgraphs. Keep discarded subgraphs to allow variables in parent scopes to be resolved... Otherwise parent pointers could be stale!

  std::vector<std::shared_ptr<Subgraph<RealType>>> vSubgraphs;
  size_t i = 0;

  do {
    // Expand subgraphs
    for ( ; i < vSubgraphs.size(); ++i) {
      const std::shared_ptr<Subgraph<RealType>> &p_clSubgraph = vSubgraphs[i];

      if (!ExpandSubgraph(p_clGraph, p_clSubgraph)) {
        std::cerr << "Error: Could not expand subgraph '" << p_clSubgraph->GetName() << "'." << std::endl;
        return std::shared_ptr<GraphType>();
      }
    }

    // Collect subgraphs
    for (const auto &p_clVertex : p_clGraph->GetAllVertices()) {
      std::shared_ptr<Subgraph<RealType>> p_clSubgraph = std::dynamic_pointer_cast<Subgraph<RealType>>(p_clVertex);

      if (p_clSubgraph != nullptr)
        vSubgraphs.push_back(p_clSubgraph);
    }
  } while (i < vSubgraphs.size());

  return p_clGraph;
}

} // end namespace bleak


#endif // !BLEAK_SADLOADER_H
