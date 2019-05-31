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

#include <cstdlib>
#include <cstring>
#include <memory>
#include "Vertex.h"
#include "Parser/bleak_graph_ast.h"
#include "Parser/bleak_expression.h"

namespace bleak {

template<typename RealType>
class Subgraph : public Vertex<RealType> {
public:
  bleakNewVertex(Subgraph, Vertex<RealType>); // Inputs/outputs/properties determined by bleak_graph

  bleakForwardVertexTypedefs();

  using SuperType::RegisterProperty;
  using SuperType::RegisterInput;
  using SuperType::RegisterOutput;

  virtual ~Subgraph() = default;

  virtual bool SetSizes() override {
    std::cerr << GetName() << ": Error: This vertex should never appear in a Graph!" << std::endl;
    return false;
  }

  virtual bool Initialize() override {
    std::cerr << GetName() << ": Error: This vertex should never appear in a Graph!" << std::endl;
    return false;
  }

  virtual void Forward() override { } // Nothing to do
  virtual void Backward() override { } // Nothing to do

  // Will instantiate on first call! Subsequent SetProperty() calls will do NOTHING to returned graph.
  const std::shared_ptr<bleak_graph> & InstantiateGraph() {
    const std::shared_ptr<bleak_graph> &p_stGraphAst = GetGraphAst();

    if (!p_stGraphAst)
      return m_p_stGraph;

    if (m_p_stGraph == nullptr)
      m_p_stGraph.reset(bleak_expr_graph_instantiate(p_stGraphAst.get()), &bleak_graph_free);

    return m_p_stGraph;
  }

  // IMPORTANT: Be sure to call this to map back changed string properties!!!!
  const std::shared_ptr<bleak_graph> & GetGraphAst() const {
    if (m_p_stGraphAst == nullptr)
      return m_p_stGraphAst;

    // Go through property table and copy back string values (lame!!!)

    std::string strValue;    

    for (bleak_key_value_pair *p_stKvp : *m_p_stGraphAst->p_stVariables) {
      // Private variables are ignored
      if (p_stKvp->iPrivate != 0)
        continue;

      bleak_value *p_stValue = p_stKvp->p_stValue;

      strValue.clear();

      if (p_stValue->eType == BLEAK_STRING && GetProperty(p_stKvp->p_cKey, strValue)) {
        char *p_cValue = strdup(strValue.c_str());

        if (p_cValue != nullptr) {
          free(p_stValue->p_cValue);
          p_stValue->p_cValue = p_cValue;
        }
      }
    }

    return m_p_stGraphAst;
  }

  // NOTE: This is a bleak_graph BEFORE instantiation
  bool SetGraphAst(const std::shared_ptr<bleak_graph> &p_stGraphAst) {
    if (p_stGraphAst == nullptr || m_p_stGraphAst != nullptr)
      return false;

    m_p_stGraphAst = p_stGraphAst;

    // Count string values so we can resize the string values vector once (no changing addresses!)
    size_t numStringValues = 0;
    for (bleak_key_value_pair *p_stKvp : *p_stGraphAst->p_stVariables) {
      // Private variables are ignored
      if (p_stKvp->iPrivate == 0) {
        bleak_value *p_stValue = p_stKvp->p_stValue;
        numStringValues += (p_stValue->eType == BLEAK_STRING);
      }
    }

    m_vStringValues.resize(numStringValues);

    size_t stringValueIndex = 0;

    // Setup properties
    for (bleak_key_value_pair *p_stKvp : *p_stGraphAst->p_stVariables) {
      // Private variables are ignored
      if (p_stKvp->iPrivate != 0)
        continue;

      if (strcmp(p_stKvp->p_cKey, "name") == 0) {
        std::cout << "Ignoring 'name' variable in subgraph." << std::endl;
        continue;
      }

      bleak_value *p_stValue = p_stKvp->p_stValue;

      switch (p_stValue->eType) {
      case BLEAK_INTEGER:
        RegisterProperty(p_stKvp->p_cKey, p_stValue->iValue);
        break;
      case BLEAK_FLOAT:
        RegisterProperty(p_stKvp->p_cKey, p_stValue->fValue);
        break;
      case BLEAK_STRING:
        m_vStringValues[stringValueIndex] = p_stValue->p_cValue;
        RegisterProperty(p_stKvp->p_cKey, m_vStringValues[stringValueIndex++]);
        break;
      case BLEAK_BOOL:
        RegisterProperty(p_stKvp->p_cKey, p_stValue->bValue);
        break;
      case BLEAK_INTEGER_VECTOR:
        RegisterProperty(p_stKvp->p_cKey, *p_stValue->p_stIVValue);
        break;
      case BLEAK_FLOAT_VECTOR:
        RegisterProperty(p_stKvp->p_cKey, *p_stValue->p_stFVValue);
        break;
      default:
        std::cerr << "Error: Unknown value type (" << (int)p_stValue->eType << ")." << std::endl;
        return false;
      }
    }

    // Setup inputs/outputs
    // Don't break it! Meanings are inverted in subgraphs!
    for (bleak_connection *p_stConnection : *p_stGraphAst->p_stConnections) {
      if (strcmp(p_stConnection->p_cSourceName, "this") == 0 && !HasInput(p_stConnection->p_cOutputName))
        RegisterInput(p_stConnection->p_cOutputName);

      if (strcmp(p_stConnection->p_cTargetName, "this") == 0 && !HasOutput(p_stConnection->p_cInputName))
        RegisterOutput(p_stConnection->p_cInputName);
    }

    return true;
  }

protected:
  Subgraph() = default;

private:
  // The AST is a "blue print", the latter is an instantiation and made a member to keep it alive with the vertex
  // This solves a complicated stale pointer problem in SadLoader.
  std::shared_ptr<bleak_graph> m_p_stGraphAst, m_p_stGraph; 
  std::vector<std::string> m_vStringValues;
};

} // end namespace bleak
