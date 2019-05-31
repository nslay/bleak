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

#include "Common.h"
#include "SadLoader.h"
#include "Parser/bleak_vector.h"
#include "Parser/bleak_expression.h"
#include "Parser/bleak_parser.h"

namespace bleak {

std::shared_ptr<bleak_graph> LoadGraphAst(const std::string &strFileName, const std::vector<std::string> &vSearchDirs) {
  std::vector<const char *> vTmpSearchDirs(vSearchDirs.size()+1, nullptr);

  for(size_t i = 0; i < vSearchDirs.size(); ++i)
    vTmpSearchDirs[i] = vSearchDirs[i].c_str();

  bleak_parser *p_stParser = bleak_parser_alloc();

  if (p_stParser == nullptr)
    return std::shared_ptr<bleak_graph>();

  p_stParser->p_cIncludeDirs = vTmpSearchDirs.data();

  bleak_graph *p_stGraph = bleak_parser_load_graph(p_stParser, strFileName.c_str());

  bleak_parser_free(p_stParser);

  if (p_stGraph == nullptr)
    return std::shared_ptr<bleak_graph>();

  bleak_graph_link_parents(p_stGraph);
  bleak_graph *p_stNewGraph = bleak_expr_graph_instantiate(p_stGraph);

  bleak_graph_free(p_stGraph);

  if(p_stNewGraph == nullptr)
    return std::shared_ptr<bleak_graph>();

  std::shared_ptr<bleak_graph> p_stGraphAst(p_stNewGraph, &bleak_graph_free);

  return p_stGraphAst;
}

template class VertexAstFactory<float>;
template class VertexAstFactory<double>;

template bool AssignProperties<float>(const std::shared_ptr<Vertex<float>> &, bleak_vertex *);
template bool AssignProperties<double>(const std::shared_ptr<Vertex<double>> &, bleak_vertex *);

template bool InstantiateGraph<float>(const std::shared_ptr<Graph<float>> &, const std::shared_ptr<bleak_graph> &, const std::string &);
template bool InstantiateGraph<double>(const std::shared_ptr<Graph<double>> &, const std::shared_ptr<bleak_graph> &, const std::string &);

template bool ExpandSubgraph<float>(const std::shared_ptr<Graph<float>> &, const std::shared_ptr<Subgraph<float>> &);
template bool ExpandSubgraph<double>(const std::shared_ptr<Graph<double>> &, const std::shared_ptr<Subgraph<double>> &);

template std::shared_ptr<Graph<float>> LoadGraph<float>(const std::string &, const std::vector<std::string> &);
template std::shared_ptr<Graph<double>> LoadGraph<double>(const std::string &, const std::vector<std::string> &);

} // end namespace bleak
