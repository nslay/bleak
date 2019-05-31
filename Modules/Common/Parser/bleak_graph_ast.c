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

#include <stdlib.h>
#include <string.h>
#include "bleak_graph_ast.h"

bleak_value * bleak_value_alloc() {
  return calloc(1, sizeof(bleak_value));
}

void bleak_value_free(bleak_value *p_stValue) {
  if (p_stValue == NULL)
    return;

  switch (p_stValue->eType) {
  case BLEAK_INTEGER:
  case BLEAK_FLOAT:
  case BLEAK_BOOL:
  case BLEAK_OPCODE:
    break;
  case BLEAK_STRING:
  case BLEAK_REFERENCE:
    free(p_stValue->p_cValue);
    break;    
  case BLEAK_INTEGER_VECTOR:
    bleak_vector_int_free(p_stValue->p_stIVValue);
    break;
  case BLEAK_FLOAT_VECTOR:
    bleak_vector_float_free(p_stValue->p_stFVValue);
    break;
  case BLEAK_VALUE_VECTOR:
  case BLEAK_EXPRESSION:
    bleak_vector_value_for_each(p_stValue->p_stValues, &bleak_value_free);
    bleak_vector_value_free(p_stValue->p_stValues);
    break;
  }

  free(p_stValue);
}

bleak_key_value_pair * bleak_kvp_alloc() {
  return calloc(1, sizeof(bleak_key_value_pair));
}

bleak_value * bleak_value_dup(bleak_value *p_stValue) {
  bleak_value *p_stNewValue = NULL;

  if (p_stValue == NULL)
    return NULL;

  p_stNewValue = bleak_value_alloc();

  if (p_stNewValue == NULL)
    return NULL;

  p_stNewValue->eType = p_stValue->eType;

  switch (p_stValue->eType) {
  case BLEAK_INTEGER:
  case BLEAK_OPCODE:
    p_stNewValue->iValue = p_stValue->iValue;
    break;
  case BLEAK_FLOAT:
    p_stNewValue->fValue = p_stValue->fValue;
    break;
  case BLEAK_BOOL:
    p_stNewValue->bValue = p_stValue->bValue;
    break;
  case BLEAK_STRING:
  case BLEAK_REFERENCE:
    p_stNewValue->p_cValue = strdup(p_stValue->p_cValue);
    if (p_stNewValue->p_cValue == NULL) {
      bleak_value_free(p_stNewValue);
      return NULL;
    }
    break;
  case BLEAK_INTEGER_VECTOR:
    p_stNewValue->p_stIVValue = bleak_vector_int_dup(p_stValue->p_stIVValue);
    if (p_stNewValue->p_stIVValue == NULL) {
      bleak_value_free(p_stNewValue);
      return NULL;
    }
    break;
  case BLEAK_FLOAT_VECTOR:
    p_stNewValue->p_stFVValue = bleak_vector_float_dup(p_stValue->p_stFVValue);
    if (p_stNewValue->p_stFVValue == NULL) {
      bleak_value_free(p_stNewValue);
      return NULL;
    }
    break;
  case BLEAK_VALUE_VECTOR:
  case BLEAK_EXPRESSION:
    p_stNewValue->p_stValues = bleak_vector_value_dup(p_stValue->p_stValues);
    if (p_stNewValue->p_stValues == NULL) {
      bleak_value_free(p_stNewValue);
      return NULL;
    }

    /* Now dup the individual values! */
    bleak_vector_value_transform(p_stNewValue->p_stValues, &bleak_value_dup);

    if (bleak_vector_value_any_of(p_stNewValue->p_stValues, &bleak_value_is_null)) {
      bleak_value_free(p_stNewValue);
      return NULL;
    }

    break;
  }

  return p_stNewValue;
}

int bleak_value_is_null(bleak_value *p_stValue) {
  return (p_stValue == NULL);
}

void bleak_kvp_free(bleak_key_value_pair *p_stKvp) {
  if (p_stKvp == NULL)
    return;

  free(p_stKvp->p_cKey);
  bleak_value_free(p_stKvp->p_stValue);

  free(p_stKvp);
}

bleak_key_value_pair * bleak_kvp_dup(bleak_key_value_pair *p_stKvp) {
  bleak_key_value_pair *p_stNewKvp = NULL;

  if (p_stKvp == NULL)
    return NULL;

  p_stNewKvp = bleak_kvp_alloc();

  if (p_stNewKvp == NULL)
    return NULL;

  p_stNewKvp->p_cKey = strdup(p_stKvp->p_cKey);
  p_stNewKvp->p_stValue = bleak_value_dup(p_stKvp->p_stValue);
  p_stNewKvp->iPrivate = p_stKvp->iPrivate;

  if (p_stNewKvp->p_cKey == NULL || p_stNewKvp->p_stValue == NULL) {
    bleak_kvp_free(p_stNewKvp);
    return NULL;
  }

  return p_stNewKvp;
}

int bleak_kvp_is_null(bleak_key_value_pair *p_stKvp) {
  return (p_stKvp == NULL);
}

bleak_connection * bleak_connection_alloc() {
  return calloc(1, sizeof(bleak_connection));
}

void bleak_connection_free(bleak_connection *p_stConnection) {
  if (p_stConnection == NULL)
    return;

  free(p_stConnection->p_cSourceName);
  free(p_stConnection->p_cTargetName);

  free(p_stConnection->p_cOutputName);
  free(p_stConnection->p_cInputName);

  free(p_stConnection);
}

bleak_connection * bleak_connection_dup(bleak_connection *p_stConnection) {
  bleak_connection *p_stNewConnection = NULL;

  if (p_stConnection == NULL)
    return NULL;

  p_stNewConnection = bleak_connection_alloc();

  if (p_stNewConnection == NULL)
    return NULL;

  p_stNewConnection->p_cSourceName = strdup(p_stConnection->p_cSourceName);
  p_stNewConnection->p_cTargetName = strdup(p_stConnection->p_cTargetName);
  p_stNewConnection->p_cOutputName = strdup(p_stConnection->p_cOutputName);
  p_stNewConnection->p_cInputName = strdup(p_stConnection->p_cInputName);

  if (p_stNewConnection->p_cSourceName == NULL || p_stNewConnection->p_cTargetName == NULL || 
    p_stNewConnection->p_cOutputName == NULL || p_stNewConnection->p_cInputName == NULL) {
    bleak_connection_free(p_stNewConnection);
    return NULL;
  }

  return p_stNewConnection;
}

int bleak_connection_is_null(bleak_connection *p_stConnection) {
  return (p_stConnection == NULL);
}

bleak_vertex * bleak_vertex_alloc() {
  return calloc(1, sizeof(bleak_vertex));
}

void bleak_vertex_free(bleak_vertex *p_stVertex) {
  if (p_stVertex == NULL)  
    return;

  free(p_stVertex->p_cType);
  free(p_stVertex->p_cName);

  bleak_vector_kvp_for_each(p_stVertex->p_stProperties, &bleak_kvp_free);
  bleak_vector_kvp_free(p_stVertex->p_stProperties);

  free(p_stVertex);
}

bleak_vertex * bleak_vertex_dup(bleak_vertex *p_stVertex) {
  bleak_vertex *p_stNewVertex = NULL;

  if (p_stVertex == NULL)
    return NULL;

  p_stNewVertex = bleak_vertex_alloc();

  if (p_stNewVertex == NULL)
    return NULL;

  p_stNewVertex->p_cType = strdup(p_stVertex->p_cType);
  p_stNewVertex->p_cName = strdup(p_stVertex->p_cName);

  if (p_stNewVertex->p_cType == NULL || p_stNewVertex->p_cName == NULL) {
    bleak_vertex_free(p_stNewVertex);
    return NULL;
  }

  p_stNewVertex->p_stProperties = bleak_vector_kvp_dup(p_stVertex->p_stProperties);

  if (p_stNewVertex == NULL) {
    bleak_vertex_free(p_stNewVertex);
    return NULL;
  }

  bleak_vector_kvp_transform(p_stNewVertex->p_stProperties, &bleak_kvp_dup);

  if (bleak_vector_kvp_any_of(p_stNewVertex->p_stProperties, &bleak_kvp_is_null)) {
    bleak_vertex_free(p_stNewVertex);
    return NULL;
  }

  return p_stNewVertex;
}

int bleak_vertex_is_null(bleak_vertex *p_stVertex) {
  return (p_stVertex == NULL);
}

bleak_graph * bleak_graph_alloc() {
  return calloc(1, sizeof(bleak_graph));
}

void bleak_graph_free(bleak_graph *p_stGraph) {
  if (p_stGraph == NULL)
    return;

  free(p_stGraph->p_cName);

  bleak_vector_kvp_for_each(p_stGraph->p_stVariables, &bleak_kvp_free);
  bleak_vector_kvp_free(p_stGraph->p_stVariables);

  bleak_vector_graph_for_each(p_stGraph->p_stSubgraphs, &bleak_graph_free);
  bleak_vector_graph_free(p_stGraph->p_stSubgraphs);

  bleak_vector_vertex_for_each(p_stGraph->p_stVertices, &bleak_vertex_free);
  bleak_vector_vertex_free(p_stGraph->p_stVertices);

  bleak_vector_connection_for_each(p_stGraph->p_stConnections, &bleak_connection_free);
  bleak_vector_connection_free(p_stGraph->p_stConnections);

  free(p_stGraph);
}

/* Relink parents of subgraphs */
void bleak_graph_link_parents(bleak_graph *p_stGraph) {
  bleak_graph **p_stSubgraph = NULL;
  size_t i, numSubgraphs = 0;

  if (p_stGraph == NULL)
    return;

  p_stSubgraph = bleak_vector_graph_data(p_stGraph->p_stSubgraphs);
  numSubgraphs = bleak_vector_graph_size(p_stGraph->p_stSubgraphs);

  for (i = 0; i < numSubgraphs; ++i) {
    p_stSubgraph[i]->p_stParent = p_stGraph;
    bleak_graph_link_parents(p_stSubgraph[i]);
  }
}

bleak_graph * bleak_graph_dup(bleak_graph *p_stGraph) {
  bleak_graph *p_stNewGraph = NULL;

  if (p_stGraph == NULL)
    return NULL;

  p_stNewGraph = bleak_graph_alloc();

  if (p_stNewGraph == NULL)
    return NULL;

  p_stNewGraph->p_stParent = p_stGraph->p_stParent;

  if (p_stGraph->p_cName != NULL) {
    p_stNewGraph->p_cName = strdup(p_stGraph->p_cName);

    if (p_stNewGraph->p_cName == NULL) {
      bleak_graph_free(p_stNewGraph);
      return NULL;
    }
  }

  /* Dup variables */
  p_stNewGraph->p_stVariables = bleak_vector_kvp_dup(p_stGraph->p_stVariables);

  if (p_stNewGraph->p_stVariables == NULL) {
    bleak_graph_free(p_stNewGraph);
    return NULL;
  }

  bleak_vector_kvp_transform(p_stNewGraph->p_stVariables, &bleak_kvp_dup);

  if (bleak_vector_kvp_any_of(p_stNewGraph->p_stVariables, &bleak_kvp_is_null)) {
    bleak_graph_free(p_stNewGraph);
    return NULL;
  }

  /* Dup subgraphs */
  p_stNewGraph->p_stSubgraphs = bleak_vector_graph_dup(p_stGraph->p_stSubgraphs);

  if (p_stNewGraph->p_stSubgraphs == NULL) {
    bleak_graph_free(p_stNewGraph);
    return NULL;
  }

  bleak_vector_graph_transform(p_stNewGraph->p_stSubgraphs, &bleak_graph_dup);

  if (bleak_vector_graph_any_of(p_stNewGraph->p_stSubgraphs, &bleak_graph_is_null)) {
    bleak_graph_free(p_stNewGraph);
    return NULL;
  }

  /* Dup vertices */
  p_stNewGraph->p_stVertices = bleak_vector_vertex_dup(p_stGraph->p_stVertices);

  if (p_stNewGraph->p_stVertices == NULL) {
    bleak_graph_free(p_stNewGraph);
    return NULL;
  }

  bleak_vector_vertex_transform(p_stNewGraph->p_stVertices, &bleak_vertex_dup);

  if (bleak_vector_vertex_any_of(p_stNewGraph->p_stVertices, &bleak_vertex_is_null)) {
    bleak_graph_free(p_stNewGraph);
    return NULL;
  }

  /* Dup connections */
  p_stNewGraph->p_stConnections = bleak_vector_connection_dup(p_stGraph->p_stConnections);

  if (p_stNewGraph->p_stConnections == NULL) {
    bleak_graph_free(p_stNewGraph);
    return NULL;
  }

  bleak_vector_connection_transform(p_stNewGraph->p_stConnections, &bleak_connection_dup);

  if (bleak_vector_connection_any_of(p_stNewGraph->p_stConnections, &bleak_connection_is_null)) {
    bleak_graph_free(p_stNewGraph);
    return NULL;
  }

  /* Relink parents of subgraphs */
  bleak_graph_link_parents(p_stNewGraph);

  return p_stNewGraph;
}

int bleak_graph_is_null(bleak_graph *p_stGraph) {
  return (p_stGraph == NULL);
}


