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

#ifndef BLEAK_GRAPH_AST_H
#define BLEAK_GRAPH_AST_H

#include "bleak_vector.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/* NOTE: Just to be clear, everything stored in these structures are expected to be allocated! */

enum bleak_data_type_ { BLEAK_INTEGER = 0, BLEAK_FLOAT, BLEAK_BOOL, BLEAK_STRING, BLEAK_INTEGER_VECTOR, BLEAK_FLOAT_VECTOR,
/* Unresolved expression types below. These are not meant to be used directly! */
BLEAK_REFERENCE, BLEAK_VALUE_VECTOR, BLEAK_EXPRESSION, BLEAK_OPCODE
};

typedef enum bleak_data_type_ bleak_data_type;

typedef struct bleak_value_ {
  bleak_data_type eType;
  union {
    int iValue;
    float fValue;
    char *p_cValue;
    int bValue; /* _Bool seemingly not supported by VS 2017 */
    bleak_vector_int *p_stIVValue;
    bleak_vector_float *p_stFVValue;
    bleak_vector_value *p_stValues;
  };
} bleak_value;

bleak_value * bleak_value_alloc();
void bleak_value_free(bleak_value *p_stValue);
bleak_value * bleak_value_dup(bleak_value *p_stValue);
int bleak_value_is_null(bleak_value *p_stValue);

typedef struct bleak_key_value_pair_ {
  char *p_cKey;
  bleak_value *p_stValue;
  int iPrivate;
} bleak_key_value_pair;

bleak_key_value_pair * bleak_kvp_alloc();
void bleak_kvp_free(bleak_key_value_pair *p_stKvp);
bleak_key_value_pair * bleak_kvp_dup(bleak_key_value_pair *p_stKvp);
int bleak_kvp_is_null(bleak_key_value_pair *p_stKvp);

typedef struct bleak_connection_ {
  char *p_cSourceName, *p_cTargetName;
  char *p_cOutputName, *p_cInputName;
} bleak_connection;

bleak_connection * bleak_connection_alloc();
void bleak_connection_free(bleak_connection *p_stConnection);
bleak_connection * bleak_connection_dup(bleak_connection *p_stConnection);
int bleak_connection_is_null(bleak_connection *p_stConnection);

typedef struct bleak_vertex_ {
  char *p_cType;
  char *p_cName;
  bleak_vector_kvp *p_stProperties;
} bleak_vertex;

bleak_vertex * bleak_vertex_alloc();
void bleak_vertex_free(bleak_vertex *p_stVertex);
bleak_vertex * bleak_vertex_dup(bleak_vertex *p_stVertex);
int bleak_vertex_is_null(bleak_vertex *p_stVertex);

typedef struct bleak_graph_ {
  char *p_cName;
  bleak_graph *p_stParent;
  bleak_vector_kvp *p_stVariables;
  bleak_vector_graph *p_stSubgraphs;
  bleak_vector_vertex *p_stVertices;
  bleak_vector_connection *p_stConnections;
} bleak_graph;

bleak_graph * bleak_graph_alloc();
void bleak_graph_free(bleak_graph *p_stGraph);
bleak_graph * bleak_graph_dup(bleak_graph *p_stGraph);
int bleak_graph_is_null(bleak_graph *p_stGraph);
void bleak_graph_link_parents(bleak_graph *p_stGraph);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* !BLEAK_GRAPH_AST_H */

