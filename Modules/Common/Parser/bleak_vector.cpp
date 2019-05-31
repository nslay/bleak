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

#include <algorithm>
#include <vector>
#include "bleak_graph_ast.h"
#include "bleak_vector.h"

#define DEFINE_WRAPPED_VECTOR(vectorType, valueType) \
vectorType * vectorType##_alloc() { \
  try { \
    return new ( vectorType )(); \
  } \
  catch (...) { \
    return NULL; \
  } \
  return NULL; \
} \
void vectorType##_free( vectorType *p_v) { \
  delete p_v; \
} \
valueType * vectorType##_data( vectorType *p_v) { \
  return p_v != NULL ? p_v->data() : NULL; \
} \
int vectorType##_push_back( vectorType *p_v, valueType value) { \
  if (p_v == NULL) \
    return -1; \
  try { \
    p_v->push_back(value); \
  } \
  catch (...) { \
    return -1; \
  } \
  return 0; \
} \
size_t vectorType##_size( vectorType *p_v) { \
  return p_v != NULL ? p_v->size() : 0; \
} \
void vectorType##_clear( vectorType *p_v) { \
  if (p_v != NULL) \
    p_v->clear(); \
} \
vectorType * vectorType##_dup( vectorType *p_v ) { \
  if (p_v == NULL) \
    return NULL; \
  try { \
    return new ( vectorType )( *p_v ); \
  } \
  catch (...) { \
    return NULL; \
  } \
  return NULL; \
} \
void vectorType##_for_each( vectorType *p_v, void (*func)( valueType )) { \
  if (p_v != NULL) \
    std::for_each(p_v->begin(), p_v->end(), func); \
} \
void vectorType##_transform( vectorType *p_v, valueType (*func)( valueType )) { \
  if (p_v != NULL) \
    std::transform(p_v->begin(), p_v->end(), p_v->begin(), func); \
} \
int vectorType##_any_of( vectorType *p_v, int (*func)( valueType )) { \
  if (p_v != NULL) \
    return std::any_of(p_v->begin(), p_v->end(), func) ? 1 : 0; \
  return 0; \
} \
int vectorType##_join( vectorType *p_v, vectorType *p_vOther) { \
  if (p_v == NULL || p_vOther == NULL) \
    return -1; \
  try { \
    p_v->insert(p_v->end(), p_vOther->begin(), p_vOther->end()); \
  } \
  catch (...) { \
    return -1; \
  } \
  return 0; \
}

DEFINE_WRAPPED_VECTOR(bleak_vector_int, int);
DEFINE_WRAPPED_VECTOR(bleak_vector_float, float);
DEFINE_WRAPPED_VECTOR(bleak_vector_kvp, bleak_key_value_pair *);
DEFINE_WRAPPED_VECTOR(bleak_vector_connection, bleak_connection *);
DEFINE_WRAPPED_VECTOR(bleak_vector_vertex, bleak_vertex *);
DEFINE_WRAPPED_VECTOR(bleak_vector_value, bleak_value *);
DEFINE_WRAPPED_VECTOR(bleak_vector_graph, bleak_graph *);

bleak_vector_int * bleak_vector_value_to_vector_int(bleak_vector_value *p_v) {
  if (p_v == NULL)
    return NULL;

  for (bleak_value *p_stValue : *p_v) {
    if (p_stValue->eType != BLEAK_INTEGER)
      return NULL;
  }

  bleak_vector_int *p_vInt = bleak_vector_int_alloc();

  if (p_vInt == NULL)
    return NULL;

  try {
    p_vInt->resize(p_v->size());
  }
  catch (...) {
    bleak_vector_int_free(p_vInt);
    return NULL;
  }

  std::transform(p_v->begin(), p_v->end(), p_vInt->begin(), 
    [](bleak_value *p_stValue) -> int {
      return p_stValue->iValue;
    });

  return p_vInt;
}

bleak_vector_float * bleak_vector_value_to_vector_float(bleak_vector_value *p_v) {
  if (p_v == NULL)
    return NULL;

  for (bleak_value *p_stValue : *p_v) {
    if (p_stValue->eType != BLEAK_INTEGER && p_stValue->eType != BLEAK_FLOAT)
      return NULL;
  }

  bleak_vector_float *p_vFloat = bleak_vector_float_alloc();

  if (p_vFloat == NULL)
    return NULL;

  try {
    p_vFloat->resize(p_v->size());
  }
  catch (...) {
    bleak_vector_float_free(p_vFloat);
    return NULL;
  }

  std::transform(p_v->begin(), p_v->end(), p_vFloat->begin(), 
    [](bleak_value *p_stValue) -> float {
      if (p_stValue->eType == BLEAK_INTEGER)
        return (float)p_stValue->iValue;

      return p_stValue->fValue;
    });

  return p_vFloat;
}

