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

#ifndef BLEAK_VECTOR_H
#define BLEAK_VECTOR_H

#include <stddef.h>

typedef struct bleak_key_value_pair_ bleak_key_value_pair;
typedef struct bleak_vertex_ bleak_vertex;
typedef struct bleak_connection_ bleak_connection;
typedef struct bleak_value_ bleak_value;
typedef struct bleak_graph_ bleak_graph;

#ifdef __cplusplus
#include <vector>

#define DECLARE_CPP_WRAPPED_VECTOR(vectorType, valueType) \
struct vectorType##_ : public std::vector< valueType > { };

DECLARE_CPP_WRAPPED_VECTOR(bleak_vector_int, int);
DECLARE_CPP_WRAPPED_VECTOR(bleak_vector_float, float);
DECLARE_CPP_WRAPPED_VECTOR(bleak_vector_kvp, bleak_key_value_pair *);
DECLARE_CPP_WRAPPED_VECTOR(bleak_vector_connection, bleak_connection *);
DECLARE_CPP_WRAPPED_VECTOR(bleak_vector_vertex, bleak_vertex *);
DECLARE_CPP_WRAPPED_VECTOR(bleak_vector_value, bleak_value *);
DECLARE_CPP_WRAPPED_VECTOR(bleak_vector_graph, bleak_graph *);

#undef DECLARE_CPP_WRAPPED_VECTOR

extern "C" {
#endif /* __cplusplus */

#define DECLARE_WRAPPED_VECTOR(vectorType, valueType) \
typedef struct vectorType##_ vectorType; \
vectorType * vectorType##_alloc(); \
void vectorType##_free( vectorType *p_v); \
valueType * vectorType##_data( vectorType *p_v); \
int vectorType##_push_back( vectorType *p_v, valueType value); \
size_t vectorType##_size( vectorType *p_v); \
void vectorType##_clear( vectorType *p_v); \
vectorType * vectorType##_dup( vectorType *p_v ); \
void vectorType##_for_each( vectorType *p_v, void (*func)( valueType )); \
void vectorType##_transform( vectorType *p_v, valueType (*func)( valueType )); \
int vectorType##_any_of( vectorType *p_v, int (*func)( valueType )); \
int vectorType##_join( vectorType *p_v, vectorType *p_vOther)

DECLARE_WRAPPED_VECTOR(bleak_vector_int, int);
DECLARE_WRAPPED_VECTOR(bleak_vector_float, float);
DECLARE_WRAPPED_VECTOR(bleak_vector_kvp, bleak_key_value_pair *);
DECLARE_WRAPPED_VECTOR(bleak_vector_connection, bleak_connection *);
DECLARE_WRAPPED_VECTOR(bleak_vector_vertex, bleak_vertex *);
DECLARE_WRAPPED_VECTOR(bleak_vector_value, bleak_value *);
DECLARE_WRAPPED_VECTOR(bleak_vector_graph, bleak_graph *);

#undef DECLARE_WRAPPED_VECTOR

bleak_vector_int * bleak_vector_value_to_vector_int(bleak_vector_value *p_v);
bleak_vector_float * bleak_vector_value_to_vector_float(bleak_vector_value *p_v);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* !BLEAK_VECTOR_H */

