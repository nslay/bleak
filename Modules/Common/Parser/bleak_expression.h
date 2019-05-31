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

#ifndef BLEAK_EXPRESSION_H
#define BLEAK_EXPRESSION_H

#include "bleak_graph_ast.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/* Unary functions */
bleak_value * bleak_expr_negate(bleak_value *p_stA);

/* Binary functions */
bleak_value * bleak_expr_plus(bleak_value *p_stA, bleak_value *p_stB);
bleak_value * bleak_expr_minus(bleak_value *p_stA, bleak_value *p_stB);
bleak_value * bleak_expr_divide(bleak_value *p_stA, bleak_value *p_stB);
bleak_value * bleak_expr_multiply(bleak_value *p_stA, bleak_value *p_stB);
bleak_value * bleak_expr_pow(bleak_value *p_stA, bleak_value *p_stB);
bleak_value * bleak_expr_modulo(bleak_value *p_stA, bleak_value *p_stB);

/* Evaluate non-private global variables in this graph */
bleak_graph * bleak_expr_graph_resolve_variables(bleak_graph *p_stGraph, int iEvaluatePrivate);

/* Evaluate all expressions in a graph and return a new graph */
bleak_graph * bleak_expr_graph_instantiate(bleak_graph *p_stGraph);

/* Dedup and evaluate KVPs */
bleak_vector_kvp * bleak_expr_kvp_dedup(bleak_vector_kvp *p_stKvps, bleak_graph *p_stParent, int iEvaluatePrivate);

/* Evaluate a value and return a new simple_value or int/float vector */
bleak_value * bleak_expr_evaluate(bleak_value *p_stValue, bleak_graph *p_stParent, int iAllowPrivateAccess);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* !BLEAK_EXPRESSION_H */

