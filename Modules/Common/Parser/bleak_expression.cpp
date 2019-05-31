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
#include <cstring>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <functional>
#include <deque>
#include "bleak_expression.h"

namespace {

template<bleak_data_type BleakType>
struct BleakTypeToNativeType { };

template<typename NativeType>
struct NativeTypeToBleakType { };

#define DECLARE_TRAITS(bleakType, valueMember) \
template<> \
struct BleakTypeToNativeType< bleakType > { \
  typedef decltype(bleak_value:: valueMember ) ValueType; \
  constexpr static ValueType bleak_value::* const member = &bleak_value:: valueMember ; \
}; \
template<> \
struct NativeTypeToBleakType< decltype(bleak_value:: valueMember ) > { \
  constexpr static const bleak_data_type eType = bleakType ; \
}

DECLARE_TRAITS(BLEAK_INTEGER, iValue);
DECLARE_TRAITS(BLEAK_FLOAT, fValue);
DECLARE_TRAITS(BLEAK_STRING, p_cValue);

template<>
struct BleakTypeToNativeType<BLEAK_BOOL> : BleakTypeToNativeType<BLEAK_INTEGER> { };

template<>
struct NativeTypeToBleakType<long> : NativeTypeToBleakType<int> { };

#define DECLARE_BINARY_OPERATION(opName, op) \
template<bleak_data_type A, bleak_data_type B> \
struct opName { \
  static bleak_value * Compute(bleak_value *p_stA, bleak_value *p_stB) { \
    if (p_stA == NULL || p_stB == NULL || p_stA->eType != A || p_stB->eType != B) \
      return NULL; \
\
    bleak_value *p_stValue = bleak_value_alloc(); \
\
    if (p_stValue == NULL) \
      return NULL; \
\
    constexpr auto memA = BleakTypeToNativeType<A>::member; \
    constexpr auto memB = BleakTypeToNativeType<B>::member; \
\
    auto value = p_stA->*memA op p_stB->*memB; \
\
    constexpr bleak_data_type eReturnType = NativeTypeToBleakType<decltype(value)>::eType; \
    constexpr auto memC = BleakTypeToNativeType<eReturnType>::member; \
\
    p_stValue->eType = eReturnType; \
    p_stValue->*memC = value; \
\
    return p_stValue;\
  }\
}

#define DECLARE_UNARY_OPERATION(opName, op) \
template<bleak_data_type A> \
struct opName { \
  static bleak_value * Compute(bleak_value *p_stA) { \
    if (p_stA == NULL || p_stA->eType != A) \
      return NULL; \
\
    bleak_value *p_stValue = bleak_value_alloc(); \
\
    if (p_stValue == NULL) \
      return NULL; \
\
    constexpr auto memA = BleakTypeToNativeType<A>::member; \
\
    auto value = op (p_stA->*memA); \
\
    constexpr bleak_data_type eReturnType = NativeTypeToBleakType<decltype(value)>::eType; \
    constexpr auto memC = BleakTypeToNativeType<eReturnType>::member; \
\
    p_stValue->eType = eReturnType; \
    p_stValue->*memC = value; \
\
    return p_stValue; \
  } \
}

DECLARE_BINARY_OPERATION(Plus, +);
DECLARE_BINARY_OPERATION(Minus, -);
DECLARE_BINARY_OPERATION(Divide, /);
DECLARE_BINARY_OPERATION(Multiply, *);
DECLARE_BINARY_OPERATION(Modulo, %);

DECLARE_UNARY_OPERATION(Negate, -);

template<bleak_data_type A, bleak_data_type B>
struct Pow {
  static bleak_value * Compute(bleak_value *p_stA, bleak_value *p_stB) { 
    if (p_stA == NULL || p_stB == NULL || p_stA->eType != A || p_stB->eType != B) 
      return NULL; 

    bleak_value *p_stValue = bleak_value_alloc(); 

    if (p_stValue == NULL) 
      return NULL; 

    constexpr auto memA = BleakTypeToNativeType<A>::member; 
    constexpr auto memB = BleakTypeToNativeType<B>::member; 

    float fValue = (float)std::pow(p_stA->*memA, p_stB->*memB);

    p_stValue->eType = BLEAK_FLOAT;
    p_stValue->fValue = fValue;

    return p_stValue;
  }
};

////////////////////////////// Specializations Below //////////////////////////////

template<>
struct Plus<BLEAK_STRING, BLEAK_STRING> {
  static bleak_value * Compute(bleak_value *p_stA, bleak_value *p_stB) {
    if (p_stA == NULL || p_stB == NULL || p_stA->eType != BLEAK_STRING || p_stB->eType != BLEAK_STRING || p_stA->p_cValue == NULL || p_stB->p_cValue == NULL)
      return NULL;

    bleak_value *p_stValue = bleak_value_alloc();

    if (p_stValue == NULL)
      return NULL;

    char *p_cValue = (char *)malloc(strlen(p_stA->p_cValue) + strlen(p_stB->p_cValue) + 1);

    if (p_cValue == NULL) {
      bleak_value_free(p_stValue);
      return NULL;
    }

    strcpy(p_cValue, p_stA->p_cValue);
    strcat(p_cValue, p_stB->p_cValue);

    p_stValue->eType = BLEAK_STRING;
    p_stValue->p_cValue = p_cValue;

    return p_stValue;
  }
};

template<>
struct Pow<BLEAK_INTEGER, BLEAK_INTEGER> {
  static bleak_value * Compute(bleak_value *p_stA, bleak_value *p_stB) { 
    if (p_stA == NULL || p_stB == NULL || p_stA->eType != BLEAK_INTEGER || p_stB->eType != BLEAK_INTEGER) 
      return NULL; 

    bleak_value *p_stValue = bleak_value_alloc(); 

    if (p_stValue == NULL) 
      return NULL; 

    int iValue = (int)std::pow(p_stA->iValue, p_stB->iValue);

    p_stValue->eType = BLEAK_INTEGER;
    p_stValue->iValue = iValue;

    return p_stValue;
  }
};

////////////////////////////// Dispatchers //////////////////////////////

#define ALLOW_ARGS(typeA, typeB, opName) \
  forAllowArgsMacro_; \
  m_a_funTable[ typeA ][ typeB ] = & opName < typeA, typeB >::Compute; \
  forAllowArgsMacro_

#define ALLOW_ARG(typeA, opName) \
  forAllowArgsMacro_; \
  m_a_funTable[ typeA ] = & opName < typeA >::Compute; \
  forAllowArgsMacro_

#define ALLOW_ARGS_COMBO(typeA, typeB, opName) \
  forAllowArgsMacro_; \
  m_a_funTable[ typeA ][ typeB ] = & opName < typeA, typeB >::Compute; \
  m_a_funTable[ typeB ][ typeA ] = & opName < typeB, typeA >::Compute; \
  forAllowArgsMacro_

#define DECLARE_BINARY_DISPATCH(className, ...) \
class className { \
public: \
  typedef bleak_value * (*ComputeFunc)(bleak_value *, bleak_value *); \
\
  className () { \
    std::memset(m_a_funTable, 0, sizeof(m_a_funTable)); \
    int forAllowArgsMacro_ = 0; \
    __VA_ARGS__ ; \
  } \
\
  bleak_value * operator()(bleak_value *p_stA, bleak_value *p_stB) const { \
    if (p_stA == NULL || p_stB == NULL) \
      return NULL; \
\
    if (p_stA->eType < 0 || p_stA->eType >= m_s_iMaxTypes || p_stB->eType < 0 || p_stB->eType >= m_s_iMaxTypes) \
      return NULL; \
\
    ComputeFunc fun = m_a_funTable[p_stA->eType][p_stB->eType]; \
\
    return fun != NULL ? (*fun)(p_stA, p_stB) : NULL; \
  }\
\
private: \
  constexpr static const int m_s_iMaxTypes = 6; \
  ComputeFunc m_a_funTable[m_s_iMaxTypes][m_s_iMaxTypes]; \
}

#define DECLARE_UNARY_DISPATCH(className, ...) \
class className { \
public: \
  typedef bleak_value * (*ComputeFunc)(bleak_value *); \
\
  className () { \
    std::memset(m_a_funTable, 0, sizeof(m_a_funTable)); \
    int forAllowArgsMacro_ = 0; \
    __VA_ARGS__ ; \
  } \
\
  bleak_value * operator()(bleak_value *p_stA) const { \
    if (p_stA == NULL) \
      return NULL; \
\
    if (p_stA->eType < 0 || p_stA->eType >= m_s_iMaxTypes) \
      return NULL; \
\
    ComputeFunc fun = m_a_funTable[p_stA->eType]; \
\
    return fun != NULL ? (*fun)(p_stA) : NULL; \
  } \
\
private: \
  constexpr static const int m_s_iMaxTypes = 6; \
  ComputeFunc m_a_funTable[m_s_iMaxTypes]; \
}

DECLARE_BINARY_DISPATCH(PlusDispatch,
  ALLOW_ARGS(BLEAK_STRING, BLEAK_STRING, Plus),
  ALLOW_ARGS(BLEAK_INTEGER, BLEAK_INTEGER, Plus),
  ALLOW_ARGS(BLEAK_FLOAT, BLEAK_FLOAT, Plus),
  ALLOW_ARGS(BLEAK_BOOL, BLEAK_BOOL, Plus),
  ALLOW_ARGS_COMBO(BLEAK_INTEGER, BLEAK_FLOAT, Plus),
  ALLOW_ARGS_COMBO(BLEAK_INTEGER, BLEAK_BOOL, Plus),
  ALLOW_ARGS_COMBO(BLEAK_FLOAT, BLEAK_BOOL, Plus)
);

DECLARE_BINARY_DISPATCH(MinusDispatch,
  ALLOW_ARGS(BLEAK_INTEGER, BLEAK_INTEGER, Minus),
  ALLOW_ARGS(BLEAK_FLOAT, BLEAK_FLOAT, Minus),
  ALLOW_ARGS(BLEAK_BOOL, BLEAK_BOOL, Minus),
  ALLOW_ARGS_COMBO(BLEAK_INTEGER, BLEAK_FLOAT, Minus),
  ALLOW_ARGS_COMBO(BLEAK_INTEGER, BLEAK_BOOL, Minus),
  ALLOW_ARGS_COMBO(BLEAK_FLOAT, BLEAK_BOOL, Minus)
);

DECLARE_BINARY_DISPATCH(MultiplyDispatch,
  ALLOW_ARGS(BLEAK_INTEGER, BLEAK_INTEGER, Multiply),
  ALLOW_ARGS(BLEAK_FLOAT, BLEAK_FLOAT, Multiply),
  ALLOW_ARGS(BLEAK_BOOL, BLEAK_BOOL, Multiply),
  ALLOW_ARGS_COMBO(BLEAK_INTEGER, BLEAK_FLOAT, Multiply),
  ALLOW_ARGS_COMBO(BLEAK_INTEGER, BLEAK_BOOL, Multiply),
  ALLOW_ARGS_COMBO(BLEAK_FLOAT, BLEAK_BOOL, Multiply)
);

DECLARE_BINARY_DISPATCH(DivideDispatch,
  ALLOW_ARGS(BLEAK_INTEGER, BLEAK_INTEGER, Divide),
  ALLOW_ARGS(BLEAK_FLOAT, BLEAK_FLOAT, Divide),
  ALLOW_ARGS(BLEAK_BOOL, BLEAK_BOOL, Divide),
  ALLOW_ARGS_COMBO(BLEAK_INTEGER, BLEAK_FLOAT, Divide),
  ALLOW_ARGS_COMBO(BLEAK_INTEGER, BLEAK_BOOL, Divide),
  ALLOW_ARGS_COMBO(BLEAK_FLOAT, BLEAK_BOOL, Divide)
);

DECLARE_BINARY_DISPATCH(PowDispatch,
  ALLOW_ARGS(BLEAK_INTEGER, BLEAK_INTEGER, Pow),
  ALLOW_ARGS(BLEAK_FLOAT, BLEAK_FLOAT, Pow),
  ALLOW_ARGS_COMBO(BLEAK_INTEGER, BLEAK_FLOAT, Pow)
);

DECLARE_BINARY_DISPATCH(ModuloDispatch,
  ALLOW_ARGS(BLEAK_INTEGER, BLEAK_INTEGER, Modulo)
);

DECLARE_UNARY_DISPATCH(NegateDispatch,
  ALLOW_ARG(BLEAK_INTEGER, Negate),
  ALLOW_ARG(BLEAK_FLOAT, Negate),
  ALLOW_ARG(BLEAK_BOOL, Negate)
);

} // end anonymous namespace

bleak_value * bleak_expr_negate(bleak_value *p_stA) {
  static NegateDispatch clDispatch;
  return clDispatch(p_stA);
}

bleak_value * bleak_expr_plus(bleak_value *p_stA, bleak_value *p_stB) {
  static PlusDispatch clDispatch;
  return clDispatch(p_stA, p_stB);
}

bleak_value * bleak_expr_minus(bleak_value *p_stA, bleak_value *p_stB) {
  static MinusDispatch clDispatch;
  return clDispatch(p_stA, p_stB);
}

bleak_value * bleak_expr_divide(bleak_value *p_stA, bleak_value *p_stB) {
  static DivideDispatch clDispatch;
  return clDispatch(p_stA, p_stB);
}

bleak_value * bleak_expr_multiply(bleak_value *p_stA, bleak_value *p_stB) {
  static MultiplyDispatch clDispatch;
  return clDispatch(p_stA, p_stB);
}

bleak_value * bleak_expr_pow(bleak_value *p_stA, bleak_value *p_stB) {
  static PowDispatch clDispatch;
  return clDispatch(p_stA, p_stB);
}

bleak_value * bleak_expr_modulo(bleak_value *p_stA, bleak_value *p_stB) {
  static ModuloDispatch clDispatch;
  return clDispatch(p_stA, p_stB);
}

bleak_graph * bleak_expr_graph_resolve_variables(bleak_graph *p_stGraph, int iEvaluatePrivate) {
   bleak_graph *p_stNewGraph = bleak_graph_dup(p_stGraph);

  if (p_stNewGraph == NULL)
    return NULL;

  p_stNewGraph->p_stParent = p_stGraph->p_stParent;

  bleak_vector_kvp *p_stNewVariables = bleak_expr_kvp_dedup(p_stNewGraph->p_stVariables, p_stNewGraph, iEvaluatePrivate);

  if (p_stNewVariables == NULL) {
    bleak_graph_free(p_stNewGraph);
    return NULL;
  }

  bleak_vector_kvp_for_each(p_stNewGraph->p_stVariables, &bleak_kvp_free);
  bleak_vector_kvp_free(p_stNewGraph->p_stVariables);

  p_stNewGraph->p_stVariables = p_stNewVariables;

  return p_stNewGraph;
}

bleak_graph * bleak_expr_graph_instantiate(bleak_graph *p_stGraph) {
  bleak_graph *p_stNewGraph = bleak_expr_graph_resolve_variables(p_stGraph, 1);

  if (p_stNewGraph == NULL)
    return NULL;

  p_stNewGraph->p_stParent = p_stGraph->p_stParent;

  for (bleak_vertex *p_stVertex : *p_stNewGraph->p_stVertices) {
    bleak_vector_kvp *p_stNewVariables = bleak_expr_kvp_dedup(p_stVertex->p_stProperties, p_stNewGraph, 1);

    if (p_stNewVariables == NULL) {
      bleak_graph_free(p_stNewGraph);
      return NULL;
    }

    bleak_vector_kvp_for_each(p_stVertex->p_stProperties, &bleak_kvp_free);
    bleak_vector_kvp_free(p_stVertex->p_stProperties);

    p_stVertex->p_stProperties = p_stNewVariables;
  }

  return p_stNewGraph;
}

bleak_vector_kvp * bleak_expr_kvp_dedup(bleak_vector_kvp *p_stKvps, bleak_graph *p_stParent, int iEvaluatePrivate) {
  if (p_stKvps == NULL || p_stParent == NULL)
    return NULL;

  bleak_vector_kvp *p_stNewKvps = bleak_vector_kvp_alloc();

  if (p_stNewKvps == NULL)
    return NULL;

  bleak_vector_kvp * const p_stParentVariables = p_stParent->p_stVariables;

  /* Setup variable table in parent if we are deduping the same KVPs */
  if (p_stParentVariables == p_stKvps)
    p_stParent->p_stVariables = p_stNewKvps;

  for (bleak_key_value_pair *p_stKvp : *p_stKvps) {
    auto itr = std::find_if(p_stNewKvps->begin(), p_stNewKvps->end(),
      [&p_stKvp](bleak_key_value_pair *p_stOtherKvp) -> bool {
        return strcmp(p_stOtherKvp->p_cKey, p_stKvp->p_cKey) == 0;
      });

    // Is/was this variable declared private?
    bool bPrivate = (p_stKvp->iPrivate != 0);

    if (itr != p_stNewKvps->end()) {
      // Sanity check: A previously declared public variable can't be declared private
      if (p_stKvp->iPrivate != 0 && (*itr)->iPrivate == 0) {
        std::cerr << "Error: Public key '" << p_stKvp->p_cKey << "' declared private." << std::endl;

        p_stParent->p_stVariables = p_stParentVariables; // Put it back the way it was

        std::for_each(p_stNewKvps->begin(), p_stNewKvps->end(), &bleak_kvp_free);
        bleak_vector_kvp_free(p_stNewKvps);

        return NULL;
      }

      bPrivate = (bPrivate || ((*itr)->iPrivate != 0));
    }

    bleak_value *p_stNewValue = NULL;

    if (iEvaluatePrivate != 0 || !bPrivate)
      p_stNewValue = bleak_expr_evaluate(p_stKvp->p_stValue, p_stParent, iEvaluatePrivate);
    else
      p_stNewValue = bleak_value_dup(p_stKvp->p_stValue);

    if (p_stNewValue == NULL) {
      std::cerr << "Error: Could not evaluate key '" << p_stKvp->p_cKey << "'." << std::endl;

      p_stParent->p_stVariables = p_stParentVariables; // Put it back the way it was

      std::for_each(p_stNewKvps->begin(), p_stNewKvps->end(), &bleak_kvp_free);
      bleak_vector_kvp_free(p_stNewKvps);

      return NULL;
    }

    // Only replace existing variable if we can evaluate everything or the variable isn't private
    if (itr != p_stNewKvps->end() && (iEvaluatePrivate != 0 || !bPrivate)) {
      bleak_value_free((*itr)->p_stValue);
      (*itr)->p_stValue = p_stNewValue;
    }
    else {
      bleak_key_value_pair *p_stNewKvp = bleak_kvp_alloc();
      char *p_cKey = strdup(p_stKvp->p_cKey);

      if (p_stNewKvp == NULL || p_cKey == NULL) {
        bleak_value_free(p_stNewValue);
        bleak_kvp_free(p_stNewKvp);
        free(p_cKey);

        p_stParent->p_stVariables = p_stParentVariables; // Put it back the way it was

        std::for_each(p_stNewKvps->begin(), p_stNewKvps->end(), &bleak_kvp_free);
        bleak_vector_kvp_free(p_stNewKvps);

        return NULL;        
      }

      p_stNewKvp->p_cKey = p_cKey;
      p_stNewKvp->p_stValue = p_stNewValue;
      p_stNewKvp->iPrivate = bPrivate ? 1 : 0;

      p_stNewKvps->push_back(p_stNewKvp);
    }
  }

  p_stParent->p_stVariables = p_stParentVariables; // Put it back the way it was

  return p_stNewKvps;
}

bleak_value * bleak_expr_evaluate(bleak_value *p_stValue, bleak_graph *p_stParent, int iAllowPrivateAccess) {
  if (p_stValue == NULL || p_stParent == NULL)
    return NULL;

  switch (p_stValue->eType) {
  case BLEAK_REFERENCE:
    {
      bleak_graph *p_stScope = p_stParent;

      while (p_stScope != NULL) {
        bleak_vector_kvp *p_stKvps = p_stScope->p_stVariables;

        auto itr = std::find_if(p_stKvps->begin(), p_stKvps->end(), 
          [&p_stValue](bleak_key_value_pair *p_stKvp) -> bool {
            return strcmp(p_stKvp->p_cKey, p_stValue->p_cValue) == 0;
          });

        if (itr != p_stKvps->end() && (iAllowPrivateAccess != 0 || (*itr)->iPrivate == 0))
          return bleak_value_dup((*itr)->p_stValue);


        p_stScope = p_stScope->p_stParent;

        iAllowPrivateAccess = 0; // Private variables are graph-local
      }

      std::cerr << "Error: Could not resolve reference '$" << p_stValue->p_cValue << "'." << std::endl;
    }
    break;
  case BLEAK_VALUE_VECTOR:
    {
      bleak_value *p_stNewValue = bleak_value_alloc();

      if (p_stNewValue == NULL)
        return NULL;

      // Re-use variable for easier freeing of resources
      p_stNewValue->eType = BLEAK_VALUE_VECTOR;
      p_stNewValue->p_stValues = bleak_vector_value_dup(p_stValue->p_stValues);

      // NOTE: Pointers in vector are not duped themselves!

      auto p_stValues = p_stNewValue->p_stValues;

      if (p_stValues == NULL) {
        bleak_value_free(p_stNewValue);
        return NULL;
      }

      std::transform(p_stValues->begin(), p_stValues->end(), p_stValues->begin(),
        [&p_stParent,&iAllowPrivateAccess](bleak_value *p_stV) -> bleak_value * {
          return bleak_expr_evaluate(p_stV, p_stParent, iAllowPrivateAccess);
        });

      if (std::find(p_stValues->begin(), p_stValues->end(), nullptr) != p_stValues->end()) {
        bleak_value_free(p_stNewValue);
        return NULL;
      }

      bleak_vector_int *p_stIVValue = bleak_vector_value_to_vector_int(p_stValues);
      if (p_stIVValue != NULL) {
        bleak_vector_value_for_each(p_stValues, &bleak_value_free);
        bleak_vector_value_free(p_stValues);

        p_stNewValue->eType = BLEAK_INTEGER_VECTOR;
        p_stNewValue->p_stIVValue = p_stIVValue;

        return p_stNewValue;
      }

      bleak_vector_float *p_stFVValue = bleak_vector_value_to_vector_float(p_stValues);
      if (p_stFVValue != NULL) {
        bleak_vector_value_for_each(p_stValues, &bleak_value_free);
        bleak_vector_value_free(p_stValues);

        p_stNewValue->eType = BLEAK_FLOAT_VECTOR;
        p_stNewValue->p_stFVValue = p_stFVValue;

        return p_stNewValue;
      }

      // Don't forget that we failed if we reach here 
      bleak_value_free(p_stNewValue);

      std::cerr << "Error: Not an integer or float vector." << std::endl;
    }
    break;
  case BLEAK_EXPRESSION:
    {
      auto p_stValues = p_stValue->p_stValues;

      std::deque<bleak_value *> dqStack;

      for (bleak_value *p_stTmp : *p_stValues) {

        if (p_stTmp->eType == BLEAK_OPCODE) {
          // Evaluate an operation

          size_t argsNeeded = 0;

          switch (p_stTmp->iValue) {
          case '+':
          case '-':
          case '*':
          case '/':
          case '%':
          case '^':
            argsNeeded = 2;
            break;
          case 'N': // Negate
            argsNeeded = 1;
            break;
          default:
            std::for_each(dqStack.begin(), dqStack.end(), &bleak_value_free);
            std::cerr << "Error: Unknown op '" << (char)(p_stTmp->iValue) << "'." << std::endl;
            return NULL;
          }

          if (dqStack.size() < argsNeeded) {
            std::for_each(dqStack.begin(), dqStack.end(), &bleak_value_free);
            std::cerr << "Error: Insufficient parameters for op '" << (char)(p_stTmp->iValue) << "'." << std::endl;
            return NULL;
          }

          bleak_value *p_stResult = NULL;

          switch (p_stTmp->iValue) {
          case '+':
            p_stResult = bleak_expr_plus(dqStack[1], dqStack[0]);
            break;
          case '-':
            p_stResult = bleak_expr_minus(dqStack[1], dqStack[0]);
            break;
          case '*':
            p_stResult = bleak_expr_multiply(dqStack[1], dqStack[0]);
            break;
          case '/':
            p_stResult = bleak_expr_divide(dqStack[1], dqStack[0]);
            break;
          case '%':
            p_stResult = bleak_expr_modulo(dqStack[1], dqStack[0]);
            break;
          case '^':
            p_stResult = bleak_expr_pow(dqStack[1], dqStack[0]);
            break;
          case 'N': // Negate
            p_stResult = bleak_expr_negate(dqStack[0]);
            break;
          }

          if (p_stResult == NULL) {
            std::for_each(dqStack.begin(), dqStack.end(), &bleak_value_free);
            std::cerr << "Error: Evaluation failed." << std::endl;
            return NULL;
          }

          for (size_t i = 0; i < argsNeeded; ++i) {
            bleak_value_free(dqStack.front());
            dqStack.pop_front();
          }

          dqStack.push_front(p_stResult);
        }
        else {
          // Push a value onto the stack

          bleak_value *p_stResult = bleak_expr_evaluate(p_stTmp, p_stParent, iAllowPrivateAccess);

          if (p_stResult == NULL) {
            std::for_each(dqStack.begin(), dqStack.end(), &bleak_value_free);
            std::cerr << "Error: Failed to evaluate an expression." << std::endl;
            return NULL;
          }

          dqStack.push_front(p_stResult);
        }
      }

      if (dqStack.size() != 1) {
        std::for_each(dqStack.begin(), dqStack.end(), &bleak_value_free);
        std::cerr << "Error: Expected 1 value on the stack." << std::endl;
        return NULL;
      }

      return dqStack.front();
    }
    break;
  case BLEAK_OPCODE:
    std::cerr << "Error: Op codes are not values." << std::endl;
    break;
  default:
    return bleak_value_dup(p_stValue); // Nothing to do 
  }

  return NULL;
}


