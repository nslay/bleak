#ifndef _yy_defines_h_
#define _yy_defines_h_

#define LEFT_ARROW 257
#define RIGHT_ARROW 258
#define SUBGRAPH 259
#define INCLUDE 260
#define THIS 261
#define PRIVATE 262
#define INTEGER 263
#define FLOAT 264
#define BOOL 265
#define STRING 266
#define IDENTIFIER 267
#define REFERENCE 268
#define NEGATE 269
#ifdef YYSTYPE
#undef  YYSTYPE_IS_DECLARED
#define YYSTYPE_IS_DECLARED 1
#endif
#ifndef YYSTYPE_IS_DECLARED
#define YYSTYPE_IS_DECLARED 1
typedef union {
  int iValue;
  float fValue;
  int bValue; /* VS 2017 does not seem to work with _Bool */
  char *p_cValue;
  bleak_graph *p_stGraph;
  bleak_value *p_stValue;
  bleak_vector_value *p_stValues;
  bleak_key_value_pair *p_stKvp;
  bleak_vector_kvp *p_stKvps;
  bleak_vertex *p_stVertex;
  bleak_vector_vertex *p_stVertices;
  bleak_connection *p_stConnection;
  bleak_vector_connection *p_stConnections;
  bleak_vector_graph *p_stGraphs;
} YYSTYPE;
#endif /* !YYSTYPE_IS_DECLARED */

#endif /* _yy_defines_h_ */
