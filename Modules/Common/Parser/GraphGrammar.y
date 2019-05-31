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
 
%{
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include "bleak_parser.h"
#include "bleak_vector.h"
#include "bleak_graph_ast.h"
#include "bleak_expression.h"
#include "y.tab.h"
%}

%start graph

%union {
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
}

%pure-parser
%parse-param { bleak_parser *p_stParser } { yyscan_t scanner }
%lex-param { bleak_parser *p_stParser } { yyscan_t scanner }

%token LEFT_ARROW RIGHT_ARROW SUBGRAPH INCLUDE THIS PRIVATE
%token <iValue> INTEGER
%token <fValue> FLOAT
%token <bValue> BOOL
%token <p_cValue> STRING IDENTIFIER REFERENCE

%left '+' '-'
%left '*' '/' '%'
%left '^'
%left NEGATE

%type <p_cValue> vertex_identifier
%type <p_stGraph> graph subgraph
%type <p_stValue> value simple_value vector simple_expression
%type <p_stValues> simple_values
%type <p_stKvp> variable private_variable
%type <p_stKvps> variables all_variables
%type <p_stVertex> vertex
%type <p_stVertices> vertices
%type <p_stConnection> connection
%type <p_stConnections> connections
%type <p_stGraphs> subgraphs

%destructor { 
  if ($$ != p_stParser->p_stGraph)
    bleak_graph_free($$); 
} subgraph graph

%destructor { 
  size_t i, numSubgraphs = bleak_vector_graph_size($$);
  bleak_graph **p_stSubgraphs = bleak_vector_graph_data($$);

  for (i = 0; i < numSubgraphs; ++i) {
    bleak_graph_free(p_stSubgraphs[i]);

    /* Prevent double-free */
    if (p_stParser->p_stGraph == p_stSubgraphs[i])
      p_stParser->p_stGraph = NULL;
  }

  bleak_vector_graph_free($$);
} subgraphs

%destructor { free($$); } STRING IDENTIFIER REFERENCE vertex_identifier
%destructor { bleak_value_free($$); } value vector simple_value simple_expression
%destructor {
  bleak_vector_value_for_each($$, &bleak_value_free);
  bleak_vector_value_free($$);
} simple_values

%destructor { bleak_kvp_free($$); } variable private_variable
%destructor { 
  bleak_vector_kvp_for_each($$, &bleak_kvp_free);
  bleak_vector_kvp_free($$);
} variables all_variables

%destructor { bleak_vertex_free($$); } vertex
%destructor { 
  bleak_vector_vertex_for_each($$, &bleak_vertex_free);
  bleak_vector_vertex_free($$);
} vertices

%destructor { bleak_connection_free($$); } connection
%destructor {
  bleak_vector_connection_for_each($$, &bleak_connection_free);
  bleak_vector_connection_free($$);
} connections

%%

graph   :   all_variables vertices connections
          { 
            bleak_graph *p_stNewGraph = bleak_graph_alloc();

            p_stNewGraph->p_stVariables = $1 ;
            p_stNewGraph->p_stSubgraphs = bleak_vector_graph_alloc();
            p_stNewGraph->p_stVertices = $2 ;
            p_stNewGraph->p_stConnections = $3 ;

            p_stParser->p_stGraph = p_stNewGraph;

            $$ = p_stNewGraph;
          }
        |   all_variables subgraphs vertices connections
          {
            bleak_graph *p_stNewGraph = bleak_graph_alloc();

            p_stNewGraph->p_stVariables = $1 ;
            p_stNewGraph->p_stSubgraphs = $2 ;
            p_stNewGraph->p_stVertices = $3 ;
            p_stNewGraph->p_stConnections = $4 ;

            p_stParser->p_stGraph = p_stNewGraph;

            $$ = p_stNewGraph;
          }
        |   all_variables vertices
          { 
            bleak_graph *p_stNewGraph = bleak_graph_alloc();

            p_stNewGraph->p_stVariables = $1 ;
            p_stNewGraph->p_stSubgraphs = bleak_vector_graph_alloc();
            p_stNewGraph->p_stVertices = $2 ;
            p_stNewGraph->p_stConnections = bleak_vector_connection_alloc();

            p_stParser->p_stGraph = p_stNewGraph;

            $$ = p_stNewGraph;
          }
        |   all_variables subgraphs vertices
          {
            bleak_graph *p_stNewGraph = bleak_graph_alloc();

            p_stNewGraph->p_stVariables = $1 ;
            p_stNewGraph->p_stSubgraphs = $2 ;
            p_stNewGraph->p_stVertices = $3 ;
            p_stNewGraph->p_stConnections = bleak_vector_connection_alloc();

            p_stParser->p_stGraph = p_stNewGraph;

            $$ = p_stNewGraph;
          }
        ;

subgraphs   :   subgraph
              {
                bleak_vector_graph *p_stSubgraphs = bleak_vector_graph_alloc();
                bleak_vector_graph_push_back(p_stSubgraphs, $1);
                $$ = p_stSubgraphs;
              }
            |   subgraphs subgraph
              {
                bleak_vector_graph_push_back($1, $2);
                $$ = $1 ;
              }
            ; 

subgraph    :   SUBGRAPH IDENTIFIER '{' graph '}' ';'
              {
                /* Need to get parent */
                bleak_graph *p_stSubgraph = $4 ;
                p_stSubgraph->p_cName = $2 ;
                $$ = p_stSubgraph;
              }
            ;

private_variable  : PRIVATE variable
                    {
                      $2->iPrivate = 1;
                      $$ = $2 ;
                    }
                  ;

all_variables : /* empty */
                {
                  $$ = bleak_vector_kvp_alloc();
                }
              | all_variables variable
                {
                  bleak_vector_kvp_push_back($1, $2);
                  $$ = $1;
                }
              | all_variables private_variable
                {
                  bleak_vector_kvp_push_back($1, $2);
                  $$ = $1;
                }
              ;

variables   :   /* empty */
              { 
                $$ = bleak_vector_kvp_alloc(); 
              }
            |   variables variable
              {
                bleak_vector_kvp_push_back($1, $2);
                $$ = $1;
              }
            ;

variable    :   IDENTIFIER '=' value ';'
              {
                bleak_key_value_pair *p_stVariable = bleak_kvp_alloc();
                p_stVariable->p_cKey = $1 ;
                p_stVariable->p_stValue = $3 ;
                $$ = p_stVariable;
              }
            ;

simple_expression   :   simple_value
                      { 
                        bleak_value *p_stNewValue = bleak_value_alloc();
                        p_stNewValue->eType = BLEAK_EXPRESSION;
                        p_stNewValue->p_stValues = bleak_vector_value_alloc();
                        bleak_vector_value_push_back(p_stNewValue->p_stValues, $1);

                        $$ = p_stNewValue;
                      }
                    |   '(' simple_expression ')'
                      { $$ = $2 ; }
                    |   '-' simple_expression   %prec NEGATE
                      { 
                        bleak_value *p_stOpValue = bleak_value_alloc();
                        p_stOpValue->eType = BLEAK_OPCODE;
                        p_stOpValue->iValue = 'N';

                        bleak_vector_value_push_back($2->p_stValues , p_stOpValue);

                        $$ = $2 ;
                      }
                    |   simple_expression '+' simple_expression
                      {
                        bleak_value *p_stA = $1;
                        bleak_value *p_stB = $3;

                        bleak_vector_value_join(p_stA->p_stValues, p_stB->p_stValues);

                        bleak_vector_value_free(p_stB->p_stValues); /* Does not free elements of vector */
                        p_stB->p_stValues = NULL;

                        bleak_value_free(p_stB);

                        bleak_value *p_stOpValue = bleak_value_alloc();
                        p_stOpValue->eType = BLEAK_OPCODE;
                        p_stOpValue->iValue = '+';

                        bleak_vector_value_push_back(p_stA->p_stValues, p_stOpValue);

                        $$ = p_stA;
                      }
                    |   simple_expression '-' simple_expression
                      {
                        bleak_value *p_stA = $1;
                        bleak_value *p_stB = $3;

                        bleak_vector_value_join(p_stA->p_stValues, p_stB->p_stValues);

                        bleak_vector_value_free(p_stB->p_stValues); /* Does not free elements of vector */
                        p_stB->p_stValues = NULL;

                        bleak_value_free(p_stB);

                        bleak_value *p_stOpValue = bleak_value_alloc();
                        p_stOpValue->eType = BLEAK_OPCODE;
                        p_stOpValue->iValue = '-';

                        bleak_vector_value_push_back(p_stA->p_stValues, p_stOpValue);

                        $$ = p_stA;
                      }
                    |   simple_expression '/' simple_expression
                      {
                        bleak_value *p_stA = $1;
                        bleak_value *p_stB = $3;

                        bleak_vector_value_join(p_stA->p_stValues, p_stB->p_stValues);

                        bleak_vector_value_free(p_stB->p_stValues); /* Does not free elements of vector */
                        p_stB->p_stValues = NULL;

                        bleak_value_free(p_stB);

                        bleak_value *p_stOpValue = bleak_value_alloc();
                        p_stOpValue->eType = BLEAK_OPCODE;
                        p_stOpValue->iValue = '/';

                        bleak_vector_value_push_back(p_stA->p_stValues, p_stOpValue);

                        $$ = p_stA;
                      }
                    |   simple_expression '*' simple_expression
                      {
                        bleak_value *p_stA = $1;
                        bleak_value *p_stB = $3;

                        bleak_vector_value_join(p_stA->p_stValues, p_stB->p_stValues);

                        bleak_vector_value_free(p_stB->p_stValues); /* Does not free elements of vector */
                        p_stB->p_stValues = NULL;

                        bleak_value_free(p_stB);

                        bleak_value *p_stOpValue = bleak_value_alloc();
                        p_stOpValue->eType = BLEAK_OPCODE;
                        p_stOpValue->iValue = '*';

                        bleak_vector_value_push_back(p_stA->p_stValues, p_stOpValue);

                        $$ = p_stA;
                      }
                    |   simple_expression '^' simple_expression
                      {
                        bleak_value *p_stA = $1;
                        bleak_value *p_stB = $3;

                        bleak_vector_value_join(p_stA->p_stValues, p_stB->p_stValues);

                        bleak_vector_value_free(p_stB->p_stValues); /* Does not free elements of vector */
                        p_stB->p_stValues = NULL;

                        bleak_value_free(p_stB);

                        bleak_value *p_stOpValue = bleak_value_alloc();
                        p_stOpValue->eType = BLEAK_OPCODE;
                        p_stOpValue->iValue = '^';

                        bleak_vector_value_push_back(p_stA->p_stValues, p_stOpValue);

                        $$ = p_stA;
                      }
                    |   simple_expression '%' simple_expression
                      {
                        bleak_value *p_stA = $1;
                        bleak_value *p_stB = $3;

                        bleak_vector_value_join(p_stA->p_stValues, p_stB->p_stValues);

                        bleak_vector_value_free(p_stB->p_stValues); /* Does not free elements of vector */
                        p_stB->p_stValues = NULL;

                        bleak_value_free(p_stB);

                        bleak_value *p_stOpValue = bleak_value_alloc();
                        p_stOpValue->eType = BLEAK_OPCODE;
                        p_stOpValue->iValue = '%';

                        bleak_vector_value_push_back(p_stA->p_stValues, p_stOpValue);

                        $$ = p_stA;
                      }
                    ;

simple_value    :   INTEGER
                  {
                    bleak_value *p_stValue = bleak_value_alloc();
                    p_stValue->eType = BLEAK_INTEGER;
                    p_stValue->iValue = $1 ;
                    $$ = p_stValue;
                  }
                |   FLOAT
                  {
                    bleak_value *p_stValue = bleak_value_alloc();
                    p_stValue->eType = BLEAK_FLOAT;
                    p_stValue->fValue = $1 ;
                    $$ = p_stValue;
                  }
                |   BOOL
                  {
                    bleak_value *p_stValue = bleak_value_alloc();
                    p_stValue->eType = BLEAK_BOOL;
                    p_stValue->bValue = $1 ;
                    $$ = p_stValue;
                  }
                |   STRING
                  {
                    bleak_value *p_stValue = bleak_value_alloc();
                    p_stValue->eType = BLEAK_STRING;
                    p_stValue->p_cValue = $1 ;
                    $$ = p_stValue;
                  }
                |   REFERENCE
                  {
                    bleak_value *p_stValue = bleak_value_alloc();
                    p_stValue->eType = BLEAK_REFERENCE;
                    p_stValue->p_cValue = $1 ;
                    $$ = p_stValue;
                  }
                ;


simple_values   :   simple_expression
                  {
                    $$ = bleak_vector_value_alloc();
                    bleak_vector_value_push_back( $$, $1 );
                  }
                |   simple_values ',' simple_expression
                  {
                    $$ = $1 ;
                    bleak_vector_value_push_back( $$ , $3 );
                  }
                ;

vector    :   '[' ']'
            {
              bleak_value *p_stValue = bleak_value_alloc();
              p_stValue->eType = BLEAK_INTEGER_VECTOR;
              p_stValue->p_stIVValue = bleak_vector_int_alloc();
              $$ = p_stValue;
            }
          |   '[' simple_values ']'
            {
              bleak_value *p_stValue = bleak_value_alloc();
              p_stValue->eType = BLEAK_VALUE_VECTOR;
              p_stValue->p_stValues = $2 ;
              $$ = p_stValue;
            }
          ;

value   :   simple_expression
          {
            $$ = $1 ;
          }
        |   vector
          {
            $$ = $1 ;
          }
        ;

vertices    :   vertex
              { 
                $$ = bleak_vector_vertex_alloc(); 
                bleak_vector_vertex_push_back( $$ , $1 );
              }
            |   vertices vertex
              {
                $$ = $1 ;
                bleak_vector_vertex_push_back( $$ , $2 );
              }
            ;

vertex      : IDENTIFIER '{' variables '}' IDENTIFIER ';'
              {
                bleak_vertex *p_stVertex = bleak_vertex_alloc();
                p_stVertex->p_cType = $1 ;
                p_stVertex->p_cName = $5 ;
                p_stVertex->p_stProperties = $3 ;
                $$ = p_stVertex;
              }
            | IDENTIFIER IDENTIFIER ';'
              {
                bleak_vertex *p_stVertex = bleak_vertex_alloc();
                p_stVertex->p_cType = $1 ;
                p_stVertex->p_cName = $2 ;
                p_stVertex->p_stProperties = bleak_vector_kvp_alloc(); /* Empty properties */
                $$ = p_stVertex;
              }
            ;

connections   :   connection
                { 
                  $$ = bleak_vector_connection_alloc(); 
                  bleak_vector_connection_push_back( $$ , $1 );
                }
              |   connections connection
                {
                  $$ = $1 ;
                  bleak_vector_connection_push_back( $$ , $2 );
                }
              ;

connection    :   vertex_identifier '.' IDENTIFIER RIGHT_ARROW vertex_identifier '.' IDENTIFIER ';'
                {
                  bleak_connection *p_stConnection = bleak_connection_alloc();
                  p_stConnection->p_cSourceName = $1 ;
                  p_stConnection->p_cOutputName = $3 ;
                  p_stConnection->p_cTargetName = $5 ;
                  p_stConnection->p_cInputName = $7 ;
                  $$ = p_stConnection;
                }
              |   vertex_identifier '.' IDENTIFIER LEFT_ARROW vertex_identifier '.' IDENTIFIER ';'
                {
                  bleak_connection *p_stConnection = bleak_connection_alloc();
                  p_stConnection->p_cSourceName = $5 ;
                  p_stConnection->p_cOutputName = $7 ;
                  p_stConnection->p_cTargetName = $1 ;
                  p_stConnection->p_cInputName = $3 ;
                  $$ = p_stConnection;
                }
              ;

vertex_identifier   :   IDENTIFIER
                      { $$ = $1 ; }
                    |   THIS
                      { $$ = strdup("this") ; }
                    ;

%%

void yyerror(bleak_parser *p_stParser, yyscan_t scanner, const char *p_cErrorMsg, ...) {
  va_list ap;

  fputs("Error: ", stderr);

  va_start(ap, p_cErrorMsg);
  vfprintf(stderr, p_cErrorMsg, ap);
  va_end(ap);

  fputc('\n', stderr);
}

