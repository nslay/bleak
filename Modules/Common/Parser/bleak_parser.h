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

#ifndef BLEAK_PARSER_H
#define BLEAK_PARSER_H

#include <stdio.h>
#include "bleak_graph_ast.h"
#include "y.tab.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#define YY_DECL int yylex(YYSTYPE *yylval_param, bleak_parser *p_stParser, yyscan_t yyscanner)

#ifndef FLEX_SCANNER
#include "lex.yy.h"
#endif /* !FLEX_SCANNER */

#define BLEAK_MAX_INCLUDE_DEPTH 10

typedef struct bleak_parser_ {
  bleak_graph *p_stGraph; /* This will be freed in bleak_parser_free() */
  YY_BUFFER_STATE a_stIncludeStack[BLEAK_MAX_INCLUDE_DEPTH];
  char *a_cFileStack[BLEAK_MAX_INCLUDE_DEPTH+1]; /* File paths of currently opened files starting with the first graph file. These should be allocated! */
  char *a_cDirStack[BLEAK_MAX_INCLUDE_DEPTH+1]; /* dirnames of include files starting with the first graph file. These should be allocated! */
  int iStackSize;
  const char **p_cIncludeDirs; /* NULL terminated! */
  yyscan_t lexScanner;
} bleak_parser;

/* Functions to allocate/clean up */
bleak_parser * bleak_parser_alloc();
void bleak_parser_free(bleak_parser *p_stParser);

/* Load bleak_graph from a given file. p_stParser should only be used once! Caller may manually set p_cIncludeDirs pointer. */
bleak_graph * bleak_parser_load_graph(bleak_parser *p_stParser, const char *p_cFileName);


/* Find and open an include file, return the dirname of the path in p_cDirName */
FILE * bleak_parser_open_include(bleak_parser *p_stParser, const char *p_cFileName, char **p_cDirName, char **p_cResolvedFileName);

/* Function prototypes for flex and byacc */

YY_DECL;
int yyparse(bleak_parser *p_stParser, yyscan_t scanner);
void yyerror(bleak_parser *p_stParser, yyscan_t scanner, const char *p_cErrorMsg, ...);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* !BLEAK_PARSER_H */

