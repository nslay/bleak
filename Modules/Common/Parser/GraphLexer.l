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
#include "bleak_parser.h"
#include "bleak_graph_ast.h"
#include "y.tab.h"
%}

%option noyywrap
%option reentrant
%option bison-bridge
%option yylineno
%option never-interactive

%%

[ \t\r\n]+ /* whitespace */
#.*$      /* comment */

"true" { 
  yylval->bValue = 1; 
  return BOOL; 
}

"false" { 
  yylval->bValue = 0; 
  return BOOL; 
}

"include" {
  return INCLUDE; /* Reserved keyword */
}

"include"[ \t]*"\""[^"]*"\""[ \t\r]*$ {
  /* Implement this here since it makes less sense in the grammar! */

  char *p_cFileName = strdup(strchr(yytext, '"')+1);
  char *p = strchr(p_cFileName, '"');

  *p = '\0';

  if (p_stParser->iStackSize >= BLEAK_MAX_INCLUDE_DEPTH) {
    free(p_cFileName);
    yyerror(p_stParser, yyscanner, "Include files nested too deeply.");

    yyterminate();
  }

  p_stParser->a_stIncludeStack[p_stParser->iStackSize] = YY_CURRENT_BUFFER;
  yyin = bleak_parser_open_include(p_stParser, p_cFileName, p_stParser->a_cDirStack + p_stParser->iStackSize + 1, p_stParser->a_cFileStack + p_stParser->iStackSize + 1);
  
  if (yyin == NULL) {
    yyerror(p_stParser, yyscanner, "Could not open include file '%s'.", p_cFileName);

    free(p_cFileName);

    yyterminate();
  }

  ++p_stParser->iStackSize; /* This has to be done AFTER bleak_parser_open_include */

  free(p_cFileName);

  yy_switch_to_buffer(yy_create_buffer(yyin, YY_BUF_SIZE, yyscanner), yyscanner);

  BEGIN(INITIAL);
}

<<EOF>> {
  if (--p_stParser->iStackSize < 0) {
    yyterminate();
  }
  else {
    /* lex does not close files */
    if (YY_CURRENT_BUFFER->yy_input_file != NULL) {
      fclose(YY_CURRENT_BUFFER->yy_input_file);
      YY_CURRENT_BUFFER->yy_input_file = NULL;
    }
    yy_delete_buffer(YY_CURRENT_BUFFER, yyscanner);

    /* We decremented above! */
    free(p_stParser->a_cDirStack[p_stParser->iStackSize+1]);
    p_stParser->a_cDirStack[p_stParser->iStackSize+1] = NULL;

    free(p_stParser->a_cFileStack[p_stParser->iStackSize+1]);
    p_stParser->a_cFileStack[p_stParser->iStackSize+1] = NULL;

    yy_switch_to_buffer(p_stParser->a_stIncludeStack[p_stParser->iStackSize], yyscanner);
  }
}

"subgraph" {
  return SUBGRAPH;
}

"this" {
  return THIS;
}

"private" {
  return PRIVATE;
}

[a-zA-Z_][a-zA-Z0-9_]* { 
  yylval->p_cValue = strdup(yytext); 
  return IDENTIFIER;
}

"$"[a-zA-Z_][a-zA-Z0-9_]* {
  yylval->p_cValue = strdup(yytext+1); 
  return REFERENCE;
}

"\""[^"]*"\"" {
  char *p_cValue = strdup(yytext+1);
  p_cValue[strlen(p_cValue)-1] = '\0';  
  yylval->p_cValue = p_cValue;
  return STRING;
}

[0-9]+ {
  yylval->iValue = strtol(yytext,NULL,10);
  return INTEGER;
}

(([0-9]*"."[0-9]+)|([0-9]+"."[0-9]*)) {
  /* Formats: x.y or x. or .x */
  yylval->fValue = strtof(yytext,NULL);
  return FLOAT;
}

(([0-9]*"."[0-9]+)|([0-9]+"."[0-9]*)|[0-9]+)([eE][+-]?[0-9]+) {
  /* Formats: x.yEz or .xEz or x.Ez */
  yylval->fValue = strtof(yytext,NULL);
  return FLOAT;
}

"->" { return RIGHT_ARROW; }
"<-" { return LEFT_ARROW; }
"**" { return '^'; }
. { return yytext[0]; }

