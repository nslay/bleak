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
#include <ctype.h>
#include <string.h>
#include "bleak_parser.h"

#ifdef _WIN32
/* Return position in path where drive letter/hostname (UNC) are no longer visible */
static char * bleak_ignore_windows(char *p_cPath) {
  size_t length = strlen(p_cPath);

  if (length >= 2) {
    /* drive letter? */
    if (isalpha(p_cPath[0]) && p_cPath[1] == ':')
      return p_cPath+2;

    /* UNC path? */
    if (p_cPath[0] == '\\' && p_cPath[1] == '\\') {
      char *p = strpbrk(p_cPath+2, "/\\");

      if (p == NULL)
        return p_cPath + length;

      return p;
    }
  }

  return p_cPath;
}
#endif /* _WIN32 */

static void bleak_basename(char *p_cFullFilePath) {
  size_t length = 0;
  char *p_cDelim = NULL;
  char *p_cFilePath = p_cFullFilePath;

  if (p_cFullFilePath == NULL)
    return;

#ifdef _WIN32
  p_cFilePath = bleak_ignore_windows(p_cFullFilePath);
#endif /* _WIN32 */

  length = strlen(p_cFilePath);

  /* Remove trailing delimiters */
#ifdef _WIN32
  while (length > 1 && (p_cFilePath[length-1] == '/' || p_cFilePath[length-1] == '\\'))
    p_cFilePath[--length] = '\0';
#else /* !_WIN32 */
  while (length > 1 && p_cFilePath[length-1] == '/')
    p_cFilePath[--length] = '\0';
#endif /* _WIN32 */

  if (length <= 1)
    return;

  p_cDelim = strrchr(p_cFilePath, '/');

#ifdef _WIN32
  {
    char *p_cDelim2 = strrchr(p_cFilePath, '\\');

    if (p_cDelim == NULL || p_cDelim2 > p_cDelim)
      p_cDelim = p_cDelim2;
  }
#endif /* _WIN32 */

  if (p_cDelim != NULL)
    memmove(p_cFullFilePath, p_cDelim+1, strlen(p_cDelim+1)+1); /* +1 for '\0' */
}

static void bleak_dirname(char *p_cFullFilePath) {
  size_t length = 0;
  char *p_cDelim = NULL;
  char *p_cFilePath = p_cFullFilePath;

  if (p_cFullFilePath == NULL)
    return;

#ifdef _WIN32
  p_cFilePath = bleak_ignore_windows(p_cFullFilePath);
#endif /* _WIN32 */

  length = strlen(p_cFilePath);

  if (length == 0)
    return;

  /* Remove trailing delimiters */
#ifdef _WIN32
  while (length > 1 && (p_cFilePath[length-1] == '/' || p_cFilePath[length-1] == '\\'))
    p_cFilePath[--length] = '\0';
#else /* !_WIN32 */
  while (length > 1 && p_cFilePath[length-1] == '/')
    p_cFilePath[--length] = '\0';
#endif /* _WIN32 */

  p_cDelim = strrchr(p_cFilePath, '/');

#ifdef _WIN32
  {
    char *p_cDelim2 = strrchr(p_cFilePath, '\\');

    if (p_cDelim == NULL || p_cDelim2 > p_cDelim)
      p_cDelim = p_cDelim2;
  }
#endif /* _WIN32 */

  if (p_cDelim == NULL) {
    if (p_cFilePath == p_cFullFilePath)
      strcpy(p_cFilePath, "."); /* Size guaranteed to be at least 1 + \0 */
    else
      *p_cFilePath = '\0'; /* Windows drive/UNC prefix is present already */
  }
  else if (p_cDelim == p_cFilePath)
    *(p_cDelim+1) = '\0'; /* Cover case where dirname is / */
  else
    *p_cDelim = '\0';
}

bleak_parser * bleak_parser_alloc() {
  bleak_parser *p_stParser = calloc(1, sizeof(bleak_parser));

  if (p_stParser == NULL)
    return NULL;

  if (yylex_init(&p_stParser->lexScanner) != 0) {
    free(p_stParser);
    return NULL;
  }

  return p_stParser;
}

bleak_graph * bleak_parser_load_graph(bleak_parser *p_stParser, const char *p_cFileName) {
  char *p_cFileNameForStack = NULL;
  char *p_cDirName = NULL;
  FILE *pFile = NULL;
  int iRet = 0;
  bleak_graph *p_stGraph = NULL;

  /* Check for invalid or re-used parser state */
  if (p_stParser == NULL || p_stParser->iStackSize != 0 || p_stParser->p_stGraph != NULL || p_stParser->a_cDirStack[0] != NULL)
    return NULL;

  pFile = fopen(p_cFileName, "r");

  if (pFile == NULL)
    return NULL;

  p_cFileNameForStack = strdup(p_cFileName);
  p_cDirName = strdup(p_cFileName);

  if (p_cDirName == NULL || p_cFileNameForStack == NULL) {
    free(p_cDirName); /* One of these could be non-NULL */
    free(p_cFileNameForStack); /* Ditto */
    fclose(pFile);
    return NULL;
  }

  bleak_dirname(p_cDirName);

  p_stParser->a_cFileStack[0] = p_cFileNameForStack;
  p_stParser->a_cDirStack[0] = p_cDirName;

  yyset_in(pFile, p_stParser->lexScanner);

  iRet = yyparse(p_stParser, p_stParser->lexScanner);

  fclose(pFile);

  if (iRet != 0)
    return NULL;

  p_stGraph = p_stParser->p_stGraph;
  p_stParser->p_stGraph = NULL; /* Prevent the loaded graph from being freed */

  return p_stGraph;
}

void bleak_parser_free(bleak_parser *p_stParser) {
  int i = 0;

  if (p_stParser == NULL)
    return;

  bleak_graph_free(p_stParser->p_stGraph);

  free(p_stParser->a_cDirStack[0]); /* This will always be set (see bleak_parser_load_graph) */
  free(p_stParser->a_cFileStack[0]); /* Ditto */

  for (i = 1; i < p_stParser->iStackSize; ++i) {
    free(p_stParser->a_cDirStack[i]);
    free(p_stParser->a_cFileStack[i]);

    /* lex does not close files. The first file handle will be closed by the caller */
    if (p_stParser->a_stIncludeStack[i]->yy_input_file != NULL) {
      fclose(p_stParser->a_stIncludeStack[i]->yy_input_file);
      p_stParser->a_stIncludeStack[i]->yy_input_file = NULL;
    }

    /* We do no need to call this explicitly as flex will delete its own stack of buffer states */
    /*yy_delete_buffer(p_stParser->a_stIncludeStack[i], p_stParser->lexScanner);*/
  }

  /* This could be decremented to -1 on last closed file, so need to check */
  if (p_stParser->iStackSize > 0) {
    /* +1 element on a_cDirStack */
    free(p_stParser->a_cDirStack[p_stParser->iStackSize]);
    free(p_stParser->a_cFileStack[p_stParser->iStackSize]);
  }

  yylex_destroy(p_stParser->lexScanner);

  free(p_stParser);
}

FILE * bleak_parser_open_include(bleak_parser *p_stParser, const char *p_cFileName, char **p_cDirName, char **p_cResolvedFileName) {
  FILE *pFile = NULL;
  size_t fileNameLen = 0;
  char *p_cFilePath = NULL;
  const char *p_cSearchDir = NULL;
  int i = 0;

  if (p_cDirName == NULL || p_cResolvedFileName == NULL)
    return NULL;

  *p_cDirName = NULL;
  *p_cResolvedFileName = NULL;

  if (p_stParser == NULL || p_cFileName == NULL)
    return NULL;

  fileNameLen = strlen(p_cFileName);
  p_cSearchDir = p_stParser->a_cDirStack[p_stParser->iStackSize]; /* +1 element on a_cDirStack */

  while (pFile == NULL && p_cSearchDir != NULL) {
    free(p_cFilePath);

    p_cFilePath = malloc(strlen(p_cSearchDir) + fileNameLen + 2); /* Extra space for '/' and '\0' considered */

    if (p_cFilePath == NULL)
      return NULL;

    strcpy(p_cFilePath, p_cSearchDir);
    strcat(p_cFilePath, "/");
    strcat(p_cFilePath, p_cFileName);

    pFile = fopen(p_cFilePath, "r");

    if (p_stParser->p_cIncludeDirs == NULL)
      p_cSearchDir = NULL;
    else
      p_cSearchDir = p_stParser->p_cIncludeDirs[i++];
  } 

  if (pFile == NULL) {
    free(p_cFilePath);
    return NULL;
  }

  *p_cDirName = strdup(p_cFilePath);

  if (*p_cDirName == NULL) {
    free(p_cFilePath);
    fclose(pFile);
    return NULL;
  }

  bleak_dirname(*p_cDirName);

  *p_cResolvedFileName = p_cFilePath;

  return pFile;
}

