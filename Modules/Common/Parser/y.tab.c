/* original parser id follows */
/* yysccsid[] = "@(#)yaccpar	1.9 (Berkeley) 02/21/93" */
/* (use YYMAJOR/YYMINOR for ifdefs dependent on parser version) */

#define YYBYACC 1
#define YYMAJOR 1
#define YYMINOR 9
#define YYPATCH 20191125

#define YYEMPTY        (-1)
#define yyclearin      (yychar = YYEMPTY)
#define yyerrok        (yyerrflag = 0)
#define YYRECOVERING() (yyerrflag != 0)
#define YYENOMEM       (-2)
#define YYEOF          0
#undef YYBTYACC
#define YYBTYACC 1
#define YYDEBUGSTR (yytrial ? YYPREFIX "debug(trial)" : YYPREFIX "debug")
#define YYPREFIX "yy"

#define YYPURE 1

#line 27 "GraphGrammar.y"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include "bleak_parser.h"
#include "bleak_vector.h"
#include "bleak_graph_ast.h"
#include "bleak_expression.h"
#include "y.tab.h"
#ifdef YYSTYPE
#undef  YYSTYPE_IS_DECLARED
#define YYSTYPE_IS_DECLARED 1
#endif
#ifndef YYSTYPE_IS_DECLARED
#define YYSTYPE_IS_DECLARED 1
#line 40 "GraphGrammar.y"
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
#line 58 "y.tab.c"

/* compatibility with bison */
#ifdef YYPARSE_PARAM
/* compatibility with FreeBSD */
# ifdef YYPARSE_PARAM_TYPE
#  define YYPARSE_DECL() yyparse(YYPARSE_PARAM_TYPE YYPARSE_PARAM)
# else
#  define YYPARSE_DECL() yyparse(void *YYPARSE_PARAM)
# endif
#else
# define YYPARSE_DECL() yyparse(bleak_parser *p_stParser, yyscan_t scanner)
#endif

/* Parameters sent to lex. */
#ifdef YYLEX_PARAM
# ifdef YYLEX_PARAM_TYPE
#  define YYLEX_DECL() yylex(YYSTYPE *yylval, YYLEX_PARAM_TYPE YYLEX_PARAM)
# else
#  define YYLEX_DECL() yylex(YYSTYPE *yylval, void * YYLEX_PARAM)
# endif
# define YYLEX yylex(&yylval, YYLEX_PARAM)
#else
# define YYLEX_DECL() yylex(YYSTYPE *yylval, bleak_parser *p_stParser, yyscan_t scanner)
# define YYLEX yylex(&yylval, p_stParser, scanner)
#endif

/* Parameters sent to yyerror. */
#ifndef YYERROR_DECL
#define YYERROR_DECL() yyerror(bleak_parser *p_stParser, yyscan_t scanner, const char *s)
#endif
#ifndef YYERROR_CALL
#define YYERROR_CALL(msg) yyerror(p_stParser, scanner, msg)
#endif

#ifndef YYDESTRUCT_DECL
#define YYDESTRUCT_DECL() yydestruct(const char *msg, int psymb, YYSTYPE *val, bleak_parser *p_stParser, yyscan_t scanner)
#endif
#ifndef YYDESTRUCT_CALL
#define YYDESTRUCT_CALL(msg, psymb, val) yydestruct(msg, psymb, val, p_stParser, scanner)
#endif

extern int YYPARSE_DECL();

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
#define YYERRCODE 256
typedef short YYINT;
static const YYINT yylhs[] = {                           -1,
    0,    0,    0,    0,   16,   16,    2,    9,   11,   11,
   11,   10,   10,    8,    6,    6,    6,    6,    6,    6,
    6,    6,    6,    4,    4,    4,    4,    4,    7,    7,
    5,    5,    3,    3,   13,   13,   12,   12,   15,   15,
   14,   14,    1,    1,
};
static const YYINT yylen[] = {                            2,
    3,    4,    2,    3,    1,    2,    6,    2,    0,    2,
    2,    0,    2,    4,    1,    3,    2,    3,    3,    3,
    3,    3,    3,    1,    1,    1,    1,    1,    1,    3,
    2,    3,    1,    1,    1,    2,    6,    3,    1,    2,
    8,    8,    1,    1,
};
static const YYINT yydefred[] = {                         9,
    0,    0,    0,    0,    0,    5,   10,   11,   35,    0,
    0,    0,    0,    8,    0,   12,    0,   44,    0,    0,
   36,   39,    0,    0,    6,    0,    9,   38,    0,   24,
   25,   26,   27,   28,    0,    0,    0,    0,   15,   34,
    0,    0,   43,   40,    0,    0,    0,   13,   17,    0,
   31,    0,    0,   14,    0,    0,    0,    0,    0,    0,
    0,    0,    0,   16,    0,   32,    0,    0,    0,    0,
    0,   22,    0,    0,    7,   37,    0,    0,    0,    0,
    0,    0,    0,   42,   41,
};
#if defined(YYDESTRUCT_CALL) || defined(YYSTYPE_TOSTRING)
static const YYINT yystos[] = {                           0,
  271,  282,  259,  262,  267,  273,  279,  280,  283,  284,
  287,  267,  267,  279,  267,  123,   61,  261,  267,  272,
  283,  285,  286,  267,  273,  284,  123,   59,  281,  263,
  264,  265,  266,  268,   45,   40,   91,  274,  275,  276,
  277,   46,  267,  285,  286,  271,  125,  279,  277,  277,
   93,  277,  278,   59,   43,   45,   42,   47,   37,   94,
  267,  125,  267,   41,   44,   93,  277,  277,  277,  277,
  277,  277,  257,  258,   59,   59,  277,  272,  272,   46,
   46,  267,  267,   59,   59,
};
#endif /* YYDESTRUCT_CALL || YYSTYPE_TOSTRING */
static const YYINT yydgoto[] = {                          1,
   20,    6,   38,   39,   40,   41,   53,    7,    8,   29,
    2,    9,   10,   22,   23,   11,
};
static const YYINT yysindex[] = {                         0,
    0, -174, -254, -207,  -60,    0,    0,    0,    0, -184,
 -167,  -49,   20,    0,   35,    0,  -40,    0, -121,   51,
    0,    0, -165, -121,    0, -184,    0,    0, -122,    0,
    0,    0,    0,    0,  -28,  -28,  -34,   39,    0,    0,
  -18, -168,    0,    0, -165,  -24, -164,    0,    0,  -27,
    0,  -18,  -14,    0,  -28,  -28,  -28,  -28,  -28,  -28,
 -235,   46,   47,    0,  -28,    0,  -16,  -16,   13,   13,
   13,    0, -165, -165,    0,    0,  -18,   62,   63, -157,
 -156,   54,   56,    0,    0,
};
static const YYINT yyrindex[] = {                         0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    4,
    0,    0,    0,    0,    0,    0,    0,    0,   66,    0,
    0,    0,    7,    0,    0,    8,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
   57,    0,    0,    0,    9,    0,    0,    0,    0,    0,
    0,   -7,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,   21,   28,   -9,    2,
   11,    0,    0,    0,    0,    0,   -4,    0,    0,    0,
    0,    0,    0,    0,    0,
};
#if YYBTYACC
static const YYINT yycindex[] = {                         0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,
};
#endif
static const YYINT yygindex[] = {                        93,
  -32,  111,    0,    0,    0,   82,    0,   53,    0,    0,
    0,   65,  112,   45,   98,    0,
};
#define YYTABLESIZE 240
static const YYINT yytable[] = {                         36,
   17,   16,   47,    3,   35,   36,    1,    4,    2,   59,
   35,   36,   12,   64,   57,   55,   35,   56,   59,   58,
   59,   73,   74,   57,   55,   57,   56,   21,   58,   65,
   58,   21,   21,   21,   21,   21,   29,   21,   20,   30,
   78,   79,   20,   20,   20,   20,   20,   23,   20,   21,
   37,   23,   23,   23,   23,   23,   14,   23,   51,   13,
   20,   18,   16,   18,   18,   18,   60,   44,   19,   23,
   19,   19,   19,   27,   21,   60,   18,   60,   66,   18,
   17,   48,   19,   21,    3,   29,   19,    4,   30,   44,
   21,    3,    5,   28,   20,   18,   42,   54,   61,   24,
   62,   43,   63,   23,   75,   76,   60,   80,   81,   82,
   83,   43,   84,   18,   85,   33,   49,   50,   52,   46,
   19,   25,   26,   45,    0,    0,    0,    0,    3,    0,
    0,    1,    4,    2,    0,    0,   67,   68,   69,   70,
   71,   72,    0,    0,   13,   15,   77,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,   15,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,   30,   31,   32,   33,    0,   34,   30,   31,
   32,   33,    0,   34,   30,   31,   32,   33,    0,   34,
};
static const YYINT yycheck[] = {                         40,
   61,  123,  125,    0,   45,   40,    0,    0,    0,   37,
   45,   40,  267,   41,   42,   43,   45,   45,   37,   47,
   37,  257,  258,   42,   43,   42,   45,   37,   47,   44,
   47,   41,   42,   43,   44,   45,   44,   47,   37,   44,
   73,   74,   41,   42,   43,   44,   45,   37,   47,   59,
   91,   41,   42,   43,   44,   45,    4,   47,   93,  267,
   59,   41,  123,   43,   44,   45,   94,   23,   41,   59,
   43,   44,   45,  123,   10,   94,  261,   94,   93,   59,
   61,   29,  267,   93,  259,   93,   59,  262,   93,   45,
   26,  259,  267,   59,   93,  261,   46,   59,  267,  267,
  125,  267,  267,   93,   59,   59,   94,   46,   46,  267,
  267,   46,   59,   93,   59,   59,   35,   36,   37,   27,
   93,   11,   11,   26,   -1,   -1,   -1,   -1,  125,   -1,
   -1,  125,  125,  125,   -1,   -1,   55,   56,   57,   58,
   59,   60,   -1,   -1,  267,  267,   65,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,  267,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,  263,  264,  265,  266,   -1,  268,  263,  264,
  265,  266,   -1,  268,  263,  264,  265,  266,   -1,  268,
};
#if YYBTYACC
static const YYINT yyctable[] = {                        -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
};
#endif
#define YYFINAL 1
#ifndef YYDEBUG
#define YYDEBUG 0
#endif
#define YYMAXTOKEN 269
#define YYUNDFTOKEN 288
#define YYTRANSLATE(a) ((a) > YYMAXTOKEN ? YYUNDFTOKEN : (a))
#if YYDEBUG
static const char *const yyname[] = {

"$end",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
"'%'",0,0,"'('","')'","'*'","'+'","','","'-'","'.'","'/'",0,0,0,0,0,0,0,0,0,0,0,
"';'",0,"'='",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"'['",0,
"']'","'^'",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"'{'",0,
"'}'",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,"error","LEFT_ARROW","RIGHT_ARROW","SUBGRAPH",
"INCLUDE","THIS","PRIVATE","INTEGER","FLOAT","BOOL","STRING","IDENTIFIER",
"REFERENCE","NEGATE","$accept","graph","vertex_identifier","subgraph","value",
"simple_value","vector","simple_expression","simple_values","variable",
"private_variable","variables","all_variables","vertex","vertices","connection",
"connections","subgraphs","illegal-symbol",
};
static const char *const yyrule[] = {
"$accept : graph",
"graph : all_variables vertices connections",
"graph : all_variables subgraphs vertices connections",
"graph : all_variables vertices",
"graph : all_variables subgraphs vertices",
"subgraphs : subgraph",
"subgraphs : subgraphs subgraph",
"subgraph : SUBGRAPH IDENTIFIER '{' graph '}' ';'",
"private_variable : PRIVATE variable",
"all_variables :",
"all_variables : all_variables variable",
"all_variables : all_variables private_variable",
"variables :",
"variables : variables variable",
"variable : IDENTIFIER '=' value ';'",
"simple_expression : simple_value",
"simple_expression : '(' simple_expression ')'",
"simple_expression : '-' simple_expression",
"simple_expression : simple_expression '+' simple_expression",
"simple_expression : simple_expression '-' simple_expression",
"simple_expression : simple_expression '/' simple_expression",
"simple_expression : simple_expression '*' simple_expression",
"simple_expression : simple_expression '^' simple_expression",
"simple_expression : simple_expression '%' simple_expression",
"simple_value : INTEGER",
"simple_value : FLOAT",
"simple_value : BOOL",
"simple_value : STRING",
"simple_value : REFERENCE",
"simple_values : simple_expression",
"simple_values : simple_values ',' simple_expression",
"vector : '[' ']'",
"vector : '[' simple_values ']'",
"value : simple_expression",
"value : vector",
"vertices : vertex",
"vertices : vertices vertex",
"vertex : IDENTIFIER '{' variables '}' IDENTIFIER ';'",
"vertex : IDENTIFIER IDENTIFIER ';'",
"connections : connection",
"connections : connections connection",
"connection : vertex_identifier '.' IDENTIFIER RIGHT_ARROW vertex_identifier '.' IDENTIFIER ';'",
"connection : vertex_identifier '.' IDENTIFIER LEFT_ARROW vertex_identifier '.' IDENTIFIER ';'",
"vertex_identifier : IDENTIFIER",
"vertex_identifier : THIS",

};
#endif

#if YYDEBUG
int      yydebug;
#endif

#if defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED)
#ifndef YYLLOC_DEFAULT
#define YYLLOC_DEFAULT(loc, rhs, n) \
do \
{ \
    if (n == 0) \
    { \
        (loc).first_line   = YYRHSLOC(rhs, 0).last_line; \
        (loc).first_column = YYRHSLOC(rhs, 0).last_column; \
        (loc).last_line    = YYRHSLOC(rhs, 0).last_line; \
        (loc).last_column  = YYRHSLOC(rhs, 0).last_column; \
    } \
    else \
    { \
        (loc).first_line   = YYRHSLOC(rhs, 1).first_line; \
        (loc).first_column = YYRHSLOC(rhs, 1).first_column; \
        (loc).last_line    = YYRHSLOC(rhs, n).last_line; \
        (loc).last_column  = YYRHSLOC(rhs, n).last_column; \
    } \
} while (0)
#endif /* YYLLOC_DEFAULT */
#endif /* defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED) */
#if YYBTYACC

#ifndef YYLVQUEUEGROWTH
#define YYLVQUEUEGROWTH 32
#endif
#endif /* YYBTYACC */

/* define the initial stack-sizes */
#ifdef YYSTACKSIZE
#undef YYMAXDEPTH
#define YYMAXDEPTH  YYSTACKSIZE
#else
#ifdef YYMAXDEPTH
#define YYSTACKSIZE YYMAXDEPTH
#else
#define YYSTACKSIZE 10000
#define YYMAXDEPTH  10000
#endif
#endif

#ifndef YYINITSTACKSIZE
#define YYINITSTACKSIZE 200
#endif

typedef struct {
    unsigned stacksize;
    YYINT    *s_base;
    YYINT    *s_mark;
    YYINT    *s_last;
    YYSTYPE  *l_base;
    YYSTYPE  *l_mark;
#if defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED)
    YYLTYPE  *p_base;
    YYLTYPE  *p_mark;
#endif
} YYSTACKDATA;
#if YYBTYACC

struct YYParseState_s
{
    struct YYParseState_s *save;    /* Previously saved parser state */
    YYSTACKDATA            yystack; /* saved parser stack */
    int                    state;   /* saved parser state */
    int                    errflag; /* saved error recovery status */
    int                    lexeme;  /* saved index of the conflict lexeme in the lexical queue */
    YYINT                  ctry;    /* saved index in yyctable[] for this conflict */
};
typedef struct YYParseState_s YYParseState;
#endif /* YYBTYACC */
#line 538 "GraphGrammar.y"

void yyerror(bleak_parser *p_stParser, yyscan_t scanner, const char *p_cErrorMsg, ...) {
  va_list ap;

  fputs("Error: ", stderr);

  va_start(ap, p_cErrorMsg);
  vfprintf(stderr, p_cErrorMsg, ap);
  va_end(ap);

  fputc('\n', stderr);
}

#line 442 "y.tab.c"

/* Release memory associated with symbol. */
#if ! defined YYDESTRUCT_IS_DECLARED
static void
YYDESTRUCT_DECL()
{
    switch (psymb)
    {
	case 266:
#line 104 "GraphGrammar.y"
	{ free((*val).p_cValue); }
	break;
#line 455 "y.tab.c"
	case 267:
#line 104 "GraphGrammar.y"
	{ free((*val).p_cValue); }
	break;
#line 460 "y.tab.c"
	case 268:
#line 104 "GraphGrammar.y"
	{ free((*val).p_cValue); }
	break;
#line 465 "y.tab.c"
	case 271:
#line 84 "GraphGrammar.y"
	{ 
  if ((*val).p_stGraph != p_stParser->p_stGraph)
    bleak_graph_free((*val).p_stGraph); 
}
	break;
#line 473 "y.tab.c"
	case 272:
#line 104 "GraphGrammar.y"
	{ free((*val).p_cValue); }
	break;
#line 478 "y.tab.c"
	case 273:
#line 84 "GraphGrammar.y"
	{ 
  if ((*val).p_stGraph != p_stParser->p_stGraph)
    bleak_graph_free((*val).p_stGraph); 
}
	break;
#line 486 "y.tab.c"
	case 274:
#line 105 "GraphGrammar.y"
	{ bleak_value_free((*val).p_stValue); }
	break;
#line 491 "y.tab.c"
	case 275:
#line 105 "GraphGrammar.y"
	{ bleak_value_free((*val).p_stValue); }
	break;
#line 496 "y.tab.c"
	case 276:
#line 105 "GraphGrammar.y"
	{ bleak_value_free((*val).p_stValue); }
	break;
#line 501 "y.tab.c"
	case 277:
#line 105 "GraphGrammar.y"
	{ bleak_value_free((*val).p_stValue); }
	break;
#line 506 "y.tab.c"
	case 278:
#line 106 "GraphGrammar.y"
	{
  bleak_vector_value_for_each((*val).p_stValues, &bleak_value_free);
  bleak_vector_value_free((*val).p_stValues);
}
	break;
#line 514 "y.tab.c"
	case 279:
#line 111 "GraphGrammar.y"
	{ bleak_kvp_free((*val).p_stKvp); }
	break;
#line 519 "y.tab.c"
	case 280:
#line 111 "GraphGrammar.y"
	{ bleak_kvp_free((*val).p_stKvp); }
	break;
#line 524 "y.tab.c"
	case 281:
#line 112 "GraphGrammar.y"
	{ 
  bleak_vector_kvp_for_each((*val).p_stKvps, &bleak_kvp_free);
  bleak_vector_kvp_free((*val).p_stKvps);
}
	break;
#line 532 "y.tab.c"
	case 282:
#line 112 "GraphGrammar.y"
	{ 
  bleak_vector_kvp_for_each((*val).p_stKvps, &bleak_kvp_free);
  bleak_vector_kvp_free((*val).p_stKvps);
}
	break;
#line 540 "y.tab.c"
	case 283:
#line 117 "GraphGrammar.y"
	{ bleak_vertex_free((*val).p_stVertex); }
	break;
#line 545 "y.tab.c"
	case 284:
#line 118 "GraphGrammar.y"
	{ 
  bleak_vector_vertex_for_each((*val).p_stVertices, &bleak_vertex_free);
  bleak_vector_vertex_free((*val).p_stVertices);
}
	break;
#line 553 "y.tab.c"
	case 285:
#line 123 "GraphGrammar.y"
	{ bleak_connection_free((*val).p_stConnection); }
	break;
#line 558 "y.tab.c"
	case 286:
#line 124 "GraphGrammar.y"
	{
  bleak_vector_connection_for_each((*val).p_stConnections, &bleak_connection_free);
  bleak_vector_connection_free((*val).p_stConnections);
}
	break;
#line 566 "y.tab.c"
	case 287:
#line 89 "GraphGrammar.y"
	{ 
  size_t i, numSubgraphs = bleak_vector_graph_size((*val).p_stGraphs);
  bleak_graph **p_stSubgraphs = bleak_vector_graph_data((*val).p_stGraphs);

  for (i = 0; i < numSubgraphs; ++i) {
    bleak_graph_free(p_stSubgraphs[i]);

    /* Prevent double-free */
    if (p_stParser->p_stGraph == p_stSubgraphs[i])
      p_stParser->p_stGraph = NULL;
  }

  bleak_vector_graph_free((*val).p_stGraphs);
}
	break;
#line 584 "y.tab.c"
    }
}
#define YYDESTRUCT_IS_DECLARED 1
#endif

/* For use in generated program */
#define yydepth (int)(yystack.s_mark - yystack.s_base)
#if YYBTYACC
#define yytrial (yyps->save)
#endif /* YYBTYACC */

#if YYDEBUG
#include <stdio.h>	/* needed for printf */
#endif

#include <stdlib.h>	/* needed for malloc, etc */
#include <string.h>	/* needed for memset */

/* allocate initial stack or double stack size, up to YYMAXDEPTH */
static int yygrowstack(YYSTACKDATA *data)
{
    int i;
    unsigned newsize;
    YYINT *newss;
    YYSTYPE *newvs;
#if defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED)
    YYLTYPE *newps;
#endif

    if ((newsize = data->stacksize) == 0)
        newsize = YYINITSTACKSIZE;
    else if (newsize >= YYMAXDEPTH)
        return YYENOMEM;
    else if ((newsize *= 2) > YYMAXDEPTH)
        newsize = YYMAXDEPTH;

    i = (int) (data->s_mark - data->s_base);
    newss = (YYINT *)realloc(data->s_base, newsize * sizeof(*newss));
    if (newss == 0)
        return YYENOMEM;

    data->s_base = newss;
    data->s_mark = newss + i;

    newvs = (YYSTYPE *)realloc(data->l_base, newsize * sizeof(*newvs));
    if (newvs == 0)
        return YYENOMEM;

    data->l_base = newvs;
    data->l_mark = newvs + i;

#if defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED)
    newps = (YYLTYPE *)realloc(data->p_base, newsize * sizeof(*newps));
    if (newps == 0)
        return YYENOMEM;

    data->p_base = newps;
    data->p_mark = newps + i;
#endif

    data->stacksize = newsize;
    data->s_last = data->s_base + newsize - 1;

#if YYDEBUG
    if (yydebug)
        fprintf(stderr, "%sdebug: stack size increased to %d\n", YYPREFIX, newsize);
#endif
    return 0;
}

#if YYPURE || defined(YY_NO_LEAKS)
static void yyfreestack(YYSTACKDATA *data)
{
    free(data->s_base);
    free(data->l_base);
#if defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED)
    free(data->p_base);
#endif
    memset(data, 0, sizeof(*data));
}
#else
#define yyfreestack(data) /* nothing */
#endif /* YYPURE || defined(YY_NO_LEAKS) */
#if YYBTYACC

static YYParseState *
yyNewState(unsigned size)
{
    YYParseState *p = (YYParseState *) malloc(sizeof(YYParseState));
    if (p == NULL) return NULL;

    p->yystack.stacksize = size;
    if (size == 0)
    {
        p->yystack.s_base = NULL;
        p->yystack.l_base = NULL;
#if defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED)
        p->yystack.p_base = NULL;
#endif
        return p;
    }
    p->yystack.s_base    = (YYINT *) malloc(size * sizeof(YYINT));
    if (p->yystack.s_base == NULL) return NULL;
    p->yystack.l_base    = (YYSTYPE *) malloc(size * sizeof(YYSTYPE));
    if (p->yystack.l_base == NULL) return NULL;
    memset(p->yystack.l_base, 0, size * sizeof(YYSTYPE));
#if defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED)
    p->yystack.p_base    = (YYLTYPE *) malloc(size * sizeof(YYLTYPE));
    if (p->yystack.p_base == NULL) return NULL;
    memset(p->yystack.p_base, 0, size * sizeof(YYLTYPE));
#endif

    return p;
}

static void
yyFreeState(YYParseState *p)
{
    yyfreestack(&p->yystack);
    free(p);
}
#endif /* YYBTYACC */

#define YYABORT  goto yyabort
#define YYREJECT goto yyabort
#define YYACCEPT goto yyaccept
#define YYERROR  goto yyerrlab
#if YYBTYACC
#define YYVALID        do { if (yyps->save)            goto yyvalid; } while(0)
#define YYVALID_NESTED do { if (yyps->save && \
                                yyps->save->save == 0) goto yyvalid; } while(0)
#endif /* YYBTYACC */

int
YYPARSE_DECL()
{
    int      yyerrflag;
    int      yychar;
    YYSTYPE  yyval;
    YYSTYPE  yylval;
    int      yynerrs;

#if defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED)
    YYLTYPE  yyloc; /* position returned by actions */
    YYLTYPE  yylloc; /* position from the lexer */
#endif

    /* variables for the parser stack */
    YYSTACKDATA yystack;
#if YYBTYACC

    /* Current parser state */
    static YYParseState *yyps = 0;

    /* yypath != NULL: do the full parse, starting at *yypath parser state. */
    static YYParseState *yypath = 0;

    /* Base of the lexical value queue */
    static YYSTYPE *yylvals = 0;

    /* Current position at lexical value queue */
    static YYSTYPE *yylvp = 0;

    /* End position of lexical value queue */
    static YYSTYPE *yylve = 0;

    /* The last allocated position at the lexical value queue */
    static YYSTYPE *yylvlim = 0;

#if defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED)
    /* Base of the lexical position queue */
    static YYLTYPE *yylpsns = 0;

    /* Current position at lexical position queue */
    static YYLTYPE *yylpp = 0;

    /* End position of lexical position queue */
    static YYLTYPE *yylpe = 0;

    /* The last allocated position at the lexical position queue */
    static YYLTYPE *yylplim = 0;
#endif

    /* Current position at lexical token queue */
    static YYINT  *yylexp = 0;

    static YYINT  *yylexemes = 0;
#endif /* YYBTYACC */
    int yym, yyn, yystate, yyresult;
#if YYBTYACC
    int yynewerrflag;
    YYParseState *yyerrctx = NULL;
#endif /* YYBTYACC */
#if defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED)
    YYLTYPE  yyerror_loc_range[3]; /* position of error start/end (0 unused) */
#endif
#if YYDEBUG
    const char *yys;

    if ((yys = getenv("YYDEBUG")) != 0)
    {
        yyn = *yys;
        if (yyn >= '0' && yyn <= '9')
            yydebug = yyn - '0';
    }
    if (yydebug)
        fprintf(stderr, "%sdebug[<# of symbols on state stack>]\n", YYPREFIX);
#endif
#if defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED)
    memset(yyerror_loc_range, 0, sizeof(yyerror_loc_range));
#endif

    yyerrflag = 0;
    yychar = 0;
    memset(&yyval,  0, sizeof(yyval));
    memset(&yylval, 0, sizeof(yylval));
#if defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED)
    memset(&yyloc,  0, sizeof(yyloc));
    memset(&yylloc, 0, sizeof(yylloc));
#endif

#if YYBTYACC
    yyps = yyNewState(0); if (yyps == 0) goto yyenomem;
    yyps->save = 0;
#endif /* YYBTYACC */
    yym = 0;
    yyn = 0;
    yynerrs = 0;
    yyerrflag = 0;
    yychar = YYEMPTY;
    yystate = 0;

#if YYPURE
    memset(&yystack, 0, sizeof(yystack));
#endif

    if (yystack.s_base == NULL && yygrowstack(&yystack) == YYENOMEM) goto yyoverflow;
    yystack.s_mark = yystack.s_base;
    yystack.l_mark = yystack.l_base;
#if defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED)
    yystack.p_mark = yystack.p_base;
#endif
    yystate = 0;
    *yystack.s_mark = 0;

yyloop:
    if ((yyn = yydefred[yystate]) != 0) goto yyreduce;
    if (yychar < 0)
    {
#if YYBTYACC
        do {
        if (yylvp < yylve)
        {
            /* we're currently re-reading tokens */
            yylval = *yylvp++;
#if defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED)
            yylloc = *yylpp++;
#endif
            yychar = *yylexp++;
            break;
        }
        if (yyps->save)
        {
            /* in trial mode; save scanner results for future parse attempts */
            if (yylvp == yylvlim)
            {   /* Enlarge lexical value queue */
                size_t p = (size_t) (yylvp - yylvals);
                size_t s = (size_t) (yylvlim - yylvals);

                s += YYLVQUEUEGROWTH;
                if ((yylexemes = (YYINT *)realloc(yylexemes, s * sizeof(YYINT))) == NULL) goto yyenomem;
                if ((yylvals   = (YYSTYPE *)realloc(yylvals, s * sizeof(YYSTYPE))) == NULL) goto yyenomem;
#if defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED)
                if ((yylpsns   = (YYLTYPE *)realloc(yylpsns, s * sizeof(YYLTYPE))) == NULL) goto yyenomem;
#endif
                yylvp   = yylve = yylvals + p;
                yylvlim = yylvals + s;
#if defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED)
                yylpp   = yylpe = yylpsns + p;
                yylplim = yylpsns + s;
#endif
                yylexp  = yylexemes + p;
            }
            *yylexp = (YYINT) YYLEX;
            *yylvp++ = yylval;
            yylve++;
#if defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED)
            *yylpp++ = yylloc;
            yylpe++;
#endif
            yychar = *yylexp++;
            break;
        }
        /* normal operation, no conflict encountered */
#endif /* YYBTYACC */
        yychar = YYLEX;
#if YYBTYACC
        } while (0);
#endif /* YYBTYACC */
        if (yychar < 0) yychar = YYEOF;
#if YYDEBUG
        if (yydebug)
        {
            if ((yys = yyname[YYTRANSLATE(yychar)]) == NULL) yys = yyname[YYUNDFTOKEN];
            fprintf(stderr, "%s[%d]: state %d, reading token %d (%s)",
                            YYDEBUGSTR, yydepth, yystate, yychar, yys);
#ifdef YYSTYPE_TOSTRING
#if YYBTYACC
            if (!yytrial)
#endif /* YYBTYACC */
                fprintf(stderr, " <%s>", YYSTYPE_TOSTRING(yychar, yylval));
#endif
            fputc('\n', stderr);
        }
#endif
    }
#if YYBTYACC

    /* Do we have a conflict? */
    if (((yyn = yycindex[yystate]) != 0) && (yyn += yychar) >= 0 &&
        yyn <= YYTABLESIZE && yycheck[yyn] == (YYINT) yychar)
    {
        YYINT ctry;

        if (yypath)
        {
            YYParseState *save;
#if YYDEBUG
            if (yydebug)
                fprintf(stderr, "%s[%d]: CONFLICT in state %d: following successful trial parse\n",
                                YYDEBUGSTR, yydepth, yystate);
#endif
            /* Switch to the next conflict context */
            save = yypath;
            yypath = save->save;
            save->save = NULL;
            ctry = save->ctry;
            if (save->state != yystate) YYABORT;
            yyFreeState(save);

        }
        else
        {

            /* Unresolved conflict - start/continue trial parse */
            YYParseState *save;
#if YYDEBUG
            if (yydebug)
            {
                fprintf(stderr, "%s[%d]: CONFLICT in state %d. ", YYDEBUGSTR, yydepth, yystate);
                if (yyps->save)
                    fputs("ALREADY in conflict, continuing trial parse.\n", stderr);
                else
                    fputs("Starting trial parse.\n", stderr);
            }
#endif
            save                  = yyNewState((unsigned)(yystack.s_mark - yystack.s_base + 1));
            if (save == NULL) goto yyenomem;
            save->save            = yyps->save;
            save->state           = yystate;
            save->errflag         = yyerrflag;
            save->yystack.s_mark  = save->yystack.s_base + (yystack.s_mark - yystack.s_base);
            memcpy (save->yystack.s_base, yystack.s_base, (size_t) (yystack.s_mark - yystack.s_base + 1) * sizeof(YYINT));
            save->yystack.l_mark  = save->yystack.l_base + (yystack.l_mark - yystack.l_base);
            memcpy (save->yystack.l_base, yystack.l_base, (size_t) (yystack.l_mark - yystack.l_base + 1) * sizeof(YYSTYPE));
#if defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED)
            save->yystack.p_mark  = save->yystack.p_base + (yystack.p_mark - yystack.p_base);
            memcpy (save->yystack.p_base, yystack.p_base, (size_t) (yystack.p_mark - yystack.p_base + 1) * sizeof(YYLTYPE));
#endif
            ctry                  = yytable[yyn];
            if (yyctable[ctry] == -1)
            {
#if YYDEBUG
                if (yydebug && yychar >= YYEOF)
                    fprintf(stderr, "%s[%d]: backtracking 1 token\n", YYDEBUGSTR, yydepth);
#endif
                ctry++;
            }
            save->ctry = ctry;
            if (yyps->save == NULL)
            {
                /* If this is a first conflict in the stack, start saving lexemes */
                if (!yylexemes)
                {
                    yylexemes = (YYINT *) malloc((YYLVQUEUEGROWTH) * sizeof(YYINT));
                    if (yylexemes == NULL) goto yyenomem;
                    yylvals   = (YYSTYPE *) malloc((YYLVQUEUEGROWTH) * sizeof(YYSTYPE));
                    if (yylvals == NULL) goto yyenomem;
                    yylvlim   = yylvals + YYLVQUEUEGROWTH;
#if defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED)
                    yylpsns   = (YYLTYPE *) malloc((YYLVQUEUEGROWTH) * sizeof(YYLTYPE));
                    if (yylpsns == NULL) goto yyenomem;
                    yylplim   = yylpsns + YYLVQUEUEGROWTH;
#endif
                }
                if (yylvp == yylve)
                {
                    yylvp  = yylve = yylvals;
#if defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED)
                    yylpp  = yylpe = yylpsns;
#endif
                    yylexp = yylexemes;
                    if (yychar >= YYEOF)
                    {
                        *yylve++ = yylval;
#if defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED)
                        *yylpe++ = yylloc;
#endif
                        *yylexp  = (YYINT) yychar;
                        yychar   = YYEMPTY;
                    }
                }
            }
            if (yychar >= YYEOF)
            {
                yylvp--;
#if defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED)
                yylpp--;
#endif
                yylexp--;
                yychar = YYEMPTY;
            }
            save->lexeme = (int) (yylvp - yylvals);
            yyps->save   = save;
        }
        if (yytable[yyn] == ctry)
        {
#if YYDEBUG
            if (yydebug)
                fprintf(stderr, "%s[%d]: state %d, shifting to state %d\n",
                                YYDEBUGSTR, yydepth, yystate, yyctable[ctry]);
#endif
            if (yychar < 0)
            {
                yylvp++;
#if defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED)
                yylpp++;
#endif
                yylexp++;
            }
            if (yystack.s_mark >= yystack.s_last && yygrowstack(&yystack) == YYENOMEM)
                goto yyoverflow;
            yystate = yyctable[ctry];
            *++yystack.s_mark = (YYINT) yystate;
            *++yystack.l_mark = yylval;
#if defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED)
            *++yystack.p_mark = yylloc;
#endif
            yychar  = YYEMPTY;
            if (yyerrflag > 0) --yyerrflag;
            goto yyloop;
        }
        else
        {
            yyn = yyctable[ctry];
            goto yyreduce;
        }
    } /* End of code dealing with conflicts */
#endif /* YYBTYACC */
    if (((yyn = yysindex[yystate]) != 0) && (yyn += yychar) >= 0 &&
            yyn <= YYTABLESIZE && yycheck[yyn] == (YYINT) yychar)
    {
#if YYDEBUG
        if (yydebug)
            fprintf(stderr, "%s[%d]: state %d, shifting to state %d\n",
                            YYDEBUGSTR, yydepth, yystate, yytable[yyn]);
#endif
        if (yystack.s_mark >= yystack.s_last && yygrowstack(&yystack) == YYENOMEM) goto yyoverflow;
        yystate = yytable[yyn];
        *++yystack.s_mark = yytable[yyn];
        *++yystack.l_mark = yylval;
#if defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED)
        *++yystack.p_mark = yylloc;
#endif
        yychar = YYEMPTY;
        if (yyerrflag > 0)  --yyerrflag;
        goto yyloop;
    }
    if (((yyn = yyrindex[yystate]) != 0) && (yyn += yychar) >= 0 &&
            yyn <= YYTABLESIZE && yycheck[yyn] == (YYINT) yychar)
    {
        yyn = yytable[yyn];
        goto yyreduce;
    }
    if (yyerrflag != 0) goto yyinrecovery;
#if YYBTYACC

    yynewerrflag = 1;
    goto yyerrhandler;
    goto yyerrlab; /* redundant goto avoids 'unused label' warning */

yyerrlab:
    /* explicit YYERROR from an action -- pop the rhs of the rule reduced
     * before looking for error recovery */
    yystack.s_mark -= yym;
    yystate = *yystack.s_mark;
    yystack.l_mark -= yym;
#if defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED)
    yystack.p_mark -= yym;
#endif

    yynewerrflag = 0;
yyerrhandler:
    while (yyps->save)
    {
        int ctry;
        YYParseState *save = yyps->save;
#if YYDEBUG
        if (yydebug)
            fprintf(stderr, "%s[%d]: ERROR in state %d, CONFLICT BACKTRACKING to state %d, %d tokens\n",
                            YYDEBUGSTR, yydepth, yystate, yyps->save->state,
                    (int)(yylvp - yylvals - yyps->save->lexeme));
#endif
        /* Memorize most forward-looking error state in case it's really an error. */
        if (yyerrctx == NULL || yyerrctx->lexeme < yylvp - yylvals)
        {
            /* Free old saved error context state */
            if (yyerrctx) yyFreeState(yyerrctx);
            /* Create and fill out new saved error context state */
            yyerrctx                 = yyNewState((unsigned)(yystack.s_mark - yystack.s_base + 1));
            if (yyerrctx == NULL) goto yyenomem;
            yyerrctx->save           = yyps->save;
            yyerrctx->state          = yystate;
            yyerrctx->errflag        = yyerrflag;
            yyerrctx->yystack.s_mark = yyerrctx->yystack.s_base + (yystack.s_mark - yystack.s_base);
            memcpy (yyerrctx->yystack.s_base, yystack.s_base, (size_t) (yystack.s_mark - yystack.s_base + 1) * sizeof(YYINT));
            yyerrctx->yystack.l_mark = yyerrctx->yystack.l_base + (yystack.l_mark - yystack.l_base);
            memcpy (yyerrctx->yystack.l_base, yystack.l_base, (size_t) (yystack.l_mark - yystack.l_base + 1) * sizeof(YYSTYPE));
#if defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED)
            yyerrctx->yystack.p_mark = yyerrctx->yystack.p_base + (yystack.p_mark - yystack.p_base);
            memcpy (yyerrctx->yystack.p_base, yystack.p_base, (size_t) (yystack.p_mark - yystack.p_base + 1) * sizeof(YYLTYPE));
#endif
            yyerrctx->lexeme         = (int) (yylvp - yylvals);
        }
        yylvp          = yylvals   + save->lexeme;
#if defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED)
        yylpp          = yylpsns   + save->lexeme;
#endif
        yylexp         = yylexemes + save->lexeme;
        yychar         = YYEMPTY;
        yystack.s_mark = yystack.s_base + (save->yystack.s_mark - save->yystack.s_base);
        memcpy (yystack.s_base, save->yystack.s_base, (size_t) (yystack.s_mark - yystack.s_base + 1) * sizeof(YYINT));
        yystack.l_mark = yystack.l_base + (save->yystack.l_mark - save->yystack.l_base);
        memcpy (yystack.l_base, save->yystack.l_base, (size_t) (yystack.l_mark - yystack.l_base + 1) * sizeof(YYSTYPE));
#if defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED)
        yystack.p_mark = yystack.p_base + (save->yystack.p_mark - save->yystack.p_base);
        memcpy (yystack.p_base, save->yystack.p_base, (size_t) (yystack.p_mark - yystack.p_base + 1) * sizeof(YYLTYPE));
#endif
        ctry           = ++save->ctry;
        yystate        = save->state;
        /* We tried shift, try reduce now */
        if ((yyn = yyctable[ctry]) >= 0) goto yyreduce;
        yyps->save     = save->save;
        save->save     = NULL;
        yyFreeState(save);

        /* Nothing left on the stack -- error */
        if (!yyps->save)
        {
#if YYDEBUG
            if (yydebug)
                fprintf(stderr, "%sdebug[%d,trial]: trial parse FAILED, entering ERROR mode\n",
                                YYPREFIX, yydepth);
#endif
            /* Restore state as it was in the most forward-advanced error */
            yylvp          = yylvals   + yyerrctx->lexeme;
#if defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED)
            yylpp          = yylpsns   + yyerrctx->lexeme;
#endif
            yylexp         = yylexemes + yyerrctx->lexeme;
            yychar         = yylexp[-1];
            yylval         = yylvp[-1];
#if defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED)
            yylloc         = yylpp[-1];
#endif
            yystack.s_mark = yystack.s_base + (yyerrctx->yystack.s_mark - yyerrctx->yystack.s_base);
            memcpy (yystack.s_base, yyerrctx->yystack.s_base, (size_t) (yystack.s_mark - yystack.s_base + 1) * sizeof(YYINT));
            yystack.l_mark = yystack.l_base + (yyerrctx->yystack.l_mark - yyerrctx->yystack.l_base);
            memcpy (yystack.l_base, yyerrctx->yystack.l_base, (size_t) (yystack.l_mark - yystack.l_base + 1) * sizeof(YYSTYPE));
#if defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED)
            yystack.p_mark = yystack.p_base + (yyerrctx->yystack.p_mark - yyerrctx->yystack.p_base);
            memcpy (yystack.p_base, yyerrctx->yystack.p_base, (size_t) (yystack.p_mark - yystack.p_base + 1) * sizeof(YYLTYPE));
#endif
            yystate        = yyerrctx->state;
            yyFreeState(yyerrctx);
            yyerrctx       = NULL;
        }
        yynewerrflag = 1;
    }
    if (yynewerrflag == 0) goto yyinrecovery;
#endif /* YYBTYACC */

    YYERROR_CALL("syntax error");
#if defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED)
    yyerror_loc_range[1] = yylloc; /* lookahead position is error start position */
#endif

#if !YYBTYACC
    goto yyerrlab; /* redundant goto avoids 'unused label' warning */
yyerrlab:
#endif
    ++yynerrs;

yyinrecovery:
    if (yyerrflag < 3)
    {
        yyerrflag = 3;
        for (;;)
        {
            if (((yyn = yysindex[*yystack.s_mark]) != 0) && (yyn += YYERRCODE) >= 0 &&
                    yyn <= YYTABLESIZE && yycheck[yyn] == (YYINT) YYERRCODE)
            {
#if YYDEBUG
                if (yydebug)
                    fprintf(stderr, "%s[%d]: state %d, error recovery shifting to state %d\n",
                                    YYDEBUGSTR, yydepth, *yystack.s_mark, yytable[yyn]);
#endif
                if (yystack.s_mark >= yystack.s_last && yygrowstack(&yystack) == YYENOMEM) goto yyoverflow;
                yystate = yytable[yyn];
                *++yystack.s_mark = yytable[yyn];
                *++yystack.l_mark = yylval;
#if defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED)
                /* lookahead position is error end position */
                yyerror_loc_range[2] = yylloc;
                YYLLOC_DEFAULT(yyloc, yyerror_loc_range, 2); /* position of error span */
                *++yystack.p_mark = yyloc;
#endif
                goto yyloop;
            }
            else
            {
#if YYDEBUG
                if (yydebug)
                    fprintf(stderr, "%s[%d]: error recovery discarding state %d\n",
                                    YYDEBUGSTR, yydepth, *yystack.s_mark);
#endif
                if (yystack.s_mark <= yystack.s_base) goto yyabort;
#if defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED)
                /* the current TOS position is the error start position */
                yyerror_loc_range[1] = *yystack.p_mark;
#endif
#if defined(YYDESTRUCT_CALL)
#if YYBTYACC
                if (!yytrial)
#endif /* YYBTYACC */
#if defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED)
                    YYDESTRUCT_CALL("error: discarding state",
                                    yystos[*yystack.s_mark], yystack.l_mark, yystack.p_mark);
#else
                    YYDESTRUCT_CALL("error: discarding state",
                                    yystos[*yystack.s_mark], yystack.l_mark);
#endif /* defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED) */
#endif /* defined(YYDESTRUCT_CALL) */
                --yystack.s_mark;
                --yystack.l_mark;
#if defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED)
                --yystack.p_mark;
#endif
            }
        }
    }
    else
    {
        if (yychar == YYEOF) goto yyabort;
#if YYDEBUG
        if (yydebug)
        {
            if ((yys = yyname[YYTRANSLATE(yychar)]) == NULL) yys = yyname[YYUNDFTOKEN];
            fprintf(stderr, "%s[%d]: state %d, error recovery discarding token %d (%s)\n",
                            YYDEBUGSTR, yydepth, yystate, yychar, yys);
        }
#endif
#if defined(YYDESTRUCT_CALL)
#if YYBTYACC
        if (!yytrial)
#endif /* YYBTYACC */
#if defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED)
            YYDESTRUCT_CALL("error: discarding token", yychar, &yylval, &yylloc);
#else
            YYDESTRUCT_CALL("error: discarding token", yychar, &yylval);
#endif /* defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED) */
#endif /* defined(YYDESTRUCT_CALL) */
        yychar = YYEMPTY;
        goto yyloop;
    }

yyreduce:
    yym = yylen[yyn];
#if YYDEBUG
    if (yydebug)
    {
        fprintf(stderr, "%s[%d]: state %d, reducing by rule %d (%s)",
                        YYDEBUGSTR, yydepth, yystate, yyn, yyrule[yyn]);
#ifdef YYSTYPE_TOSTRING
#if YYBTYACC
        if (!yytrial)
#endif /* YYBTYACC */
            if (yym > 0)
            {
                int i;
                fputc('<', stderr);
                for (i = yym; i > 0; i--)
                {
                    if (i != yym) fputs(", ", stderr);
                    fputs(YYSTYPE_TOSTRING(yystos[yystack.s_mark[1-i]],
                                           yystack.l_mark[1-i]), stderr);
                }
                fputc('>', stderr);
            }
#endif
        fputc('\n', stderr);
    }
#endif
    if (yym > 0)
        yyval = yystack.l_mark[1-yym];
    else
        memset(&yyval, 0, sizeof yyval);
#if defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED)

    /* Perform position reduction */
    memset(&yyloc, 0, sizeof(yyloc));
#if YYBTYACC
    if (!yytrial)
#endif /* YYBTYACC */
    {
        YYLLOC_DEFAULT(yyloc, &yystack.p_mark[-yym], yym);
        /* just in case YYERROR is invoked within the action, save
           the start of the rhs as the error start position */
        yyerror_loc_range[1] = yystack.p_mark[1-yym];
    }
#endif

    switch (yyn)
    {
case 1:
  if (!yytrial)
#line 132 "GraphGrammar.y"
	{ 
            bleak_graph *p_stNewGraph = bleak_graph_alloc();

            p_stNewGraph->p_stVariables = yystack.l_mark[-2].p_stKvps ;
            p_stNewGraph->p_stSubgraphs = bleak_vector_graph_alloc();
            p_stNewGraph->p_stVertices = yystack.l_mark[-1].p_stVertices ;
            p_stNewGraph->p_stConnections = yystack.l_mark[0].p_stConnections ;

            p_stParser->p_stGraph = p_stNewGraph;

            yyval.p_stGraph = p_stNewGraph;
          }
break;
case 2:
  if (!yytrial)
#line 145 "GraphGrammar.y"
	{
            bleak_graph *p_stNewGraph = bleak_graph_alloc();

            p_stNewGraph->p_stVariables = yystack.l_mark[-3].p_stKvps ;
            p_stNewGraph->p_stSubgraphs = yystack.l_mark[-2].p_stGraphs ;
            p_stNewGraph->p_stVertices = yystack.l_mark[-1].p_stVertices ;
            p_stNewGraph->p_stConnections = yystack.l_mark[0].p_stConnections ;

            p_stParser->p_stGraph = p_stNewGraph;

            yyval.p_stGraph = p_stNewGraph;
          }
break;
case 3:
  if (!yytrial)
#line 158 "GraphGrammar.y"
	{ 
            bleak_graph *p_stNewGraph = bleak_graph_alloc();

            p_stNewGraph->p_stVariables = yystack.l_mark[-1].p_stKvps ;
            p_stNewGraph->p_stSubgraphs = bleak_vector_graph_alloc();
            p_stNewGraph->p_stVertices = yystack.l_mark[0].p_stVertices ;
            p_stNewGraph->p_stConnections = bleak_vector_connection_alloc();

            p_stParser->p_stGraph = p_stNewGraph;

            yyval.p_stGraph = p_stNewGraph;
          }
break;
case 4:
  if (!yytrial)
#line 171 "GraphGrammar.y"
	{
            bleak_graph *p_stNewGraph = bleak_graph_alloc();

            p_stNewGraph->p_stVariables = yystack.l_mark[-2].p_stKvps ;
            p_stNewGraph->p_stSubgraphs = yystack.l_mark[-1].p_stGraphs ;
            p_stNewGraph->p_stVertices = yystack.l_mark[0].p_stVertices ;
            p_stNewGraph->p_stConnections = bleak_vector_connection_alloc();

            p_stParser->p_stGraph = p_stNewGraph;

            yyval.p_stGraph = p_stNewGraph;
          }
break;
case 5:
  if (!yytrial)
#line 186 "GraphGrammar.y"
	{
                bleak_vector_graph *p_stSubgraphs = bleak_vector_graph_alloc();
                bleak_vector_graph_push_back(p_stSubgraphs, yystack.l_mark[0].p_stGraph);
                yyval.p_stGraphs = p_stSubgraphs;
              }
break;
case 6:
  if (!yytrial)
#line 192 "GraphGrammar.y"
	{
                bleak_vector_graph_push_back(yystack.l_mark[-1].p_stGraphs, yystack.l_mark[0].p_stGraph);
                yyval.p_stGraphs = yystack.l_mark[-1].p_stGraphs ;
              }
break;
case 7:
  if (!yytrial)
#line 199 "GraphGrammar.y"
	{
                /* Need to get parent */
                bleak_graph *p_stSubgraph = yystack.l_mark[-2].p_stGraph ;
                p_stSubgraph->p_cName = yystack.l_mark[-4].p_cValue ;
                yyval.p_stGraph = p_stSubgraph;
              }
break;
case 8:
  if (!yytrial)
#line 208 "GraphGrammar.y"
	{
                      yystack.l_mark[0].p_stKvp->iPrivate = 1;
                      yyval.p_stKvp = yystack.l_mark[0].p_stKvp ;
                    }
break;
case 9:
  if (!yytrial)
#line 215 "GraphGrammar.y"
	{
                  yyval.p_stKvps = bleak_vector_kvp_alloc();
                }
break;
case 10:
  if (!yytrial)
#line 219 "GraphGrammar.y"
	{
                  bleak_vector_kvp_push_back(yystack.l_mark[-1].p_stKvps, yystack.l_mark[0].p_stKvp);
                  yyval.p_stKvps = yystack.l_mark[-1].p_stKvps;
                }
break;
case 11:
  if (!yytrial)
#line 224 "GraphGrammar.y"
	{
                  bleak_vector_kvp_push_back(yystack.l_mark[-1].p_stKvps, yystack.l_mark[0].p_stKvp);
                  yyval.p_stKvps = yystack.l_mark[-1].p_stKvps;
                }
break;
case 12:
  if (!yytrial)
#line 231 "GraphGrammar.y"
	{ 
                yyval.p_stKvps = bleak_vector_kvp_alloc(); 
              }
break;
case 13:
  if (!yytrial)
#line 235 "GraphGrammar.y"
	{
                bleak_vector_kvp_push_back(yystack.l_mark[-1].p_stKvps, yystack.l_mark[0].p_stKvp);
                yyval.p_stKvps = yystack.l_mark[-1].p_stKvps;
              }
break;
case 14:
  if (!yytrial)
#line 242 "GraphGrammar.y"
	{
                bleak_key_value_pair *p_stVariable = bleak_kvp_alloc();
                p_stVariable->p_cKey = yystack.l_mark[-3].p_cValue ;
                p_stVariable->p_stValue = yystack.l_mark[-1].p_stValue ;
                yyval.p_stKvp = p_stVariable;
              }
break;
case 15:
  if (!yytrial)
#line 251 "GraphGrammar.y"
	{ 
                        bleak_value *p_stNewValue = bleak_value_alloc();
                        p_stNewValue->eType = BLEAK_EXPRESSION;
                        p_stNewValue->p_stValues = bleak_vector_value_alloc();
                        bleak_vector_value_push_back(p_stNewValue->p_stValues, yystack.l_mark[0].p_stValue);

                        yyval.p_stValue = p_stNewValue;
                      }
break;
case 16:
  if (!yytrial)
#line 260 "GraphGrammar.y"
	{ yyval.p_stValue = yystack.l_mark[-1].p_stValue ; }
break;
case 17:
  if (!yytrial)
#line 262 "GraphGrammar.y"
	{ 
                        bleak_value *p_stOpValue = bleak_value_alloc();
                        p_stOpValue->eType = BLEAK_OPCODE;
                        p_stOpValue->iValue = 'N';

                        bleak_vector_value_push_back(yystack.l_mark[0].p_stValue->p_stValues , p_stOpValue);

                        yyval.p_stValue = yystack.l_mark[0].p_stValue ;
                      }
break;
case 18:
  if (!yytrial)
#line 272 "GraphGrammar.y"
	{
                        bleak_value *p_stA = yystack.l_mark[-2].p_stValue;
                        bleak_value *p_stB = yystack.l_mark[0].p_stValue;

                        bleak_vector_value_join(p_stA->p_stValues, p_stB->p_stValues);

                        bleak_vector_value_free(p_stB->p_stValues); /* Does not free elements of vector */
                        p_stB->p_stValues = NULL;

                        bleak_value_free(p_stB);

                        bleak_value *p_stOpValue = bleak_value_alloc();
                        p_stOpValue->eType = BLEAK_OPCODE;
                        p_stOpValue->iValue = '+';

                        bleak_vector_value_push_back(p_stA->p_stValues, p_stOpValue);

                        yyval.p_stValue = p_stA;
                      }
break;
case 19:
  if (!yytrial)
#line 292 "GraphGrammar.y"
	{
                        bleak_value *p_stA = yystack.l_mark[-2].p_stValue;
                        bleak_value *p_stB = yystack.l_mark[0].p_stValue;

                        bleak_vector_value_join(p_stA->p_stValues, p_stB->p_stValues);

                        bleak_vector_value_free(p_stB->p_stValues); /* Does not free elements of vector */
                        p_stB->p_stValues = NULL;

                        bleak_value_free(p_stB);

                        bleak_value *p_stOpValue = bleak_value_alloc();
                        p_stOpValue->eType = BLEAK_OPCODE;
                        p_stOpValue->iValue = '-';

                        bleak_vector_value_push_back(p_stA->p_stValues, p_stOpValue);

                        yyval.p_stValue = p_stA;
                      }
break;
case 20:
  if (!yytrial)
#line 312 "GraphGrammar.y"
	{
                        bleak_value *p_stA = yystack.l_mark[-2].p_stValue;
                        bleak_value *p_stB = yystack.l_mark[0].p_stValue;

                        bleak_vector_value_join(p_stA->p_stValues, p_stB->p_stValues);

                        bleak_vector_value_free(p_stB->p_stValues); /* Does not free elements of vector */
                        p_stB->p_stValues = NULL;

                        bleak_value_free(p_stB);

                        bleak_value *p_stOpValue = bleak_value_alloc();
                        p_stOpValue->eType = BLEAK_OPCODE;
                        p_stOpValue->iValue = '/';

                        bleak_vector_value_push_back(p_stA->p_stValues, p_stOpValue);

                        yyval.p_stValue = p_stA;
                      }
break;
case 21:
  if (!yytrial)
#line 332 "GraphGrammar.y"
	{
                        bleak_value *p_stA = yystack.l_mark[-2].p_stValue;
                        bleak_value *p_stB = yystack.l_mark[0].p_stValue;

                        bleak_vector_value_join(p_stA->p_stValues, p_stB->p_stValues);

                        bleak_vector_value_free(p_stB->p_stValues); /* Does not free elements of vector */
                        p_stB->p_stValues = NULL;

                        bleak_value_free(p_stB);

                        bleak_value *p_stOpValue = bleak_value_alloc();
                        p_stOpValue->eType = BLEAK_OPCODE;
                        p_stOpValue->iValue = '*';

                        bleak_vector_value_push_back(p_stA->p_stValues, p_stOpValue);

                        yyval.p_stValue = p_stA;
                      }
break;
case 22:
  if (!yytrial)
#line 352 "GraphGrammar.y"
	{
                        bleak_value *p_stA = yystack.l_mark[-2].p_stValue;
                        bleak_value *p_stB = yystack.l_mark[0].p_stValue;

                        bleak_vector_value_join(p_stA->p_stValues, p_stB->p_stValues);

                        bleak_vector_value_free(p_stB->p_stValues); /* Does not free elements of vector */
                        p_stB->p_stValues = NULL;

                        bleak_value_free(p_stB);

                        bleak_value *p_stOpValue = bleak_value_alloc();
                        p_stOpValue->eType = BLEAK_OPCODE;
                        p_stOpValue->iValue = '^';

                        bleak_vector_value_push_back(p_stA->p_stValues, p_stOpValue);

                        yyval.p_stValue = p_stA;
                      }
break;
case 23:
  if (!yytrial)
#line 372 "GraphGrammar.y"
	{
                        bleak_value *p_stA = yystack.l_mark[-2].p_stValue;
                        bleak_value *p_stB = yystack.l_mark[0].p_stValue;

                        bleak_vector_value_join(p_stA->p_stValues, p_stB->p_stValues);

                        bleak_vector_value_free(p_stB->p_stValues); /* Does not free elements of vector */
                        p_stB->p_stValues = NULL;

                        bleak_value_free(p_stB);

                        bleak_value *p_stOpValue = bleak_value_alloc();
                        p_stOpValue->eType = BLEAK_OPCODE;
                        p_stOpValue->iValue = '%';

                        bleak_vector_value_push_back(p_stA->p_stValues, p_stOpValue);

                        yyval.p_stValue = p_stA;
                      }
break;
case 24:
  if (!yytrial)
#line 394 "GraphGrammar.y"
	{
                    bleak_value *p_stValue = bleak_value_alloc();
                    p_stValue->eType = BLEAK_INTEGER;
                    p_stValue->iValue = yystack.l_mark[0].iValue ;
                    yyval.p_stValue = p_stValue;
                  }
break;
case 25:
  if (!yytrial)
#line 401 "GraphGrammar.y"
	{
                    bleak_value *p_stValue = bleak_value_alloc();
                    p_stValue->eType = BLEAK_FLOAT;
                    p_stValue->fValue = yystack.l_mark[0].fValue ;
                    yyval.p_stValue = p_stValue;
                  }
break;
case 26:
  if (!yytrial)
#line 408 "GraphGrammar.y"
	{
                    bleak_value *p_stValue = bleak_value_alloc();
                    p_stValue->eType = BLEAK_BOOL;
                    p_stValue->bValue = yystack.l_mark[0].bValue ;
                    yyval.p_stValue = p_stValue;
                  }
break;
case 27:
  if (!yytrial)
#line 415 "GraphGrammar.y"
	{
                    bleak_value *p_stValue = bleak_value_alloc();
                    p_stValue->eType = BLEAK_STRING;
                    p_stValue->p_cValue = yystack.l_mark[0].p_cValue ;
                    yyval.p_stValue = p_stValue;
                  }
break;
case 28:
  if (!yytrial)
#line 422 "GraphGrammar.y"
	{
                    bleak_value *p_stValue = bleak_value_alloc();
                    p_stValue->eType = BLEAK_REFERENCE;
                    p_stValue->p_cValue = yystack.l_mark[0].p_cValue ;
                    yyval.p_stValue = p_stValue;
                  }
break;
case 29:
  if (!yytrial)
#line 432 "GraphGrammar.y"
	{
                    yyval.p_stValues = bleak_vector_value_alloc();
                    bleak_vector_value_push_back( yyval.p_stValues, yystack.l_mark[0].p_stValue );
                  }
break;
case 30:
  if (!yytrial)
#line 437 "GraphGrammar.y"
	{
                    yyval.p_stValues = yystack.l_mark[-2].p_stValues ;
                    bleak_vector_value_push_back( yyval.p_stValues , yystack.l_mark[0].p_stValue );
                  }
break;
case 31:
  if (!yytrial)
#line 444 "GraphGrammar.y"
	{
              bleak_value *p_stValue = bleak_value_alloc();
              p_stValue->eType = BLEAK_INTEGER_VECTOR;
              p_stValue->p_stIVValue = bleak_vector_int_alloc();
              yyval.p_stValue = p_stValue;
            }
break;
case 32:
  if (!yytrial)
#line 451 "GraphGrammar.y"
	{
              bleak_value *p_stValue = bleak_value_alloc();
              p_stValue->eType = BLEAK_VALUE_VECTOR;
              p_stValue->p_stValues = yystack.l_mark[-1].p_stValues ;
              yyval.p_stValue = p_stValue;
            }
break;
case 33:
  if (!yytrial)
#line 460 "GraphGrammar.y"
	{
            yyval.p_stValue = yystack.l_mark[0].p_stValue ;
          }
break;
case 34:
  if (!yytrial)
#line 464 "GraphGrammar.y"
	{
            yyval.p_stValue = yystack.l_mark[0].p_stValue ;
          }
break;
case 35:
  if (!yytrial)
#line 470 "GraphGrammar.y"
	{ 
                yyval.p_stVertices = bleak_vector_vertex_alloc(); 
                bleak_vector_vertex_push_back( yyval.p_stVertices , yystack.l_mark[0].p_stVertex );
              }
break;
case 36:
  if (!yytrial)
#line 475 "GraphGrammar.y"
	{
                yyval.p_stVertices = yystack.l_mark[-1].p_stVertices ;
                bleak_vector_vertex_push_back( yyval.p_stVertices , yystack.l_mark[0].p_stVertex );
              }
break;
case 37:
  if (!yytrial)
#line 482 "GraphGrammar.y"
	{
                bleak_vertex *p_stVertex = bleak_vertex_alloc();
                p_stVertex->p_cType = yystack.l_mark[-5].p_cValue ;
                p_stVertex->p_cName = yystack.l_mark[-1].p_cValue ;
                p_stVertex->p_stProperties = yystack.l_mark[-3].p_stKvps ;
                yyval.p_stVertex = p_stVertex;
              }
break;
case 38:
  if (!yytrial)
#line 490 "GraphGrammar.y"
	{
                bleak_vertex *p_stVertex = bleak_vertex_alloc();
                p_stVertex->p_cType = yystack.l_mark[-2].p_cValue ;
                p_stVertex->p_cName = yystack.l_mark[-1].p_cValue ;
                p_stVertex->p_stProperties = bleak_vector_kvp_alloc(); /* Empty properties */
                yyval.p_stVertex = p_stVertex;
              }
break;
case 39:
  if (!yytrial)
#line 500 "GraphGrammar.y"
	{ 
                  yyval.p_stConnections = bleak_vector_connection_alloc(); 
                  bleak_vector_connection_push_back( yyval.p_stConnections , yystack.l_mark[0].p_stConnection );
                }
break;
case 40:
  if (!yytrial)
#line 505 "GraphGrammar.y"
	{
                  yyval.p_stConnections = yystack.l_mark[-1].p_stConnections ;
                  bleak_vector_connection_push_back( yyval.p_stConnections , yystack.l_mark[0].p_stConnection );
                }
break;
case 41:
  if (!yytrial)
#line 512 "GraphGrammar.y"
	{
                  bleak_connection *p_stConnection = bleak_connection_alloc();
                  p_stConnection->p_cSourceName = yystack.l_mark[-7].p_cValue ;
                  p_stConnection->p_cOutputName = yystack.l_mark[-5].p_cValue ;
                  p_stConnection->p_cTargetName = yystack.l_mark[-3].p_cValue ;
                  p_stConnection->p_cInputName = yystack.l_mark[-1].p_cValue ;
                  yyval.p_stConnection = p_stConnection;
                }
break;
case 42:
  if (!yytrial)
#line 521 "GraphGrammar.y"
	{
                  bleak_connection *p_stConnection = bleak_connection_alloc();
                  p_stConnection->p_cSourceName = yystack.l_mark[-3].p_cValue ;
                  p_stConnection->p_cOutputName = yystack.l_mark[-1].p_cValue ;
                  p_stConnection->p_cTargetName = yystack.l_mark[-7].p_cValue ;
                  p_stConnection->p_cInputName = yystack.l_mark[-5].p_cValue ;
                  yyval.p_stConnection = p_stConnection;
                }
break;
case 43:
  if (!yytrial)
#line 532 "GraphGrammar.y"
	{ yyval.p_cValue = yystack.l_mark[0].p_cValue ; }
break;
case 44:
  if (!yytrial)
#line 534 "GraphGrammar.y"
	{ yyval.p_cValue = strdup("this") ; }
break;
#line 1822 "y.tab.c"
    default:
        break;
    }
    yystack.s_mark -= yym;
    yystate = *yystack.s_mark;
    yystack.l_mark -= yym;
#if defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED)
    yystack.p_mark -= yym;
#endif
    yym = yylhs[yyn];
    if (yystate == 0 && yym == 0)
    {
#if YYDEBUG
        if (yydebug)
        {
            fprintf(stderr, "%s[%d]: after reduction, ", YYDEBUGSTR, yydepth);
#ifdef YYSTYPE_TOSTRING
#if YYBTYACC
            if (!yytrial)
#endif /* YYBTYACC */
                fprintf(stderr, "result is <%s>, ", YYSTYPE_TOSTRING(yystos[YYFINAL], yyval));
#endif
            fprintf(stderr, "shifting from state 0 to final state %d\n", YYFINAL);
        }
#endif
        yystate = YYFINAL;
        *++yystack.s_mark = YYFINAL;
        *++yystack.l_mark = yyval;
#if defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED)
        *++yystack.p_mark = yyloc;
#endif
        if (yychar < 0)
        {
#if YYBTYACC
            do {
            if (yylvp < yylve)
            {
                /* we're currently re-reading tokens */
                yylval = *yylvp++;
#if defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED)
                yylloc = *yylpp++;
#endif
                yychar = *yylexp++;
                break;
            }
            if (yyps->save)
            {
                /* in trial mode; save scanner results for future parse attempts */
                if (yylvp == yylvlim)
                {   /* Enlarge lexical value queue */
                    size_t p = (size_t) (yylvp - yylvals);
                    size_t s = (size_t) (yylvlim - yylvals);

                    s += YYLVQUEUEGROWTH;
                    if ((yylexemes = (YYINT *)realloc(yylexemes, s * sizeof(YYINT))) == NULL)
                        goto yyenomem;
                    if ((yylvals   = (YYSTYPE *)realloc(yylvals, s * sizeof(YYSTYPE))) == NULL)
                        goto yyenomem;
#if defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED)
                    if ((yylpsns   = (YYLTYPE *)realloc(yylpsns, s * sizeof(YYLTYPE))) == NULL)
                        goto yyenomem;
#endif
                    yylvp   = yylve = yylvals + p;
                    yylvlim = yylvals + s;
#if defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED)
                    yylpp   = yylpe = yylpsns + p;
                    yylplim = yylpsns + s;
#endif
                    yylexp  = yylexemes + p;
                }
                *yylexp = (YYINT) YYLEX;
                *yylvp++ = yylval;
                yylve++;
#if defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED)
                *yylpp++ = yylloc;
                yylpe++;
#endif
                yychar = *yylexp++;
                break;
            }
            /* normal operation, no conflict encountered */
#endif /* YYBTYACC */
            yychar = YYLEX;
#if YYBTYACC
            } while (0);
#endif /* YYBTYACC */
            if (yychar < 0) yychar = YYEOF;
#if YYDEBUG
            if (yydebug)
            {
                if ((yys = yyname[YYTRANSLATE(yychar)]) == NULL) yys = yyname[YYUNDFTOKEN];
                fprintf(stderr, "%s[%d]: state %d, reading token %d (%s)\n",
                                YYDEBUGSTR, yydepth, YYFINAL, yychar, yys);
            }
#endif
        }
        if (yychar == YYEOF) goto yyaccept;
        goto yyloop;
    }
    if (((yyn = yygindex[yym]) != 0) && (yyn += yystate) >= 0 &&
            yyn <= YYTABLESIZE && yycheck[yyn] == (YYINT) yystate)
        yystate = yytable[yyn];
    else
        yystate = yydgoto[yym];
#if YYDEBUG
    if (yydebug)
    {
        fprintf(stderr, "%s[%d]: after reduction, ", YYDEBUGSTR, yydepth);
#ifdef YYSTYPE_TOSTRING
#if YYBTYACC
        if (!yytrial)
#endif /* YYBTYACC */
            fprintf(stderr, "result is <%s>, ", YYSTYPE_TOSTRING(yystos[yystate], yyval));
#endif
        fprintf(stderr, "shifting from state %d to state %d\n", *yystack.s_mark, yystate);
    }
#endif
    if (yystack.s_mark >= yystack.s_last && yygrowstack(&yystack) == YYENOMEM) goto yyoverflow;
    *++yystack.s_mark = (YYINT) yystate;
    *++yystack.l_mark = yyval;
#if defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED)
    *++yystack.p_mark = yyloc;
#endif
    goto yyloop;
#if YYBTYACC

    /* Reduction declares that this path is valid. Set yypath and do a full parse */
yyvalid:
    if (yypath) YYABORT;
    while (yyps->save)
    {
        YYParseState *save = yyps->save;
        yyps->save = save->save;
        save->save = yypath;
        yypath = save;
    }
#if YYDEBUG
    if (yydebug)
        fprintf(stderr, "%s[%d]: state %d, CONFLICT trial successful, backtracking to state %d, %d tokens\n",
                        YYDEBUGSTR, yydepth, yystate, yypath->state, (int)(yylvp - yylvals - yypath->lexeme));
#endif
    if (yyerrctx)
    {
        yyFreeState(yyerrctx);
        yyerrctx = NULL;
    }
    yylvp          = yylvals + yypath->lexeme;
#if defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED)
    yylpp          = yylpsns + yypath->lexeme;
#endif
    yylexp         = yylexemes + yypath->lexeme;
    yychar         = YYEMPTY;
    yystack.s_mark = yystack.s_base + (yypath->yystack.s_mark - yypath->yystack.s_base);
    memcpy (yystack.s_base, yypath->yystack.s_base, (size_t) (yystack.s_mark - yystack.s_base + 1) * sizeof(YYINT));
    yystack.l_mark = yystack.l_base + (yypath->yystack.l_mark - yypath->yystack.l_base);
    memcpy (yystack.l_base, yypath->yystack.l_base, (size_t) (yystack.l_mark - yystack.l_base + 1) * sizeof(YYSTYPE));
#if defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED)
    yystack.p_mark = yystack.p_base + (yypath->yystack.p_mark - yypath->yystack.p_base);
    memcpy (yystack.p_base, yypath->yystack.p_base, (size_t) (yystack.p_mark - yystack.p_base + 1) * sizeof(YYLTYPE));
#endif
    yystate        = yypath->state;
    goto yyloop;
#endif /* YYBTYACC */

yyoverflow:
    YYERROR_CALL("yacc stack overflow");
#if YYBTYACC
    goto yyabort_nomem;
yyenomem:
    YYERROR_CALL("memory exhausted");
yyabort_nomem:
#endif /* YYBTYACC */
    yyresult = 2;
    goto yyreturn;

yyabort:
    yyresult = 1;
    goto yyreturn;

yyaccept:
#if YYBTYACC
    if (yyps->save) goto yyvalid;
#endif /* YYBTYACC */
    yyresult = 0;

yyreturn:
#if defined(YYDESTRUCT_CALL)
    if (yychar != YYEOF && yychar != YYEMPTY)
#if defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED)
        YYDESTRUCT_CALL("cleanup: discarding token", yychar, &yylval, &yylloc);
#else
        YYDESTRUCT_CALL("cleanup: discarding token", yychar, &yylval);
#endif /* defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED) */

    {
        YYSTYPE *pv;
#if defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED)
        YYLTYPE *pp;

        for (pv = yystack.l_base, pp = yystack.p_base; pv <= yystack.l_mark; ++pv, ++pp)
             YYDESTRUCT_CALL("cleanup: discarding state",
                             yystos[*(yystack.s_base + (pv - yystack.l_base))], pv, pp);
#else
        for (pv = yystack.l_base; pv <= yystack.l_mark; ++pv)
             YYDESTRUCT_CALL("cleanup: discarding state",
                             yystos[*(yystack.s_base + (pv - yystack.l_base))], pv);
#endif /* defined(YYLTYPE) || defined(YYLTYPE_IS_DECLARED) */
    }
#endif /* defined(YYDESTRUCT_CALL) */

#if YYBTYACC
    if (yyerrctx)
    {
        yyFreeState(yyerrctx);
        yyerrctx = NULL;
    }
    while (yyps)
    {
        YYParseState *save = yyps;
        yyps = save->save;
        save->save = NULL;
        yyFreeState(save);
    }
    while (yypath)
    {
        YYParseState *save = yypath;
        yypath = save->save;
        save->save = NULL;
        yyFreeState(save);
    }
#endif /* YYBTYACC */
    yyfreestack(&yystack);
    return (yyresult);
}
