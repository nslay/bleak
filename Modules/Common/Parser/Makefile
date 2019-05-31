yacc=$(HOME)/bin/yacc
flex=$(HOME)/bin/flex

all : y.tab.c lex.yy.c

y.tab.c : GraphGrammar.y
	$(yacc) -B -d -v -P GraphGrammar.y

lex.yy.c : GraphLexer.l y.tab.h
	$(flex) --header-file=lex.yy.h GraphLexer.l


