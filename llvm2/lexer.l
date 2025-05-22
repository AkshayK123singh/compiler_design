%option noyywrap
%option nounput
%option noinput

%{
#include "parser.hpp"
#include <cstdio>
#include <cstdlib> // For atoi
#include <cstring> // For strdup

extern YYSTYPE yylval;
void yyerror(const char *s); // Declare yyerror
%}

DIGIT     [0-9]
ID        [a-zA-Z_][a-zA-Z0-9_]*
WS        [ \t\r\n]+

%%

"int"           { return INT; }
"char"          { return CHAR; }
"return"        { return RETURN; }
"if"            { return IF; }
"else"          { return ELSE; }
"while"         { return WHILE; }
"for"           { return FOR; }
"void"          { return VOID; }
"main"          { return MAIN; } // This returns the token MAIN

{DIGIT}+        {
                    yylval.ival = atoi(yytext);
                    return NUMBER;
                }
\"([^"\\]|\\.)*\" { // String literal handling
                    char* str_content = (char*)malloc(yyleng - 1);
                    if (str_content) {
                        strncpy(str_content, yytext + 1, yyleng - 2);
                        str_content[yyleng - 2] = '\0';
                        yylval.sval = str_content;
                    } else {
                        yyerror("Memory allocation failed for string literal");
                    }
                    return STRING_LITERAL;
                }
{ID}            {
                    yylval.sval = strdup(yytext);
                    return IDENTIFIER;
                }
"=="            { return EQ; }
"!="            { return NE; }
"<="            { return LE; }
">="            { return GE; }
"<"             { return '<'; }
">"             { return '>'; }
"="             { return '='; }
"+"             { return '+'; }
"-"             { return '-'; }
"*"             { return '*'; }
"/"             { return '/'; }
"("             { return '('; }
")"             { return ')'; }
"{"             { return '{'; }
"}"             { return '}'; }
";"             { return ';'; }
","             { return ','; }
"&"             { return AMPERSAND; }

"//".* { /* ignore single-line comments */ }
"/*"([^*]|\*+[^*/])*\*+"/" { /* ignore multi-line comments */ }

{WS}            { /* ignore whitespace */ }

.               { fprintf(stderr,"Unknown char: %s\n", yytext); return yytext[0]; }

%%