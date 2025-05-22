%code requires {
    #include <string>    // For std::string
    #include <vector>    // For std::vector
    #include <memory>    // For std::unique_ptr
    #include <utility>   // For std::move

    // Add necessary LLVM headers here for types used in %union and AST structs
    #include "llvm/IR/Type.h"
    #include "llvm/IR/Value.h"
    #include "llvm/IR/Instructions.h" // For AllocaInst etc.
    // Potentially other headers if needed by other AST fields, e.g.,
    // #include "llvm/IR/BasicBlock.h"

    // Forward declarations for AST classes.
    struct ExprAST;
    struct StatementAST;
    struct PrototypeAST;
    struct FunctionAST;
}

%{
#include <cstdio>
#include <cstdlib>
#include <cstring> // For strdup, strlen
#include <iostream>
#include <map>

// Include the generated parser header *here* so token definitions are visible
// to the C++ code within this block.
#include "parser.hpp"

#include "llvm/IR/Value.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
// #include "llvm/IR/Instructions.h" // Already included via %code requires for global AST structures


// Global LLVM objects defined in main.cpp, now unique_ptrs
extern std::unique_ptr<llvm::LLVMContext> TheContext;
extern std::unique_ptr<llvm::IRBuilder<>> Builder;
extern std::unique_ptr<llvm::Module> TheModule;
extern std::map<std::string, llvm::AllocaInst*> NamedValues;
extern std::unique_ptr<llvm::orc::LLJIT> TheJIT;

extern int yylex();
extern int yyparse();
extern FILE* yyin;
void yyerror(const char *s);

llvm::Value *LogErrorV(const char *Str) {
    fprintf(stderr, "Error: %s\n", Str);
    return nullptr;
}

llvm::AllocaInst *CreateEntryBlockAlloca(llvm::Function *TheFunction, const std::string &VarName, llvm::Type *VarType) {
    llvm::IRBuilder<> TmpB(&TheFunction->getEntryBlock(), TheFunction->getEntryBlock().begin());
    return TmpB.CreateAlloca(VarType, nullptr, VarName.c_str());
}

llvm::Function *getFunction(const std::string &Name) {
    if (auto *F = TheModule->getFunction(Name))
        return F;
    return nullptr;
}

// Full definitions of AST classes
struct ExprAST {
    virtual ~ExprAST() = default;
    virtual llvm::Value *codegen() = 0;
};

struct NumberExprAST : public ExprAST {
    int Val;
    NumberExprAST(int Val) : Val(Val) {}
    llvm::Value *codegen() override {
        return llvm::ConstantInt::get(llvm::Type::getInt32Ty(*TheContext), Val);
    }
};

struct VariableExprAST : public ExprAST {
    std::string Name;
    VariableExprAST(const char *N) : Name(N) {}
    llvm::Value *codegen() override {
        llvm::AllocaInst *A = NamedValues[Name];
        if (!A) return LogErrorV("Unknown variable name");
        return Builder->CreateLoad(A->getAllocatedType(), A, Name.c_str());
    }
};

// New AST node for String Literals
struct StringLiteralExprAST : public ExprAST {
    std::string Value;
    StringLiteralExprAST(const char *Str) : Value(Str) {}
    llvm::Value *codegen() override {
        return Builder->CreateGlobalStringPtr(Value.c_str(), "str");
    }
};

// New AST node for Call Expressions (e.g., printf, scanf, user-defined functions)
struct CallExprAST : public ExprAST {
    std::string Callee;
    std::vector<std::unique_ptr<ExprAST>> Args;
    // Constructor changed to take vector by value for easy move semantics
    CallExprAST(const char *callee, std::vector<ExprAST*> args) : Callee(callee) {
        for (auto A : args) Args.push_back(std::unique_ptr<ExprAST>(A));
    }
    llvm::Value *codegen() override {
        llvm::Function *CalleeF = getFunction(Callee);
        if (!CalleeF) return LogErrorV("Unknown function referenced");

        if (CalleeF->arg_size() != Args.size() && !CalleeF->isVarArg())
            return LogErrorV("Incorrect number of arguments passed to function");

        std::vector<llvm::Value*> ArgsV;
        for (unsigned i = 0, e = Args.size(); i != e; ++i) {
            ArgsV.push_back(Args[i]->codegen());
            if (!ArgsV.back()) return nullptr;
        }

        return Builder->CreateCall(CalleeF, ArgsV, "calltmp");
    }
};

// New AST node for Address-of operator (&)
struct AddressOfExprAST : public ExprAST {
    std::string VarName;
    AddressOfExprAST(const char *N) : VarName(N) {}
    llvm::Value *codegen() override {
        llvm::AllocaInst *A = NamedValues[VarName];
        if (!A) return LogErrorV("Unknown variable name for address-of");
        return A;
    }
};


struct BinaryExprAST : public ExprAST {
    int Op;
    std::unique_ptr<ExprAST> LHS, RHS;
    BinaryExprAST(int op, ExprAST *lhs, ExprAST *rhs) : Op(op), LHS(lhs), RHS(rhs) {}
    llvm::Value *codegen() override {
        llvm::Value *L = LHS->codegen();
        llvm::Value *R = RHS->codegen();
        if (!L || !R) return nullptr;

        switch (Op) {
            case '+': return Builder->CreateAdd(L, R, "addtmp");
            case '-': return Builder->CreateSub(L, R, "subtmp");
            case '*': return Builder->CreateMul(L, R, "multmp");
            case '/': return Builder->CreateSDiv(L, R, "divtmp");
            case '<':
                L = Builder->CreateICmpSLT(L, R, "cmptmp");
                return Builder->CreateZExt(L, llvm::Type::getInt32Ty(*TheContext), "booltmp");
            case '>':
                L = Builder->CreateICmpSGT(L, R, "cmptmp");
                return Builder->CreateZExt(L, llvm::Type::getInt32Ty(*TheContext), "booltmp");
            case EQ: {
                L = Builder->CreateICmpEQ(L, R, "cmptmp");
                return Builder->CreateZExt(L, llvm::Type::getInt32Ty(*TheContext), "booltmp");
            }
            case NE: {
                L = Builder->CreateICmpNE(L, R, "cmptmp");
                return Builder->CreateZExt(L, llvm::Type::getInt32Ty(*TheContext), "booltmp");
            }
            case LE: {
                L = Builder->CreateICmpSLE(L, R, "cmptmp");
                return Builder->CreateZExt(L, llvm::Type::getInt32Ty(*TheContext), "booltmp");
            }
            case GE: {
                L = Builder->CreateICmpSGE(L, R, "cmptmp");
                return Builder->CreateZExt(L, llvm::Type::getInt32Ty(*TheContext), "booltmp");
            }
            default:
                return LogErrorV("Invalid binary operator");
        }
    }
};

struct AssignmentExprAST : public ExprAST {
    std::string VarName;
    std::unique_ptr<ExprAST> Expr;
    AssignmentExprAST(const char *Name, ExprAST *E) : VarName(Name), Expr(E) {}
    llvm::Value *codegen() override {
        llvm::Value *Val = Expr->codegen();
        if (!Val) return nullptr;
        llvm::AllocaInst *Var = NamedValues[VarName];
        if (!Var) return LogErrorV("Unknown variable name in assignment");
        Builder->CreateStore(Val, Var);
        return Val;
    }
};

struct StatementAST {
    virtual ~StatementAST() = default;
    virtual llvm::Value *codegen() = 0;
};

// New AST node for Variable Declarations (e.g., int a;)
struct VarDeclStatementAST : public StatementAST {
    std::string VarName;
    llvm::Type *VarType; // Store the LLVM Type directly
    VarDeclStatementAST(const char *N, llvm::Type *T) : VarName(N), VarType(T) {}
    llvm::Value *codegen() override {
        llvm::Function *TheFunction = Builder->GetInsertBlock()->getParent();
        llvm::AllocaInst *Alloca = CreateEntryBlockAlloca(TheFunction, VarName, VarType);
        NamedValues[VarName] = Alloca;
        return Alloca; // Return the alloca instruction
    }
};


struct ExprStatementAST : public StatementAST {
    std::unique_ptr<ExprAST> Expr;
    ExprStatementAST(ExprAST *E) : Expr(E) {}
    llvm::Value *codegen() override {
        if (Expr) return Expr->codegen();
        return llvm::Constant::getNullValue(llvm::Type::getInt32Ty(*TheContext)); // Or void type if applicable
    }
};

struct ReturnStatementAST : public StatementAST {
    std::unique_ptr<ExprAST> Expr;
    ReturnStatementAST(ExprAST *E) : Expr(E) {}
    llvm::Value *codegen() override {
        llvm::Value *RetVal = nullptr;
        if (Expr) {
            RetVal = Expr->codegen();
            if (!RetVal) return nullptr;
        } else {
            RetVal = llvm::Constant::getNullValue(llvm::Type::getInt32Ty(*TheContext)); // Default to 0 for now
        }
        return Builder->CreateRet(RetVal);
    }
};

struct CompoundStatementAST : public StatementAST {
    std::vector<std::unique_ptr<StatementAST>> Statements;
    CompoundStatementAST(const std::vector<StatementAST*>& stmts) {
        for (auto s : stmts) Statements.push_back(std::unique_ptr<StatementAST>(s));
    }
    llvm::Value *codegen() override {
        llvm::Value *Last = nullptr;
        for (auto &s : Statements) {
            Last = s->codegen();
            if (!Last) return nullptr;
        }
        return Last;
    }
};

struct IfStatementAST : public StatementAST {
    std::unique_ptr<ExprAST> Cond;
    std::unique_ptr<StatementAST> Then, Else;
    IfStatementAST(ExprAST *cond, StatementAST *thenStmt, StatementAST *elseStmt)
        : Cond(cond), Then(thenStmt), Else(elseStmt) {}
    llvm::Value *codegen() override {
        llvm::Value *CondV = Cond->codegen();
        if (!CondV) return nullptr;

        CondV = Builder->CreateICmpNE(CondV, llvm::ConstantInt::get(llvm::Type::getInt32Ty(*TheContext), 0), "ifcond");

        llvm::Function *TheFunction = Builder->GetInsertBlock()->getParent();

        llvm::BasicBlock *ThenBB = llvm::BasicBlock::Create(*TheContext, "then", TheFunction);
        llvm::BasicBlock *ElseBB = llvm::BasicBlock::Create(*TheContext, "else", TheFunction);
        llvm::BasicBlock *MergeBB = llvm::BasicBlock::Create(*TheContext, "ifcont", TheFunction);

        if (Else) {
            Builder->CreateCondBr(CondV, ThenBB, ElseBB);
        } else {
            Builder->CreateCondBr(CondV, ThenBB, MergeBB);
        }

        Builder->SetInsertPoint(ThenBB);
        llvm::Value *ThenV = Then->codegen();
        if (!ThenV) return nullptr;
        if (Builder->GetInsertBlock()->getTerminator() == nullptr)
            Builder->CreateBr(MergeBB);

        if (Else) {
            Builder->SetInsertPoint(ElseBB);
            llvm::Value *ElseV = Else->codegen();
            if (!ElseV) return nullptr;
            if (Builder->GetInsertBlock()->getTerminator() == nullptr)
                Builder->CreateBr(MergeBB);
        }

        Builder->SetInsertPoint(MergeBB);

        return llvm::Constant::getNullValue(llvm::Type::getInt32Ty(*TheContext));
    }
};

struct WhileStatementAST : public StatementAST {
    std::unique_ptr<ExprAST> Cond;
    std::unique_ptr<StatementAST> Body;
    WhileStatementAST(ExprAST *cond, StatementAST *body) : Cond(cond), Body(body) {}
    llvm::Value *codegen() override {
        llvm::Function *TheFunction = Builder->GetInsertBlock()->getParent();

        llvm::BasicBlock *CondBB = llvm::BasicBlock::Create(*TheContext, "loopcond", TheFunction);
        llvm::BasicBlock *LoopBB = llvm::BasicBlock::Create(*TheContext, "loop", TheFunction);
        llvm::BasicBlock *AfterBB = llvm::BasicBlock::Create(*TheContext, "afterloop", TheFunction);

        Builder->CreateBr(CondBB);

        Builder->SetInsertPoint(CondBB);
        llvm::Value *CondV = Cond->codegen();
        if (!CondV) return nullptr;

        CondV = Builder->CreateICmpNE(CondV, llvm::ConstantInt::get(llvm::Type::getInt32Ty(*TheContext), 0), "loopcond");

        Builder->CreateCondBr(CondV, LoopBB, AfterBB);

        Builder->SetInsertPoint(LoopBB);
        if (!Body->codegen()) return nullptr;

        if (Builder->GetInsertBlock()->getTerminator() == nullptr)
            Builder->CreateBr(CondBB);

        Builder->SetInsertPoint(AfterBB);

        return llvm::Constant::getNullValue(llvm::Type::getInt32Ty(*TheContext));
    }
};

struct ForStatementAST : public StatementAST {
    std::unique_ptr<StatementAST> Init;
    std::unique_ptr<ExprAST> Cond;
    std::unique_ptr<StatementAST> Step;
    std::unique_ptr<StatementAST> Body;
    ForStatementAST(StatementAST *init, ExprAST *cond, StatementAST *step, StatementAST *body)
        : Init(init), Cond(cond), Step(step), Body(body) {}
    llvm::Value *codegen() override {
        llvm::Function *TheFunction = Builder->GetInsertBlock()->getParent();

        if (Init) {
            if (!Init->codegen()) return nullptr;
        }

        llvm::BasicBlock *CondBB = llvm::BasicBlock::Create(*TheContext, "forcond", TheFunction);
        llvm::BasicBlock *LoopBB = llvm::BasicBlock::Create(*TheContext, "forloop", TheFunction);
        llvm::BasicBlock *AfterBB = llvm::BasicBlock::Create(*TheContext, "afterfor", TheFunction);

        Builder->CreateBr(CondBB);

        Builder->SetInsertPoint(CondBB);
        llvm::Value *CondV = nullptr;
        if (Cond) {
            CondV = Cond->codegen();
            if (!CondV) return nullptr;
            CondV = Builder->CreateICmpNE(CondV, llvm::ConstantInt::get(llvm::Type::getInt32Ty(*TheContext), 0), "forcond");
        } else {
            CondV = llvm::ConstantInt::getTrue(*TheContext);
        }

        Builder->CreateCondBr(CondV, LoopBB, AfterBB);

        Builder->SetInsertPoint(LoopBB);
        if (!Body->codegen()) return nullptr;

        if (Step) {
            if (!Step->codegen()) return nullptr;
        }

        if (Builder->GetInsertBlock()->getTerminator() == nullptr)
            Builder->CreateBr(CondBB);

        Builder->SetInsertPoint(AfterBB);

        return llvm::Constant::getNullValue(llvm::Type::getInt32Ty(*TheContext));
    }
};


struct PrototypeAST {
    std::string Name;
    std::vector<std::string> Args;
    llvm::Type *RetType; // New: return type
    PrototypeAST(const std::string &name, const std::vector<std::string> &args, llvm::Type *retType)
        : Name(name), Args(args), RetType(retType) {}
    llvm::Function *codegen() {
        std::vector<llvm::Type*> ArgTypes(Args.size(), llvm::Type::getInt32Ty(*TheContext));
        llvm::FunctionType *FT = llvm::FunctionType::get(RetType, ArgTypes, false);
        llvm::Function *F = llvm::Function::Create(FT, llvm::Function::ExternalLinkage, Name, TheModule.get());

        unsigned Idx = 0;
        for (auto &Arg : F->args())
            Arg.setName(Args[Idx++]);

        return F;
    }
};

struct FunctionAST {
    std::unique_ptr<PrototypeAST> Proto;
    std::unique_ptr<StatementAST> Body;
    FunctionAST(PrototypeAST *proto, StatementAST *body) : Proto(proto), Body(body) {}
    llvm::Function *codegen() {
        NamedValues.clear();

        llvm::Function *TheFunction = getFunction(Proto->Name);
        if (!TheFunction) {
            TheFunction = Proto->codegen();
        }

        if (!TheFunction) return nullptr;

        llvm::BasicBlock *BB = llvm::BasicBlock::Create(*TheContext, "entry", TheFunction);
        Builder->SetInsertPoint(BB);

        for (auto &Arg : TheFunction->args()) {
            llvm::AllocaInst *Alloca = CreateEntryBlockAlloca(TheFunction, Arg.getName().str(), Arg.getType());
            Builder->CreateStore(&Arg, Alloca);
            NamedValues[Arg.getName().str()] = Alloca;
        }

        if (llvm::Value *RetVal = Body->codegen()) {
            if (Builder->GetInsertBlock()->getTerminator() == nullptr) {
                if (Proto->RetType->isVoidTy()) {
                    Builder->CreateRetVoid();
                } else {
                    Builder->CreateRet(llvm::Constant::getNullValue(Proto->RetType));
                }
            }
            llvm::verifyFunction(*TheFunction);
            return TheFunction;
        }

        TheFunction->eraseFromParent();
        return nullptr;
    }
};

%}

%union {
    int ival;
    char* sval;
    ExprAST* expr;
    StatementAST* stmt;
    PrototypeAST* proto;
    FunctionAST* func;
    std::vector<std::string>* strvec;
    std::string* std_string_ptr;
    std::vector<StatementAST*>* stmtvec;
    std::vector<ExprAST*>* exprvec;
    llvm::Type* llvm_type;
}

%token INT CHAR RETURN IF ELSE WHILE FOR VOID MAIN
%token IDENTIFIER NUMBER
%token STRING_LITERAL
%token EQ NE LE GE AMPERSAND

%left '+' '-'
%left '*' '/'
%left '<' '>' LE GE EQ NE
%right '='

%nonassoc THEN
%nonassoc ELSE

%type <expr> expression assignment_expression additive_expression multiplicative_expression relational_expression primary_expression call_expression
%type <exprvec> argument_list argument_list_opt
%type <stmt> statement compound_statement expression_statement selection_statement iteration_statement jump_statement declaration_statement
%type <proto> function_prototype
%type <func> function_definition
%type <stmtvec> statement_list statement_list_opt
%type <strvec> parameter_list parameter_list_opt
%type <std_string_ptr> parameter_declaration
%type <sval> IDENTIFIER STRING_LITERAL
%type <ival> NUMBER
%type <expr> expression_opt
%type <llvm_type> type_specifier

%%

program:
    /* empty */
  | program function_definition
  ;

function_definition:
    function_prototype compound_statement {
        $$ = new FunctionAST($1, $2);
        if (!($$)->codegen()) {
            yyerror("Function code generation failed");
            delete $$;
            $$ = nullptr;
        }
    }
  ;

function_prototype:
    type_specifier IDENTIFIER '(' parameter_list_opt ')' {
        $$ = new PrototypeAST(std::string($2), *$4, $1);
        free($2);
        delete $4;
    }
  | type_specifier MAIN '(' parameter_list_opt ')' { // NEW RULE: Handle 'main' keyword explicitly
        $$ = new PrototypeAST("main", *$4, $1); // Hardcode "main" as function name
        delete $4;
    }
  ;

parameter_list_opt:
    parameter_list { $$ = $1; }
  | /* empty */ { $$ = new std::vector<std::string>(); }
  ;

parameter_list:
    parameter_declaration {
        $$ = new std::vector<std::string>();
        $$->push_back(*$1);
        delete $1;
    }
  | parameter_list ',' parameter_declaration {
        $$ = $1;
        $$->push_back(std::string(*$3));
        delete $3;
    }
  ;

parameter_declaration:
    type_specifier IDENTIFIER {
        $$ = new std::string($2);
        free($2);
    }
  ;

type_specifier:
    INT { $$ = llvm::Type::getInt32Ty(*TheContext); }
  | VOID { $$ = llvm::Type::getVoidTy(*TheContext); }
  | CHAR { $$ = llvm::Type::getInt8Ty(*TheContext); }
  ;


compound_statement:
    '{' statement_list_opt '}' {
        $$ = new CompoundStatementAST(*$2);
        delete $2;
    }
  ;

statement_list_opt:
    statement_list { $$ = $1; }
  | /* empty */ { $$ = new std::vector<StatementAST*>(); }
  ;

statement_list:
    statement {
        $$ = new std::vector<StatementAST*>();
        $$->push_back($1);
    }
  | statement_list statement {
        $$ = $1;
        $$->push_back($2);
    }
  ;

statement:
    declaration_statement
  | expression_statement
  | compound_statement
  | selection_statement
  | iteration_statement
  | jump_statement
  ;

declaration_statement:
    type_specifier IDENTIFIER ';' {
        $$ = new VarDeclStatementAST($2, $1);
        free($2);
    }
  ;

expression_statement:
    expression ';' {
        $$ = new ExprStatementAST($1);
    }
  | ';' {
        $$ = new ExprStatementAST(nullptr);
    }
  ;

selection_statement:
    IF '(' expression ')' statement %prec THEN {
        $$ = new IfStatementAST($3, $5, nullptr);
    }
  | IF '(' expression ')' statement ELSE statement {
        $$ = new IfStatementAST($3, $5, $7);
    }
  ;

iteration_statement:
    WHILE '(' expression ')' statement {
        $$ = new WhileStatementAST($3, $5);
    }
  | FOR '(' expression_statement expression_opt ';' expression_opt ')' statement {
        $$ = new ForStatementAST(
            $3,
            $4,
            new ExprStatementAST($6),
            $8);
    }
  ;

expression_opt:
    expression { $$ = $1; }
  | /* empty */ { $$ = nullptr; }
  ;

jump_statement:
    RETURN expression_opt ';' {
        $$ = new ReturnStatementAST($2);
    }
  ;

expression:
    assignment_expression {
        $$ = $1;
    }
;

assignment_expression:
    IDENTIFIER '=' assignment_expression {
        $$ = new AssignmentExprAST($1, $3);
        free($1);
    }
  | relational_expression {
        $$ = $1;
    }
;

relational_expression:
    additive_expression {
        $$ = $1;
    }
  | relational_expression EQ additive_expression {
        $$ = new BinaryExprAST(EQ, $1, $3);
    }
  | relational_expression NE additive_expression {
        $$ = new BinaryExprAST(NE, $1, $3);
    }
  | relational_expression LE additive_expression {
        $$ = new BinaryExprAST(LE, $1, $3);
    }
  | relational_expression GE additive_expression {
        $$ = new BinaryExprAST(GE, $1, $3);
    }
  | relational_expression '<' additive_expression {
        $$ = new BinaryExprAST('<', $1, $3);
    }
  | relational_expression '>' additive_expression {
        $$ = new BinaryExprAST('>', $1, $3);
    }
;

additive_expression:
    multiplicative_expression {
        $$ = $1;
    }
  | additive_expression '+' multiplicative_expression {
        $$ = new BinaryExprAST('+', $1, $3);
    }
  | additive_expression '-' multiplicative_expression {
        $$ = new BinaryExprAST('-', $1, $3);
    }
;

multiplicative_expression:
    primary_expression {
        $$ = $1;
    }
  | multiplicative_expression '*' primary_expression {
        $$ = new BinaryExprAST('*', $1, $3);
    }
  | multiplicative_expression '/' primary_expression {
        $$ = new BinaryExprAST('/', $1, $3);
    }
;

primary_expression:
    IDENTIFIER {
        $$ = new VariableExprAST($1);
        free($1);
    }
  | NUMBER {
        $$ = new NumberExprAST($1);
    }
  | STRING_LITERAL {
        $$ = new StringLiteralExprAST($1);
        free($1);
    }
  | '(' expression ')' {
        $$ = $2;
    }
  | AMPERSAND IDENTIFIER {
        $$ = new AddressOfExprAST($2);
        free($2);
    }
  | call_expression {
        $$ = $1;
    }
;


call_expression:
    IDENTIFIER '(' argument_list_opt ')' {
        $$ = new CallExprAST($1, *$3);
        free($1);
        delete $3;
    }
  ;

argument_list_opt:
    argument_list { $$ = $1; }
  | /* empty */ { $$ = new std::vector<ExprAST*>(); }
  ;

argument_list:
    expression {
        $$ = new std::vector<ExprAST*>();
        $$->push_back($1);
    }
  | argument_list ',' expression {
        $$ = $1;
        $$->push_back($3);
    }
  ;

%%

void yyerror(const char *s) {
    fprintf(stderr, "Error: %s\n", s);
}