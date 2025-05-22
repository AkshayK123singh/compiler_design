#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IR/Instructions.h" // For AllocaInst
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/IR/DataLayout.h"

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

extern int yyparse();
extern FILE* yyin;

// Global LLVM objects - now as unique_ptrs to manage ownership for the JIT
std::unique_ptr<llvm::LLVMContext> TheContext;
std::unique_ptr<llvm::IRBuilder<>> Builder;
std::unique_ptr<llvm::Module> TheModule;
std::map<std::string, llvm::AllocaInst*> NamedValues;
std::unique_ptr<llvm::orc::LLJIT> TheJIT;

int main(int argc, char *argv[]) {
    llvm::InitLLVM X(argc, argv);

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
        return 1;
    }

    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();

    // Initialize TheContext and Builder here
    TheContext = std::make_unique<llvm::LLVMContext>();
    Builder = std::make_unique<llvm::IRBuilder<>>(*TheContext);

    // Create a JIT instance using LLJITBuilder *first*
    auto JIT_builder = llvm::orc::LLJITBuilder();
    auto JIT_result = JIT_builder.create();
    if (!JIT_result) {
        llvm::errs() << "Error creating JIT: " << toString(JIT_result.takeError()) << "\n";
        return 1;
    }
    TheJIT = std::move(*JIT_result); // TheJIT is now valid and initialized

    // Now, create TheModule and set its DataLayout using the valid TheJIT
    TheModule = std::make_unique<llvm::Module>("my cool jit", *TheContext);
    TheModule->setDataLayout(TheJIT->getDataLayout());

    // Declare external functions (intrinsics) like printf and scanf
    // This needs to be done before parsing if they are to be called.
    // Declare printf: int printf(const char* format, ...)
    llvm::FunctionType *PrintfFT = llvm::FunctionType::get(Builder->getInt32Ty(), {Builder->getInt8Ty()->getPointerTo()}, true); // Variadic
    llvm::Function::Create(PrintfFT, llvm::Function::ExternalLinkage, "printf", TheModule.get());

    // Declare scanf: int scanf(const char* format, ...)
    llvm::FunctionType *ScanfFT = llvm::FunctionType::get(Builder->getInt32Ty(), {Builder->getInt8Ty()->getPointerTo()}, true); // Variadic
    llvm::Function::Create(ScanfFT, llvm::Function::ExternalLinkage, "scanf", TheModule.get());


    // Open the input file
    yyin = fopen(argv[1], "r");
    if (!yyin) {
        perror("Error opening input file");
        return 1;
    }

    yyparse();

    fclose(yyin);

    TheModule->print(llvm::outs(), nullptr);

    // After parsing, add the module to the JIT
    llvm::orc::ThreadSafeContext TSCtx(std::move(TheContext)); // Move ownership of TheContext
    llvm::orc::ThreadSafeModule TSM(std::move(TheModule), std::move(TSCtx)); // Move ownership of TheModule and TSCtx

    if (auto Err = TheJIT->addIRModule(std::move(TSM))) {
        llvm::errs() << "Error adding module to JIT: " << toString(std::move(Err)) << "\n";
        return 1;
    }

    auto main_addr_result = TheJIT->lookup("main");
    if (!main_addr_result) {
        llvm::errs() << "Error finding main function: " << toString(main_addr_result.takeError()) << "\n";
        return 1;
    }

    llvm::JITTargetAddress main_addr = main_addr_result->getValue();

    int (*main_func)() = (int (*)())(main_addr);
    int result = main_func();

    std::cout << "Execution Result: " << result << std::endl;

    return 0;
}