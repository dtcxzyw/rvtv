/*
    SPDX-License-Identifier: Apache-2.0

    Copyright 2024 Yingwei Zheng
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
        http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/CodeGen/MIRPrinter.h>
#include <llvm/CodeGen/MachineFunctionPass.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/CodeGen/Passes.h>
#include <llvm/CodeGen/TargetPassConfig.h>
#include <llvm/IR/AssemblyAnnotationWriter.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstVisitor.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/PatternMatch.h>
#include <llvm/IR/User.h>
#include <llvm/IR/Value.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRPrinter/IRPrintingPasses.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/InitializePasses.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Pass.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/CodeGen.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/FormattedStream.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/ToolOutputFile.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/CGPassBuilderOption.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/TargetParser/Host.h>
#include <cstdlib>

using namespace llvm;

static cl::OptionCategory RVTVCategory("rvtv options");

static cl::opt<std::string>
    InputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"),
                  cl::value_desc("filename"), cl::cat(RVTVCategory));

static cl::opt<std::string> OutputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::cat(RVTVCategory));

static cl::opt<std::string>
    TargetTriple("mtriple", cl::desc("Override target triple for module"),
                 cl::value_desc("triple"), cl::init("riscv64-linux-gnu"),
                 cl::cat(RVTVCategory));

static cl::opt<std::string> TargetCPU("mcpu", cl::desc("Target CPU"),
                                      cl::value_desc("cpu"),
                                      cl::init("generic"),
                                      cl::cat(RVTVCategory));

static cl::opt<std::string> TargetFeatures("mattr", cl::desc("Target features"),
                                           cl::value_desc("features"),
                                           cl::init(""), cl::cat(RVTVCategory));

struct RISCVLiftPass : public MachineFunctionPass {
  static char ID;
  Module &RefM;
  Module &M;

  RISCVLiftPass(Module &RefM, Module &M)
      : MachineFunctionPass(ID), RefM(RefM), M(M) {}

  StringRef getPassName() const override {
    return "RISCV MIR -> LLVM IR Lifting Pass";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool doFinalization(Module &M) override { return verifyModule(M, &errs()); }

  bool runOnMachineFunction(MachineFunction &MF) override { return false; }
};

char RISCVLiftPass::ID;

int main(int argc, char **argv) {
  InitLLVM Init{argc, argv};
  LLVMInitializeRISCVTarget();
  LLVMInitializeRISCVTargetInfo();
  LLVMInitializeRISCVTargetMC();
  LLVMInitializeRISCVAsmPrinter();

  setBugReportMsg("PLEASE submit a bug report to "
                  "https://github.com/dtcxzyw/rvtv "
                  "and include the crash backtrace, preprocessed "
                  "source, and associated run script.\n");
  cl::ParseCommandLineOptions(argc, argv,
                              "rvtv RISCV codegen Translation Validator\n");

  LLVMContext Context;
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseIRFile(InputFilename, Err, Context);

  if (!M) {
    Err.print(argv[0], errs());
    return EXIT_FAILURE;
  }

  M->setTargetTriple(TargetTriple);

  std::string Error;
  auto *Target = TargetRegistry::lookupTarget(TargetTriple, Error);
  if (!Target) {
    errs() << Error << '\n';
    return EXIT_FAILURE;
  }

  CodeGenOptLevel Opt = CodeGenOptLevel::Default;
  TargetOptions TargetOptions;
  auto TargetMachine = std::unique_ptr<LLVMTargetMachine>(
      static_cast<LLVMTargetMachine *>(Target->createTargetMachine(
          TargetTriple, TargetCPU, TargetFeatures, TargetOptions, std::nullopt,
          std::nullopt, Opt)));

  legacy::PassManager PM;
  TargetLibraryInfoImpl TLII(Triple{TargetTriple});
  PM.add(new TargetLibraryInfoWrapperPass(TLII));
  //   TargetMachine->addPassesToEmitFile(PM, errs(), nullptr,
  //                                      CodeGenFileType::AssemblyFile);
  //   PM.run(*M);
  auto *PassConfig = TargetMachine->createPassConfig(PM);
  PassConfig->setDisableVerify(true);
  PM.add(PassConfig);
  PM.add(new MachineModuleInfoWrapperPass(TargetMachine.get()));

  if (PassConfig->addISelPasses())
    return EXIT_FAILURE;
  // PassConfig->addMachinePasses();
  PassConfig->setInitialized();
  // PM.add(createPrintMIRPass(errs()));
  Module NewM(M->getName(), Context);
  PM.add(new RISCVLiftPass(*M, NewM));

  PM.run(*M);

  NewM.dump();

  return EXIT_SUCCESS;
}
