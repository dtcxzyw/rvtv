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

#include <llvm-19/llvm/CodeGen/MachineBasicBlock.h>
#include <llvm-19/llvm/IR/Instruction.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/CodeGen/ISDOpcodes.h>
#include <llvm/CodeGen/MIRPrinter.h>
#include <llvm/CodeGen/MachineBasicBlock.h>
#include <llvm/CodeGen/MachineFunction.h>
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
#include <llvm/lib/Target/RISCV/RISCVInstrInfo.h>
#include <llvm/lib/Target/RISCV/RISCVRegisterInfo.h>
#include <cstdlib>

using namespace llvm;

static cl::OptionCategory RVTVCategory("rvtv options");

static cl::opt<std::string>
    InputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"),
                  cl::value_desc("filename"), cl::cat(RVTVCategory));

static cl::opt<std::string> OutputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"),
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
  uint32_t XLen;

  explicit RISCVLiftPass(Module &RefM, Module &M, uint32_t XLen)
      : MachineFunctionPass(ID), RefM(RefM), M(M), XLen(XLen) {}

  StringRef getPassName() const override {
    return "RISCV MIR -> LLVM IR Lifting Pass";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool doFinalization(Module &M) override { return false; }

  Type *getTypeFromRegClass(const TargetRegisterClass *RC) {
    if (RC == &RISCV::GPRRegClass)
      return Type::getIntNTy(M.getContext(), XLen);

    if (RC == &RISCV::FPR32RegClass)
      return Type::getFloatTy(M.getContext());

    if (RC == &RISCV::FPR64RegClass)
      return Type::getDoubleTy(M.getContext());

    llvm_unreachable("Unsupported register class");
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    assert(MF.getProperties().hasProperty(
               MachineFunctionProperties::Property::IsSSA) &&
           "MachineFunction is not in SSA form");

    auto *RefF = RefM.getFunction(MF.getName());
    if (!RefF) {
      errs() << "Function " << MF.getName()
             << " not found in reference module\n";
      return true;
    }

    auto *F = cast<Function>(
        M.getOrInsertFunction(MF.getName(), RefF->getFunctionType())
            .getCallee());
    F->copyAttributesFrom(RefF);

    MF.print(errs());
    errs() << '\n';

    MachineRegisterInfo &MRI = MF.getRegInfo();
    const TargetRegisterInfo &TRI = *static_cast<const RISCVRegisterInfo *>(
        MF.getSubtarget().getRegisterInfo());

    DenseMap<MachineBasicBlock *, BasicBlock *> BBMap;

    DenseMap<Register, Value *> RegMap;
    for (auto &MBB : MF) {
      auto *BB = BasicBlock::Create(M.getContext(), MBB.getName(), F);
      BBMap[&MBB] = BB;
      IRBuilder<> Builder(BB);
      for (auto &MI : MBB) {
        if (MI.getOpcode() != TargetOpcode::PHI)
          continue;
        auto Def = MI.getOperand(0).getReg();
        RegMap[Def] = Builder.CreatePHI(
            getTypeFromRegClass(MRI.getRegClass(Def)), MI.getNumOperands() / 2);
      }
    }

    for (auto &MBB : MF) {
      auto *BB = BBMap.at(&MBB);
      IRBuilder<> Builder(BB);
      DenseMap<Register, Value *> RetMap;

      for (auto &MI : MBB) {
        auto GetOperand = [&](uint32_t Id) -> Value * {
          auto &Reg = MI.getOperand(Id);
          assert(Reg.isUse() && "Operand is not a use");

          if (Reg.getReg().id() == RISCV::X0)
            return Builder.getIntN(XLen, 0);

          return RegMap.at(Reg.getReg());
        };

        auto SetGPR = [&](Value *Val) {
          assert(Val->getType()->isIntegerTy(XLen));
          auto &Reg = MI.getOperand(0);
          assert(Reg.isDef() && "Operand is not a def");
          RegMap[Reg.getReg()] = Val;
        };

        auto SetFPR = [&](Value *Val) {
          assert(Val->getType()->isFloatingPointTy());
          auto &Reg = MI.getOperand(0);
          assert(Reg.isDef() && "Operand is not a def");
          RegMap[Reg.getReg()] = Val;
        };

        auto TruncW = [&](uint32_t Id) {
          return Builder.CreateTrunc(GetOperand(Id), Builder.getInt32Ty());
        };

        auto SExt = [&](Value *Val) {
          return Builder.CreateSExt(Val, Builder.getIntNTy(XLen));
        };

        auto ZExt = [&](Value *Val) {
          return Builder.CreateZExt(Val, Builder.getIntNTy(XLen));
        };

        auto Ext = [&](Value *Val, bool IsSigned) {
          return Builder.CreateIntCast(Val, Builder.getIntNTy(XLen), IsSigned);
        };

        // auto UImm = [&](uint32_t Id) {
        //   return ConstantInt::get(Builder.getIntNTy(XLen),
        //                           APInt(XLen, MI.getOperand(Id).getImm(),
        //                                 /*isSigned=*/false));
        // };

        auto SImm = [&](uint32_t Id) {
          return ConstantInt::get(Builder.getIntNTy(XLen),
                                  APInt(XLen, MI.getOperand(Id).getImm(),
                                        /*isSigned=*/true));
        };
        auto SImmW = [&](uint32_t Id) {
          return ConstantInt::get(Builder.getInt32Ty(),
                                  APInt(32, MI.getOperand(Id).getImm(),
                                        /*isSigned=*/true));
        };

        auto Cast = [&](Value *Val, Type *DstTy) {
          auto *SrcTy = Val->getType();
          if (SrcTy == DstTy)
            return Val;

          if (SrcTy->isIntOrIntVectorTy(1) && DstTy->isIntOrIntVectorTy())
            return Builder.CreateZExt(Val, DstTy);
          if (SrcTy->isIntOrIntVectorTy() && DstTy->isIntOrIntVectorTy())
            return Builder.CreateSExtOrTrunc(Val, DstTy);
          if (SrcTy->isPtrOrPtrVectorTy() && DstTy->isIntOrIntVectorTy())
            return Builder.CreatePtrToInt(Val, DstTy);
          if (SrcTy->isIntOrIntVectorTy() && DstTy->isPtrOrPtrVectorTy())
            return Builder.CreateIntToPtr(Val, DstTy);

          errs() << "Unsupported cast from " << *SrcTy << " to " << *DstTy
                 << '\n';
          llvm_unreachable("Unsupported cast");
        };

        auto BinOp = [&](Instruction::BinaryOps Opcode, Value *LHS,
                         Value *RHS) {
          SetGPR(Builder.CreateBinOp(Opcode, LHS, RHS));
        };

        auto FPBinOp = [&](Instruction::BinaryOps Opcode) {
          SetFPR(Builder.CreateBinOp(Opcode, GetOperand(1), GetOperand(2)));
        };

        auto BinOpXLen = [&](Instruction::BinaryOps Opcode) {
          SetGPR(Builder.CreateBinOp(Opcode, GetOperand(1), GetOperand(2)));
        };

        auto BinOpXLenImm = [&](Instruction::BinaryOps Opcode) {
          SetGPR(Builder.CreateBinOp(Opcode, GetOperand(1), SImm(2)));
        };

        auto BinOpW = [&](Instruction::BinaryOps Opcode) {
          SetGPR(SExt(Builder.CreateBinOp(Opcode, TruncW(1), TruncW(2))));
        };

        auto BinOpWImm = [&](Instruction::BinaryOps Opcode) {
          SetGPR(SExt(Builder.CreateBinOp(Opcode, TruncW(1), SImmW(2))));
        };

        auto ICmp = [&](CmpInst::Predicate Predicate, Value *LHS, Value *RHS) {
          SetGPR(ZExt(Builder.CreateICmp(Predicate, LHS, RHS)));
        };

        auto BranchICmp = [&](CmpInst::Predicate Pred, Value *LHS, Value *RHS) {
          auto *Cond = Builder.CreateICmp(Pred, LHS, RHS);
          auto *TrueBB = BBMap.at(MI.getOperand(2).getMBB());
          auto *FalseBB = BBMap.at(MBB.getNextNode());
          Builder.CreateCondBr(Cond, TrueBB, FalseBB);
        };

        auto Mul2XLen = [&](bool Signed1, bool Signed2) {
          auto *Ty = Builder.getIntNTy(XLen);
          auto *DoubleTy = Builder.getIntNTy(XLen * 2);
          auto *LHS = Builder.CreateIntCast(GetOperand(1), DoubleTy, Signed1);
          auto *RHS = Builder.CreateIntCast(GetOperand(2), DoubleTy, Signed2);
          SetGPR(Builder.CreateTrunc(
              Builder.CreateLShr(Builder.CreateMul(LHS, RHS), XLen), Ty));
        };

        auto Load = [&](Type *Ty, bool IsSigned) {
          auto &Offset = MI.getOperand(2);
          auto &Base = MI.getOperand(1);
          // SetGPR(Ext(Builder.getIntN(XLen, 0), IsSigned));
        };

        switch (MI.getOpcode()) {
          // Pseudos
        case TargetOpcode::PHI:
          break;
        case TargetOpcode::COPY: {
          auto &Dst = MI.getOperand(0);
          auto &Src = MI.getOperand(1);

          // auto *RegClass = MRI.getRegClass(Src.getReg());
          // Type *Ty = getTypeFromRegClass(RegClass);
          Value *Val = nullptr;

          if (Src.getReg().isPhysical() && Src.getReg().id() != RISCV::X0) {
            auto Id = Src.getReg().id();

            uint32_t GPRCount = 0;
            uint32_t FPRCount = 0;

            for (auto &Arg : F->args()) {
              if (Id == RISCV::X10 + GPRCount)
                Val = &Arg;
              if (Id == RISCV::F10_F + FPRCount)
                Val = &Arg;
              if (Id == RISCV::F10_D + FPRCount)
                Val = &Arg;
              if (Id == RISCV::F10_H + FPRCount)
                Val = &Arg;
              if (Val)
                break;

              if (Arg.getType()->isIntOrPtrTy())
                GPRCount++;
              else if (Arg.getType()->isFloatingPointTy())
                FPRCount++;
            }

            if (!Val)
              llvm_unreachable("Unsupported argument");
          } else
            Val = GetOperand(1);

          Type *TgtTy = nullptr;

          if (Dst.getReg().isPhysical()) {
            auto Id = Dst.getReg().id();
            if (Id >= RISCV::X10 && Id <= RISCV::X31)
              TgtTy = Builder.getIntNTy(XLen);
            else if (Id >= RISCV::F10_F && Id <= RISCV::F31_F)
              TgtTy = Builder.getFloatTy();
            else if (Id >= RISCV::F10_D && Id <= RISCV::F31_D)
              TgtTy = Builder.getDoubleTy();
            else if (Id >= RISCV::F10_H && Id <= RISCV::F31_H)
              TgtTy = Builder.getHalfTy();
            else
              llvm_unreachable("Unsupported physical register");
          } else {
            TgtTy = getTypeFromRegClass(MRI.getRegClass(Dst.getReg()));
          }

          Val = Cast(Val, TgtTy);

          if (Dst.getReg().isPhysical())
            RetMap[Dst.getReg()] = Val;
          else
            RegMap[Dst.getReg()] = Val;

          break;
        }
        case RISCV::PseudoRET: {
          if (F->getReturnType()->isVoidTy()) {
            Builder.CreateRetVoid();
          } else {
            auto &Ret = MI.getOperand(0);
            assert(Ret.isUse() && "Operand is not a use");
            assert(Ret.isImplicit() && "Operand is not implicit");
            auto *Val = RetMap.at(Ret.getReg());
            Builder.CreateRet(Cast(Val, F->getReturnType()));
          }
          break;
        }
        // RV32I Base
        case RISCV::LUI:
          SetGPR(ConstantInt::get(Builder.getIntNTy(XLen),
                                  MI.getOperand(1).getImm() << 12));
          break;
        case RISCV::ADDI:
          BinOpXLenImm(Instruction::Add);
          break;
        case RISCV::SLTI:
          ICmp(CmpInst::ICMP_SLT, GetOperand(1), SImm(2));
          break;
        case RISCV::SLTIU:
          ICmp(CmpInst::ICMP_ULT, GetOperand(1), SImm(2));
          break;
        case RISCV::ANDI:
          BinOpXLenImm(Instruction::And);
          break;
        case RISCV::ORI:
          BinOpXLenImm(Instruction::Or);
          break;
        case RISCV::XORI:
          BinOpXLenImm(Instruction::Xor);
          break;
        case RISCV::SLLI:
          BinOpXLenImm(Instruction::Shl);
          break;
        case RISCV::SRLI:
          BinOpXLenImm(Instruction::LShr);
          break;
        case RISCV::SRAI:
          BinOpXLenImm(Instruction::AShr);
          break;
        case RISCV::ADD:
          BinOpXLen(Instruction::Add);
          break;
        case RISCV::SLT:
          ICmp(CmpInst::ICMP_SLT, GetOperand(1), GetOperand(2));
          break;
        case RISCV::SLTU:
          ICmp(CmpInst::ICMP_ULT, GetOperand(1), GetOperand(2));
          break;
        case RISCV::AND:
          BinOpXLen(Instruction::And);
          break;
        case RISCV::OR:
          BinOpXLen(Instruction::Or);
          break;
        case RISCV::XOR:
          BinOpXLen(Instruction::Xor);
          break;
        case RISCV::SLL:
          BinOpXLen(Instruction::Shl);
          break;
        case RISCV::SRL:
          BinOpXLen(Instruction::LShr);
          break;
        case RISCV::SUB:
          BinOpXLen(Instruction::Sub);
          break;
        case RISCV::SRA:
          BinOpXLen(Instruction::AShr);
          break;
        case RISCV::BEQ:
          BranchICmp(ICmpInst::ICMP_EQ, GetOperand(0), GetOperand(1));
          break;
        case RISCV::BNE:
          BranchICmp(ICmpInst::ICMP_NE, GetOperand(0), GetOperand(1));
          break;
        case RISCV::BLT:
          BranchICmp(ICmpInst::ICMP_SLT, GetOperand(0), GetOperand(1));
          break;
        case RISCV::BGE:
          BranchICmp(ICmpInst::ICMP_SGE, GetOperand(0), GetOperand(1));
          break;
        case RISCV::BLTU:
          BranchICmp(ICmpInst::ICMP_ULT, GetOperand(0), GetOperand(1));
          break;
        case RISCV::BGEU:
          BranchICmp(ICmpInst::ICMP_UGE, GetOperand(0), GetOperand(1));
          break;
        case RISCV::LB:
          Load(Builder.getInt8Ty(), /*IsSigned=*/true);
          break;
        case RISCV::LBU:
          Load(Builder.getInt8Ty(), /*IsSigned=*/false);
          break;
        case RISCV::LH:
          Load(Builder.getInt16Ty(), /*IsSigned=*/true);
          break;
        case RISCV::LHU:
          Load(Builder.getInt16Ty(), /*IsSigned=*/false);
          break;
        case RISCV::LW:
          Load(Builder.getInt32Ty(), /*IsSigned=*/true);
          break;
        // RV64I Base
        case RISCV::ADDIW:
          BinOpWImm(Instruction::Add);
          break;
        case RISCV::SLLIW:
          BinOpWImm(Instruction::Shl);
          break;
        case RISCV::SRLIW:
          BinOpWImm(Instruction::LShr);
          break;
        case RISCV::SRAIW:
          BinOpWImm(Instruction::AShr);
          break;
        case RISCV::ADDW:
          BinOpW(Instruction::Add);
          break;
        case RISCV::SUBW:
          BinOpW(Instruction::Sub);
          break;
        case RISCV::SLLW:
          BinOpW(Instruction::Shl);
          break;
        case RISCV::SRLW:
          BinOpW(Instruction::LShr);
          break;
        case RISCV::SRAW:
          BinOpW(Instruction::AShr);
          break;
        case RISCV::LWU:
          Load(Builder.getInt32Ty(), /*IsSigned=*/false);
          break;
        case RISCV::LD:
          Load(Builder.getInt64Ty(), /*IsSigned=*/true);
          break;
        // M
        case RISCV::MUL:
          BinOpXLen(Instruction::Mul);
          break;
        case RISCV::MULW:
          BinOpW(Instruction::Mul);
          break;
        case RISCV::MULH:
          Mul2XLen(/*Signed1=*/true, /*Signed2=*/true);
          break;
        case RISCV::MULHSU:
          Mul2XLen(/*Signed1=*/true, /*Signed2=*/false);
          break;
        case RISCV::MULHU:
          Mul2XLen(/*Signed1=*/false, /*Signed2=*/false);
          break;
        case RISCV::DIV:
          BinOpXLen(Instruction::SDiv);
          break;
        case RISCV::DIVU:
          BinOpXLen(Instruction::UDiv);
          break;
        case RISCV::DIVW:
          BinOpW(Instruction::SDiv);
          break;
        case RISCV::DIVUW:
          BinOpW(Instruction::UDiv);
          break;
        // F
        case RISCV::FMV_W_X:
          SetFPR(Builder.CreateBitCast(TruncW(1), Builder.getFloatTy()));
          break;
        case RISCV::FMV_X_W:
          SetGPR(
              SExt(Builder.CreateBitCast(GetOperand(1), Builder.getInt32Ty())));
          break;
        case RISCV::FADD_S:
          FPBinOp(Instruction::FAdd);
          break;
        case RISCV::FSUB_S:
          FPBinOp(Instruction::FSub);
          break;
        case RISCV::FMUL_S:
          FPBinOp(Instruction::FMul);
          break;
        case RISCV::FDIV_S:
          FPBinOp(Instruction::FDiv);
          break;

        default:
          errs() << "Unsupported opcode: " << MI << '\n';
          return true;
        }
      }

      if (BB->empty() || !BB->back().isTerminator())
        Builder.CreateBr(BBMap.at(MBB.getNextNode()));
    }

    for (auto &MBB : MF) {
      auto *BB = BBMap.at(&MBB);

      for (auto &MI : MBB) {
        if (MI.getOpcode() != TargetOpcode::PHI)
          continue;

        auto *PHI = cast<PHINode>(RegMap.at(MI.getOperand(0).getReg()));

        for (uint32_t I = 1; I < MI.getNumOperands(); I += 2) {
          auto *IncomingBB = BBMap.at(MI.getOperand(I + 1).getMBB());
          auto *IncomingVal = RegMap.at(MI.getOperand(I).getReg());
          PHI->addIncoming(IncomingVal, IncomingBB);
        }
      }
    }

    return false;
  }
};

char RISCVLiftPass::ID;

static void runOpt(Module &M) {
  llvm::LoopAnalysisManager LAM;
  llvm::FunctionAnalysisManager FAM;
  llvm::CGSCCAnalysisManager CGAM;
  llvm::ModuleAnalysisManager MAM;

  llvm::PassBuilder PB;
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  llvm::ModulePassManager MPM =
      PB.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3);

  // TODO: canonicalize ptrtoint/inttoptr -> ptradd
  MPM.run(M, MAM);
}

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
  M->setTargetTriple(TargetMachine->getTargetTriple().getTriple());
  M->setDataLayout(TargetMachine->createDataLayout());

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
  NewM.setTargetTriple(M->getTargetTriple());
  NewM.setDataLayout(M->getDataLayout());
  PM.add(new RISCVLiftPass(*M, NewM,
                           StringRef{TargetTriple}.contains("64") ? 64 : 32));

  PM.run(*M);

  NewM.dump();
  if (verifyModule(NewM, &errs()))
    return EXIT_FAILURE;
  runOpt(NewM);
  if (verifyModule(NewM, &errs()))
    return EXIT_FAILURE;

  std::error_code EC;
  auto Out =
      std::make_unique<ToolOutputFile>(OutputFilename, EC, sys::fs::OF_None);
  if (EC) {
    errs() << EC.message() << '\n';
    return EXIT_FAILURE;
  }

  NewM.setTargetTriple("");
  // TODO: attach MIR
  NewM.print(Out->os(), nullptr);
  Out->keep();

  return EXIT_SUCCESS;
}
