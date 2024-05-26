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

#include <llvm/ADT/FloatingPointMode.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/CodeGen/ISDOpcodes.h>
#include <llvm/CodeGen/MIRPrinter.h>
#include <llvm/CodeGen/MachineBasicBlock.h>
#include <llvm/CodeGen/MachineConstantPool.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineFunctionPass.h>
#include <llvm/CodeGen/MachineMemOperand.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/CodeGen/Passes.h>
#include <llvm/CodeGen/TargetPassConfig.h>
#include <llvm/IR/Argument.h>
#include <llvm/IR/AssemblyAnnotationWriter.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/GlobalVariable.h>
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
#include <llvm/IR/NoFolder.h>
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
#include <llvm/Transforms/Utils/Local.h>
#include <llvm/lib/Target/RISCV/MCTargetDesc/RISCVBaseInfo.h>
#include <llvm/lib/Target/RISCV/RISCV.h>
#include <llvm/lib/Target/RISCV/RISCVInstrInfo.h>
#include <llvm/lib/Target/RISCV/RISCVRegisterInfo.h>
#include <llvm/lib/Target/RISCV/RISCVSubtarget.h>
#include <cstdlib>
#include <string>

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
  uint32_t GlobalCount;

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

    if (RC == &RISCV::FPR16RegClass)
      return Type::getHalfTy(M.getContext());

    if (RC == &RISCV::FPR32RegClass)
      return Type::getFloatTy(M.getContext());

    if (RC == &RISCV::FPR64RegClass)
      return Type::getDoubleTy(M.getContext());

    llvm_unreachable("Unsupported register class");
  }

  bool isValidType(const RISCVSubtarget &ST, const Type *Ty) {
    if (!Ty->isSingleValueType())
      return false;
    if (Ty->isIntegerTy() && Ty->getScalarSizeInBits() > XLen)
      return false;

    if (Ty->isHalfTy() && !ST.hasStdExtZfh())
      return false;

    return true;
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

    auto &SubTarget = MF.getSubtarget<RISCVSubtarget>();

    // skip functions with unsupported types
    if (!isValidType(SubTarget, RefF->getReturnType()) ||
        any_of(RefF->args(),
               [&](Argument &Arg) {
                 return !isValidType(SubTarget, Arg.getType());
               }) ||
        RefF->isVarArg()) {
      errs() << "Unsupported function signature: " << *RefF->getFunctionType()
             << '\n';
      return false;
    }
    // skip functions with libcall
    for (auto &MBB : MF) {
      for (auto &MI : MBB) {
        if (MI.getOpcode() == RISCV::ADJCALLSTACKDOWN ||
            MI.getOpcode() == RISCV::ADJCALLSTACKUP ||
            MI.getOpcode() == RISCV::PseudoTAIL ||
            MI.getOpcode() == RISCV::PseudoCALL) {
          errs() << "Unsupported function with libcall: " << MF.getName()
                 << '\n';
          return false;
        }
      }
    }

    // skip functions with inttoptr
    using namespace PatternMatch;
    for (auto &BB : *RefF)
      for (auto &I : BB)
        if (I.getOpcode() == Instruction::IntToPtr ||
            (I.getOpcode() == Instruction::GetElementPtr &&
             match(cast<GetElementPtrInst>(I).getPointerOperand(), m_Zero()))) {
          errs() << "Unsupported function with inttoptr: " << MF.getName()
                 << '\n';
          return false;
        }

    auto *F = cast<Function>(
        M.getOrInsertFunction(MF.getName(), RefF->getFunctionType())
            .getCallee());
    F->copyAttributesFrom(RefF);

    // MF.print(errs());
    // errs() << '\n';

    MachineRegisterInfo &MRI = MF.getRegInfo();
    const TargetRegisterInfo &TRI = *static_cast<const RISCVRegisterInfo *>(
        MF.getSubtarget().getRegisterInfo());

    auto *ConstantPool = MF.getConstantPool();
    SmallVector<Value *> Constants;
    for (auto &Entry : ConstantPool->getConstants()) {
      assert(!Entry.isMachineConstantPoolEntry());
      auto *Val = const_cast<Constant *>(Entry.Val.ConstVal);
      auto Name = "constantpool" + std::to_string(GlobalCount++);
      Constants.push_back(M.getOrInsertGlobal(Name, Val->getType(), [&] {
        return new GlobalVariable(M, Val->getType(), /*isConstant=*/true,
                                  GlobalValue::InternalLinkage, Val, Name);
      }));
    }

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
      auto *XLenTy = Builder.getIntNTy(XLen);

      for (auto &MI : MBB) {
        // errs() << MI << '\n';

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

        auto SExt = [&](Value *Val) { return Builder.CreateSExt(Val, XLenTy); };

        auto ZExt = [&](Value *Val) { return Builder.CreateZExt(Val, XLenTy); };

        auto Ext = [&](Value *Val, bool IsSigned) {
          return Builder.CreateIntCast(Val, XLenTy, IsSigned);
        };

        // auto UImm = [&](uint32_t Id) {
        //   return ConstantInt::get(XLenTy,
        //                           APInt(XLen, MI.getOperand(Id).getImm(),
        //                                 /*isSigned=*/false));
        // };

        auto SImm = [&](uint32_t Id) {
          return ConstantInt::get(XLenTy,
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

        auto GetType = [&](uint32_t Id) {
          return getTypeFromRegClass(
              MRI.getRegClass(MI.getOperand(Id).getReg()));
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

        auto Shift = [&](Instruction::BinaryOps Opcode) {
          SetGPR(Builder.CreateBinOp(Opcode, GetOperand(1),
                                     Builder.CreateAnd(GetOperand(2), 63)));
        };

        auto ShiftW = [&](Instruction::BinaryOps Opcode) {
          SetGPR(SExt(Builder.CreateBinOp(Opcode, TruncW(1),
                                          Builder.CreateAnd(TruncW(2), 31))));
        };

        auto ICmp = [&](CmpInst::Predicate Predicate, Value *LHS, Value *RHS) {
          SetGPR(ZExt(Builder.CreateICmp(Predicate, LHS, RHS)));
        };

        auto FCmp = [&](CmpInst::Predicate Predicate) {
          SetGPR(ZExt(
              Builder.CreateFCmp(Predicate, GetOperand(1), GetOperand(2))));
        };

        auto BranchICmp = [&](CmpInst::Predicate Pred) {
          auto *Cond = Builder.CreateICmp(Pred, GetOperand(0), GetOperand(1));
          auto *TrueBB = BBMap.at(MI.getOperand(2).getMBB());
          auto *FalseBB = BBMap.at(MBB.getNextNode());
          Builder.CreateCondBr(Cond, TrueBB, FalseBB);
        };

        auto Mul2XLen = [&](bool Signed1, bool Signed2) {
          auto *DoubleTy = Builder.getIntNTy(XLen * 2);
          auto *LHS = Builder.CreateIntCast(GetOperand(1), DoubleTy, Signed1);
          auto *RHS = Builder.CreateIntCast(GetOperand(2), DoubleTy, Signed2);
          SetGPR(Builder.CreateTrunc(
              Builder.CreateLShr(Builder.CreateMul(LHS, RHS), XLen), XLenTy));
        };

        auto Load = [&](Type *Ty, bool IsSigned) {
          auto &Offset = MI.getOperand(2);
          auto &Base = MI.getOperand(1);

          if (Offset.isCPI()) {
            auto &Entry = ConstantPool->getConstants()[Offset.getIndex()];
            assert(!Entry.isMachineConstantPoolEntry());
            SetGPR(Builder.CreateIntCast(
                const_cast<Constant *>(Entry.Val.ConstVal), XLenTy, IsSigned));
            return;
          }

          auto *BasePtr =
              Builder.CreateIntToPtr(GetOperand(1), Builder.getPtrTy());
          auto *Ptr = Builder.CreatePtrAdd(
              BasePtr, Offset.isImm()
                           ? ConstantInt::get(XLenTy, Offset.getImm())
                           : GetOperand(2));
          SetGPR(Ext(Builder.CreateLoad(Ty, Ptr), IsSigned));
        };

        auto FPLoad = [&] {
          auto &Offset = MI.getOperand(2);
          auto &Base = MI.getOperand(1);

          if (Offset.isCPI()) {
            auto &Entry = ConstantPool->getConstants()[Offset.getIndex()];
            assert(!Entry.isMachineConstantPoolEntry());
            SetFPR(const_cast<Constant *>(Entry.Val.ConstVal));
            return;
          }

          auto *BasePtr =
              Builder.CreateIntToPtr(GetOperand(1), Builder.getPtrTy());
          auto *Ptr = Builder.CreatePtrAdd(
              BasePtr, Offset.isImm()
                           ? ConstantInt::get(XLenTy, Offset.getImm())
                           : GetOperand(2));
          SetFPR(Builder.CreateLoad(GetType(0), Ptr));
        };

        auto FMinMax = [&](Intrinsic::ID BaseIID, Intrinsic::ID ZeroIID) {
          auto *LHS = GetOperand(1);
          auto *RHS = GetOperand(2);
          auto *BaseRes = Builder.CreateBinaryIntrinsic(BaseIID, LHS, RHS);
          auto *ZeroRes = Builder.CreateBinaryIntrinsic(ZeroIID, LHS, RHS);
          auto *Cond = Builder.CreateAnd(Builder.createIsFPClass(LHS, fcZero),
                                         Builder.createIsFPClass(RHS, fcZero));
          SetFPR(Builder.CreateSelect(Cond, ZeroRes, BaseRes));
        };

        auto Div = [&](bool IsSigned, Value *LHS, Value *RHS) {
          auto *IsZero = Builder.CreateIsNull(RHS);
          auto *SignedMin = ConstantInt::get(
              LHS->getType(),
              APInt::getSignedMinValue(LHS->getType()->getScalarSizeInBits()));
          auto *AllOnes = Constant::getAllOnesValue(LHS->getType());

          auto *IsOverflow =
              IsSigned ? Builder.CreateAnd(Builder.CreateICmpEQ(LHS, SignedMin),
                                           Builder.CreateICmpEQ(RHS, AllOnes))
                       : Builder.getFalse();
          auto *SafeRHS =
              Builder.CreateSelect(Builder.CreateOr(IsZero, IsOverflow),
                                   ConstantInt::get(RHS->getType(), 1), RHS);
          auto *Div = Builder.CreateBinOp(
              IsSigned ? Instruction::SDiv : Instruction::UDiv, LHS, SafeRHS);
          Div = Builder.CreateSelect(IsZero, AllOnes, Div);
          Div = Builder.CreateSelect(IsOverflow, SignedMin, Div);
          return Div;
        };

        auto Rem = [&](bool IsSigned, Value *LHS, Value *RHS) {
          auto *IsZero = Builder.CreateIsNull(RHS);
          auto *SignedMin = ConstantInt::get(
              LHS->getType(),
              APInt::getSignedMinValue(LHS->getType()->getScalarSizeInBits()));
          auto *AllOnes = Constant::getAllOnesValue(LHS->getType());

          auto *IsOverflow =
              IsSigned ? Builder.CreateAnd(Builder.CreateICmpEQ(LHS, SignedMin),
                                           Builder.CreateICmpEQ(RHS, AllOnes))
                       : Builder.getFalse();
          auto *SafeRHS =
              Builder.CreateSelect(Builder.CreateOr(IsZero, IsOverflow),
                                   ConstantInt::get(RHS->getType(), 1), RHS);
          auto *Rem = Builder.CreateBinOp(
              IsSigned ? Instruction::SRem : Instruction::URem, LHS, SafeRHS);
          Rem = Builder.CreateSelect(IsZero, LHS, Rem);
          Rem = Builder.CreateSelect(
              IsOverflow, Constant::getNullValue(LHS->getType()), Rem);
          return Rem;
        };

        auto CZero = [&](ICmpInst::Predicate Pred) {
          Value *Val = GetOperand(1);
          Value *Cond = GetOperand(2);
          Value *Zero = Constant::getNullValue(Cond->getType());
          return Builder.CreateSelect(Builder.CreateICmp(Pred, Cond, Zero),
                                      Zero, Val);
        };

        auto BinaryIntrinsicXLen = [&](Intrinsic::ID IID) {
          SetGPR(
              Builder.CreateBinaryIntrinsic(IID, GetOperand(1), GetOperand(2)));
        };

        auto ShXAdd = [&](uint32_t ShAmt, bool HasUW) {
          SetGPR(Builder.CreateAdd(
              GetOperand(2),
              Builder.CreateShl(HasUW ? ZExt(TruncW(1)) : GetOperand(1),
                                ShAmt)));
        };

        auto Rotate = [&](Intrinsic::ID IID, Value *X, Value *Y) {
          return Builder.CreateIntrinsic(IID, {X->getType()}, {X, X, Y});
        };

        auto ClearBit = [&](Value *Shamt) {
          auto *SafeShamt = Builder.CreateAnd(
              Shamt, ConstantInt::get(Shamt->getType(), XLen - 1));
          SetGPR(Builder.CreateAnd(GetOperand(1),
                                   Builder.CreateNot(Builder.CreateShl(
                                       Builder.getIntN(XLen, 1), SafeShamt))));
        };

        auto ExtractBit = [&](Value *Shamt) {
          auto *SafeShamt = Builder.CreateAnd(
              Shamt, ConstantInt::get(Shamt->getType(), XLen - 1));
          SetGPR(Builder.CreateAnd(Builder.CreateLShr(GetOperand(1), SafeShamt),
                                   1));
        };

        auto InvertBit = [&](Value *Shamt) {
          auto *SafeShamt = Builder.CreateAnd(
              Shamt, ConstantInt::get(Shamt->getType(), XLen - 1));
          SetGPR(Builder.CreateXor(
              GetOperand(1),
              Builder.CreateShl(Builder.getIntN(XLen, 1), SafeShamt)));
        };

        auto SetBit = [&](Value *Shamt) {
          auto *SafeShamt = Builder.CreateAnd(
              Shamt, ConstantInt::get(Shamt->getType(), XLen - 1));
          SetGPR(Builder.CreateOr(
              GetOperand(1),
              Builder.CreateShl(Builder.getIntN(XLen, 1), SafeShamt)));
        };

        auto Pack = [&](uint32_t Size) {
          uint32_t Half = Size / 2;
          auto *HalfTy = Builder.getIntNTy(Half);
          auto *Lo = ZExt(Builder.CreateTrunc(GetOperand(1), HalfTy));
          auto *Hi = ZExt(Builder.CreateTrunc(GetOperand(2), HalfTy));
          SetGPR(Builder.CreateOr(Builder.CreateShl(Hi, Half), Lo));
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
              if (Id == RISCV::X10 + GPRCount && Arg.getType()->isIntOrPtrTy())
                Val = &Arg;
              if (Arg.getType()->isFloatingPointTy()) {
                if (Id == RISCV::F10_F + FPRCount)
                  Val = &Arg;
                if (Id == RISCV::F10_D + FPRCount)
                  Val = &Arg;
                if (Id == RISCV::F10_H + FPRCount)
                  Val = &Arg;
              }
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
            if (Id >= RISCV::X0 && Id <= RISCV::X31)
              TgtTy = XLenTy;
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
        case RISCV::PseudoBR: {
          auto *DstBB = BBMap.at(MI.getOperand(0).getMBB());
          if (!BB->empty() && BB->back().isTerminator()) {
            if (auto *Br = dyn_cast<BranchInst>(BB->getTerminator())) {
              if (Br->isConditional() && Br->getSuccessor(1) == DstBB)
                break;
            }
          }
          Builder.CreateBr(DstBB);
          break;
        }
        // RV32I Base
        case RISCV::LUI:
          if (MI.getOperand(1).isImm())
            SetGPR(SExt(Builder.CreateTrunc(
                ConstantInt::get(XLenTy, MI.getOperand(1).getImm() << 12),
                Builder.getInt32Ty())));
          break;
        case RISCV::ADDI:
          if (MI.getOperand(2).isImm())
            BinOpXLenImm(Instruction::Add);
          else if (MI.getOperand(2).isCPI())
            SetGPR(Builder.CreatePtrToInt(
                Constants[MI.getOperand(2).getIndex()], XLenTy));
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
          Shift(Instruction::Shl);
          break;
        case RISCV::SRL:
          Shift(Instruction::LShr);
          break;
        case RISCV::SUB:
          BinOpXLen(Instruction::Sub);
          break;
        case RISCV::SRA:
          Shift(Instruction::AShr);
          break;
        case RISCV::BEQ:
          BranchICmp(ICmpInst::ICMP_EQ);
          break;
        case RISCV::BNE:
          BranchICmp(ICmpInst::ICMP_NE);
          break;
        case RISCV::BLT:
          BranchICmp(ICmpInst::ICMP_SLT);
          break;
        case RISCV::BGE:
          BranchICmp(ICmpInst::ICMP_SGE);
          break;
        case RISCV::BLTU:
          BranchICmp(ICmpInst::ICMP_ULT);
          break;
        case RISCV::BGEU:
          BranchICmp(ICmpInst::ICMP_UGE);
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
          ShiftW(Instruction::Shl);
          break;
        case RISCV::SRLW:
          ShiftW(Instruction::LShr);
          break;
        case RISCV::SRAW:
          ShiftW(Instruction::AShr);
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
          SetGPR(Div(/*IsSigned=*/true, GetOperand(1), GetOperand(2)));
          break;
        case RISCV::DIVU:
          SetGPR(Div(/*IsSigned=*/false, GetOperand(1), GetOperand(2)));
          break;
        case RISCV::DIVW:
          SetGPR(SExt(Div(/*IsSigned=*/true, TruncW(1), TruncW(2))));
          break;
        case RISCV::DIVUW:
          SetGPR(SExt(Div(/*IsSigned=*/false, TruncW(1), TruncW(2))));
          break;
        case RISCV::REM:
          SetGPR(Rem(/*IsSigned=*/true, GetOperand(1), GetOperand(2)));
          break;
        case RISCV::REMU:
          SetGPR(Rem(/*IsSigned=*/false, GetOperand(1), GetOperand(2)));
          break;
        case RISCV::REMW:
          SetGPR(SExt(Rem(/*IsSigned=*/true, TruncW(1), TruncW(2))));
          break;
        case RISCV::REMUW:
          SetGPR(SExt(Rem(/*IsSigned=*/false, TruncW(1), TruncW(2))));
          break;
        // F/D/ZFH
        case RISCV::FLH:
        case RISCV::FLW:
        case RISCV::FLD:
          FPLoad();
          break;
        case RISCV::FMV_H_X:
          SetFPR(Builder.CreateBitCast(
              Builder.CreateTrunc(GetOperand(1), Builder.getInt16Ty()),
              Builder.getHalfTy()));
          break;
        case RISCV::FMV_X_H:
          SetGPR(
              SExt(Builder.CreateBitCast(GetOperand(1), Builder.getInt16Ty())));
          break;
        case RISCV::FMV_W_X:
          SetFPR(Builder.CreateBitCast(TruncW(1), Builder.getFloatTy()));
          break;
        case RISCV::FMV_X_W:
          SetGPR(
              SExt(Builder.CreateBitCast(GetOperand(1), Builder.getInt32Ty())));
          break;
        case RISCV::FMV_X_D:
          SetGPR(Builder.CreateBitCast(GetOperand(1), Builder.getInt64Ty()));
          break;
        case RISCV::FMV_D_X:
          SetFPR(Builder.CreateBitCast(GetOperand(1), Builder.getDoubleTy()));
          break;
        // FIXME: See Table 28. Domains of float-to-integer conversions and
        // behavior for invalid inputs
        case RISCV::FCVT_W_H:
        case RISCV::FCVT_W_S:
        case RISCV::FCVT_W_D:
          SetGPR(SExt(Builder.CreateFreeze(
              Builder.CreateFPToSI(GetOperand(1), Builder.getInt32Ty()))));
          break;
        case RISCV::FCVT_WU_H:
        case RISCV::FCVT_WU_S:
        case RISCV::FCVT_WU_D:
          SetGPR(SExt(Builder.CreateFreeze(
              Builder.CreateFPToUI(GetOperand(1), Builder.getInt32Ty()))));
          break;
        case RISCV::FCVT_L_H:
        case RISCV::FCVT_L_S:
        case RISCV::FCVT_L_D:
          SetGPR(Builder.CreateFreeze(
              Builder.CreateFPToSI(GetOperand(1), Builder.getInt64Ty())));
          break;
        case RISCV::FCVT_LU_H:
        case RISCV::FCVT_LU_S:
        case RISCV::FCVT_LU_D:
          SetGPR(Builder.CreateFreeze(
              Builder.CreateFPToUI(GetOperand(1), Builder.getInt64Ty())));
          break;
        case RISCV::FCVT_S_W:
          SetFPR(Builder.CreateSIToFP(TruncW(1), Builder.getFloatTy()));
          break;
        case RISCV::FCVT_S_L:
          SetFPR(Builder.CreateSIToFP(GetOperand(1), Builder.getFloatTy()));
          break;
        case RISCV::FCVT_S_WU:
          SetFPR(Builder.CreateUIToFP(TruncW(1), Builder.getFloatTy()));
          break;
        case RISCV::FCVT_S_LU:
          SetFPR(Builder.CreateUIToFP(GetOperand(1), Builder.getFloatTy()));
          break;
        case RISCV::FCVT_D_W:
          SetFPR(Builder.CreateSIToFP(TruncW(1), Builder.getDoubleTy()));
          break;
        case RISCV::FCVT_D_L:
          SetFPR(Builder.CreateSIToFP(GetOperand(1), Builder.getDoubleTy()));
          break;
        case RISCV::FCVT_D_WU:
          SetFPR(Builder.CreateUIToFP(TruncW(1), Builder.getDoubleTy()));
          break;
        case RISCV::FCVT_D_LU:
          SetFPR(Builder.CreateUIToFP(GetOperand(1), Builder.getDoubleTy()));
          break;
        case RISCV::FCVT_H_S:
        case RISCV::FCVT_H_D:
          SetFPR(Builder.CreateFPCast(GetOperand(1), Builder.getHalfTy()));
          break;
        case RISCV::FCVT_S_H:
        case RISCV::FCVT_S_D:
          SetFPR(Builder.CreateFPCast(GetOperand(1), Builder.getFloatTy()));
          break;
        case RISCV::FCVT_D_S:
        case RISCV::FCVT_D_H:
          SetFPR(Builder.CreateFPCast(GetOperand(1), Builder.getDoubleTy()));
          break;
        case RISCV::FADD_H:
        case RISCV::FADD_S:
        case RISCV::FADD_D:
          FPBinOp(Instruction::FAdd);
          break;
        case RISCV::FSUB_H:
        case RISCV::FSUB_S:
        case RISCV::FSUB_D:
          FPBinOp(Instruction::FSub);
          break;
        case RISCV::FMUL_H:
        case RISCV::FMUL_S:
        case RISCV::FMUL_D:
          FPBinOp(Instruction::FMul);
          break;
        case RISCV::FDIV_H:
        case RISCV::FDIV_S:
        case RISCV::FDIV_D:
          FPBinOp(Instruction::FDiv);
          break;
        case RISCV::FLT_H:
        case RISCV::FLT_S:
        case RISCV::FLT_D:
        case RISCV::FLTQ_H:
        case RISCV::FLTQ_S:
        case RISCV::FLTQ_D:
          FCmp(CmpInst::FCMP_OLT);
          break;
        case RISCV::FLE_H:
        case RISCV::FLE_S:
        case RISCV::FLE_D:
        case RISCV::FLEQ_H:
        case RISCV::FLEQ_S:
        case RISCV::FLEQ_D:
          FCmp(CmpInst::FCMP_OLE);
          break;
        case RISCV::FEQ_H:
        case RISCV::FEQ_S:
        case RISCV::FEQ_D:
          FCmp(CmpInst::FCMP_OEQ);
          break;
        case RISCV::FMADD_H:
        case RISCV::FMADD_S:
        case RISCV::FMADD_D:
          SetFPR(Builder.CreateIntrinsic(
              GetType(0), Intrinsic::fma,
              {GetOperand(1), GetOperand(2), GetOperand(3)}));
          break;
        case RISCV::FNMADD_H:
        case RISCV::FNMADD_S:
        case RISCV::FNMADD_D:
          SetFPR(Builder.CreateIntrinsic(GetType(0), Intrinsic::fma,
                                         {GetOperand(1),
                                          Builder.CreateFNeg(GetOperand(2)),
                                          Builder.CreateFNeg(GetOperand(3))}));
          break;
        case RISCV::FMSUB_H:
        case RISCV::FMSUB_S:
        case RISCV::FMSUB_D:
          SetFPR(Builder.CreateIntrinsic(GetType(0), Intrinsic::fma,
                                         {GetOperand(1), GetOperand(2),
                                          Builder.CreateFNeg(GetOperand(3))}));
          break;
        case RISCV::FNMSUB_H:
        case RISCV::FNMSUB_S:
        case RISCV::FNMSUB_D:
          SetFPR(Builder.CreateIntrinsic(GetType(0), Intrinsic::fma,
                                         {GetOperand(1),
                                          Builder.CreateFNeg(GetOperand(2)),
                                          GetOperand(3)}));
          break;
        case RISCV::FSGNJ_H:
        case RISCV::FSGNJ_S:
        case RISCV::FSGNJ_D:
          SetFPR(Builder.CreateCopySign(GetOperand(1), GetOperand(2)));
          break;
        case RISCV::FSGNJX_H:
        case RISCV::FSGNJX_S:
        case RISCV::FSGNJX_D: {
          auto *LHS = GetOperand(1);
          auto *RHS = GetOperand(2);

          // fabs idiom
          if (LHS == RHS) {
            SetFPR(Builder.CreateUnaryIntrinsic(Intrinsic::fabs, LHS));
            break;
          }

          llvm_unreachable("todo");
          break;
        }
        case RISCV::FSGNJN_H:
        case RISCV::FSGNJN_S:
        case RISCV::FSGNJN_D: {
          auto *LHS = GetOperand(1);
          auto *RHS = GetOperand(2);

          // fneg idiom
          if (LHS == RHS) {
            SetFPR(Builder.CreateFNeg(LHS));
            break;
          }

          SetFPR(Builder.CreateCopySign(LHS, Builder.CreateFNeg(RHS)));
          break;
        }
        case RISCV::FCLASS_H:
        case RISCV::FCLASS_S:
        case RISCV::FCLASS_D: {
          auto *Val = GetOperand(1);
          Value *Res = Builder.getIntN(XLen, 1ULL << 0); // fcNegInf
          auto TestClass = [&](FPClassTest Test, uint32_t Bit) {
            Res = Builder.CreateSelect(Builder.createIsFPClass(Val, Test),
                                       Builder.getIntN(XLen, 1U << Bit), Res);
          };
          TestClass(fcNegNormal, 1);
          TestClass(fcNegSubnormal, 2);
          TestClass(fcNegZero, 3);
          TestClass(fcPosZero, 4);
          TestClass(fcPosSubnormal, 5);
          TestClass(fcPosNormal, 6);
          TestClass(fcPosInf, 7);
          TestClass(fcSNan, 8);
          TestClass(fcQNan, 9);
          SetGPR(Res);
          break;
        }
        case RISCV::FMAX_H:
        case RISCV::FMAX_S:
        case RISCV::FMAX_D:
          FMinMax(Intrinsic::maxnum, Intrinsic::maximum);
          break;
        case RISCV::FMIN_H:
        case RISCV::FMIN_S:
        case RISCV::FMIN_D:
          FMinMax(Intrinsic::minnum, Intrinsic::minimum);
          break;
        // Zicond
        case RISCV::CZERO_EQZ:
          SetGPR(CZero(ICmpInst::ICMP_EQ));
          break;
        case RISCV::CZERO_NEZ:
          SetGPR(CZero(ICmpInst::ICMP_NE));
          break;
        // Zba
        case RISCV::ADD_UW:
          SetGPR(Builder.CreateAdd(GetOperand(2), ZExt(TruncW(1))));
          break;
        case RISCV::SH1ADD:
          ShXAdd(/*ShAmt=*/1, /*HasUW=*/false);
          break;
        case RISCV::SH1ADD_UW:
          ShXAdd(/*ShAmt=*/1, /*HasUW=*/true);
          break;
        case RISCV::SH2ADD:
          ShXAdd(/*ShAmt=*/2, /*HasUW=*/false);
          break;
        case RISCV::SH2ADD_UW:
          ShXAdd(/*ShAmt=*/2, /*HasUW=*/true);
          break;
        case RISCV::SH3ADD:
          ShXAdd(/*ShAmt=*/3, /*HasUW=*/false);
          break;
        case RISCV::SH3ADD_UW:
          ShXAdd(/*ShAmt=*/3, /*HasUW=*/true);
          break;
        case RISCV::SLLI_UW:
          SetGPR(Builder.CreateShl(ZExt(TruncW(1)), SImm(2)));
          break;
        // Zbb
        case RISCV::ANDN:
          SetGPR(Builder.CreateAnd(GetOperand(1),
                                   Builder.CreateNot(GetOperand(2))));
          break;
        case RISCV::ORN:
          SetGPR(Builder.CreateOr(GetOperand(1),
                                  Builder.CreateNot(GetOperand(2))));
          break;
        case RISCV::XNOR:
          SetGPR(Builder.CreateNot(
              Builder.CreateXor(GetOperand(1), GetOperand(2))));
          break;
        case RISCV::CLZ:
          SetGPR(Builder.CreateBinaryIntrinsic(Intrinsic::ctlz, GetOperand(1),
                                               Builder.getFalse()));
          break;
        case RISCV::CLZW:
          SetGPR(ZExt(Builder.CreateBinaryIntrinsic(Intrinsic::ctlz, TruncW(1),
                                                    Builder.getFalse())));
          break;
        case RISCV::CTZ:
          SetGPR(Builder.CreateBinaryIntrinsic(Intrinsic::cttz, GetOperand(1),
                                               Builder.getFalse()));
          break;
        case RISCV::CTZW:
          SetGPR(ZExt(Builder.CreateBinaryIntrinsic(Intrinsic::cttz, TruncW(1),
                                                    Builder.getFalse())));
          break;
        case RISCV::CPOP:
          SetGPR(Builder.CreateUnaryIntrinsic(Intrinsic::ctpop, GetOperand(1)));
          break;
        case RISCV::CPOPW:
          SetGPR(
              ZExt(Builder.CreateUnaryIntrinsic(Intrinsic::ctpop, TruncW(1))));
          break;
        case RISCV::MAX:
          BinaryIntrinsicXLen(Intrinsic::smax);
          break;
        case RISCV::MAXU:
          BinaryIntrinsicXLen(Intrinsic::umax);
          break;
        case RISCV::MIN:
          BinaryIntrinsicXLen(Intrinsic::smin);
          break;
        case RISCV::MINU:
          BinaryIntrinsicXLen(Intrinsic::umin);
          break;
        case RISCV::SEXT_B:
          SetGPR(SExt(Builder.CreateTrunc(GetOperand(1), Builder.getInt8Ty())));
          break;
        case RISCV::SEXT_H:
          SetGPR(
              SExt(Builder.CreateTrunc(GetOperand(1), Builder.getInt16Ty())));
          break;
        case RISCV::ZEXT_H_RV32:
        case RISCV::ZEXT_H_RV64:
          SetGPR(
              ZExt(Builder.CreateTrunc(GetOperand(1), Builder.getInt16Ty())));
          break;
        case RISCV::ROL:
          SetGPR(Rotate(Intrinsic::fshl, GetOperand(1), GetOperand(2)));
          break;
        case RISCV::ROLW:
          SetGPR(SExt(Rotate(Intrinsic::fshl, TruncW(1), TruncW(2))));
          break;
        case RISCV::ROR:
          SetGPR(Rotate(Intrinsic::fshr, GetOperand(1), GetOperand(2)));
          break;
        case RISCV::RORW:
          SetGPR(SExt(Rotate(Intrinsic::fshr, TruncW(1), TruncW(2))));
          break;
        case RISCV::RORI:
          SetGPR(Rotate(Intrinsic::fshr, GetOperand(1), SImm(2)));
          break;
        case RISCV::RORIW:
          SetGPR(SExt(Rotate(Intrinsic::fshr, TruncW(1), SImmW(2))));
          break;
        case RISCV::ORC_B:
          llvm_unreachable("todo");
          break;
        case RISCV::REV8_RV32:
        case RISCV::REV8_RV64:
          SetGPR(Builder.CreateUnaryIntrinsic(Intrinsic::bswap, GetOperand(1)));
          break;
        // Zbs
        case RISCV::BCLR:
          ClearBit(GetOperand(2));
          break;
        case RISCV::BCLRI:
          ClearBit(SImm(2));
          break;
        case RISCV::BEXT:
          ExtractBit(GetOperand(2));
          break;
        case RISCV::BEXTI:
          ExtractBit(SImm(2));
          break;
        case RISCV::BINV:
          InvertBit(GetOperand(2));
          break;
        case RISCV::BINVI:
          InvertBit(SImm(2));
          break;
        case RISCV::BSET:
          SetBit(GetOperand(2));
          break;
        case RISCV::BSETI:
          SetBit(SImm(2));
          break;
        // Zfa
        case RISCV::FLI_H:
        case RISCV::FLI_S:
        case RISCV::FLI_D: {
          auto Imm = MI.getOperand(1).getImm();
          auto *Ty = GetType(0);
          if (Imm == 1)
            SetFPR(ConstantFP::get(
                Ty, APFloat::getSmallestNormalized(Ty->getFltSemantics())));
          else if (Imm == 30)
            SetFPR(ConstantFP::getInfinity(Ty));
          else if (Imm == 31)
            SetFPR(ConstantFP::getNaN(Ty));
          else
            SetFPR(ConstantFP::get(Ty, RISCVLoadFPImm::getFPImm(Imm)));
        } break;
        case RISCV::FMINM_H:
        case RISCV::FMINM_S:
        case RISCV::FMINM_D:
          SetFPR(Builder.CreateBinaryIntrinsic(Intrinsic::minimum,
                                               GetOperand(1), GetOperand(2)));
          break;
        case RISCV::FMAXM_H:
        case RISCV::FMAXM_S:
        case RISCV::FMAXM_D:
          SetFPR(Builder.CreateBinaryIntrinsic(Intrinsic::maximum,
                                               GetOperand(1), GetOperand(2)));
          break;
        case RISCV::FROUND_H:
        case RISCV::FROUND_S:
        case RISCV::FROUND_D:
        case RISCV::FROUNDNX_H:
        case RISCV::FROUNDNX_S:
        case RISCV::FROUNDNX_D:
          llvm_unreachable("todo");
          break;
        case RISCV::FCVTMOD_W_D:
          llvm_unreachable("todo");
          break;
        case RISCV::FMVH_X_D:
          llvm_unreachable("todo");
          break;
        case RISCV::FMVP_D_X:
          llvm_unreachable("todo");
          break;
        // Zbkb
        case RISCV::BREV8:
          SetGPR(Builder.CreateUnaryIntrinsic(
              Intrinsic::bswap, Builder.CreateUnaryIntrinsic(
                                    Intrinsic::bitreverse, GetOperand(1))));
          break;
        case RISCV::PACK:
          Pack(XLen);
          break;
        case RISCV::PACKH:
          Pack(16);
          break;
        case RISCV::PACKW:
          Pack(32);
          break;

        default:
          errs() << "Unsupported opcode: " << MI << '\n';
          llvm_unreachable("Unsupported opcode");
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

static void decomposeAddSubTree(Value *V, DenseMap<Value *, int32_t> &Row,
                                APInt &C, bool Add) {
  using namespace PatternMatch;
  const APInt *RHSC;
  if (match(V, m_APInt(RHSC))) {
    if (Add)
      C += *RHSC;
    else
      C -= *RHSC;
    return;
  }

  Value *A, *B;
  if (match(V, m_AddLike(m_Value(A), m_Value(B)))) {
    decomposeAddSubTree(A, Row, C, Add);
    decomposeAddSubTree(B, Row, C, Add);
    return;
  }

  uint64_t X;
  if (match(V, m_Shl(m_Value(A), m_ConstantInt(X))) && X <= 15) {
    if (Add)
      Row[A] += 1 << X;
    else
      Row[A] -= 1 << X;
    return;
  }

  if (match(V, m_Sub(m_Value(A), m_Value(B)))) {
    decomposeAddSubTree(A, Row, C, Add);
    decomposeAddSubTree(B, Row, C, !Add);
    return;
  }

  Row[V] += Add ? 1 : -1;
}

struct MaybePtr_match {
  Value *&ResV;
  uint64_t PtrSize;

  explicit MaybePtr_match(const DataLayout &DL, Value *&V)
      : ResV(V), PtrSize(DL.getPointerSizeInBits()) {}

  template <typename OpTy> bool match(OpTy *V) {
    if (auto *O = dyn_cast<Instruction>(V)) {
      DenseMap<Value *, int32_t> Row;
      APInt C(PtrSize, 0);
      decomposeAddSubTree(O, Row, C, /*Add=*/true);

      for (auto &[Val, Coeff] : Row)
        if (isa<PtrToIntInst>(Val) && Coeff >= 1) {
          ResV = V;
          return true;
        }
    }
    return false;
  }
};

static MaybePtr_match m_MaybePtr(const DataLayout &DL, Value *&V) {
  return MaybePtr_match{DL, V};
}

static bool canonicalize(Function &F) {
  using namespace PatternMatch;
  bool Changed = false;

  auto &DL = F.getParent()->getDataLayout();

  for (auto &BB : F)
    for (auto &I : make_early_inc_range(BB)) {

      if (RecursivelyDeleteTriviallyDeadInstructions(&I))
        continue;

      Value *A, *B, *C;
      if (match(&I, m_IntToPtr(m_c_Add(m_PtrToIntSameSize(DL, m_Value(A)),
                                       m_Value(B))))) {
        IRBuilder<> Builder(&I);
        I.replaceAllUsesWith(Builder.CreatePtrAdd(A, B));
        I.eraseFromParent();
        Changed = true;
        continue;
      }
      if (match(&I, m_IntToPtr(m_Sub(m_Value(A), m_Value(B))))) {
        IRBuilder<> Builder(&I);
        I.replaceAllUsesWith(
            Builder.CreatePtrAdd(Builder.CreateIntToPtr(A, Builder.getPtrTy()),
                                 Builder.CreateNeg(B)));
        I.eraseFromParent();
        Changed = true;
        continue;
      }
      if (match(&I, m_IntToPtr(m_c_Add(
                        m_c_Add(m_PtrToIntSameSize(DL, m_Value(A)), m_Value(B)),
                        m_Value(C))))) {
        IRBuilder<> Builder(&I);
        I.replaceAllUsesWith(Builder.CreatePtrAdd(A, Builder.CreateAdd(B, C)));
        I.eraseFromParent();
        Changed = true;
        continue;
      }
      if (match(&I, m_IntToPtr(m_Select(m_Value(A), m_Value(B), m_Value(C))))) {
        IRBuilder<> Builder(&I);
        auto *DstTy = I.getType();
        I.replaceAllUsesWith(
            Builder.CreateSelect(A, Builder.CreateIntToPtr(B, DstTy),
                                 Builder.CreateIntToPtr(C, DstTy)));
        I.eraseFromParent();
        Changed = true;
        continue;
      }
      if (match(&I, m_Select(m_Value(A), m_PtrToIntSameSize(DL, m_Value(B)),
                             m_Zero()))) {
        IRBuilder<> Builder(&I);
        auto *DstTy = I.getType();
        I.replaceAllUsesWith(Builder.CreatePtrToInt(
            Builder.CreateSelect(A, B, Constant::getNullValue(B->getType())),
            DstTy));
        I.eraseFromParent();
        Changed = true;
        continue;
      }
      if (match(&I, m_Select(m_Value(A), m_Zero(),
                             m_PtrToIntSameSize(DL, m_Value(B))))) {
        IRBuilder<> Builder(&I);
        auto *DstTy = I.getType();
        I.replaceAllUsesWith(Builder.CreatePtrToInt(
            Builder.CreateSelect(A, Constant::getNullValue(B->getType()), B),
            DstTy));
        I.eraseFromParent();
        Changed = true;
        continue;
      }

      if (match(&I, m_IntToPtr(m_UMax(m_Value(A), m_Value(B))))) {
        IRBuilder<> Builder(&I);
        auto *AP = Builder.CreateIntToPtr(A, I.getType());
        auto *BP = Builder.CreateIntToPtr(B, I.getType());
        I.replaceAllUsesWith(
            Builder.CreateSelect(Builder.CreateICmpUGT(AP, BP), AP, BP));
        I.eraseFromParent();
        Changed = true;
        continue;
      }

      if (match(&I, m_IntToPtr(m_UMin(m_Value(A), m_Value(B))))) {
        IRBuilder<> Builder(&I);
        auto *AP = Builder.CreateIntToPtr(A, I.getType());
        auto *BP = Builder.CreateIntToPtr(B, I.getType());
        I.replaceAllUsesWith(
            Builder.CreateSelect(Builder.CreateICmpULT(AP, BP), AP, BP));
        I.eraseFromParent();
        Changed = true;
        continue;
      }

      const APInt *RHSC;
      if (match(&I, m_IntToPtr(m_And(m_Value(A), m_APInt(RHSC))))) {
        IRBuilder<> Builder(&I);
        auto *PtrTy = Builder.getPtrTy();
        auto *IntTy = Builder.getIntNTy(RHSC->getBitWidth());
        I.replaceAllUsesWith(
            Builder.CreateIntrinsic(Intrinsic::ptrmask, {PtrTy, IntTy},
                                    {Builder.CreateIntToPtr(A, PtrTy),
                                     ConstantInt::get(IntTy, *RHSC)}));
        I.eraseFromParent();
        Changed = true;
        continue;
      }
      if (match(&I, m_IntToPtr(m_c_And(m_MaybePtr(DL, A), m_Value(B))))) {
        IRBuilder<> Builder(&I);
        auto *PtrTy = Builder.getPtrTy();
        auto *IntTy = B->getType();
        I.replaceAllUsesWith(
            Builder.CreateIntrinsic(Intrinsic::ptrmask, {PtrTy, IntTy},
                                    {Builder.CreateIntToPtr(A, PtrTy), B}));
        I.eraseFromParent();
        Changed = true;
        continue;
      }

      if (match(&I, m_IntToPtr(m_Value(A)))) {
        DenseMap<Value *, int32_t> Row;
        APInt C(DL.getPointerSizeInBits(), 0);
        decomposeAddSubTree(A, Row, C, /*Add=*/true);
        if (Row.size() > 1 ||
            (Row.size() == 1 && (!C.isZero() || Row.begin()->second != 1))) {
          Value *Base = nullptr;
          for (auto &[V, Coeff] : Row)
            if (Coeff >= 1 && match(V, m_PtrToIntSameSize(DL, m_Value(B)))) {
              Base = B;
              Coeff -= 1;
              break;
            }
          IRBuilder<> Builder(&I);
          if (!Base) {
            Value *Candidate = nullptr;
            uint32_t Weight = 0;
            auto GetWeight = [](Value *X) {
              if (isa<SelectInst>(X))
                return 4;
              if (match(X, m_MaxOrMin(m_Value(), m_Value())))
                return 3;
              if (match(X, m_And(m_Value(), m_Value())))
                return 2;
              if (isa<Argument>(X))
                return 1;
              return 0;
            };

            for (auto &[V, Coeff] : Row)
              if (Coeff >= 1 && !isa<Constant>(V)) {
                auto NewWeight = GetWeight(V);
                if (Weight < NewWeight) {
                  Candidate = V;
                  Weight = NewWeight;
                }
              }

            if (Candidate) {
              Base = Builder.CreateIntToPtr(Candidate, Builder.getPtrTy());
              Row[Candidate] -= 1;
            }
          }
          if (Base) {
            Value *Offset = ConstantInt::get(F.getContext(), C);
            for (auto &[V, Coeff] : Row) {
              if (Coeff == 0)
                continue;
              if (Coeff == 1) {
                Offset = Builder.CreateAdd(Offset, V);
                continue;
              }
              if (Coeff == -1) {
                Offset = Builder.CreateSub(Offset, V);
                continue;
              }
              Offset = Builder.CreateAdd(
                  Offset, Builder.CreateMul(
                              V, ConstantInt::get(Offset->getType(), Coeff)));
            }
            auto *Ptr = Builder.CreatePtrAdd(Base, Offset);
            I.replaceAllUsesWith(Ptr);
            I.eraseFromParent();
            Changed = true;
            continue;
          }
        }
      }
    }

  return Changed;
}

static bool canonicalize(Module &M) {
  bool Changed = false;
  for (auto &F : M) {
    if (F.empty())
      continue;
    while (canonicalize(F))
      Changed = true;
  }
  if (Changed) {
    assert(!verifyModule(M, &errs()));
    // M.dump();
  }
  return Changed;
}

static bool postCanonicalize(Function &F) {
  using namespace PatternMatch;
  bool Changed = false;

  for (auto &BB : F)
    for (auto &I : make_early_inc_range(BB)) {
      IRBuilder<NoFolder> Builder(&I);
      for (auto &Op : I.operands()) {
        Constant *C;
        if (match(Op.get(), m_IntToPtr(m_ImmConstant(C)))) {
          Op.set(
              Builder.CreatePtrAdd(Constant::getNullValue(Op->getType()), C));
          Changed = true;
        }
      }

      Value *X;
      if (match(&I, m_IntToPtr(m_Value(X)))) {
        auto *V = Builder.CreatePtrAdd(Constant::getNullValue(I.getType()), X);
        I.replaceAllUsesWith(V);
        I.eraseFromParent();
        Changed = true;
      }
    }

  return Changed;
}

static bool postCanonicalize(Module &M) {
  bool Changed = false;
  for (auto &F : M) {
    if (F.empty())
      continue;
    Changed |= postCanonicalize(F);
  }
  if (Changed) {
    assert(!verifyModule(M, &errs()));
    // M.dump();
  }
  return Changed;
}

static void verifyCanonicalization(Module &M) {
  using namespace PatternMatch;
  for (auto &F : M) {
    for (auto &BB : F) {
      for (auto &I : BB) {
        switch (I.getOpcode()) {
        // case Instruction::Load:
        case Instruction::IntToPtr:
          if (!isa<Argument>(I.getOperand(0)) &&
              // or disjoint
              !match(I.getOperand(0), m_Or(m_Value(), m_Value()))) {
            errs() << F << '\n';
            llvm_unreachable("Unsupported instruction");
          }
          break;
        default:
          break;
        }
      }
    }
  }
}

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

  // NewM.dump();
  if (verifyModule(NewM, &errs()))
    return EXIT_FAILURE;
  runOpt(NewM);
  // NewM.dump();
  if (canonicalize(NewM)) {
    // NewM.dump();
    runOpt(NewM);
  }
  if (!TargetMachine->getTargetFeatureString().contains("zicond"))
    verifyCanonicalization(NewM);
  postCanonicalize(NewM);
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

  if (OutputFilename != "-") {
    auto SrcOut = std::make_unique<ToolOutputFile>(OutputFilename + ".src", EC,
                                                   sys::fs::OF_None);
    if (EC) {
      errs() << EC.message() << '\n';
      return EXIT_FAILURE;
    }
    M->setTargetTriple("");
    M->print(SrcOut->os(), nullptr);
    SrcOut->keep();
  }

  return EXIT_SUCCESS;
}
