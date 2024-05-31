# rvtv

Risc-V codegen Translation Validator (RISC-V MIR -> LLVM IR Lifter)

## Prerequisites

+ C++ compiler with C++20 support
+ LLVM **build from source** with RISC-V target support
+ Alive2

## Installation

```bash
git clone https://github.com/dtcxzyw/rvtv.git
cd rvtv
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DLLVM_DIR=/path/to/llvm-install/lib/cmake/llvm -DLLVM_SOURCE_DIR=/path/to/llvm-src -DLLVM_BUILD_DIR=/path/to/llvm-build
cmake --build . -j
```

## Usage
```
bin/rvtv -mtriple=<triple> -mattr=<attr> -mcpu=<target-cpu> input.ll -o output.ll
alive-tv --tgt-is-asm --disable-undef-input --disable-poison-input output.ll.src output.ll
```

## License

This repository is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
