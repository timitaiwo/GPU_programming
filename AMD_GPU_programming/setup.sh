#!/bin/bash

# Parse command-line arguments
RESET=0
for arg in "$@"; do
    if [ "$arg" = "--reset" ]; then
        RESET=1
    fi

    if [ "$arg" == "--set-path" ]; then 
      SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
      LLVM_PATH="$SCRIPT_DIR/third-party/llvm-project/llvm/bin"
      export PATH="$LLVM_PATH:$PATH"

      echo "LLVM path set. exiting...."
      exit 0
    fi
done

git submodule update --init

if [ $RESET -eq 1 ] || [ ! -d "build" ]; then
  rm -rf build
  mkdir build
  cd build
  cmake -G Ninja "-DLLVM_ENABLE_PROJECTS=llvm;clang;lldb" ..
else
  cd build
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLVM_PATH="$SCRIPT_DIR/third-party/llvm-project/llvm/bin"
export PATH="$LLVM_PATH:$PATH"

ninja  -j 8

cd ..
