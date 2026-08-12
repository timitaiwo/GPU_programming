git submodule update --init
rm -rf build
mkdir build

cd build

cmake -G Ninja "-DLLVM_ENABLE_PROJECTS=llvm;clang;lldb" ..

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLVM_PATH="$SCRIPT_DIR/third-party/llvm-project/llvm/bin"
export PATH="$LLVM_PATH:$PATH"

ninja  -J 8

cd ..
