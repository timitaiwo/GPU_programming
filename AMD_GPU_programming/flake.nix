{
  description = "LLDB exploration";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        clang = pkgs.llvmPackages_22.clang;
        # clang-unwrapped = pkgs.symlinkJoin {
        #   name = "clang-unwrapped-merged";
        #   paths = [
        #     pkgs.llvmPackages_22.clang-unwrapped.dev
        #     pkgs.llvmPackages_22.clang-unwrapped.lib
        #     pkgs.llvmPackages_22.clang-unwrapped.out
        #   ];
        # };
      in
      {
        devShells.default = pkgs.mkShell {
          name = "lldb exploration";

          buildInputs = [
            # Build system
            pkgs.cmake
            pkgs.ninja
            # pkgs.libffi
            # pkgs.libxml2
            # pkgs.swig
            # pkgs.tree-sitter
            # pkgs.xz

            # Languages
            # pkgs.python3
            # pkgs.python3Packages.six
            # pkgs.nodejs

            # LLVM toolchain
            clang
            # clang-unwrapped

            # Libraries
            # pkgs.libedit
            # pkgs.zstd
            # pkgs.curl
          ];

          # Ensure the shell knows where to find LLVM
          shellHook = ''
            export CC=clang
            export CXX=clang++
            export CLANG_RESOURCE_DIR="$(clang -print-resource-dir)"

            export CFLAGS="$NIX_CFLAGS_COMPILE"
            export CXXFLAGS="$NIX_CFLAGS_COMPILE"

            echo "   CMake:   $(cmake --version | head -1)"
            echo "   Ninja:   $(ninja --version)"
          '';
        };
      }
    );
}
