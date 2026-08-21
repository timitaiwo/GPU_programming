# Introduction

This directory explores the AMD HIP gpu compute build chain with a sample kernel that squares the input vector.

I am also curious to read/understand/breakdown the output ELF file that is the result of compilation with `hipcc`.

**Note:** This code has very simple one way logic, nothing fancy. The kernel either executes or doesn't. While compiling for Nvidia is possible with AMD's HIP, early attempts show that CUDA needs to be installed on the system.

My observations from deconstructing the elf files are [here](#observations) and the benchmark process [here](#benchmarking).


### Files

`main.hip`: the GPU kernel 

`kernel.readelf.v1`: a sample readelf output that does not have a hip kernel in the code

`kernel.readelf.v2`: a sample readelf output that has a hip kernel in the code


## Setup Instructions
The below needs to be installed according to your platform instructions

- `cmake`
- `ninja` (recommended)
- `hipcc` 
- `readelf` (needed to inspect the generated elf file in human readable format)
- `rocprofv3` (needed for benchmarking / profiling)

To build the kernel run the below instructions

```bash
mkdir build && cd build
cmake -G Ninja ..
ninja
```

Expect to see an ELF file named `kernel` in this `sample_kernel` directory. Use readelf to inspect ELF file.

```bash
cd ..  # Do this to leave the build directory
readelf -a kernel > kernel.readelf
```

# Observations

These are my observations / learnings when comparing the ELF file generated when the kernel is not in the code vs when it is as well as learning about the elf format.

1. Sections gives the view of the linker when combining separate object files into a single elf file
    Previously known info:
    - `text` section holds the computer program
    - `rodata` section holds the readonly data

2. Segments gives the view of the elf file as it is loaded by the kernel for execution

3. Since the kernel was compiled with `hipcc` there is still a dependency on the `amdhip64` shared library highlighted below. I assume that a program compiled with Nvidia's `nvcc` would have a similar CUDA dependency.

```txt
Dynamic section at offset 0x830 contains 31 entries:
  Tag        Type                         Name/Value
 0x0000000000000001 (NEEDED)             Shared library: [libamdhip64.so.7]
```

Attempting to run this on an Nvidia/CUDA system gave the error

```txt
root@27e41395f72a:/GPU_programming/sample_kernel# ./kernel
./kernel: error while loading shared libraries: libamdhip64.so.7: cannot open shared object file: No such file or directory
```


# Benchmarking

<TODO>
