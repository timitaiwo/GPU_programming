# Introduction

This directory explores the AMD HIP gpu compute build chain with a sample kernel that squares the input vector.

I am also curious to read/understand/breakdown the output ELF file that is the result of compilation with `hipcc`.

**Note:** This code has very simple one way logic, nothing fancy. The kernel either executes or doesn't.

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

2. Segments gives the view of the elf file as it is loaded by the kernel for execution


# Benchmarking

<TODO>
