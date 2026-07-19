# Introduction

```infrastructure_as_code:``` holds the infrastructure I use to run the GPU programs / experiements

```thunder_kittens:``` holds my forray into CUDA GPU programming with the ThunderKittens library


```vulkan:``` holds my forray into Vulkan development


# GPU Programming Challenges

The below are some GPU programming challenges. Note that the strategy used to move around the matrix can affect performance i.e. if you iterate over the matrix as a row, threads don't need to jump around memory looking for the next element because it would be next to the current one.

## Exercises

1. Given an array of values, add 10 to each element of the array e.g. [1, 3, 8, 2] becomes [11, 13, 18, 12]

2. Given an 2D array of values, add 10 to each element of the array e.g.

```
[
 [3, 6, 9],
 [2, 4, 0],
 [4, 1, 5]
]
```

becomes

```
[
 [13, 16, 19],
 [12, 14, 10],
 [14, 11, 15]
]
```

3. Given an array of values, add 10 to each element of the array e.g. [1, 3, 8, 2, 7, 3, 1, 10] becomes [11, 13, 18, 12, 17, 13, 11, 20]. The constraint is to use two threadblocks/workgroups 5 threads each that access the vector using shared memory

