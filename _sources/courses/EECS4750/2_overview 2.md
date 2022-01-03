## Overview
This is an overview


**1.2.HC.HeterogeneousParallelComputing.20200924.pdf**

GPUs
- Throughput oriented cores
- Single Instruction Multiple Data


CPUs
- Latency
- Perform many different types of operations
- Large caches
- Branch prediction


**1.3.HC.PortabilityAndScalability.SWCost.20200924.pdf**

Parallel programming work flow
- Identify compute instensive parts of application
- Optimize data arrangement to maximus locality
- performance tuning

Software should be scalable and portable

Any algorithm complexity higher than linear is not data scalable

- A sequential algorithm with linear data scalability can outperform a parallel algorithm with nlog(n) complexity
- Parallelism cannot overcome complexity for large data sets

Load balance is important

**1.4.HC.CUDA.DataParallelismAndThreads.20200924.pdf**

Programming hierchy

Natural Language
Algorithm
High LEvel Language (C / C++)
<--- Compiler ---->
Instruction Set (Contract between hardware and software)
Microarchitecture
Circuits
Electrons


CUDA Overview
- Kernel executed by a grid of threads.
- All threads in a grid run the same kernel code (SPMD)
- Each thread has an index that is uses to compute memory addresses and make control decisions
- Threads within a thread block cooperate via shared memory atomic operations and barrier synchronization
- Blockidx and threadidx simplifies memory addressing when processing multidimensional data

**1.5.HC.CUDA.DataAllocationandMovement.APIs.20200924.pdf**

CUDA Continued

Host Code
1. Allocate device memory
2. Kernel launch code
3. Copy result data

cudaMalloc
- Allocates object in the device global memory
- Take addresses of a pointer to the allocated object and size of allocated object in terms of bytes

cudaFree
- Frees object from device glocal memory 
- Pointer to freed object

cudaMemcpy()
- Memory data transfer
- Requries four parameters
  - Pointer to destination
  - pointer to source
  - number of bytes coped
  - type / direction of transfer

**1.6.HC.CUDA.Kernel-BasedParallelProgramming.20200924.pdf**

CUDA Continued

Example vector addition kernel

```
__global__ void vector_addition(float* A_d, float* B_d, float* C_d, int n)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){
        C_d[i] = A_d[i] + B_d[i];
    }
```
num_blocks = (len_vec // blocksize + 1) 
num_threads_per_block = blocksize

dim_block = (blocksize, 1, 1)
dim_grid = (num_blocks, 1, 1)
To run this, run {num_blocks} blocks of {num_threads_per_block} threads each


Cuda function declarations

|   | Executed On  | Only callable from the:  |
|---|---|---|
| __device__ float DeviceFunc()  |  device |  device |
| __global__ void KernelFunc()  | device  |  host |
|  __host__ float HostFunc() | host  |  host |


**1.7.HC.MultidimensionalKernelConfiguration.20201001.pdf**

Multidimensional Kernel Configuration

- Row Major layout in C/C++
  - To access i = row*width+col

Grid, block thread
- A grid is a three dimensionall array of blocks
- Each block is a three dimensional array of threads

Host code for launching a picture kernel

- 16 threads per block
- length of picture / block_size + 1 to make sure threads for all parts
```
// assume that the picture is size mxn,
// m pixels in y dimension and n pixels in x dimension
// input d_Pin has been allocated on and copied to device
// output d_Pout has been allocated on device
dim3 DimGrid ( (n-1) /16 + 1,
( (m-1) /16+1, 1);
dim3 DimBlock (16, 16, 1) ;
PictureKernel<<<DimGrid,DimBlock>>>(d Pin, d Pout, n, m) ;
```


**1.8.HC.BasicMatrix-MatrixMultiplication.20201001.pdf**

Simple Square matrix multiplication
- Each thread calculates one element of P
- Each row of M is loaded width times from global memory
- Each column is load width times from global memory


```
__global__ void matrix_mult_simple(float* A, float* B, float* P, int m, int n, int k)){
    // Matrix A is (m X n)  (m is height (num_rows), n is width (num_cols))
    // Matrix B is (n X k)  (n is height (num_rows), k is width (num_cols))
    // Return is (m X k)    (m is height (num_rows), k is width (num_cols))
    row_out = blockIdx.x*blockDim.x + threadIdx.x;
    col_out = blockIdx.y*blockDim.y + threadIdx.y;

    if ((row_out < m) & (col_out < k))
    int sum = 0;
    for (int i=0; i < row_width; i++){
        sum += M[row_out*row_width_1 + i] * N[i*row_width_2 + col_out]
    }
    P[row_out*k + col_out] = sum;
}
```


Started tiled matrix operation?

**2.1.HC.ThreadScheduling.20201001.pdf**



Instruction Level Parallelism:
- 1st gen: instructions executed sequentially in program order
- 2nd gen: Pipelined execution. Don't wait until clothes are finished in dryer before starting next wash cycle
- 3rd level: instructions executed in paralle. (Need to be independent)
- 4th gen: multi threading. Same processor
- 5th gen: multi core, multi processors 

Von Neurmann Model with SIMD Units

memory
ALU
Reg 
etc

Warps as a scheduling Unit
- Each block executed as 32 thread warps
- Hardware imlementation decision. Not part of the cuda programing model


Question: If 3 blocks are assigned to a streaming multiprocesser and each block has 256 threads, how many warps are there in a streaming multiproccesor. 

Answer: Each block is divided into 256/32 = 8 warps. There are 8*3=24 warps 

All threads in a warp execute the same instruction

For matrix multiplication using multiple blocks. 

- 8x8 blocks for fermi:
  - 64 threads per block
  - Each SM can take up to 1536 threads
  - 1536 threads / 64 threads/block = 24 blocks
  - Each SM only takes 8 blocks, only 64*8=512 threads per block
- 16x16 blocks for fermi:
  - 256 threads per block
  - Each SM can take up to 1536 threads
  - 1536 threads / 256 threads/block = 6 blocks
  - Each SM takes 6 blocks, only 256*6=1536 threads per block
- 32x32 blocks for fermi:
  - 1024 threads per block
  - Each SM can take up to 1536 threads
  - 1536 threads / 1024 threads/block =~ 1 block
  - Each SM takes 1 blocks, only 1024 threads per block


**2.2.HC.ControlDivergence.20200924.pdf**

Control divergence

Warps as scheduling units
- Implementation decision. Not part of the CUDA programming model
- Thread IDs within a warp are consecutive and increasing
- If working across warps, can't rely on any rdering. Need to use __syncthreads()
- Need to keep track of control divergence within a warp

Example with divergence:
``` if (threadIdx.x > 2) ```
- Threads within a block have different paths

Example without divergence:
``` if (blockDim.x > 2) ```
- Branch granularity is a multiple of block size. Al threads in any given warp follow the same path

**2.3.HC.CUDAmemories.modelAndLocality.20200924.pdf**


CUDA memory delay

- Read/write per threads
  - Registers (~1 cycle) per-thread
  - Shared memory (~5 cycles) per-block
  - Global memory (~500 cycles) per-grid
- Read only 
  - Constant memory (~5 cycles with cache) per-grid



|  Variable declartion | Memory  | Scope | Lifetime |
|---|---|---|---|
|  ```int local_var; ```| register  | thread  |  thread |
| ```__device__ __shared__ int shared_var;  ```|  shared |  block | block  |
|  ```__device__ int global_var; ```| global  | grid  | application  |
| ```__device__ __constant__ int const_var; ```|  constant | grid | application  |


Common programming strategy is to partition data into subsets or tiles that fit into shared memory. Use one thread block to handle each tile by
    - Loading the tile from global memory into shared memory. (Uses multiple threads bu only loaded once per block)
    - Perform operation with access to shared memory

**2.4.HC.TiledParallelAlgorithms.20200924.pdf**

Study on speed of traditional matrix multiplication without tiling
- Two memory accesses (8 bytes) per floating point multiply add
- 4 B /s of memory bandwith per FLOPS (floating point operations per section)
- Peak floating point rate is 1000 G FLOPS
- 4*1000 = 4000 GB/s required to achieve peak FLOP rating
- Reality 150 GB/s limits the code at 37.5 GFlops

**2.5.HC.TiledMatrixMultiplication.20201007.pdf**

Tiled matrix multiply:

```

__global__ void tiled_mat_mul(float* A, float* B, float* C, w){

  // Define the __shared__ mem tiles
  // Tile width should be same 
  int row = blockDim.x * blockIdx.x + threadIdx.x;
  int col = blockDim.y * blockIdx.y + threadIdx.y;

  for (int i=0; i < something; i++){


    __shared__ float load_tile_a[blockDim.y*i + threadIdx.y][blockDim.x];
    __shared__ float load_tile_b[blockDim.y][blockDim.x*i + threadIdx.x];
    
    //Loading

    load_tile_a[col][row] = A[col][row];
    load_tile_b[col][row] = B[col][row];



    __syncthreads();


    // Perform the mat mult

  }
}

```


**2.6.HC.TiledMatrixMultiplicationKernel.20201007.pdf**
**2.7.HC.BoundaryConditionsInTiling.20201007.pdf**
**2.8.HC.BoundaryConditionsTilingKernel.20201007.pdf**
**3.1.HC.DramBandwidth.20201022.pdf**
**3.2.HC.MemoryCoalescing.20211028.pdf**
**3.3.HC.Convolution.20201008.pdf**
**3.4.HC.TiledConvolution.20201008.pdf**



**3.5.HC.TileBoundaryConditionKernel.20201008.pdf**
**3.6.HC.ConvolutionReuse.20191009.pdf**
**4.1.HC.Reduction.20201022.pdf**
**4.2.HC.ReductionKernel.20201022.pdf**
**4.3.HC.BetterReductionKernel.20201022.pdf**
**4.4.HC.ScanParallelPrefixSum.20201022.pdf**
**4.5.HC.WorkInefficientScanKernel.20201022.pdf**
**4.6.HC.WorkEfficientScanKernel.20201022.pdf**
**4.7.HC.MoreOnParallelScan.20191014.pdf**
**5.1.HC.Histogramming.20201028.pdf**
**5.2.HC.AtomicOperations.20201028.pdf**
**5.3.HC.AtomicOperationsInCUDA.20201028.pdf**
**5.4.HC.AtomicOperationPerformance.20201028.pdf**
**5.5.HC.PrivatizedHistogram.20201028.pdf**
**15.1.HC.FloatingPoint.20201029.pdf**
**15.2.HC.FloatingPointGPUs.20201029.pdf**
