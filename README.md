# RSA_in_CUDA
A Parallel implementation of RSA in CUDA
----------------------
Command Line Arguments
----------------------
-make
-Enter input as prompted on the screen for prime numbers and message 
-------------------
Compiling Options
--------------------
1.Unoptimized version
nvcc rsa.cu
2. Optimized version
nvcc g nvcc –o rsamod rsamodified.cu –arch=compute53 –rdc=true 
    –use_fast_math 
-------------------------
Kernels Launches
--------------------------
1. Unoptimized version
isprime << <blocksPerGrid, threadsPerBlock >> >(deviceflag);
ce << <<blocksPerGrid, threadsPerBlock >> >(d_t,d_e,d_d);
encrypt<<<blocksPerGrid, threadsPerBlock >>>(d_e, d_msg);
decrypt<<<blocksPerGrid, threadsPerBlock >>>(d_d, d_msg);

2. Optimized version
rsa <<< 1,1>>>(d_t,d_e,d_d,d_msg,e_msg,d_len); ------> Parent kernel
ce << <blocksPerGrid, threadsPerBlock>> > (d_t,d_e,d_d);----> Child Kernel
encrypt << <<blocksPerGrid, threadsPerBlock>> > (d_e,d_msg,e_msg);------> Child Kernel
decrypt << <<blocksPerGrid, threadsPerBlock>> > (d_d,d_msg,e_msg);-------> Child Kernel
-------------------------
Design Decisions & Issues
-------------------------
1. Unoptimized version
With many independent kernel execution for each functionality gave poor performance.
There was increased usage of registers and occupancy was low.
2. Optimized Approach
With dynamic parallelisation techniques the execution time was significantly lowered.Memory access were better optimised. Compute itensive kernels were equally distributed.
----------
Profiling
----------
We used the Tegra visual profiler for visualizing the performance of the program and from various analysis it was found that :
In unoptimized approach the CE kernel took the longest and was the highest compute demanding kernel.
The optimized approach computation got distributed amongts the child kernel. The isprime kernel was removed and was replaced with a __device function. 
Register count:
parent used 32 registers per thread
ce used 26 registers per thread
encrypt used 32 registers per thread
decrypt used 26 registers per thread

nvprof results for both approaches:
1. Unoptimized approach
==6954== Profiling application: ./rsa
==6954== Profiling result:
Time(%)      Time     Calls       Avg       Min       Max  Name
 59.02%  963.26us         1  963.26us  963.26us  963.26us  ce(long*, long*, long*)
 22.24%  362.93us         1  362.93us  362.93us  362.93us  decrypt(long*, long*)
  7.66%  124.95us         2  62.475us  46.928us  78.023us  isprime(int*)
  5.43%  88.544us         1  88.544us  88.544us  88.544us  encrypt(long*, long*)
  3.37%  54.950us         6  9.1580us  6.8760us  11.458us  [CUDA memcpy DtoH]
  2.29%  37.399us        13  2.8760us  2.2910us  6.9800us  [CUDA memcpy HtoD]

2. Modified approach
==6615== Profiling application: ./rsamodified
==6615== Profiling result:
Time(%)      Time  Calls (host)  Calls (device)       Avg       Min       Max  Name
 95.17%  29.447ms             1               0  29.447ms  29.447ms  29.447ms  rsa(long*, long*, long*, long*, long*, int*)
  3.43%  1.0614ms             0               1  1.0614ms  1.0614ms  1.0614ms  ce(long*, long*, long*)
  0.70%  216.67us             0               1  216.67us  216.67us  216.67us  encrypt(long*, long*, long*)
  0.61%  188.86us             0               1  188.86us  188.86us  188.86us  decrypt(long*, long*, long*)
  0.09%  27.396us             9               -  3.0440us  2.2910us  5.6770us  [CUDA memcpy HtoD]


--------------------
Algorithmic Analysis
--------------------

Upon comparing both the approaches we found that the algorithm with multiple kernel launches performed badly compared to dynamic parallelism implementation. The use of fast math compiling option further boosted the performance. For computation of multiplication intrinsic fucntion of _mul24 was used compared to normal one which helped to reduce the clock cycles significantly for large prime numbers. Also the fast math square root option was used instead of normal sqrt function. The child kernel launch helped to effciently increase the occupancy and only release the desired number of threads as per computation avoiding launch of large unused threads.

