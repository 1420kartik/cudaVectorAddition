# cudaVectorAddition
Sample program to test CUDA utilization

1. We are starting by importing the following libraries:
numba: To write CUDA kernels
numpy: For numerical calculation


2. Then we are defining the vector addition func:

- It uses @cuda.jit that will ensure the function will be comp[iled on CUDA capable GPU

- cuda.grid(1) method is used to get the global thread index and [idx] is used to store the unique index of each thread. Each thread will process one element of the vector, which allows for parallel computation.

- The kernel then performs the vector addition for each element of the input arrays a and b, storing the result in c.

3. In our Main Body:

- We are setting the size of each vector as 1024

- We initialize vectors a and b with ones, and c with zeros.

- We then move these vectors to the GPU's memory using cuda.to_device.

- Additionally, we define the number of threads per block as 256 and the number of blocks per grid as 4.
  4 * 256 = 1024
  This ensures all elements are processed.
  
- Finally, we call the vector addition function with all the defined parameters

- The result vector c is copied back from the GPU to local system and the result of the vector addition is printed.
