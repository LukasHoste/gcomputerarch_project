#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>


#define N 65536/sizeof(int)
#define THREADS_PER_BLOCK 64
#define AMOUNT_OF_BLOCKS ((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK)


#define ITERATIONS 1000

struct TimeInfo {
    float timeCPU;
    float timeGPU;
};


struct BigTimeInfo {
    float timeGPUGPU;
    float timeGPUCPU;
    float timeCPU;
};


__global__ void flip__gpu(int *a, int *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Global thread index

    if (i < N) {  // Prevent out-of-bounds access
        b[n - i - 1] = a[i];
    }
}


// both fn return the ms elapsed
float cpuFlip(int*a, int*b, int n) {
    // start time
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < n; i++) {
        b[n - i - 1] = a[i];
    }
    // end time
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();  // Convert to milliseconds
}

TimeInfo gpuFlip(int*a , int*b, int n, int threadsPerBlock, int amountOfBlocks) {
    TimeInfo timeInfo;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
        // Allocate memory on device
    int *dev_a, *dev_b;
    cudaMalloc((void**)&dev_a, n * sizeof(int));
    cudaMalloc((void**)&dev_b, n * sizeof(int));
    // start and end of CPU using chrona
    const auto startCPU = std::chrono::steady_clock::now();
    // clock_t startCPU, endCPU;
    // startCPU = clock();


    // Copy input array to device
    cudaMemcpy(dev_a, a, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaEventRecord(start);

    // Configure kernel launch parameters
    // int blocksPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK; // Ensure all elements are covered

    // Launch the kernel with multiple blocks and threads
    flip__gpu<<<amountOfBlocks, threadsPerBlock>>>(dev_a, dev_b, n);
    cudaDeviceSynchronize(); // Ensure kernel execution completes
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Copy result back to host
    cudaMemcpy(b, dev_b, n * sizeof(int), cudaMemcpyDeviceToHost);

   
    //endCPU = clock();
    const auto endCPU = std::chrono::steady_clock::now();
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    timeInfo.timeGPU = milliseconds;
    //timeInfo.timeCPU = (float) (endCPU - startCPU) * 1000 / CLOCKS_PER_SEC;
    timeInfo.timeCPU = std::chrono::duration<double, std::milli>(endCPU - startCPU).count();  // Convert to milliseconds

     // Free device memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    return timeInfo;
}


BigTimeInfo timedIteration(int n, int threadsPerBlock, int amountOfBlocks) {
    BigTimeInfo bigTimeInfo;
    int a[n], b[n], c[n];
    // Initialize input array
    for (int i = 0; i < n; i++) {
        a[i] = i;
    }

    TimeInfo gpuInfo = gpuFlip(a, b, n, threadsPerBlock, amountOfBlocks);
    float cpuMs = cpuFlip(a, c, n);
    for (int i = 0; i < n; i++) {
        if (b[i] != c[i]) {
            printf("Error: CPU and GPU results do not match\n");
            break;
        }
    }
    bigTimeInfo.timeGPUCPU = gpuInfo.timeCPU;
    bigTimeInfo.timeGPUGPU = gpuInfo.timeGPU;
    bigTimeInfo.timeCPU = cpuMs;
    return bigTimeInfo;
}

BigTimeInfo test(int n, int threadsPerBlock, int amountOfBlocks) {

    float cpuSum = 0;
    float gpuGpuSum = 0;
    float gpuCpuSum = 0;

    for (int i = 0; i < ITERATIONS; i++) {
        BigTimeInfo bigTimeInfo = timedIteration(n, threadsPerBlock, amountOfBlocks);
        cpuSum += bigTimeInfo.timeCPU;
        gpuGpuSum += bigTimeInfo.timeGPUGPU;
        gpuCpuSum += bigTimeInfo.timeGPUCPU;
    }

    BigTimeInfo bigTimeInfo;
    bigTimeInfo.timeCPU = cpuSum / ITERATIONS;
    bigTimeInfo.timeGPUGPU = gpuGpuSum / ITERATIONS;
    bigTimeInfo.timeGPUCPU = gpuCpuSum / ITERATIONS;
    return bigTimeInfo;
}



int main() {
    //printf("ThreadsPerBlock, TimeCPU, TimeGPU\n");
    /*
    for (int i = 2; i <= 128; i+= 2) {
        int threadsPerBlock = i;
        int amountOfBlocks = ((N + threadsPerBlock - 1) / threadsPerBlock);
        BigTimeInfo info = test(N, threadsPerBlock, amountOfBlocks);
        printf("%d, %f, %f\n", threadsPerBlock, info.timeCPU, info.timeGPUGPU);

    }
    */
    printf("N, TimeCPU, TimeGPU\n");
   for (int i = 64; i <= N; i+= 64) {
    int n = i;
    int amountOfBlocks = ((n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    BigTimeInfo info = test(n, THREADS_PER_BLOCK, amountOfBlocks);
    printf("%d, %f, %f\n", n, info.timeCPU, info.timeGPUGPU);
   }
   return 0;


}