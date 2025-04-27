#include "main.cuh"
#include "arg_parser.h"
#include <cuda_runtime.h>
#include <algorithm> // For std::min
#include <stdio.h>

#define CHECK(call)                                                            \
    {                                                                          \
        const cudaError_t error = call;                                        \
        if (error != cudaSuccess)                                              \
        {                                                                      \
            printf("Error: %s:%d, ", __FILE__, __LINE__);                      \
            printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
            exit(1);                                                           \
        }                                                                      \
    }                                                                          \

// __device__ int *d_locks;
// __device__ int *d_shared_data;
// __device__ int d_size;

int main(int argc, char **argv) {
    // parse_args(argc, argv);
    char *input_file = argv[1]; 

    // if (true) {
    //     sequential(input_file);
    //     return 0;
    // }

    FILE *file = fopen(input_file, "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file %s\n", input_file);
        return 1;
    }

    int size = 0;
    fscanf(file, "%d", &size);

    int num_increments = 0;
    fscanf(file, "%d", &num_increments);

    int num_clients = 0;
    fscanf(file, "%d", &num_clients);

    int num_servers = 0;
    fscanf(file, "%d", &num_servers);

    fclose(file);

    // int *h_locks = new int[size]();
    // int *h_shared_data = new int[size]();
    // int *d_locks, *d_shared_data;
    // CHECK(cudaMalloc(&d_locks, size * sizeof(int)));
    // CHECK(cudaMalloc(&d_shared_data, size * sizeof(int)));
    // CHECK(cudaMemcpy(d_locks, h_locks, size * sizeof(int), cudaMemcpyHostToDevice));
    // CHECK(cudaMemcpy(d_shared_data, h_shared_data, size * sizeof(int), cudaMemcpyHostToDevice));

    // dim3 block_size (std::min(size, 1024));
    // dim3 grid_size ((size + block_size.x - 1) / block_size.x);
    // basic<<<grid_size, block_size>>>(num_increments, d_locks, d_shared_data, size);

    // CHECK(cudaDeviceSynchronize());
    
    // CHECK(cudaMemcpy(h_shared_data, d_shared_data, size * sizeof(int), cudaMemcpyDeviceToHost));

    printf("NUM COUNTERS: %i\n", size);
    printf("NUM INCREMENTS: %i\n", num_increments);
    gpu_buffer(size, num_servers, num_clients);

    // for (int i = 0; i < size; i++) {
    //     printf("shared_data[%d] = %d\n", i, h_shared_data[i]);
    // }

    // delete[] h_locks;
    // delete[] h_shared_data;
    // CHECK(cudaFree(d_locks));
    // CHECK(cudaFree(d_shared_data));
    return 0;
}

void gpu_buffer(int size, int num_servers, int num_clients) {
    int *h_locks = new int[size]();
    int *h_shared_data = new int[size](); 
    int *d_shared_data, *d_done;
    Buffer *d_bufs;
    // CHECK(cudaMalloc(&d_locks, size * sizeof(int)));
    CHECK(cudaMalloc(&d_done, sizeof(int)));
    CHECK(cudaMemset(d_done, 0, sizeof(int)));
    CHECK(cudaMalloc(&d_bufs, num_servers * sizeof(Buffer)));
    CHECK(cudaMemset(d_bufs, 0, num_servers * sizeof(Buffer)));
    CHECK(cudaMalloc(&d_shared_data, size * sizeof(int)));
    // CHECK(cudaMemcpy(d_locks, h_locks, size * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_shared_data, h_shared_data, size * sizeof(int), cudaMemcpyHostToDevice));

    dim3 block_size (std::min(num_clients, 256));
    dim3 grid_size ((num_clients + block_size.x - 1) / block_size.x + num_servers);
    printf("NUM COUNTERS: %i\n", size);
    printf("NUM CLIENTS: %i\n", num_clients);
    printf("BLOCK SIZE: %i\n", block_size.x);
    printf("GRID SIZE: %i\n", grid_size.x); 
    
    // Calculate operations per thread
    int operations_per_thread = std::max(1, size / num_clients);
    printf("OPERATIONS PER THREAD: %i\n", operations_per_thread);
    printf("TOTAL EXPECTED OPERATIONS: %i\n", operations_per_thread * num_clients);
    
    // Fix the shared memory allocation - it only needs to be large enough for the locks
    int shared_mem_size = size * sizeof(int); 
    counters_client_and_server_entry<<<grid_size, block_size, shared_mem_size>>>(
        d_shared_data, size, num_servers, d_bufs, d_done, num_clients);

    CHECK(cudaDeviceSynchronize());
    
    CHECK(cudaMemcpy(h_shared_data, d_shared_data, size * sizeof(int), cudaMemcpyDeviceToHost));

    CHECK(cudaDeviceSynchronize());
    
    int total = 0;
    for (int i = 0; i < size; i++) {
        printf("shared_data[%d] = %d\n", i, h_shared_data[i]); 
        total += h_shared_data[i];
    }

    printf("TOTAL: %d\n", total);
}

__device__ bool try_lock(int data_id, int *locks) {
    return atomicCAS(&locks[data_id], 0, 1) == 0;
}

__device__ void unlock(int data_id, int *locks) {
    atomicExch(&locks[data_id], 0);
}

__device__ void critical_sec(int data_id, int *shared_data) {
    shared_data[data_id] += 1;
}

__global__ void basic(int num_increments, int *locks, int *shared_data, int size) {
    int data_id = blockDim.x * blockIdx.x + threadIdx.x;

    if (data_id < size) {
        for (int i = 0; i < num_increments; i++) {
            bool success = false;
            do {
                if (try_lock(data_id, locks)) {
                    critical_sec(data_id, shared_data);
                    __threadfence();
                    unlock(data_id, locks);
                    success = true;
                }
            } while (!success);
        }
    }
}

// void init_cuda_constants(int size, int *d_shared_data) {
//     cudaMemcpyToSymbol(d_size, &size, sizeof(int));
//     cudaMalloc(&d_locks, size * sizeof(int));
//     cudaMalloc(&d_shared_data, size * sizeof(int));
// }

void sequential(char* input) {
    FILE *file = fopen(input, "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file %s\n", input);
        return;
    }

    int size = 0;
    fscanf(file, "%d", &size);

    int num_increments = 0;
    fscanf(file, "%d", &num_increments);

    int *res = new int[size];
    for (int i = 0; i < num_increments; i++) {
        for (int j = 0; j < size; j++) {
            res[j]++;
        }
    }

    for (int i = 0; i < size; i++) {
        printf("sequential_data[%d] = %d\n", i, res[i]);
    }

    delete[] res;
    fclose(file);
}