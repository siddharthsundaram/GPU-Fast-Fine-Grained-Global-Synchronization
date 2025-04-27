#include "main.h"
#include "arg_parser.h"
#include <chrono>

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
    auto start = std::chrono::high_resolution_clock::now();
    char *input_file = argv[1]; 
    parse_args(argc, argv);
    
    FILE *file = fopen(input_file, "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file %s\n", input_file);
        return 1;
    }
    
    int size = 0;
    fscanf(file, "%d", &size);
    
    int num_clients = 0;
    fscanf(file, "%d", &num_clients);
    
    int num_servers = 0;
    fscanf(file, "%d", &num_servers);
    
    fclose(file);

    // Sequential implementation
    if (seq) {
        sequential(size, num_clients);

    // Basic GPU implementation
    } else if (basic) {
        int *d_locks, *d_shared_data;
        CHECK(cudaMalloc(&d_locks, size * sizeof(int)));
        CHECK(cudaMalloc(&d_shared_data, size * sizeof(int)));
        CHECK(cudaMemset(d_locks, 0, size * sizeof(int)));
        CHECK(cudaMemset(d_shared_data, 0, size * sizeof(int)));
        dim3 block_size (std::min(num_clients, 256));
        dim3 grid_size ((num_clients + block_size.x - 1) / block_size.x + num_servers);
        basic_gpu<<<grid_size, block_size>>>(d_locks, d_shared_data, size, num_clients);
        cudaDeviceSynchronize();

        int *h_shared_data = new int[size]();
        CHECK(cudaMemcpy(h_shared_data, d_shared_data, size * sizeof(int), cudaMemcpyDeviceToHost));

        // int total = 0;
        // for (int i = 0; i < size; i++) {
        //     printf("shared_data[%d] = %d\n", i, h_shared_data[i]); 
        //     total += h_shared_data[i];
        // }

        delete[] h_shared_data;
        CHECK(cudaFree(d_shared_data));
        CHECK(cudaFree(d_locks));

        // printf("TOTAL: %d\n", total);

    // Fine grain synch GPU implementation
    } else if (fg) {
        gpu_buffer(size, num_servers, num_clients);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    printf("TIME: %lf\n", elapsed.count());

    return 0;
}

void gpu_buffer(int size, int num_servers, int num_clients) {
    int *h_locks = new int[size]();
    int *h_shared_data = new int[size](); 
    int *d_shared_data, *d_done;
    Buffer *d_bufs;
    CHECK(cudaMalloc(&d_done, sizeof(int)));
    CHECK(cudaMemset(d_done, 0, sizeof(int)));
    CHECK(cudaMalloc(&d_bufs, num_servers * sizeof(Buffer)));
    CHECK(cudaMemset(d_bufs, 0, num_servers * sizeof(Buffer)));
    CHECK(cudaMalloc(&d_shared_data, size * sizeof(int)));
    CHECK(cudaMemcpy(d_shared_data, h_shared_data, size * sizeof(int), cudaMemcpyHostToDevice));

    dim3 block_size (std::min(num_clients, 256));
    dim3 grid_size ((num_clients + block_size.x - 1) / block_size.x + num_servers);
    // printf("NUM CLIENTS: %i\n", num_clients);
    // printf("BLOCK SIZE: %i\n", block_size);
    // printf("GRID SIZE: %i\n", grid_size); 
    int shared_mem_size = num_servers * size * sizeof(int); 
    counters_client_and_server_entry<<<4, 1024, shared_mem_size>>>(d_shared_data, size, num_servers, d_bufs, d_done, num_clients);

    CHECK(cudaDeviceSynchronize());
    
    CHECK(cudaMemcpy(h_shared_data, d_shared_data, size * sizeof(int), cudaMemcpyDeviceToHost));

    CHECK(cudaDeviceSynchronize());
    
    // int total = 0;
    // for (int i = 0; i < size; i++) {
    //     printf("shared_data[%d] = %d\n", i, h_shared_data[i]); 
    //     total += h_shared_data[i];
    // }

    delete[] h_locks;
    delete[] h_shared_data;
    CHECK(cudaFree(d_shared_data));
    CHECK(cudaFree(d_done));
    CHECK(cudaFree(d_bufs));

    // printf("TOTAL: %d\n", total);
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

__global__ void basic_gpu(int *locks, int *shared_data, int size, int num_clients) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < num_clients) {
        int data_id = tid % size;

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

void sequential(int size, int num_increments) {
    int *res = new int[size]();
    for (int i = 0; i < num_increments; i++) {
        int j = i % size;
        res[j]++;
    }

    // int total = 0;
    // for (int i = 0; i < size; i++) {
    //     printf("sequential_data[%d] = %d\n", i, res[i]);
    //     total += res[i];
    // }

    // printf("TOTAL: %d\n", total);

    delete[] res;
}