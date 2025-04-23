#include "main.h"
#include "arg_parser.h"

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

    gpu_buffer(size, num_increments);

    // for (int i = 0; i < size; i++) {
    //     printf("shared_data[%d] = %d\n", i, h_shared_data[i]);
    // }

    // delete[] h_locks;
    // delete[] h_shared_data;
    // CHECK(cudaFree(d_locks));
    // CHECK(cudaFree(d_shared_data));
    return 0;
}

void gpu_buffer(int size, int num_servers) {
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

    dim3 block_size (std::min(size, 1024));
    dim3 grid_size ((size + block_size.x - 1) / block_size.x);
    int shared_mem_size = num_servers * size * sizeof(int);
    counters_client_and_server_entry<<<4, 1024, shared_mem_size>>>(d_shared_data, size, num_servers, d_bufs, d_done);

    CHECK(cudaDeviceSynchronize());
    
    CHECK(cudaMemcpy(h_shared_data, d_shared_data, size * sizeof(int), cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < size; i++) {
        printf("shared_data[%d] = %d\n", i, h_shared_data[i]);
    }
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