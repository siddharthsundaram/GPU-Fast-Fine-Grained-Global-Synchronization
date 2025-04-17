#include <main.h>

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

__device__ int *d_locks;
__device__ int *d_shared_data;
__device__ int d_size;

int main(int argc, char **argv) {
    char *input_file = argv[1];

    FILE *file = fopen(input_file, "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file %s\n", input_file);
        return 1;
    }

    int h_size = 0;
    fscanf(file, "%d", &size);

    int num_increments = 0;
    fscanf(file, "%d", &num_increments);

    int *h_locks = new int[size];
    int *h_shared_data = new int[size];

    int *d_size, **d_locks, **d_shared_data;
    cudaMalloc(&d_size, size * sizeof(int));
    cudaMalloc(&d_locks, size * sizeof(int));
    cudaMalloc(&d_shared_data, size * sizeof(int));

    cudaMemcpy(d_size, &size, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_locks, h_locks, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_shared_data, h_shared_data, size * sizeof(int), cudaMemcpyHostToDevice);

    basic<<<(size + 255) / 256, 256>>>(num_increments, 0);

    cudaMemcpy(h_shared_data, d_shared_data, size * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; i++) {
        printf("shared_data[%d] = %d\n", i, h_shared_data[i]);
    }

    delete[] h_locks;
    delete[] h_shared_data;
    cudaFree(d_locks);
    cudaFree(d_shared_data);
}

__device__ bool try_lock(int data_id) {
    return atomicCAS(&d_locks[data_id], 0, 1) == 0;
}

__device__ void unlock(int data_id) {
    atomicExch(&d_locks[data_id], 0);
}

__device__ void critical_sec(int data_id, int arg0, int arg1) {
    d_shared_data[data_id] += arg0 + arg1;
}

__global__ void basic(int arg0, int arg1) {
    int data_id = blockDim.x * blockIdx.x + threadIdx.x;

    if (data_id < d_size) {
        for (int i = 0; i < arg0; i++) {
            bool success = false;
            do {
                if (try_lock(data_id)) {
                    critical_sec(data_id, 0, arg1);
                    __threadfence();
                    unlock(data_id);
                    success = true;
                }
            } while (!success);
        }
    }
}

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
        res[i]++;
    }

    delete[] res;
    fclose(file);
}