#pragma once
#include "buffer.cuh"

__device__ void sleep(int cycles) {
    clock_t start = clock();

    while (clock() - start < cycles);
}

// used by the client to send 
__device__ void send_msg(int target_server, int index_to_increment, Buffer *bufs, int *done) {
    // get target servers buffer and call enqueue
    Buffer *buf = &bufs[target_server];
    Message msg;
    msg.counter_idx = index_to_increment;
    int delay = 1000;

    while (!enqueue(buf, msg)) {
        sleep(delay);
        delay = min(delay * 2, 64000);
    }

    atomicAdd(done, 1);
}

__global__ void counters_client_and_server_entry(int *counters, int num_counters, int num_server_blocks, Buffer *bufs, int *done, int num_threads) {
    bool is_server = blockIdx.x < num_server_blocks;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    // int num_clients = total_threads - (num_server_blocks * blockDim.x);
    
    if (is_server) {

        // TODO: Possibly move this to a separate kernel
        // do server stuff
        extern __shared__ int locks[];
        for (int i = threadIdx.x; i < num_counters; i += blockDim.x) {
            locks[i] = 0;
        }

        if (threadIdx.x == 0) {
            printf("Server %d initialized locks to 0 in shared memory\n", blockIdx.x);
        }

        __syncthreads();
        // __threadfence_block();
        
        Buffer *my_buf = &bufs[blockIdx.x];
        
        while (true) {
            int sent = atomicAdd(done, 0);

            // receive_msg
            Message msg;

            // TODO: Need to check dequeue return val
            if (dequeue(my_buf, &msg)) {

                // printf("Server received msg: counter_idx = %d\n", msg.counter_idx);

                // Acquire lock in shmem
                while (atomicCAS(&locks[msg.counter_idx], 0, 1) != 0) {
                    // printf("Waiting for lock on counter_idx %d\n", msg.counter_idx);
                }

                // Critical section: increment counter
                counters[msg.counter_idx] += 1;
                __threadfence();
                // printf("Incremented counters[%d] = %d\n", msg.counter_idx, counters[msg.counter_idx]);

                // Release lock
                atomicExch(&locks[msg.counter_idx], 0);
                // printf("Released lock on counter_idx %d\n", msg.counter_idx);
            }

            if (sent >= num_threads && isEmpty(my_buf)) {
                break;
            }
        }

    } else {
        // do client stuff
        // calculate target server
        if (tid < (num_server_blocks * blockDim.x + num_threads)) {
            int counter = tid % num_counters;
            int target_server = counter % num_server_blocks;
            // send message to server
            send_msg(target_server, counter, bufs, done);
            // atomicAdd(done, 1);
            // printf("DONE: %d\n", atomicAdd(done, 1));
        }
    }

    __syncthreads();
}