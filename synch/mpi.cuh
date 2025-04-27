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
        
        Buffer *my_buf = &bufs[blockIdx.x];
        int empty_iterations = 0;
        const int MAX_EMPTY_ITERATIONS = 1000; // Add a threshold for empty iterations
        
        while (true) {
            int sent = atomicAdd(done, 0);
            bool processed = false;

            // receive_msg
            Message msg;

            if (dequeue(my_buf, &msg)) {
                // Acquire lock in shmem
                while (atomicCAS(&locks[msg.counter_idx], 0, 1) != 0) {
                    // Spin waiting for lock
                }

                // Critical section: increment counter
                counters[msg.counter_idx] += 1;
                __threadfence();

                // Release lock
                atomicExch(&locks[msg.counter_idx], 0);
                processed = true;
                empty_iterations = 0;
            } else {
                empty_iterations++;
            }

            // Exit when all messages have been processed AND buffer is empty for a while
            // This ensures we don't exit too early when messages might still be in transit
            if (sent >= num_threads && (isEmpty(my_buf) || empty_iterations > MAX_EMPTY_ITERATIONS)) {
                if (threadIdx.x == 0) {
                    printf("Server %d exiting. Processed %d/%d operations\n", 
                          blockIdx.x, sent, num_threads);
                }
                break;
            }
            
            // Add a small delay if no messages were processed to reduce contention
            if (!processed) {
                // Short yield/backoff to avoid excessive polling
                for (int i = 0; i < 10; i++) { 
                    __threadfence_block();
                }
            }
        }
    } else {
        // Client code: each thread sends exactly one message
        // We want exactly num_threads client threads to send messages
        int client_tid = tid - (num_server_blocks * blockDim.x);
        
        if (client_tid < num_threads) {
            // Each thread is responsible for one message only
            // We use client_tid as a unique client ID, and map it to a counter
            int counter_idx = client_tid % num_counters;
            int target_server = counter_idx % num_server_blocks;
            
            // Send one message per thread
            send_msg(target_server, counter_idx, bufs, done);
            
            if (threadIdx.x == 0 && blockIdx.x == num_server_blocks) {
                printf("Client block %d started sending messages\n", blockIdx.x);
            }
        }
    }

    __syncthreads();
}