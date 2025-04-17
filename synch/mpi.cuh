#pragma once
#include "buffer.cuh"

__device__ void sleep(int cycles) {
    clock_t start = clock();

    while (clock() - start < cycles);
}

// used by the client to send 
__device__ void send_msg(int target_server, int index_to_increment, Buffer *bufs) {
    // get target servers buffer and call enqueue
    Buffer *buf = &bufs[target_server];
    Message msg;
    msg.counter_idx = index_to_increment;
    int delay = 1000;

    while (!enqueue(buf, msg)) {
        sleep(delay);
        delay = min(delay * 2, 64000);
    }
}

__global__ void counters_client_and_server_entry(int *counters, int num_counters, int num_server_blocks, Buffer *bufs, int *done) {
    bool is_server = blockIdx.x < num_server_blocks;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    int num_clients = total_threads - (num_server_blocks * blockDim.x);
    
    if (is_server) {
        // do server stuff
        extern __shared__ int locks[num_counters];
        for (int i = tid; i < num_counters; i += blockDim.x) {
            locks[i] = 0;
        }
        __syncthreads();
        
        Buffer *my_buf = &bufs[blockIdx.x];
        
        while(true) {
            int sent = atomicAdd(done, 0);

            // receive_msg
            Message msg;
            dequeue(my_buf, &msg);

            // Acquire lock in shmem
            while (atomicCAS(&locks[msg.counter_idx], 0, 1) != 0);

            // Critical section: increment counter
            counters[msg.counter_idx] += 1;

            // Release lock
            atomicExch(&locks[msg.counter_idx], 0);

            if (sent >= num_clients && isEmpty(my_buf)) {
                break;
            }
        }

    } else {
        // do client stuff
        // calculate target server
        int counter = tid % num_counters;
        int target_server = counter % num_server_blocks;
        // send message to server
        send_msg(target_server, counter, bufs);
        atomicAdd(done, 1);
    }
}