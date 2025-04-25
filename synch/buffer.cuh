#pragma once
#include "message.cuh"

#define BUF_CAP 1024

struct Buffer {
    struct Message buf[BUF_CAP];
    int bitmask[BUF_CAP];
    int write_idx;
    int read_idx;
};

// __device__ bool isFull(Buffer *b) {
//     // return ((b->write_idx + 1) % BUF_CAP) == b->read_idx;

//     // This doesn't make full use of circular buffer after write_idx circles back
//     // to index 0, but not sure how to avoid the race in the alternative
//     return b->write_idx < b->read_idx;
// }

__device__ bool isEmpty(Buffer *b) {
    return b->write_idx == b->read_idx;
}

__device__ bool enqueue(Buffer *b, Message msg) {
    int current, next;
    
    do {
        current = b->write_idx;
        next = (current + 1) % BUF_CAP;
        
        // Check if buffer is full
        if (next == b->read_idx)
            return false;
            
    } while (atomicCAS(&b->write_idx, current, next) != current);
    
    b->buf[current] = msg;
    
    // Signal that this slot has valid data
    __threadfence();
    // b->bitmask[current] = 1;
    atomicExch(&b->bitmask[current], 1);
    
    return true;
}

__device__ bool dequeue(Buffer *b, Message *out_msg) {
    int current, next;
    
    do {
        current = b->read_idx;
        
        // Check if buffer is empty
        if (current == b->write_idx)
            return false;

        // check to see valid bitmask pos
        if (atomicAdd(&b->bitmask[current], 0) == 0)
            return false;
        
        next = (current + 1) % BUF_CAP;
    } while (atomicCAS(&b->read_idx, current, next) != current);
    
    *out_msg = b->buf[current];
    // free the bitmask
    atomicExch(&b->bitmask[current], 0);
    __threadfence(); // ensure visibility of the bitmask change
    return true;
}