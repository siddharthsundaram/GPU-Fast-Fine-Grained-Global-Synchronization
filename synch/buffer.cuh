#pragma once
#include "message.cuh"

#define BUF_CAP 1024

struct Buffer {
    Message buf[BUF_CAP];
    bool bitmask[BUF_CAP];
    int write_idx;
    int read_idx;
};

__device__ bool isFull(Buffer *b) {
    // return ((b->write_idx + 1) % BUF_CAP) == b->read_idx;

    // This doesn't make full use of circular buffer after write_idx circles back
    // to index 0, but not sure how to avoid the race in the alternative
    return b->write_idx < b->read_idx;
}

__device__ bool isEmpty(Buffer *b) {
    return b->write_idx == b->read_idx;
}

__device__ void enqueue(Buffer *b, Message msg) {
    int idx = atomicAdd(&b->write_idx, 1) % BUF_CAP;

    while (atomicCAS(&b->bitmask[idx], 0, 1) != 0) {}
    b->buf[idx] = msg;

    // might need to threadfence here?
    __threadfence()
}

__device__ void dequeue(Buffer *b, Message *out_msg) {
    int idx = atomicAdd(&b->read_idx, 1) % BUF_CAP;

    while (atomicCAS(&b->bitmask[idx], 1, 0)) {}

    // might need to threadfence here?

    *out_msg = b->buf[atomicAdd(&b->read_idx, 1)];
}