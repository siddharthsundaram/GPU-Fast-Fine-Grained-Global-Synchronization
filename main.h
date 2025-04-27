#ifndef MAIN_H
#define MAIN_H

#include <stdio.h>
#include "synch/mpi.cuh"

void sequential(int size, int num_increments);
__global__ void basic_gpu(int *locks, int *shared_data, int size, int num_clients);
void gpu_buffer(int size, int num_servers, int num_clients);

#endif // MAIN_H
