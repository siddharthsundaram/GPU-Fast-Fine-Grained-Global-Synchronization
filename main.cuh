#ifndef MAIN_H
#define MAIN_H

#include <stdio.h>
#include "synch/mpi.cuh"

void sequential(char* input);
__global__ void basic(int num_increments, int *locks, int *shared_data, int size);
void gpu_buffer(int size, int num_servers, int num_clients);

#endif // MAIN_H
