#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>     // Using chrono for timing
#include <iostream>   // Added for std::cout and std::cerr
#include <cuda_runtime.h>
#include <string.h>
#include <unistd.h> 
#include "synch/buffer.cuh"
#include "synch/message.cuh"
#include "synch/mpi.cuh"
#include <boost/program_options.hpp> // Added for argument parsing