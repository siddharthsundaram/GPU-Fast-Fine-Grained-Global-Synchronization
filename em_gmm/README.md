# EM-GMM: Expectation Maximization for Gaussian Mixture Models Benchmark

## Overview
This part of the project implements the Expectation-Maximization (EM) workload for Gaussian Mixture Models (GMMs). A sequential, base GPU, fine-grained locking GPU, and fine-grained synchronization using scratchpad memory for locks and client-server architecture GPU implementations are done. 

## Folder Structure
- `src/`: Source files
- `obj/`: Build artifacts
- `out/`: Output data files
- `test/`: Test data files
- `em_results.png`: Speedup graph
- `Makefile`: Build instructions

Note: the `out/` files might not be up-to-date-were not maintained past initial testing for correctness.

## How to Build
Done at the root of the em_gmm folder.
```bash
make
```

## How to Run
```bash
./em_gmm -i <input_file> -o <output_file> -m <mode> -n <max_iterations> -t <tolerance> -s <seed> [-r <workload_ratio>]
```
Example:
```bash
./em_gmm -i test/skew_contention.txt -o out/test.txt -n 100 -t 0.1 -s 42 -m cs -r 0.99
```

### Command Line Flags
- `-i <input_file>`: Path to input data file.
- `-o <output_file>`: Path where output results will be saved - this was kept as an artifact from testing for correctness.
- `-m <mode>`: Execution mode:
  - `s`: Sequential CPU
  - `b`: Basic GPU
  - `f`: Fine-grained locking GPU
  - `cs`: Client-server GPU
- `-n <max_iterations>`: Maximum number of EM iterations.
- `-t <tolerance>`: Convergence tolerance.
- `-s <seed>`: Random seed for initialization reproducibility.
- `-r <workload_ratio>` (optional, only used for `cs` mode):  
  Ratio of server blocks to total SMs (streaming multiprocessors) when launching client-server GPU mode.  
  Example:  
  - `-r 0.2` means 20% of SMs become server blocks and the rest are clients.  
  - Internally:  
    ```cpp
    numServerBlocks = ceil(workload_ratio * numSMs)
    numClientBlocks = numSMs - numServerBlocks
    ```

