
# GPU-Fast-Fine-Grained-Global-Synchronization

## Overview
An implementation of fast, fine-grained global synchronization between threads on a GPU based on a paper by Wang et al. (2019)

## Contributors
Siddharth Sundaram
Julian Canales
Brian Nguyen

## Build Instructions
### Shared Counters
To build/compile the shared counters mini-benchmark, run the following in the command line in the outermost directory:

`make`

To test the shared counters benchmark, edit the test file `inputs/test0`, which has the following format:

`# shared counters`
`# client threads`
`# server thread blocks`

Then, run the following command to test the input:

`make test`

### Hash Table
To build/compile the hash table benchmark, run the following in the command line in the outermost directory:

`make`

To test the hash table benchmark, edit the command line arguments in the Makefile (or don't and use the configs already present) and run the following command:

`make hash_table_all`

### EM-GMM

## References
Kai Wang, Don Fussell, and Calvin Lin. “Fast Fine-Grained Global Syn-
chronization on GPUs”. In: Proceedings of the Twenty-Fourth International
Conference on Architectural Support for Programming Languages and Op-
erating Systems (ASPLOS). Apr. 2019, pp. 793–806. url: https://www.
cs.utexas.edu/~lin/papers/asplos19.pdf.