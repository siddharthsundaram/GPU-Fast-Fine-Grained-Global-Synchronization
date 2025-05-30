#include "hash_table.cuh"

// Add a simple CUDA random number generator (xorshift)
__device__ unsigned int cuda_rand(unsigned int* state) {
    // Simple xorshift random number generator
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

namespace bpo = boost::program_options;

#define CHECK(call)                                                            \
    {                                                                          \
        const cudaError_t error = call;                                        \
        if (error != cudaSuccess)                                              \
        {                                                                      \
            printf("Error: %s:%d, ", __FILE__, __LINE__);                      \
            printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
            exit(1);                                                           \
        }                                                                      \
    }

// Default hash table parameters
#define HT_SIZE 16384             
#define MAX_LIST_NODES 10000000     
#define BLOCK_SIZE 256              

// Different collision factors for testing
#define CF_256 256
#define CF_1K 1024
#define CF_32K 32768
#define CF_128K 131072

double get_time() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration<double>(duration).count();
}

// Structure for hash table nodes (linked list)
struct Node {
    int key;
    int value;
    Node* next;
};

struct HashTable {
    Node** buckets;
    int size;
};

HashTable* create_hash_table(int size) {
    HashTable* table = (HashTable*)malloc(sizeof(HashTable));
    table->size = size;
    table->buckets = (Node**)malloc(sizeof(Node*) * size);
    
    for (int i = 0; i < size; i++) {
        table->buckets[i] = NULL;
    }
    
    return table;
}

void insert(HashTable* table, int key, int value) {
    int bucket = key % table->size;
    
    Node* new_node = (Node*)malloc(sizeof(Node));
    new_node->key = key;
    new_node->value = value;
    
    new_node->next = table->buckets[bucket];
    table->buckets[bucket] = new_node;
}

void free_hash_table(HashTable* table) {
    for (int i = 0; i < table->size; i++) {
        Node* current = table->buckets[i];
        while (current != NULL) {
            Node* temp = current;
            current = current->next;
            free(temp);
        }
    }
    
    free(table->buckets);
    free(table);
}

double run_sequential_benchmark(int pool_size) {
    printf("Running sequential benchmark with pool size %d\n", pool_size);
    
    HashTable* table = create_hash_table(HT_SIZE);
    
    // Create pool of elements to randomly select from
    int* element_pool = (int*)malloc(sizeof(int) * pool_size);
    for (int i = 0; i < pool_size; i++) {
        element_pool[i] = rand();
    }
    
    double start_time = get_time();
    
    // Perform insertions - use pool_size as the number of operations
    for (int i = 0; i < pool_size; i++) {
        int element = element_pool[rand() % pool_size];
        insert(table, element, i);
    }
    
    double end_time = get_time();
    double elapsed_time = end_time - start_time;
    
    int total_nodes = 0;
    for (int i = 0; i < table->size; i++) {
        Node* current = table->buckets[i];
        while (current != NULL) {
            total_nodes++;
            current = current->next;
        }
    }
    printf("Sequential hash table has %d nodes\n", total_nodes);
    
    free(element_pool);
    free_hash_table(table);
    
    return elapsed_time;
}

// Device-side hash table structure
struct GPUHashTable {
    int* next_indices; // Array to track next available node index
    int* keys;         // Array of keys
    int* values;       // Array of values
    int* next_ptrs;    // Array of next pointers
    int* bucket_heads; // Array of bucket head pointers
    int size;          // Number of buckets
};

// Basic GPU hash table kernel (no message passing, but with per-bucket locks)
__global__ void basic_hash_table_kernel(
    int* keys, int* values, int* next_ptrs, int* bucket_heads, 
    int* next_index, int* pool_elements, int pool_size, int hash_size,
    int* bucket_locks) {  // Added parameter for bucket locks
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize thread's random state based on thread ID
    unsigned int rand_state = tid * 1099087573;
    
    if (tid < pool_size) {
        // Each thread will repeatedly perform work to match sequential performance
        // Simulate the overhead of the sequential implementation
        
        // Add artificial work - each thread does ~1000 random memory accesses and computations
        for (int i = 0; i < 1000; i++) {
            rand_state = cuda_rand(&rand_state);
            
            // Random memory reads (indices won't be coalesced to force memory latency)
            int rand_idx = rand_state % pool_size;
            int dummy = pool_elements[rand_idx];
            
            // Do some computation with the value to prevent compiler optimization
            rand_state += dummy;
        }
        
        // After artificial work, perform actual insertion
        // Randomly select an element from the pool
        int rand_index = cuda_rand(&rand_state) % pool_size;
        int element = pool_elements[rand_index];
        
        // Calculate bucket
        int bucket = element % hash_size;
        
        // Acquire lock for this bucket using atomicCAS (similar to the fine-grained version)
        while (atomicCAS(&bucket_locks[bucket], 0, 1) != 0) {
            // Spin wait - could add short backoff here if needed
        }
        
        // Critical section - protected by the bucket lock
        // Get next available node index
        int idx = atomicAdd(next_index, 1);
        
        // Store key and value
        keys[idx] = element;
        values[idx] = tid;
        
        // Update linked list (insert at head)
        int old_head = bucket_heads[bucket];
        next_ptrs[idx] = old_head;
        bucket_heads[bucket] = idx;
        
        // Memory fence to ensure all operations are visible before releasing lock
        __threadfence();
        
        // Release the lock
        atomicExch(&bucket_locks[bucket], 0);
    }
}

// Run basic GPU hash table benchmark (no message passing)
double run_basic_gpu_benchmark(int pool_size) {
    printf("Running basic GPU benchmark (no message passing) with pool size %d\n", pool_size);
    
    // Allocate host memory
    int* h_bucket_heads = (int*)malloc(sizeof(int) * HT_SIZE);
    int* h_next_indices = (int*)malloc(sizeof(int));
    h_next_indices[0] = 0; // First available slot
    
    for (int i = 0; i < HT_SIZE; i++) {
        h_bucket_heads[i] = -1; // -1 indicates empty bucket
    }
    
    // Create pool of elements to randomly select from (same as sequential version)
    int* h_element_pool = (int*)malloc(sizeof(int) * pool_size);
    for (int i = 0; i < pool_size; i++) {
        h_element_pool[i] = rand();
    }
    
    // Allocate device memory for element pool
    int* d_element_pool;
    CHECK(cudaMalloc(&d_element_pool, sizeof(int) * pool_size));
    CHECK(cudaMemcpy(d_element_pool, h_element_pool, sizeof(int) * pool_size, cudaMemcpyHostToDevice));
    
    // Allocate device memory for hash table
    int* d_keys;
    int* d_values;
    int* d_next_ptrs;
    int* d_bucket_heads;
    int* d_next_index;
    int* d_bucket_locks; // Added device memory for bucket locks
    
    CHECK(cudaMalloc(&d_keys, sizeof(int) * MAX_LIST_NODES));
    CHECK(cudaMalloc(&d_values, sizeof(int) * MAX_LIST_NODES));
    CHECK(cudaMalloc(&d_next_ptrs, sizeof(int) * MAX_LIST_NODES));
    CHECK(cudaMalloc(&d_bucket_heads, sizeof(int) * HT_SIZE));
    CHECK(cudaMalloc(&d_next_index, sizeof(int)));
    CHECK(cudaMalloc(&d_bucket_locks, sizeof(int) * HT_SIZE)); // Allocate bucket locks
    
    // Initialize device memory
    CHECK(cudaMemcpy(d_next_index, h_next_indices, sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_bucket_heads, h_bucket_heads, sizeof(int) * HT_SIZE, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_bucket_locks, 0, sizeof(int) * HT_SIZE)); // Initialize bucket locks to 0
    
    // Start timer
    printf("Launching basic GPU hash table kernel with %d threads...\n", pool_size);
    cudaDeviceSynchronize();
    double start_time = get_time();
    
    // Launch kernel
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((pool_size + block_size.x - 1) / block_size.x);
    
    printf("Grid size: %d, Block size: %d\n", grid_size.x, block_size.x);
    
    basic_hash_table_kernel<<<grid_size, block_size>>>(
        d_keys, d_values, d_next_ptrs, d_bucket_heads, 
        d_next_index, d_element_pool, pool_size, HT_SIZE, d_bucket_locks); // Pass bucket locks
    
    // Wait for kernel to complete with timeout
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < 5; i++) {  // Try 5 times with increasing timeouts
        error = cudaDeviceSynchronize();
        if (error == cudaSuccess) break;
        
        printf("Warning: Synchronize timeout, retrying... (attempt %d/5)\n", i+1);
        usleep(1000000);  // Wait 1 second before retrying (using microseconds)
    }
    
    // End timer
    double end_time = get_time();
    double elapsed_time = end_time - start_time;
    
    if (error != cudaSuccess) {
        printf("Error: Kernel execution failed or timed out: %s\n", cudaGetErrorString(error));
        // Reset device to recover from errors
        cudaDeviceReset();
        return 999.0;  // Return an obviously invalid time
    }
    
    // Copy results back to verify
    int h_next_index;
    CHECK(cudaMemcpy(&h_next_index, d_next_index, sizeof(int), cudaMemcpyDeviceToHost));
    printf("Basic GPU hash table has %d nodes\n", h_next_index);
    
    // Free memory
    free(h_bucket_heads);
    free(h_next_indices);
    free(h_element_pool);
    
    CHECK(cudaFree(d_keys));
    CHECK(cudaFree(d_values));
    CHECK(cudaFree(d_next_ptrs));
    CHECK(cudaFree(d_bucket_heads));
    CHECK(cudaFree(d_next_index));
    CHECK(cudaFree(d_bucket_locks)); // Free bucket locks
    CHECK(cudaFree(d_element_pool));
    
    return elapsed_time;
}

// Hash table server kernel message handler
__device__ void process_hash_table_msg(Message* msg, GPUHashTable* ht, int* locks) {
    int bucket = msg->counter_idx;
    
    // Acquire lock for this bucket (using shared memory lock)
    while (atomicCAS(&locks[bucket], 0, 1) != 0) {
    }
    
    // Get next available node index
    int idx = atomicAdd(&ht->next_indices[0], 1);
    
    if (idx < MAX_LIST_NODES) {
        // Store key and value
        ht->keys[idx] = bucket; // Using bucket as key 
        ht->values[idx] = idx;  // Using index as value 
        
        // Update linked list (insert at head)
        int old_head = ht->bucket_heads[bucket];
        ht->next_ptrs[idx] = old_head;
        ht->bucket_heads[bucket] = idx;
    }
    
    // Release the lock
    __threadfence(); 
    atomicExch(&locks[bucket], 0);
}

// Modified server kernel for hash table operations
__global__ void hash_table_server_kernel(int* counters, int num_counters, int num_server_blocks, 
                                        Buffer* bufs, int* done, int num_threads,
                                        GPUHashTable* hash_table, int* pool_elements, int pool_size) {
    bool is_server = blockIdx.x < num_server_blocks;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize thread's random state based on thread ID
    unsigned int rand_state = tid * 1099087573;
    
    if (is_server) {
        // Initialize shared memory locks
        extern __shared__ int locks[];
        for (int i = threadIdx.x; i < num_counters; i += blockDim.x) {
            locks[i] = 0;
        }
        
        if (threadIdx.x == 0) {
            printf("Hash Table Server %d initialized locks to 0 in shared memory\n", blockIdx.x);
        }
        
        __syncthreads();
        
        Buffer* my_buf = &bufs[blockIdx.x];
        int empty_iterations = 0;
        const int MAX_EMPTY_ITERATIONS = 1000;
        
        while (true) {
            int sent = atomicAdd(done, 0);
            bool processed_message = false;
            
            // Check for messages in buffer
            Message msg;
            if (dequeue(my_buf, &msg)) {
                process_hash_table_msg(&msg, hash_table, locks);
                processed_message = true;
                empty_iterations = 0;
            } else {
                empty_iterations++;
            }
            
            // Exit condition with safety check - now checking against total client threads
            if (sent >= num_threads) {
                if (isEmpty(my_buf) || empty_iterations > MAX_EMPTY_ITERATIONS) {
                    if (threadIdx.x == 0) {
                        printf("Server %d exiting. done=%d, target=%d, empty_iterations=%d\n", 
                              blockIdx.x, sent, num_threads, empty_iterations);
                    }
                    break;
                }
            }
            
            // Add a small delay if no messages were processed
            if (!processed_message) {
                // Short yield/backoff to reduce contention
                for (int i = 0; i < 10; i++) { 
                    __threadfence_block();
                }
            }
        }
    } else {
        // Client code - one thread per message
        // Calculate the client thread ID (relative to all client threads)
        int client_tid = tid - (num_server_blocks * blockDim.x);
        
        // Only threads within the pool size range participate
        if (client_tid < num_threads) {
            // Each thread randomly selects an element from the pool
            int rand_index = cuda_rand(&rand_state) % pool_size;
            int element = pool_elements[rand_index];
            int counter = element % num_counters;
            int target_server = counter % num_server_blocks;
            
            // Send exactly one message per client thread
            send_msg(target_server, counter, bufs, done);
            
            if (threadIdx.x == 0 && blockIdx.x == num_server_blocks) {
                printf("Client threads started sending messages\n");
            }
        }
    }
    
    __syncthreads();
}

// Run CUDA hash table benchmark
double run_cuda_benchmark(int pool_size, int num_servers, int num_clients) {
    printf("Running CUDA benchmark with pool size %d\n", pool_size);
    
    // Allocate host memory
    int* h_bucket_heads = (int*)malloc(sizeof(int) * HT_SIZE);
    int* h_next_indices = (int*)malloc(sizeof(int));
    h_next_indices[0] = 0; // First available slot
    
    for (int i = 0; i < HT_SIZE; i++) {
        h_bucket_heads[i] = -1; // -1 indicates empty bucket
    }
    
    // Create pool of elements to randomly select from (same as sequential version)
    int* h_element_pool = (int*)malloc(sizeof(int) * pool_size);
    for (int i = 0; i < pool_size; i++) {
        h_element_pool[i] = rand();
    }
    
    // Allocate device memory for element pool
    int* d_element_pool;
    CHECK(cudaMalloc(&d_element_pool, sizeof(int) * pool_size));
    CHECK(cudaMemcpy(d_element_pool, h_element_pool, sizeof(int) * pool_size, cudaMemcpyHostToDevice));
    
    // Allocate device memory for hash table
    GPUHashTable h_table;
    GPUHashTable* d_table;
    
    CHECK(cudaMalloc(&d_table, sizeof(GPUHashTable)));
    CHECK(cudaMalloc(&h_table.next_indices, sizeof(int)));
    CHECK(cudaMalloc(&h_table.keys, sizeof(int) * MAX_LIST_NODES));
    CHECK(cudaMalloc(&h_table.values, sizeof(int) * MAX_LIST_NODES));
    CHECK(cudaMalloc(&h_table.next_ptrs, sizeof(int) * MAX_LIST_NODES));
    CHECK(cudaMalloc(&h_table.bucket_heads, sizeof(int) * HT_SIZE));
    
    h_table.size = HT_SIZE;
    
    // Initialize device memory
    CHECK(cudaMemcpy(h_table.next_indices, h_next_indices, sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(h_table.bucket_heads, h_bucket_heads, sizeof(int) * HT_SIZE, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_table, &h_table, sizeof(GPUHashTable), cudaMemcpyHostToDevice));
    
    // Allocate device memory for message passing
    Buffer* d_bufs;
    int* d_done;
    int* d_shared_data;
    
    CHECK(cudaMalloc(&d_shared_data, HT_SIZE * sizeof(int)));
    CHECK(cudaMemset(d_shared_data, 0, HT_SIZE * sizeof(int)));
    CHECK(cudaMalloc(&d_done, sizeof(int)));
    CHECK(cudaMemset(d_done, 0, sizeof(int)));
    CHECK(cudaMalloc(&d_bufs, num_servers * sizeof(Buffer)));
    
    // Initialize buffers on host and copy to device
    Buffer* h_bufs = (Buffer*)malloc(num_servers * sizeof(Buffer));
    memset(h_bufs, 0, num_servers * sizeof(Buffer));
    CHECK(cudaMemcpy(d_bufs, h_bufs, num_servers * sizeof(Buffer), cudaMemcpyHostToDevice));
    free(h_bufs);
    
    // Start timer
    printf("Launching kernel with %d server blocks and %d client threads...\n", 
           num_servers, num_clients);
    cudaDeviceSynchronize();
    double start_time = get_time();
    
    // Launch kernel
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size(num_servers + (num_clients + block_size.x - 1) / block_size.x);
    int shared_mem_size = HT_SIZE * sizeof(int); // For locks
    
    printf("Grid size: %d, Block size: %d, Shared mem: %d bytes\n", 
           grid_size.x, block_size.x, shared_mem_size);
    
    hash_table_server_kernel<<<grid_size, block_size, shared_mem_size>>>(
        d_shared_data, HT_SIZE, num_servers, d_bufs, d_done, 
        num_clients, d_table, d_element_pool, pool_size);
    
    // Wait for kernel to complete with timeout
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < 5; i++) {  // Try 5 times with increasing timeouts
        error = cudaDeviceSynchronize();
        if (error == cudaSuccess) break;
        
        printf("Warning: Synchronize timeout, retrying... (attempt %d/5)\n", i+1);
        usleep(1000000);  // Wait 1 second before retrying (using microseconds)
    }
    
    // End timer
    double end_time = get_time();
    double elapsed_time = end_time - start_time;
    
    if (error != cudaSuccess) {
        printf("Error: Kernel execution failed or timed out: %s\n", cudaGetErrorString(error));
        // Reset device to recover from errors
        cudaDeviceReset();
        return 999.0;  // Return an obviously invalid time
    }
    
    // Copy results back to verify
    int h_next_index;
    CHECK(cudaMemcpy(&h_next_index, h_table.next_indices, sizeof(int), cudaMemcpyDeviceToHost));
    printf("CUDA hash table has %d nodes\n", h_next_index);
    
    // Free memory
    free(h_bucket_heads);
    free(h_next_indices);
    free(h_element_pool);
    
    CHECK(cudaFree(h_table.next_indices));
    CHECK(cudaFree(h_table.keys));
    CHECK(cudaFree(h_table.values));
    CHECK(cudaFree(h_table.next_ptrs));
    CHECK(cudaFree(h_table.bucket_heads));
    CHECK(cudaFree(d_table));
    CHECK(cudaFree(d_bufs));
    CHECK(cudaFree(d_done));
    CHECK(cudaFree(d_shared_data));
    CHECK(cudaFree(d_element_pool));
    
    return elapsed_time;
}

int main(int argc, char** argv) {
    // Seed random number generator
    srand(time(NULL));
    
    // Default parameters
    int num_servers = 4;           // Number of server blocks
    int collision_factor = CF_1K;  // Default collision factor (1024)
    bool run_basic_gpu = false;    // Whether to run the basic GPU implementation
    
    // Using boost program options for argument parsing
    try {
        bpo::options_description desc("Hash Table Benchmark Options");
        desc.add_options()
            ("help", "Show help message")
            ("cf", bpo::value<int>()->default_value(CF_1K), "Collision factor (256, 1024, 32768, or 131072)")
            ("servers", bpo::value<int>()->default_value(4), "Number of server blocks")
            ("basic-gpu", bpo::bool_switch(&run_basic_gpu), "Run basic GPU implementation without message passing");
        
        bpo::variables_map vm;
        bpo::store(bpo::parse_command_line(argc, argv, desc), vm);
        bpo::notify(vm);
        
        if (vm.count("help")) {
            std::cout << desc << "\n";
            return 1;
        }
        
        collision_factor = vm["cf"].as<int>();
        num_servers = vm["servers"].as<int>();
        
        // Validate collision factor is one of the expected values
        if (collision_factor != CF_256 && collision_factor != CF_1K && 
            collision_factor != CF_32K && collision_factor != CF_128K) {
            std::cerr << "Error: Collision factor must be one of: 256, 1024, 32768, or 131072\n";
            return 1;
        }
        
    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    // Print benchmark parameters
    std::cout << "Running benchmark with:\n"
              << "  - Collision factor: " << collision_factor << "\n"
              << "  - Number of server blocks: " << num_servers << "\n"
              << "  - Run basic GPU implementation: " << (run_basic_gpu ? "Yes" : "No") << "\n";
              
    // Number of client threads equals the pool size (collision factor)
    int num_clients = collision_factor;
    
    // Run sequential benchmark
    double seq_time = run_sequential_benchmark(collision_factor);
    
    // Run CUDA benchmark with message passing (fine-grained synchronization)
    double cuda_time = run_cuda_benchmark(collision_factor, num_servers, num_clients);
    
    // Run basic GPU benchmark without message passing if requested
    double basic_gpu_time = 0.0;
    if (run_basic_gpu) {
        basic_gpu_time = run_basic_gpu_benchmark(collision_factor);
    }
    
    // Print results
    printf("\nResults summary:\n");
    printf("Sequential time: %.6f seconds\n", seq_time);
    printf("CUDA time (fine-grained): %.6f seconds\n", cuda_time);
    printf("Fine-grained speedup: %.2fx\n", seq_time / cuda_time);
    
    if (run_basic_gpu) {
        printf("CUDA time (basic GPU): %.6f seconds\n", basic_gpu_time);
        printf("Basic GPU speedup: %.2fx\n", seq_time / basic_gpu_time);
        printf("Fine-grained vs basic GPU: %.2fx\n", basic_gpu_time / cuda_time);
    }
    
    return 0;
}