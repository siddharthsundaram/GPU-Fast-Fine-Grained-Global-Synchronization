#include "em.h"

// --- DEVICE FUNCTIONS FOR BUFFER ---

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
    __threadfence_system();
    // free the bitmask
    atomicExch(&b->bitmask[current], 0);
    return true;
}

// --- DEVICE FUNCTIONS FOR FINE-GRAINED IMPLEMENTATION ---

__device__ int mapToServerId(int dataId, int numServerTBs) {
    return dataId % numServerTBs;
}

__device__ void lock_local(int* locks, int lockIdx) {
    while (atomicCAS(&locks[lockIdx], 0, 1) != 0);
}

__device__ void unlock_local(int* locks, int lockIdx) {
    __threadfence_block();
    atomicExch(&locks[lockIdx], 0);
}

// --- FINE-GRAIN IMPLEMENTATION KERNELS ---

__global__ void eStepClientKernel(
    float* d_data,
    float* d_weights,
    float* d_means,
    float* d_covariances,
    float* d_responsibilities,
    Buffer* d_NkBuffers,
    Buffer* d_LLBuffers,
    int numData,
    int numComponents,
    int numDimensions,
    int numServerTBs,
    int* d_messagesProduced
) {
    int dataIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (dataIdx < numData) {
        float* dataPoint = &d_data[dataIdx * numDimensions];
        
        // compute log-weights + log-pdfs 
        float logT[MAX_COMPONENTS];
        float maxLog = -INFINITY;
        
        for (int k = 0; k < numComponents; k++) {
            float* mean = &d_means[k * numDimensions];
            float* covariance = &d_covariances[k * numDimensions * numDimensions];
            
            // calculate log PDF and add log weight
            logT[k] = logf(d_weights[k]) + 
                      logDeviceMultivariatePDF(dataPoint, mean, covariance, numDimensions);
            
            // track maximum log value for numerical stability
            maxLog = fmaxf(maxLog, logT[k]);
        }
        
        // log-sum-exp denominator
        float sumExp = 0.0f;
        for (int k = 0; k < numComponents; k++) {
            sumExp += expf(logT[k] - maxLog);
        }
        float logDenom = maxLog + logf(sumExp);
        
        Message llMsg;
        llMsg.operation = OP_UPDATE_LL;
        llMsg.component = 0;  // not used for LL
        llMsg.dim1 = 0;       // not used for LL
        llMsg.dim2 = 0;       // not used for LL
        llMsg.value = logDenom;
        
        int serverId = mapToServerId(0, numServerTBs);  // LL has a single accumulator
        bool enqueued = false;
        int backoff = 1;  // initial backoff value
        
        // try to enqueue with exponential backoff
        while (!enqueued) {
            enqueued = enqueue(&d_LLBuffers[serverId], llMsg);
            
            if (!enqueued) {
                // exponential backoff to reduce contention
                backoff = min(backoff * 2, 1024);
                int jitter = (threadIdx.x % 32) * 2;
                for (int i = 0; i < backoff + jitter; i++) {
                    __threadfence();
                }
            }
        }
        // Increment message counter
        atomicAdd(d_messagesProduced, 1);
        
        // fill responsibilities and send Nk updates to servers
        for (int k = 0; k < numComponents; k++) {
            float resp = expf(logT[k] - logDenom);
            d_responsibilities[dataIdx * numComponents + k] = resp;
            
            // send Nk update to the appropriate server
            Message nkMsg;
            nkMsg.operation = OP_UPDATE_NK;
            nkMsg.component = k;
            nkMsg.dim1 = -1;  // not used for Nk
            nkMsg.dim2 = -1;  // not used for Nk
            nkMsg.value = resp;
            
            serverId = mapToServerId(k, numServerTBs);
            enqueued = false;
            backoff = 1;
            
            while (!enqueued) {
                enqueued = enqueue(&d_NkBuffers[serverId], nkMsg);
                
                if (!enqueued) {
                    backoff = min(backoff * 2, 1024);
                    int jitter = (threadIdx.x % 32) * 2;
                    for (int i = 0; i < backoff + jitter; i++) {
                        __threadfence();
                    }
                }
            }
            // Increment message counter
            atomicAdd(d_messagesProduced, 1);
        }
    }
}

__global__ void updateMeansClientKernel(
    float* d_data,
    float* d_responsibilities,
    Buffer* d_meanBuffers,
    int numData,
    int numComponents,
    int numDimensions,
    int numServerTBs,
    int* d_messagesProduced
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int compIdx = idx / numDimensions;
    int dimIdx = idx % numDimensions;
    
    if (compIdx < numComponents && dimIdx < numDimensions) {
        extern __shared__ float s_sums[];
        int tid = threadIdx.x;
        s_sums[tid] = 0.0f;
        
        // each thread accumulates for its (component, dimension)
        for (int i = 0; i < numData; i++) {
            s_sums[tid] += d_responsibilities[i * numComponents + compIdx] * 
                         d_data[i * numDimensions + dimIdx];
        }
        
        __syncthreads();
        
        //  send message to server
        Message meanMsg;
        meanMsg.operation = OP_UPDATE_MEAN;
        meanMsg.component = compIdx;
        meanMsg.dim1 = dimIdx;
        meanMsg.dim2 = -1;  // Not used for means
        meanMsg.value = s_sums[tid];
        
        int serverId = mapToServerId(compIdx * numDimensions + dimIdx, numServerTBs);
        bool enqueued = false;
        int backoff = 1;
        
        while (!enqueued) {
            enqueued = enqueue(&d_meanBuffers[serverId], meanMsg);
            
            if (!enqueued) {
                backoff = min(backoff * 2, 1024);
                int jitter = (threadIdx.x % 32) * 2;
                for (int i = 0; i < backoff + jitter; i++) {
                    __threadfence();
                }
            }
        }
        // Increment message counter
        atomicAdd(d_messagesProduced, 1);
    }
}

__global__ void updateCovariancesClientKernel(
    float* d_data,
    float* d_responsibilities,
    float* d_means,
    Buffer* d_covBuffers,
    int numData,
    int numComponents,
    int numDimensions,
    int numServerTBs,
    int* d_messagesProduced
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int compIdx = idx / (numDimensions * numDimensions);
    int dimIdx1 = (idx / numDimensions) % numDimensions;
    int dimIdx2 = idx % numDimensions;
    
    if (compIdx < numComponents && dimIdx1 < numDimensions && dimIdx2 < numDimensions) {
        extern __shared__ float s_sums[];
        int tid = threadIdx.x;
        s_sums[tid] = 0.0f;
        
        // calculate covariance element
        for (int i = 0; i < numData; i++) {
            float resp = d_responsibilities[i * numComponents + compIdx];
            float diff1 = d_data[i * numDimensions + dimIdx1] - 
                        d_means[compIdx * numDimensions + dimIdx1];
            float diff2 = d_data[i * numDimensions + dimIdx2] - 
                        d_means[compIdx * numDimensions + dimIdx2];
            s_sums[tid] += resp * diff1 * diff2;
        }
        
        __syncthreads();
        
        // send message to server for aggregation
        Message covMsg;
        covMsg.operation = OP_UPDATE_COV;
        covMsg.component = compIdx;
        covMsg.dim1 = dimIdx1;
        covMsg.dim2 = dimIdx2;
        covMsg.value = s_sums[tid];
        
        int serverId = mapToServerId(compIdx * numDimensions * numDimensions + 
                                     dimIdx1 * numDimensions + dimIdx2, numServerTBs);
        bool enqueued = false;
        int backoff = 1;
        
        while (!enqueued) {
            enqueued = enqueue(&d_covBuffers[serverId], covMsg);
            
            if (!enqueued) {
                backoff = min(backoff * 2, 1024);
                int jitter = (threadIdx.x % 32) * 2;
                for (int i = 0; i < backoff + jitter; i++) {
                    __threadfence();
                }
            }
        }
        // Increment message counter
        atomicAdd(d_messagesProduced, 1);
    }
}

__global__ void updateWeightsClientKernel(
    float* d_responsibilities,
    Buffer* d_weightBuffers,
    int numData,
    int numComponents,
    int numServerTBs,
    int* d_messagesProduced
) {
    int compIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (compIdx < numComponents) {
        float sum = 0.0f;
        
        // calculate partial sum for this component's weight
        for (int i = 0; i < numData; i++) {
            sum += d_responsibilities[i * numComponents + compIdx];
        }
        
        // send message to server
        Message weightMsg;
        weightMsg.operation = OP_UPDATE_WEIGHT;
        weightMsg.component = compIdx;
        weightMsg.dim1 = -1;
        weightMsg.dim2 = -1;
        weightMsg.value = sum;
        
        int serverId = mapToServerId(compIdx, numServerTBs);
        bool enqueued = false;
        int backoff = 1;
        
        while (!enqueued) {
            enqueued = enqueue(&d_weightBuffers[serverId], weightMsg);
            
            if (!enqueued) {
                backoff = min(backoff * 2, 1024);
                int jitter = (threadIdx.x % 32) * 2;
                for (int i = 0; i < backoff + jitter; i++) {
                    __threadfence();
                }
            }
        }
        // Increment message counter
        atomicAdd(d_messagesProduced, 1);
    }
}

// Server kernel to handle all M-step updates
__global__ void serverKernel(
    float* d_weights,
    float* d_means,
    float* d_covariances,
    float* d_Nk,
    float* d_logLikelihood,
    Buffer* d_NkBuffers,
    Buffer* d_weightBuffers,
    Buffer* d_meanBuffers,
    Buffer* d_covBuffers,
    Buffer* d_LLBuffers,
    int* d_messagesProcessed,
    int* d_messagesProduced,
    int* d_serverActive,  // flag to signal if servers should continue running
    int numData,
    int numComponents,
    int numDimensions,
    int kernelType // 0 for E-step, 1 for M-step
) {
    const int serverId = blockIdx.x;

    // allocate locks in shared memory
    __shared__ int locks[MAX_LOCKS];
    
    // initialize locks to 0 
    for (int i = threadIdx.x; i < MAX_LOCKS; i += blockDim.x) {
        locks[i] = 0;
    }
    __syncthreads();
    
    // Keep track of inactive cycles
    int inactiveCycles = 0;
    const int MAX_INACTIVE_CYCLES = 1000;  // timeout to prevent infinite loops
    
    // process messages until signaled to stop or timeout
    while (atomicAdd(d_serverActive, 0) > 0) {
        bool processed = false;
        
        if (kernelType == 0) { // E-step server
            Message nkMsg;
            if (dequeue(&d_NkBuffers[serverId], &nkMsg)) {
                if (nkMsg.operation == OP_UPDATE_NK) {
                    int comp = nkMsg.component;
                    int lockIdx = comp % MAX_LOCKS;
                    
                    lock_local(locks, lockIdx);
                    d_Nk[comp] += nkMsg.value;
                    unlock_local(locks, lockIdx);
                    
                    atomicAdd(d_messagesProcessed, 1);
                    processed = true;
                    inactiveCycles = 0; 
                }
            }
            
            Message llMsg;
            if (dequeue(&d_LLBuffers[serverId], &llMsg)) {
                if (llMsg.operation == OP_UPDATE_LL) {
                    lock_local(locks, 0);  // use the first lock for LL
                    *d_logLikelihood += llMsg.value;
                    unlock_local(locks, 0);
                    
                    atomicAdd(d_messagesProcessed, 1);
                    processed = true;
                    inactiveCycles = 0;
                }
            }
        } else { // M-step server
            Message weightMsg;
            if (dequeue(&d_weightBuffers[serverId], &weightMsg)) {
                if (weightMsg.operation == OP_UPDATE_WEIGHT) {
                    int comp = weightMsg.component;
                    int lockIdx = comp % MAX_LOCKS;
                    
                    lock_local(locks, lockIdx);
                    d_weights[comp] = weightMsg.value / numData;
                    unlock_local(locks, lockIdx);
                    
                    atomicAdd(d_messagesProcessed, 1);
                    processed = true;
                    inactiveCycles = 0; 
                }
            }
            
            Message meanMsg;
            if (dequeue(&d_meanBuffers[serverId], &meanMsg)) {
                if (meanMsg.operation == OP_UPDATE_MEAN) {
                    int comp = meanMsg.component;
                    int dim = meanMsg.dim1;
                    int lockIdx = (comp * numDimensions + dim) % MAX_LOCKS;
                    
                    lock_local(locks, lockIdx);
                    int idx = comp * numDimensions + dim;
                    if (d_Nk[comp] > 0.0f) {
                        d_means[idx] = meanMsg.value / d_Nk[comp];
                    }
                    unlock_local(locks, lockIdx);
                    
                    atomicAdd(d_messagesProcessed, 1);
                    processed = true;
                    inactiveCycles = 0;
                }
            }
            
            Message covMsg;
            if (dequeue(&d_covBuffers[serverId], &covMsg)) {
                if (covMsg.operation == OP_UPDATE_COV) {
                    int comp = covMsg.component;
                    int dim1 = covMsg.dim1;
                    int dim2 = covMsg.dim2;
                    int lockIdx = (comp * numDimensions * numDimensions + dim1 * numDimensions + dim2) % MAX_LOCKS;
                    
                    lock_local(locks, lockIdx);
                    int idx = comp * numDimensions * numDimensions + dim1 * numDimensions + dim2;
                    if (d_Nk[comp] > 0.0f) {
                        float covValue = covMsg.value / d_Nk[comp];
                        
                        // add regularization to diagonal elements
                        if (dim1 == dim2) {
                            covValue += 1e-6f;
                        }
                        
                        d_covariances[idx] = covValue;
                    }
                    unlock_local(locks, lockIdx);
                    
                    atomicAdd(d_messagesProcessed, 1);
                    processed = true;
                    inactiveCycles = 0;
                }
            }
        }
        
        // ff thread 0 in block 0, check if we've processed all messages
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            if (atomicAdd(d_messagesProcessed, 0) >= atomicAdd(d_messagesProduced, 0)) {
                if (inactiveCycles > 100) {
                    // terminate after several inactive cycles
                    atomicExch(d_serverActive, 0);
                }
            }
        }
        
        // if no messages were processed, increment inactive cycle counter
        if (!processed) {
            inactiveCycles++;
            
            // delay to reduce contention
            volatile int delay = 0;
            for (int i = 0; i < 10; i++) {
                delay++;
            }
            
            // break if inactive for too long
            if (inactiveCycles > MAX_INACTIVE_CYCLES) {
                if (blockIdx.x == 0 && threadIdx.x == 0) {
                    if (atomicAdd(d_messagesProcessed, 0) >= atomicAdd(d_messagesProduced, 0)) {
                        // processed all messages -> terminate
                        atomicExch(d_serverActive, 0);
                    }
                }
                __syncthreads();
                
                if (atomicAdd(d_serverActive, 0) > 0 && 
                    blockIdx.x == 0 && threadIdx.x == 0) {
                    // force termination if we've been inactive too long
                    atomicExch(d_serverActive, 0);
                }
                break;
            }
        }
    }
}


// Main function to run the server-client approach
EMModel runServerClientGPU(const InputData& inputData, const ArgOpts& opts) {
    EMModel model;
    model.numComponents = inputData.trueComponents.size();
    model.numDimensions = inputData.points.at(0).size();

    const int numData = inputData.points.size();

    // initialize model on CPU
    initializeModel(model, inputData.points, opts.seed);

    // calculate initial log-likelihood on CPU
    model.logLikelihood = computeLogLikelihood(model, inputData.points);
    float prevLogLikelihood = model.logLikelihood;

    // flatten data for GPU
    std::vector<float> flattenedData;
    for (const auto& point : inputData.points) {
        flattenedData.insert(flattenedData.end(), point.begin(), point.end());
    }

    // flatten model parameters for GPU
    std::vector<float> weights(model.numComponents);
    std::vector<float> means(model.numComponents * model.numDimensions);
    std::vector<float> covariances(model.numComponents * model.numDimensions * model.numDimensions);

    for (int k = 0; k < model.numComponents; k++) {
        weights[k] = model.components[k].weight;

        for (int d = 0; d < model.numDimensions; d++) {
            means[k * model.numDimensions + d] = model.components[k].mean[d];

            for (int d2 = 0; d2 < model.numDimensions; d2++) {
                covariances[k * model.numDimensions * model.numDimensions + d * model.numDimensions + d2] =
                    model.components[k].covariance[d][d2];
            }
        }
    }

    int deviceId = 0;
    cudaError_t error = cudaSetDevice(deviceId);

    // get device properties
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, deviceId));

    // calculate optimal thread block
    int minGridSize, blockSizeEStep, blockSizeWeights, blockSizeMeans, blockSizeCov, blockSizeServer;

    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSizeEStep, 
        eStepClientKernel, 0, 0));
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSizeWeights, 
        updateWeightsClientKernel, 0, 0));
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSizeMeans, 
        updateMeansClientKernel, 0, 0));
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSizeCov, 
        updateCovariancesClientKernel, 0, 0));
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSizeServer, 
        serverKernel, 0, 0));

    // make sure block sizes are multiples of warp size for better efficiency
    blockSizeEStep = (blockSizeEStep / deviceProp.warpSize) * deviceProp.warpSize;
    blockSizeWeights = (blockSizeWeights / deviceProp.warpSize) * deviceProp.warpSize;
    blockSizeMeans = (blockSizeMeans / deviceProp.warpSize) * deviceProp.warpSize;
    blockSizeCov = (blockSizeCov / deviceProp.warpSize) * deviceProp.warpSize;
    blockSizeServer = (blockSizeServer / deviceProp.warpSize) * deviceProp.warpSize;

    // updated to ensure enough SMs for both clients and servers
    int numSMs = deviceProp.multiProcessorCount;
    float serverRatio = opts.workloadRatio;
    int numServerBlocks = std::max(1, 
                        std::min(numSMs - 1, 
                        static_cast<int>(std::ceil(serverRatio * numSMs))));
    int numClientBlocks = std::max(1, numSMs - numServerBlocks);

    // Log the calculated values
    std::cout << "Device: " << deviceProp.name << std::endl;
    std::cout << "Thread block sizes: E-step=" << blockSizeEStep 
            << ", Weights=" << blockSizeWeights
            << ", Means=" << blockSizeMeans
            << ", Cov=" << blockSizeCov
            << ", Server=" << blockSizeServer << std::endl;
    std::cout << "SMs: " << numSMs 
            << ", Server blocks: " << numServerBlocks 
            << ", Client SMs: " << numClientBlocks << std::endl;
    
    // allocate device memory for model parameters
    float *d_data, *d_weights, *d_means, *d_covariances, *d_responsibilities, *d_Nk, *d_logLikelihood;
    int *d_messagesProduced, *d_messagesProcessed;
    
    CUDA_CHECK(cudaMalloc(&d_data, numData * model.numDimensions * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights, model.numComponents * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_means, model.numComponents * model.numDimensions * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_covariances, model.numComponents * model.numDimensions * model.numDimensions * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_responsibilities, numData * model.numComponents * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Nk, model.numComponents * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_logLikelihood, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_messagesProduced, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_messagesProcessed, sizeof(int)));

    // allocate memory for message buffers
    Buffer *d_NkBuffers, *d_weightBuffers, *d_meanBuffers, *d_covBuffers, *d_LLBuffers;
    
    CUDA_CHECK(cudaMalloc(&d_NkBuffers, numServerBlocks * sizeof(Buffer)));
    CUDA_CHECK(cudaMalloc(&d_weightBuffers, numServerBlocks * sizeof(Buffer)));
    CUDA_CHECK(cudaMalloc(&d_meanBuffers, numServerBlocks * sizeof(Buffer)));
    CUDA_CHECK(cudaMalloc(&d_covBuffers, numServerBlocks * sizeof(Buffer)));
    CUDA_CHECK(cudaMalloc(&d_LLBuffers, numServerBlocks * sizeof(Buffer)));

    std::cout << "Server blocks: " << numServerBlocks << ", Client blocks: " << numClientBlocks << std::endl;

    // initialize all buffers on host
    std::vector<Buffer> h_buffers(numServerBlocks * 5);  // 5 buffer types
    
    for (int i = 0; i < numServerBlocks * 5; i++) {
        // zero the entire buffer
        memset(&h_buffers[i], 0, sizeof(Buffer));
        
        // explicitly set indices to ensure they match
        h_buffers[i].write_idx = 0;
        h_buffers[i].read_idx = 0;
    }
    
    // copy initialized buffers to device
    CUDA_CHECK(cudaMemcpy(d_NkBuffers, &h_buffers[0], numServerBlocks * sizeof(Buffer), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weightBuffers, &h_buffers[numServerBlocks], numServerBlocks * sizeof(Buffer), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_meanBuffers, &h_buffers[2*numServerBlocks], numServerBlocks * sizeof(Buffer), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_covBuffers, &h_buffers[3*numServerBlocks], numServerBlocks * sizeof(Buffer), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_LLBuffers, &h_buffers[4*numServerBlocks], numServerBlocks * sizeof(Buffer), cudaMemcpyHostToDevice));

    // copy data to device
    CUDA_CHECK(cudaMemcpy(d_data, flattenedData.data(), numData * model.numDimensions * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, weights.data(), model.numComponents * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_means, means.data(), model.numComponents * model.numDimensions * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_covariances, covariances.data(), model.numComponents * model.numDimensions * model.numDimensions * sizeof(float), cudaMemcpyHostToDevice));
    
    // calculate grid sizes for client kernels
    int numBlocksData = min((numData + blockSizeEStep - 1) / blockSizeEStep, numClientBlocks * 16);
    int numBlocksComponents = min((model.numComponents + blockSizeWeights - 1) / blockSizeWeights, numClientBlocks * 4);
    int numBlocksMeans = min((model.numComponents * model.numDimensions + blockSizeMeans - 1) / blockSizeMeans, numClientBlocks * 8);
    int numBlocksCovariances = min((model.numComponents * model.numDimensions * model.numDimensions + blockSizeCov - 1) / blockSizeCov, numClientBlocks * 16);

    
    // create CUDA streams for parallel execution
    // Create CUDA streams with appropriate priorities -- added as workaround for prior deadlock issue
    cudaStream_t serverStream, clientStream;
    int leastPrio, greatestPrio;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&leastPrio, &greatestPrio));

    // low priority for servers to allow clients to run first to ensure there are messages to process
    CUDA_CHECK(cudaStreamCreateWithPriority(&serverStream, cudaStreamNonBlocking, leastPrio));
    CUDA_CHECK(cudaStreamCreate(&clientStream));

    
    // create CUDA events for synchronization
    cudaEvent_t clientDone, serverDone;
    CUDA_CHECK(cudaEventCreate(&clientDone));
    CUDA_CHECK(cudaEventCreate(&serverDone));
    
    model.iterations = 0;
    bool converged = false;
    
    // start timing
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Main EM loop
    while (!converged && model.iterations < opts.maxIterations) {
        // reset Nk and log-likelihood for this iteration
        CUDA_CHECK(cudaMemset(d_Nk, 0, model.numComponents * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_logLikelihood, 0, sizeof(float)));
        
        // --- E-step Phase ---
        
        // reset message counters
        CUDA_CHECK(cudaMemset(d_messagesProduced, 0, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_messagesProcessed, 0, sizeof(int)));
        
        // allocate and set server active flag
        int* d_serverActive;
        CUDA_CHECK(cudaMalloc(&d_serverActive, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_serverActive, 1, sizeof(int)));  // 1 means active
        
        // launch E-step server kernel first to be ready to process messages ASAP
        // caused 1st iteration to not process messages 
        serverKernel<<<numServerBlocks, blockSizeServer, 0, serverStream>>>(
            d_weights, d_means, d_covariances, d_Nk, d_logLikelihood,
            d_NkBuffers, d_weightBuffers, d_meanBuffers, d_covBuffers, d_LLBuffers,
            d_messagesProcessed, d_messagesProduced, d_serverActive, 
            numData, model.numComponents, model.numDimensions, 0 // 0 for E-step
        );
        
        // launch E-step client kernel on client stream
        eStepClientKernel<<<numBlocksData, blockSizeEStep, 0, clientStream>>>(
            d_data, d_weights, d_means, d_covariances, d_responsibilities,
            d_NkBuffers, d_LLBuffers, numData, model.numComponents, model.numDimensions, 
            numServerBlocks, d_messagesProduced
        );
        
        // wait for clients to finish
        CUDA_CHECK(cudaEventRecord(clientDone, clientStream));
        CUDA_CHECK(cudaEventSynchronize(clientDone));
        
        // signal server active flag to stop
        CUDA_CHECK(cudaMemset(d_serverActive, 0, sizeof(int)));
        
        // ensure servers have finished
        CUDA_CHECK(cudaEventRecord(serverDone, serverStream));
        CUDA_CHECK(cudaEventSynchronize(serverDone));
        
        // get E-step statistics
        int messagesProducedEStep, messagesProcessedEStep;
        CUDA_CHECK(cudaMemcpy(&messagesProducedEStep, d_messagesProduced, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&messagesProcessedEStep, d_messagesProcessed, sizeof(int), cudaMemcpyDeviceToHost));
        
        // free resources
        CUDA_CHECK(cudaFree(d_serverActive));
        
        // --- M-step Phase ---
        
        CUDA_CHECK(cudaMemset(d_messagesProduced, 0, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_messagesProcessed, 0, sizeof(int)));
        
        CUDA_CHECK(cudaMalloc(&d_serverActive, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_serverActive, 1, sizeof(int)));
        
        serverKernel<<<numServerBlocks, blockSizeServer, 0, serverStream>>>(
            d_weights, d_means, d_covariances, d_Nk, d_logLikelihood,
            d_NkBuffers, d_weightBuffers, d_meanBuffers, d_covBuffers, d_LLBuffers,
            d_messagesProcessed, d_messagesProduced, d_serverActive,
            numData, model.numComponents, model.numDimensions, 1
        );
        
        updateWeightsClientKernel<<<numBlocksComponents, blockSizeWeights, 0, clientStream>>>(
            d_responsibilities, d_weightBuffers, numData, model.numComponents, 
            numServerBlocks, d_messagesProduced
        );
        
        updateMeansClientKernel<<<numBlocksMeans, blockSizeMeans, blockSizeMeans * sizeof(float), clientStream>>>(
            d_data, d_responsibilities, d_meanBuffers, numData, model.numComponents, 
            model.numDimensions, numServerBlocks, d_messagesProduced
        );
        
        updateCovariancesClientKernel<<<numBlocksCovariances, blockSizeCov, blockSizeCov * sizeof(float), clientStream>>>(
            d_data, d_responsibilities, d_means, d_covBuffers, numData, model.numComponents,
            model.numDimensions, numServerBlocks, d_messagesProduced
        );
        
        CUDA_CHECK(cudaEventRecord(clientDone, clientStream));
        CUDA_CHECK(cudaEventSynchronize(clientDone));
        
        CUDA_CHECK(cudaMemset(d_serverActive, 0, sizeof(int)));
        
        CUDA_CHECK(cudaEventRecord(serverDone, serverStream));
        CUDA_CHECK(cudaEventSynchronize(serverDone));
        
        int messagesProducedMStep, messagesProcessedMStep;
        CUDA_CHECK(cudaMemcpy(&messagesProducedMStep, d_messagesProduced, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&messagesProcessedMStep, d_messagesProcessed, sizeof(int), cudaMemcpyDeviceToHost));
        
        CUDA_CHECK(cudaFree(d_serverActive));
        
        // --- Convergence ---
        
        // copy log-likelihood to host
        float logLikelihood;
        CUDA_CHECK(cudaMemcpy(&logLikelihood, d_logLikelihood, sizeof(float), cudaMemcpyDeviceToHost));
        
        // check convergence
        converged = std::abs(logLikelihood - prevLogLikelihood) < 
                    opts.tolerance * std::abs(prevLogLikelihood);
        prevLogLikelihood = logLikelihood;
        model.logLikelihood = logLikelihood;
        
        model.iterations++;
        
        // print iteration information
        std::cout << "Iteration " << model.iterations
            << ", Log-likelihood: " << model.logLikelihood 
            << " (E-step msgs: " << messagesProducedEStep << "/" << messagesProcessedEStep
            << ", M-step msgs: " << messagesProducedMStep << "/" << messagesProcessedMStep << ")" << std::endl;
    }
    
    // end timing
    auto endTime = std::chrono::high_resolution_clock::now();
    model.timeElapsed = std::chrono::duration<double>(endTime - startTime).count();

    CUDA_CHECK(cudaEventDestroy(clientDone));
    CUDA_CHECK(cudaEventDestroy(serverDone));
    CUDA_CHECK(cudaStreamDestroy(serverStream));
    CUDA_CHECK(cudaStreamDestroy(clientStream));
    
    // copy results back to host
    CUDA_CHECK(cudaMemcpy(weights.data(), d_weights, model.numComponents * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(means.data(), d_means, model.numComponents * model.numDimensions * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(covariances.data(), d_covariances, model.numComponents * model.numDimensions * model.numDimensions * sizeof(float), cudaMemcpyDeviceToHost));
    
    // update model parameters
    for (int k = 0; k < model.numComponents; k++) {
        model.components[k].weight = weights[k];
        
        for (int d = 0; d < model.numDimensions; d++) {
            model.components[k].mean[d] = means[k * model.numDimensions + d];
            
            for (int d2 = 0; d2 < model.numDimensions; d2++) {
                model.components[k].covariance[d][d2] =
                    covariances[k * model.numDimensions * model.numDimensions + d * model.numDimensions + d2];
            }
        }
    }
    
    // free device memory
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_means));
    CUDA_CHECK(cudaFree(d_covariances));
    CUDA_CHECK(cudaFree(d_responsibilities));
    CUDA_CHECK(cudaFree(d_Nk));
    CUDA_CHECK(cudaFree(d_logLikelihood));
    CUDA_CHECK(cudaFree(d_messagesProduced));
    CUDA_CHECK(cudaFree(d_messagesProcessed));
    
    // free buffer memory
    CUDA_CHECK(cudaFree(d_NkBuffers));
    CUDA_CHECK(cudaFree(d_weightBuffers));
    CUDA_CHECK(cudaFree(d_meanBuffers));
    CUDA_CHECK(cudaFree(d_covBuffers));
    CUDA_CHECK(cudaFree(d_LLBuffers));
    
    if (converged) {
        std::cout << (converged ? "EM converged" : "Reached max iterations")
            << " after " << model.iterations << " iterations." << std::endl;
    } else {
        std::cout << "EM reached maximum iterations (" << opts.maxIterations << ")." << std::endl;
    }
    
    return model;
}