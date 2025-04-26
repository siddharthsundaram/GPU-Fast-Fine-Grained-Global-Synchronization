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

// Data-to-server mapping function
__device__ int mapToServerId(int dataId, int numServerTBs) {
    return dataId % numServerTBs;
}

// Shared memory lock operations using scratchpad memory
__device__ void lock_local(int* locks, int lockIdx) {
    while (atomicCAS(&locks[lockIdx], 0, 1) != 0);
}

__device__ void unlock_local(int* locks, int lockIdx) {
    atomicExch(&locks[lockIdx], 0);
}

// --- FINE-GRAIN IMPLEMENTATION KERNELS ---

// Client kernel for E-step
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
    int* d_backoffCounter
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
        
        // instead of directly updating the log-likelihood, send a message to the LL server
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
                backoff = min(backoff * 2, 1024);  // cap backoff at 1024 cycles
                for (int i = 0; i < backoff; i++) {
                    // simple busy-wait backoff
                    atomicAdd(d_backoffCounter, 0);
                }
            }
        }
        // assume server received message and continue processing
        
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
                    for (int i = 0; i < backoff; i++) {
                        atomicAdd(d_backoffCounter, 0);
                    }
                }
            }
        }
    }

}

// Client kernel for M-step means update
__global__ void updateMeansClientKernel(
    float* d_data,
    float* d_responsibilities,
    Buffer* d_meanBuffers,
    int numData,
    int numComponents,
    int numDimensions,
    int numServerTBs,
    int* d_backoffCounter
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
                for (int i = 0; i < backoff; i++) {
                    atomicAdd(d_backoffCounter, 0);
                }
            }
        }
    }
}

// Client kernel for M-step covariance update
__global__ void updateCovariancesClientKernel(
    float* d_data,
    float* d_responsibilities,
    float* d_means,
    Buffer* d_covBuffers,
    int numData,
    int numComponents,
    int numDimensions,
    int numServerTBs,
    int* d_backoffCounter
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
                for (int i = 0; i < backoff; i++) {
                    atomicAdd(d_backoffCounter, 0);
                }
            }
        }
    }
}

// Client kernel for M-step weight update
__global__ void updateWeightsClientKernel(
    float* d_responsibilities,
    Buffer* d_weightBuffers,
    int numData,
    int numComponents,
    int numServerTBs,
    int* d_backoffCounter
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
                for (int i = 0; i < backoff; i++) {
                    atomicAdd(d_backoffCounter, 0);
                }
            }
        }
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
    int* d_terminationFlag,
    int numData,
    int numComponents,
    int numDimensions,
    int serverId
) {
    // allocate locks in shared memory
    __shared__ int locks[MAX_LOCKS];
    
    // initialize locks to 0 
    for (int i = threadIdx.x; i < MAX_LOCKS; i += blockDim.x) {
        locks[i] = 0;
    }
    __syncthreads();
    
    
    __shared__ volatile bool terminate;

    if (threadIdx.x == 0) {
        terminate = false;
    }
    __syncthreads();

    while (!terminate) {
        bool processedMessage = false;
        
        // process Nk updates
        Message nkMsg;
        if (dequeue(&d_NkBuffers[serverId], &nkMsg)) {
            if (nkMsg.operation == OP_UPDATE_NK) {
                int comp = nkMsg.component;
                int lockIdx = comp % MAX_LOCKS;
                
                lock_local(locks, lockIdx);
                d_Nk[comp] += nkMsg.value;
                unlock_local(locks, lockIdx);
                
                processedMessage = true;
            }
        }
        
        // process weight updates
        Message weightMsg;
        if (dequeue(&d_weightBuffers[serverId], &weightMsg)) {
            if (weightMsg.operation == OP_UPDATE_WEIGHT) {
                int comp = weightMsg.component;
                int lockIdx = comp % MAX_LOCKS;
                
                lock_local(locks, lockIdx);
                d_weights[comp] = weightMsg.value / numData;
                unlock_local(locks, lockIdx);
                
                processedMessage = true;
            }
        }
        
        // process mean updates
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
                
                processedMessage = true;
            }
        }
        
        // process covariance updates
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
                
                processedMessage = true;
            }
        }
        
        // process log-likelihood updates
        Message llMsg;
        if (dequeue(&d_LLBuffers[serverId], &llMsg)) {
            if (llMsg.operation == OP_UPDATE_LL) {
                lock_local(locks, 0);  // use the first lock for LL
                *d_logLikelihood += llMsg.value;
                unlock_local(locks, 0);
                
                processedMessage = true;
            }
        }
        
        // check termination flag
        if (threadIdx.x == 0 && !processedMessage) {
            int flag = *d_terminationFlag;
            if (flag == 1) {
                terminate = true;
            }
        }
        __syncthreads();

        if (terminate) break;
        
        // if no messages were processed, add a small delay to reduce contention
        if (!processedMessage) {
            // simple delay using volatile to prevent compiler optimization
            volatile int delay = 0;
            for (int i = 0; i < 100; i++) {
                delay++;
            }
        }
        
    }
}

// --- HELPER FOR MEMORY CONSISTENCY ---

struct Idx { 
    int write_idx;
    int read_idx;
};

bool buffersEmpty(Buffer *d_buf, int n) {
    std::vector<Idx> tmp(n);

    // copy only the two indices of each buffer
    CUDA_CHECK(cudaMemcpy(tmp.data(),
                          (char*)d_buf + offsetof(Buffer, write_idx),
                          n * sizeof(Idx),
                          cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; ++i)
        if (tmp[i].write_idx != tmp[i].read_idx)
            return false;
    return true;
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

    // calculate maximum number of concurrent blocks based on device properties
    int maxBlocksPerSM = deviceProp.maxBlocksPerMultiProcessor;
    int numSMs = deviceProp.multiProcessorCount;
    int maxBlocks = maxBlocksPerSM * numSMs;

    // Log the calculated values
    std::cout << "Device: " << deviceProp.name << std::endl;
    std::cout << "Thread block sizes: E-step=" << blockSizeEStep 
            << ", Weights=" << blockSizeWeights
            << ", Means=" << blockSizeMeans
            << ", Cov=" << blockSizeCov
            << ", Server=" << blockSizeServer << std::endl;
    std::cout << "Maximum concurrent blocks: " << maxBlocks 
            << " (" << numSMs << " SMs * " << maxBlocksPerSM << " blocks/SM)" << std::endl;
    
    // allocate device memory for model parameters
    float *d_data, *d_weights, *d_means, *d_covariances, *d_responsibilities, *d_Nk, *d_logLikelihood;
    int *d_terminationFlag, *d_backoffCounter;
    
    CUDA_CHECK(cudaMalloc(&d_data, numData * model.numDimensions * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights, model.numComponents * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_means, model.numComponents * model.numDimensions * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_covariances, model.numComponents * model.numDimensions * model.numDimensions * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_responsibilities, numData * model.numComponents * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Nk, model.numComponents * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_logLikelihood, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_terminationFlag, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_backoffCounter, sizeof(int)));

    // allocate memory for message buffers
    Buffer *d_NkBuffers, *d_weightBuffers, *d_meanBuffers, *d_covBuffers, *d_LLBuffers;

    // Determine server/client block split based on workload ratio
    float serverRatio = opts.workloadRatio;
    int numServerBlocks = max(1, (int)(maxBlocks * serverRatio));
    int numClientBlocks = maxBlocks - numServerBlocks;
    
    CUDA_CHECK(cudaMalloc(&d_NkBuffers, numServerBlocks * sizeof(Buffer)));
    CUDA_CHECK(cudaMalloc(&d_weightBuffers, numServerBlocks * sizeof(Buffer)));
    CUDA_CHECK(cudaMalloc(&d_meanBuffers, numServerBlocks * sizeof(Buffer)));
    CUDA_CHECK(cudaMalloc(&d_covBuffers, numServerBlocks * sizeof(Buffer)));
    CUDA_CHECK(cudaMalloc(&d_LLBuffers, numServerBlocks * sizeof(Buffer)));

    std::cout << "Maximum concurrent blocks: " << maxBlocks << std::endl;
    std::cout << "Server blocks: " << numServerBlocks << ", Client blocks: " << numClientBlocks << std::endl;


    // initialize all buffers on host
    std::vector<Buffer> h_buffers(numServerBlocks * 5);  // 5 buffer types
    
    for (int i = 0; i < numServerBlocks * 5; i++) {
        memset(&h_buffers[i], 0, sizeof(Buffer));
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
    
    // initialize counters and flags
    CUDA_CHECK(cudaMemset(d_terminationFlag, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_backoffCounter, 0, sizeof(int)));

    // calculate grid sizes for client kernels
    int numBlocksData = min(numClientBlocks, (numData + blockSizeEStep - 1) / blockSizeEStep);
    int numBlocksComponents = min(numClientBlocks, (model.numComponents + blockSizeWeights - 1) / blockSizeWeights);
    int numBlocksMeans = min(numClientBlocks, (model.numComponents * model.numDimensions + blockSizeMeans - 1) / blockSizeMeans);
    int numBlocksCovariances = min(numClientBlocks, (model.numComponents * model.numDimensions * model.numDimensions +
                               blockSizeCov - 1) / blockSizeCov);
    
    // create CUDA streams for parallel execution
    cudaStream_t clientStream, serverStream;
    CUDA_CHECK(cudaStreamCreate(&clientStream));
    CUDA_CHECK(cudaStreamCreate(&serverStream));
    
    model.iterations = 0;
    bool converged = false;
    
    // start timing
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // launch server kernels - they will run in parallel with client kernels
    for (int i = 0; i < numServerBlocks; i++) {
        serverKernel<<<1, blockSizeServer, 0, serverStream>>>(
            d_weights, d_means, d_covariances, d_Nk, d_logLikelihood,
            d_NkBuffers, d_weightBuffers, d_meanBuffers, d_covBuffers, d_LLBuffers,
            d_terminationFlag, numData, model.numComponents, model.numDimensions, i
        );
    }
    
    // Main EM loop
    while (!converged && model.iterations < opts.maxIterations) {
        // reset Nk and log-likelihood for this iteration
        CUDA_CHECK(cudaMemset(d_Nk, 0, model.numComponents * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_logLikelihood, 0, sizeof(float)));
        
        cudaStreamSynchronize(clientStream);
        while (!buffersEmpty(d_NkBuffers, numServerBlocks) ||
                !buffersEmpty(d_weightBuffers, numServerBlocks) ||
                !buffersEmpty(d_meanBuffers, numServerBlocks) ||
                !buffersEmpty(d_covBuffers, numServerBlocks)) {
        }

        // E-step
        eStepClientKernel<<<numBlocksData, blockSizeEStep, 0, clientStream>>>(
            d_data, d_weights, d_means, d_covariances, d_responsibilities,
            d_NkBuffers, d_LLBuffers, numData, model.numComponents, model.numDimensions, 
            numServerBlocks, d_backoffCounter
        );
        // ensure that the clients have sent messages and Nk buffer is empty -> all processed
        CUDA_CHECK(cudaStreamSynchronize(clientStream));
        while (!buffersEmpty(d_NkBuffers, numServerBlocks)) {
        }
        
        // M-step (client kernels)
        updateWeightsClientKernel<<<numBlocksComponents, blockSizeWeights, 0, clientStream>>>(
            d_responsibilities, d_weightBuffers, numData, model.numComponents, 
            numServerBlocks, d_backoffCounter
        );
        
        updateMeansClientKernel<<<numBlocksMeans, blockSizeMeans, blockSizeMeans * sizeof(float), clientStream>>>(
            d_data, d_responsibilities, d_meanBuffers, numData, model.numComponents, 
            model.numDimensions, numServerBlocks, d_backoffCounter
        );
        
        updateCovariancesClientKernel<<<numBlocksCovariances, blockSizeCov, blockSizeCov * sizeof(float), clientStream>>>(
            d_data, d_responsibilities, d_means, d_covBuffers, numData, model.numComponents,
            model.numDimensions, numServerBlocks, d_backoffCounter
        );
        
        // need to synchronize with server stream to ensure all messages are processed here 
        // before computing the log-likelihood and checking convergence else correctness might
        // be detrimentally affected
        CUDA_CHECK(cudaStreamSynchronize(clientStream));
        while (!buffersEmpty(d_LLBuffers, numServerBlocks)) {
        }
        // Copy log-likelihood to host
        float logLikelihood;
        CUDA_CHECK(cudaMemcpy(&logLikelihood, d_logLikelihood, sizeof(float), cudaMemcpyDeviceToHost));
        
        // Check convergence
        converged = std::abs(logLikelihood - prevLogLikelihood) < 
                    opts.tolerance * std::abs(prevLogLikelihood);
        prevLogLikelihood = logLikelihood;
        model.logLikelihood = logLikelihood;
        
        model.iterations++;
        
        // Print iteration information
        std::cout << "Iteration " << model.iterations
            << ", Log-likelihood: " << model.logLikelihood << std::endl;
            
    }
    
    // Signal servers to terminate
    int terminationFlag = 1;
    CUDA_CHECK(cudaMemcpy(d_terminationFlag, &terminationFlag, sizeof(int), cudaMemcpyHostToDevice));
    
    // Wait for servers to terminate
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // End timing
    auto endTime = std::chrono::high_resolution_clock::now();
    model.timeElapsed = std::chrono::duration<double>(endTime - startTime).count();
    
    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(weights.data(), d_weights, model.numComponents * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(means.data(), d_means, model.numComponents * model.numDimensions * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(covariances.data(), d_covariances, model.numComponents * model.numDimensions * model.numDimensions * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Update model parameters
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
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_means));
    CUDA_CHECK(cudaFree(d_covariances));
    CUDA_CHECK(cudaFree(d_responsibilities));
    CUDA_CHECK(cudaFree(d_Nk));
    CUDA_CHECK(cudaFree(d_logLikelihood));
    CUDA_CHECK(cudaFree(d_terminationFlag));
    CUDA_CHECK(cudaFree(d_backoffCounter));
    
    // Free buffer memory
    CUDA_CHECK(cudaFree(d_NkBuffers));
    CUDA_CHECK(cudaFree(d_weightBuffers));
    CUDA_CHECK(cudaFree(d_meanBuffers));
    CUDA_CHECK(cudaFree(d_covBuffers));
    CUDA_CHECK(cudaFree(d_LLBuffers));
    
    // Destroy streams
    CUDA_CHECK(cudaStreamDestroy(clientStream));
    CUDA_CHECK(cudaStreamDestroy(serverStream));
    
    if (converged) {
        std::cout << (converged ? "EM converged" : "Reached max iterations")
            << " after " << model.iterations << " iterations." << std::endl;
    } else {
        std::cout << "EM reached maximum iterations (" << opts.maxIterations << ")." << std::endl;
    }
    
    return model;
}
