#include "em.h"

// --- DEVICE FUNCTIONS FOR LOCK IMPLEMENTATION ---

__device__ void lock(int* mutex) {
    while (atomicCAS(mutex, 0, 1) != 0);
}

__device__ void unlock(int* mutex) {
    atomicExch(mutex, 0);
}

// --- UTILITY FUNCTIONS ---

__global__ void calculateNkKernel(float* d_responsibilities, float* d_Nk, 
    int* d_compLocks, int numData, int numComponents) {
    
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (k < numComponents) {
        // shared memory for block-level reduction
        extern __shared__ float s_partialSums[];
        s_partialSums[threadIdx.x] = 0.0f;
        
        // each thread processes a subset of data points for its component
        for (int i = 0; i < numData; i++) {
            s_partialSums[threadIdx.x] += d_responsibilities[i * numComponents + k];
        }
        
        __syncthreads();
        
        // block reduction
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                s_partialSums[threadIdx.x] += s_partialSums[threadIdx.x + stride];
            }
            __syncthreads();
        }
        
        // single thread per block updates the component's Nk using lock
        if (threadIdx.x == 0) {
            lock(&d_compLocks[k]);
            d_Nk[k] += s_partialSums[0];
            unlock(&d_compLocks[k]);
        }
    }
}

// --- FINE GPU IMPLEMENTATION KERNELS ---

__global__ void eStepKernelFine(
    float* d_data, 
    float* d_weights, 
    float* d_means, 
    float* d_covariances, 
    float* d_responsibilities,
    int numData,
    int numComponents,
    int numDimensions
) {
    int dataIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (dataIdx < numData) {
        float* dataPoint = &d_data[dataIdx * numDimensions];
        
        float logT[MAX_COMPONENTS];
        float maxLog = -INFINITY;
        
        for (int k = 0; k < numComponents; k++) {
            float* mean = &d_means[k * numDimensions];
            float* covariance = &d_covariances[k * numDimensions * numDimensions];
            
            logT[k] = logf(d_weights[k]) + 
                      logDeviceMultivariatePDF(dataPoint, mean, covariance, numDimensions);
            
            maxLog = fmaxf(maxLog, logT[k]);
        }
        
        float sumExp = 0.0f;
        for (int k = 0; k < numComponents; k++) {
            sumExp += expf(logT[k] - maxLog);
        }
        float logDenom = maxLog + logf(sumExp);
        
        for (int k = 0; k < numComponents; k++) {
            d_responsibilities[dataIdx * numComponents + k] = expf(logT[k] - logDenom);
        }
    }
}

__global__ void updateWeightsKernelFine(
    float* d_responsibilities, 
    float* d_weights,
    int numData,
    int numComponents
) {
    int compIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (compIdx < numComponents) {
        float sum = 0.0f;
        
        for (int i = 0; i < numData; i++) {
            sum += d_responsibilities[i * numComponents + compIdx];
        }
        
        d_weights[compIdx] = sum / numData;
    }
}

__global__ void updateMeansKernel(
    float* d_data,
    float* d_responsibilities, 
    float* d_means,
    float* d_Nk,
    int* d_meanLocks,
    int numData,
    int numComponents,
    int numDimensions
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
        
        // each thread handles its own (component, dimension) pair
        int lockIndex = compIdx * numDimensions + dimIdx;
        if (d_Nk[compIdx] > 0.0f) {
            lock(&d_meanLocks[lockIndex]);
            d_means[compIdx * numDimensions + dimIdx] = s_sums[tid] / d_Nk[compIdx];
            unlock(&d_meanLocks[lockIndex]);
        }
    }
}

__global__ void updateCovariancesKernel(
    float* d_data,
    float* d_responsibilities,
    float* d_means,
    float* d_covariances,
    float* d_Nk,
    int* d_covLocks,
    int numData,
    int numComponents,
    int numDimensions
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
        
        // each thread handles its own (component, dim1, dim2) element
        int lockIndex = compIdx * numDimensions * numDimensions + dimIdx1 * numDimensions + dimIdx2;
        
        if (d_Nk[compIdx] > 0.0f) {
            lock(&d_covLocks[lockIndex]);
            float covValue = s_sums[tid] / d_Nk[compIdx];
            
            // add regularization to diagonal elements
            if (dimIdx1 == dimIdx2) {
                covValue += 1e-6f;
            }
            
            d_covariances[compIdx * numDimensions * numDimensions + 
                           dimIdx1 * numDimensions + dimIdx2] = covValue;
            unlock(&d_covLocks[lockIndex]);
        }
    }
}

__global__ void logLikelihoodKernel(
    float* d_data, 
    float* d_weights, 
    float* d_means, 
    float* d_covariances,
    float* d_logLikelihood,
    int* d_llLock,
    int numData,
    int numComponents,
    int numDimensions
) {
    int dataIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    extern __shared__ float s_partialSums[];
    
    if (threadIdx.x < blockDim.x) {
        s_partialSums[threadIdx.x] = 0.0f;
    }
    
    if (dataIdx < numData) {
        float* dataPoint = &d_data[dataIdx * numDimensions];
        
        float logT[MAX_COMPONENTS]; 
        float maxLog = -INFINITY;
        
        for (int k = 0; k < numComponents; k++) {
            float* mean = &d_means[k * numDimensions];
            float* covariance = &d_covariances[k * numDimensions * numDimensions];
            
            logT[k] = logf(d_weights[k]) + 
                      logDeviceMultivariatePDF(dataPoint, mean, covariance, numDimensions);
            
            maxLog = fmaxf(maxLog, logT[k]);
        }
        
        float sumExp = 0.0f;
        for (int k = 0; k < numComponents; k++) {
            sumExp += expf(logT[k] - maxLog);
        }
        float logSum = maxLog + logf(sumExp);

        // store in shared memory
        s_partialSums[threadIdx.x] = logSum;
    }
    
    __syncthreads();
    
    // parallel reduction within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_partialSums[threadIdx.x] += s_partialSums[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    // only first thread in each block updates the global sum using lock
    if (threadIdx.x == 0) {
        lock(d_llLock);
        *d_logLikelihood += s_partialSums[0];
        unlock(d_llLock);
    }
}


EMModel runFineGPU(const InputData& inputData, const ArgOpts& opts) {
    EMModel model;
    model.numComponents = inputData.trueComponents.size();
    model.numDimensions = inputData.points.at(0).size();

    const int numData = inputData.points.size();

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

    // allocate device memory
    float *d_data, *d_weights, *d_means, *d_covariances, *d_responsibilities, *d_Nk, *d_logLikelihood;
    
    // create locks for fine-grained synchronization
    int *d_compLocks; // Locks for each component (used in Nk calculation)
    int *d_meanLocks; // Locks for each mean element
    int *d_covLocks; // Locks for each covariance element
    int *d_llLock; // Lock for log-likelihood

    CUDA_CHECK(cudaMalloc(&d_data, numData * model.numDimensions * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights, model.numComponents * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_means, model.numComponents * model.numDimensions * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_covariances, model.numComponents * model.numDimensions * model.numDimensions * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_responsibilities, numData * model.numComponents * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Nk, model.numComponents * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_logLikelihood, sizeof(float)));
    
    // Allocate memory for locks
    CUDA_CHECK(cudaMalloc(&d_compLocks, model.numComponents * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_meanLocks, model.numComponents * model.numDimensions * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_covLocks, model.numComponents * model.numDimensions * model.numDimensions * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_llLock, sizeof(int)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_data, flattenedData.data(), numData * model.numDimensions * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, weights.data(), model.numComponents * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_means, means.data(), model.numComponents * model.numDimensions * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_covariances, covariances.data(), model.numComponents * model.numDimensions * model.numDimensions * sizeof(float), cudaMemcpyHostToDevice));

    // Initialize locks to 0
    CUDA_CHECK(cudaMemset(d_compLocks, 0, model.numComponents * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_meanLocks, 0, model.numComponents * model.numDimensions * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_covLocks, 0, model.numComponents * model.numDimensions * model.numDimensions * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_llLock, 0, sizeof(int)));

    model.iterations = 0;
    bool converged = false;

    // Calculate optimal thread block sizes for each kernel
    int minGridSize, blockSizeEStep, blockSizeWeights, blockSizeNk, blockSizeMeans, blockSizeCov, blockSizeLL;

    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSizeEStep,
        eStepKernelFine, 0, 0));
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSizeWeights,
        updateWeightsKernelFine, 0, 0));
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSizeNk,
        calculateNkKernel, 0, 0));
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSizeMeans,
        updateMeansKernel, 0, 0));
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSizeCov,
        updateCovariancesKernel, 0, 0));
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSizeLL,
        logLikelihoodKernel, 0, 0));

    // calculate grid sizes based on optimal block sizes
    int numBlocksData = (numData + blockSizeEStep - 1) / blockSizeEStep;
    int numBlocksComponents = (model.numComponents + blockSizeWeights - 1) / blockSizeWeights;
    int numBlocksNk = (model.numComponents + blockSizeNk - 1) / blockSizeNk;
    int numBlocksMeans = (model.numComponents * model.numDimensions + blockSizeMeans - 1) / blockSizeMeans;
    int numBlocksCovariances = (model.numComponents * model.numDimensions * model.numDimensions +
        blockSizeCov - 1) / blockSizeCov;
    int numBlocksLL = (numData + blockSizeLL - 1) / blockSizeLL;

    std::cout << "Using optimal block sizes: E-step=" << blockSizeEStep
        << ", Weights=" << blockSizeWeights
        << ", Nk=" << blockSizeNk
        << ", Means=" << blockSizeMeans
        << ", Covariances=" << blockSizeCov
        << ", Log-likelihood=" << blockSizeLL << std::endl;

    // start timing
    auto startTime = std::chrono::high_resolution_clock::now();

    // Main EM loop
    while (!converged && model.iterations < opts.maxIterations) {
        // E-step
        eStepKernelFine<<<numBlocksData, blockSizeEStep>>>(
            d_data, d_weights, d_means, d_covariances, d_responsibilities,
            numData, model.numComponents, model.numDimensions
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        // M-step
        CUDA_CHECK(cudaMemset(d_Nk, 0, model.numComponents * sizeof(float)));
        calculateNkKernel<<<numBlocksNk, blockSizeNk, blockSizeNk * sizeof(float)>>>(
            d_responsibilities, d_Nk, d_compLocks, numData, model.numComponents
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        // Update weights
        updateWeightsKernelFine<<<numBlocksComponents, blockSizeWeights>>>(
            d_responsibilities, d_weights, numData, model.numComponents
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        // Update means with fine-grained locking
        updateMeansKernel<<<numBlocksMeans, blockSizeMeans, blockSizeMeans * sizeof(float)>>>(
            d_data, d_responsibilities, d_means, d_Nk, d_meanLocks,
            numData, model.numComponents, model.numDimensions
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        // Update covariances with fine-grained locking
        updateCovariancesKernel<<<numBlocksCovariances, blockSizeCov, blockSizeCov * sizeof(float)>>>(
            d_data, d_responsibilities, d_means, d_covariances, d_Nk, d_covLocks,
            numData, model.numComponents, model.numDimensions
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        // Compute log-likelihood with fine-grained locking
        CUDA_CHECK(cudaMemset(d_logLikelihood, 0, sizeof(float)));
        logLikelihoodKernel<<<numBlocksLL, blockSizeLL, blockSizeLL * sizeof(float)>>>(
            d_data, d_weights, d_means, d_covariances, d_logLikelihood, d_llLock,
            numData, model.numComponents, model.numDimensions
        );
        CUDA_CHECK(cudaDeviceSynchronize());

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

    // End timing
    auto endTime = std::chrono::high_resolution_clock::now();
    model.timeElapsed = std::chrono::duration<double>(endTime - startTime).count();

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
    
    // free locks
    CUDA_CHECK(cudaFree(d_compLocks));
    CUDA_CHECK(cudaFree(d_meanLocks));
    CUDA_CHECK(cudaFree(d_covLocks));
    CUDA_CHECK(cudaFree(d_llLock));

    if (converged) {
        std::cout << (converged ? "EM converged" : "Reached max iterations")
            << " after " << model.iterations << " iterations." << std::endl;
    } else {
        std::cout << "EM reached maximum iterations (" << opts.maxIterations << ")." << std::endl;
    }

    return model;
}

