#include "em.h"

// --- UTILITY FUNCTIONS ---

__global__ void calculateNkKernel(float* d_responsibilities, float* d_Nk, 
                                int numData, int numComponents) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (k < numComponents) {
        float sum = 0.0f;
        for (int i = 0; i < numData; i++) {
            sum += d_responsibilities[i * numComponents + k];
        }
        d_Nk[k] = sum;
    }
}

// --- BASIC GPU IMPLEMENTATION KERNELS ---

__global__ void eStepKernel(
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
            
            // log PDF and add log weight
            logT[k] = logf(d_weights[k]) + 
                      logDeviceMultivariatePDF(dataPoint, mean, covariance, numDimensions);
            
            // track maximum log value - numerical stability
            maxLog = fmaxf(maxLog, logT[k]);
        }
        
        // log-sum-exp denominator
        float sumExp = 0.0f;
        for (int k = 0; k < numComponents; k++) {
            sumExp += expf(logT[k] - maxLog);
        }
        float logDenom = maxLog + logf(sumExp);
        
        // fill responsibilities
        for (int k = 0; k < numComponents; k++) {
            d_responsibilities[dataIdx * numComponents + k] = expf(logT[k] - logDenom);
        }
    }
}

__global__ void updateWeightsKernel(
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
    int numData,
    int numComponents,
    int numDimensions
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int compIdx = idx / numDimensions;
    int dimIdx = idx % numDimensions;
    
    if (compIdx < numComponents && dimIdx < numDimensions) {
        float sum = 0.0f;
        
        for (int i = 0; i < numData; i++) {
            sum += d_responsibilities[i * numComponents + compIdx] * d_data[i * numDimensions + dimIdx];
        }
        
        if (d_Nk[compIdx] > 0.0f) {
            d_means[compIdx * numDimensions + dimIdx] = sum / d_Nk[compIdx];
        }
    }
}

__global__ void updateCovariancesKernel(
    float* d_data,
    float* d_responsibilities,
    float* d_means,
    float* d_covariances, 
    float* d_Nk,
    int numData,
    int numComponents,
    int numDimensions
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int compIdx = idx / (numDimensions * numDimensions);
    int dimIdx1 = (idx / numDimensions) % numDimensions;
    int dimIdx2 = idx % numDimensions;
    
    if (compIdx < numComponents && dimIdx1 < numDimensions && dimIdx2 < numDimensions) {
        float sum = 0.0f;
        
        for (int i = 0; i < numData; i++) {
            float resp = d_responsibilities[i * numComponents + compIdx];
            float diff1 = d_data[i * numDimensions + dimIdx1] - d_means[compIdx * numDimensions + dimIdx1];
            float diff2 = d_data[i * numDimensions + dimIdx2] - d_means[compIdx * numDimensions + dimIdx2];
            sum += resp * diff1 * diff2;
        }
        
        if (d_Nk[compIdx] > 0.0f) {
            d_covariances[compIdx * numDimensions * numDimensions + dimIdx1 * numDimensions + dimIdx2] = 
                sum / d_Nk[compIdx];
        }
        
        // add regularization to diagonal elements
        if (dimIdx1 == dimIdx2) {
            d_covariances[compIdx * numDimensions * numDimensions + dimIdx1 * numDimensions + dimIdx2] += 1e-6f;
        }
    }
}

__global__ void logLikelihoodKernel(
    float* d_data, 
    float* d_weights, 
    float* d_means, 
    float* d_covariances,
    float* d_logLikelihood,
    int numData,
    int numComponents,
    int numDimensions
) {
    int dataIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (dataIdx < numData) {
        float* dataPoint = &d_data[dataIdx * numDimensions];
        
        // calculate logWeights + logPDFs
        float logT[MAX_COMPONENTS];
        float maxLog = -INFINITY;
        
        for (int k = 0; k < numComponents; k++) {
            float* mean = &d_means[k * numDimensions];
            float* covariance = &d_covariances[k * numDimensions * numDimensions];
            
            logT[k] = logf(d_weights[k]) + 
                      logDeviceMultivariatePDF(dataPoint, mean, covariance, numDimensions);
            
            maxLog = fmaxf(maxLog, logT[k]);
        }
        
        // log-sum-exp calculation
        float sumExp = 0.0f;
        for (int k = 0; k < numComponents; k++) {
            sumExp += expf(logT[k] - maxLog);
        }
        float logSum = maxLog + logf(sumExp);
        
        // add to log-likelihood
        atomicAdd(d_logLikelihood, logSum);
    }
}

EMModel runBasicGPU(const InputData& inputData, const ArgOpts& opts) {
    EMModel model;
    model.numComponents = inputData.trueComponents.size();
    model.numDimensions = inputData.points.at(0).size();

    const int numData = inputData.points.size();

    // initialize model on CPU
    initializeModel(model, inputData.points, opts.seed);

    // calculate initial log-likelihood on CPU before GPU iterations
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

    CUDA_CHECK(cudaMalloc(&d_data, numData * model.numDimensions * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights, model.numComponents * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_means, model.numComponents * model.numDimensions * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_covariances, model.numComponents * model.numDimensions * model.numDimensions * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_responsibilities, numData * model.numComponents * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Nk, model.numComponents * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_logLikelihood, sizeof(float)));

    // copy data to device
    CUDA_CHECK(cudaMemcpy(d_data, flattenedData.data(), numData * model.numDimensions * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, weights.data(), model.numComponents * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_means, means.data(), model.numComponents * model.numDimensions * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_covariances, covariances.data(), model.numComponents * model.numDimensions * model.numDimensions * sizeof(float), cudaMemcpyHostToDevice));

    model.iterations = 0;
    bool converged = false;

    // calculate optimal thread block sizes for each kernel
    int minGridSize, blockSizeEStep, blockSizeWeights, blockSizeNk, blockSizeMeans, blockSizeCov, blockSizeLL;

    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSizeEStep,
        eStepKernel, 0, 0));
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSizeWeights,
        updateWeightsKernel, 0, 0));
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
        eStepKernel<<<numBlocksData, blockSizeEStep>>>(
            d_data, d_weights, d_means, d_covariances, d_responsibilities,
            numData, model.numComponents, model.numDimensions
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        // M-step
        // Calculate Nk (sum of responsibilities for each component)
        CUDA_CHECK(cudaMemset(d_Nk, 0, model.numComponents * sizeof(float)));
        calculateNkKernel<<<numBlocksNk, blockSizeNk>>>(
            d_responsibilities, d_Nk, numData, model.numComponents
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        // Update weights
        updateWeightsKernel<<<numBlocksComponents, blockSizeWeights>>>(
            d_responsibilities, d_weights, numData, model.numComponents
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        // Update means
        updateMeansKernel<<<numBlocksMeans, blockSizeMeans>>>(
            d_data, d_responsibilities, d_means, d_Nk,
            numData, model.numComponents, model.numDimensions
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        // Update covariances
        updateCovariancesKernel<<<numBlocksCovariances, blockSizeCov>>>(
            d_data, d_responsibilities, d_means, d_covariances, d_Nk,
            numData, model.numComponents, model.numDimensions
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        // Compute log-likelihood
        CUDA_CHECK(cudaMemset(d_logLikelihood, 0, sizeof(float)));
        logLikelihoodKernel<<<numBlocksLL, blockSizeLL>>>(
            d_data, d_weights, d_means, d_covariances, d_logLikelihood,
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

    if (converged) {
        std::cout << (converged ? "EM converged" : "Reached max iterations")
            << " after " << model.iterations << " iterations." << std::endl;
    } else {
        std::cout << "EM reached maximum iterations (" << opts.maxIterations << ")." << std::endl;
    }

    return model;
}