#ifndef EM_H
#define EM_H

#include <string>
#include <vector>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <unistd.h> 
#include <cmath>
#include <limits>
#include <ctime>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <chrono>
#include <map>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
// #include "../../synch/buffer.cuh"

// CUDA error checking macro
#define CUDA_CHECK(call)                                  \
    do {                                                  \
        cudaError_t err = call;                           \
        if (err != cudaSuccess) {                        \
            std::cerr << "CUDA error in " << __FILE__     \
                      << " at line " << __LINE__ << ": "  \
                      << cudaGetErrorString(err) << "\n"; \
            exit(EXIT_FAILURE);                           \
        }                                                 \
    } while (0)

// Constants
#define MAX_MSG_SIZE 1024
#define MAX_QUEUE_SIZE 10240

#define MAX_DIM 32
#define MAX_COMPONENTS 32

#define MAX_LOCKS 1024

#define BUF_CAP 1024

// Operation types for message passing
enum OperationType {
    OP_UPDATE_NK,            // Update Nk for a component
    OP_UPDATE_WEIGHT,        // Update weight for a component
    OP_UPDATE_MEAN,          // Update mean for a component and dimension
    OP_UPDATE_COV,           // Update covariance for a component and dimensions
    OP_UPDATE_LL,            // Update log-likelihood
    OP_TERMINATE             // Signal termination
};

// Message structure for client-server communication
struct Message {
    OperationType operation;       // Operation type from OperationType
    int component;       // Component index
    int dim1;            // First dimension index
    int dim2;            // Second dimension index (for covariance)
    float value;         // Value to update or contribution
};


// Structure for command line arguments
struct ArgOpts { 
    std::string inputFilePath; // path to the input file
    std::string outputFilePath; // path to the output file
    std::string mode; // mode of operation (sequential, basic_gpu, fine_grained)
    int maxIterations; // maximum number of iterations
    float tolerance; // tolerance for convergence
    int seed; // random seed
    float workloadRatio; // ratio between client and server blocks
};

// Structure for a Gaussian component
struct GaussianComponent {
    std::vector<float> mean; // Mean vector
    std::vector<std::vector<float>> covariance; // Covariance matrix
    float weight; // Mixture weight
};

// Structure for EM model
struct EMModel {
    std::vector<GaussianComponent> components;  // Gaussian components
    int numComponents;                          // Number of components
    int numDimensions;                          // Dimensionality of data
    float logLikelihood;                        // Current log-likelihood
    
    // Performance measurement
    double timeElapsed;                         // Time taken for execution
    int iterations;                             // Number of iterations performed
};

// Structure for input data with ground truth
struct InputData {
    std::vector<std::vector<float>> points;       // Data points
    std::vector<GaussianComponent> trueComponents; // True Gaussian components
};

// Command line parsing functions
ArgOpts parseCommandLine(int argc, char* argv[]);
int parseOptionToInt(const char* optarg, char option);
float parseOptionToFloat(const char* optarg, char option);

// Input/Output functions
InputData inputReader(const std::string& filePath);
void saveResults(const EMModel& model, const InputData& inputData, const std::string& filePath);

// Sequential EM implementation
EMModel runSequentialEM(const InputData& inputData, const ArgOpts& opts);
void initializeModel(EMModel& model, const std::vector<std::vector<float>>& data, int seed);
void eStep(EMModel& model, const std::vector<std::vector<float>>& data, std::vector<std::vector<float>>& responsibilities);
void mStep(EMModel& model, const std::vector<std::vector<float>>& data, const std::vector<std::vector<float>>& responsibilities);
float computeLogLikelihood(const EMModel& model, const std::vector<std::vector<float>>& data);
bool checkConvergence(const EMModel& model, float prevLogLikelihood, float tolerance);

// GPU implementation wrappers
EMModel runFineGPU(const InputData& inputData, const ArgOpts& opts);
EMModel runBasicGPU(const InputData& inputData, const ArgOpts& opts);
EMModel runServerClientGPU(const InputData& inputData, const ArgOpts& opts);

// Utility functions
float multivariatePDF(const std::vector<float>& x, const std::vector<float>& mean, 
                     const std::vector<std::vector<float>>& covariance);
float logMultivariatePDF(const std::vector<float>& x,
                        const std::vector<float>& mean,
                        const std::vector<std::vector<float>>& covariance);
float determinant(const std::vector<std::vector<float>>& matrix);
std::vector<std::vector<float>> inverse(const std::vector<std::vector<float>>& matrix);
float compareWithGroundTruth(const EMModel& model, const std::vector<GaussianComponent>& trueComponents);

// --- UTILITY FUNCTIONS FOR CUDA USED ACROSS ALL IMPLEMENTATIONS ---

__device__ inline bool deviceLUDecompose(const float* A, float* LU, int* pivots, int& pivSign, int n) {
    for (int i = 0; i < n*n; i++) {
        LU[i] = A[i];
    }
    
    for (int i = 0; i < n; i++) {
        pivots[i] = i;
    }
    pivSign = 1;
    
    for (int j = 0; j < n; j++) {
        // apply previous transformations
        for (int i = 0; i < n; i++) {
            int kmax = (i < j) ? i : j;
            float s = 0.0f;
            for (int k = 0; k < kmax; k++) {
                s += LU[i*n + k] * LU[k*n + j];
            }
            LU[i*n + j] -= s;
        }
        
        // find pivot
        int p = j;
        float maxA = fabsf(LU[j*n + j]);
        for (int i = j+1; i < n; i++) {
            float absA = fabsf(LU[i*n + j]);
            if (absA > maxA) {
                maxA = absA;
                p = i;
            }
        }
        
        // Check for singularity
        if (maxA < 1e-12f) {
            return false;  // Singular or nearly singular
        }
        
        // Exchange rows if needed
        if (p != j) {
            for (int k = 0; k < n; k++) {
                float temp = LU[p*n + k];
                LU[p*n + k] = LU[j*n + k];
                LU[j*n + k] = temp;
            }
            
            int temp = pivots[p];
            pivots[p] = pivots[j];
            pivots[j] = temp;
            
            pivSign = -pivSign;
        }
        
        // Compute multipliers
        if (j < n-1) {
            float diag = LU[j*n + j];
            for (int i = j+1; i < n; i++) {
                LU[i*n + j] /= diag;
            }
        }
    }
    
    return true;
}

__device__ inline float deviceDeterminant(const float* matrix, int n) {
    float LU[MAX_DIM * MAX_DIM];
    int pivots[MAX_DIM];
    int pivSign;
    
    if (!deviceLUDecompose(matrix, LU, pivots, pivSign, n)) {
        return 0.0f;
    }
    
    float det = static_cast<float>(pivSign);
    for (int i = 0; i < n; i++) {
        det *= LU[i*n + i];
    }
    
    return det;
}

__device__ inline bool deviceMatrixInverse(const float* matrix, float* inverse, int n) {
    float LU[MAX_DIM * MAX_DIM];
    int pivots[MAX_DIM];
    int pivSign;
    
    if (!deviceLUDecompose(matrix, LU, pivots, pivSign, n)) {
        // return identity for singular matrix
        for (int i = 0; i < n*n; i++) {
            inverse[i] = 0.0f;
        }
        for (int i = 0; i < n; i++) {
            inverse[i*n + i] = 1.0f;
        }
        return false;
    }
    
    // for each column of the inverse
    for (int j = 0; j < n; j++) {
        // set up the right-hand side
        float b[MAX_DIM];
        for (int i = 0; i < n; i++) {
            b[i] = (pivots[i] == j) ? 1.0f : 0.0f;
        }
        
        // forward solve L*y = b
        float y[MAX_DIM];
        for (int i = 0; i < n; i++) {
            float sum = b[i];
            for (int k = 0; k < i; k++) {
                sum -= LU[i*n + k] * y[k];
            }
            y[i] = sum;
        }
        
        // backward solve U*x = y
        float x[MAX_DIM];
        for (int i = n-1; i >= 0; i--) {
            float sum = y[i];
            for (int k = i+1; k < n; k++) {
                sum -= LU[i*n + k] * x[k];
            }
            x[i] = sum / LU[i*n + i];
        }
        
        // store the column in the result
        for (int i = 0; i < n; i++) {
            inverse[i*n + j] = x[i];
        }
    }
    
    return true;
}

__device__ inline float logDeviceMultivariatePDF(const float* x, const float* mean, const float* covariance, int d) {
    float diff[MAX_DIM];
    for (int i = 0; i < d; i++) {
        diff[i] = x[i] - mean[i];
    }

    float invCov[MAX_DIM * MAX_DIM];
    deviceMatrixInverse(covariance, invCov, d);

    // calculate quadratic form (x - mean)^T * inv(covariance) * (x - mean)
    float quadForm = 0.0f;
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            quadForm += diff[i] * invCov[i*d + j] * diff[j];
        }
    }

    // calculate determinant of covariance matrix
    float detCov = deviceDeterminant(covariance, d);

    // ensure positive determinant
    if (detCov <= 0.0f) {
        detCov = 1e-6f;
    }

    // log normalization constant
    float logNorm = -0.5f * d * logf(2.0f * M_PI) - 0.5f * logf(detCov);
    
    // return log of PDF
    return logNorm - 0.5f * quadForm;
}

#endif