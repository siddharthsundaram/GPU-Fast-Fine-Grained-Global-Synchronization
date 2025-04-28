#include "em.h"

int main(int argc, char* argv[]) {
    auto start = std::chrono::steady_clock::now();

    // parse command line arguments
    ArgOpts opts = parseCommandLine(argc, argv);

    // read input data
    InputData inputData = inputReader(opts.inputFilePath);
    if (inputData.points.empty()) {
        std::cerr << "Error: No data read from input file." << std::endl;
        return EXIT_FAILURE;
    }
        
    std::cout << "Running EM algorithm on " << inputData.points.size() << " data points with " 
              << inputData.points.at(0).size() << " dimensions, " << inputData.trueComponents.size() << " components"
              << " in " << opts.mode << " mode." << std::endl;
    
    std::cout << "Ground truth available with " << inputData.trueComponents.size() << " components." << std::endl;
    
    // run appropriate implementation based on mode
    EMModel result;
    if (opts.mode == "s") {
        result = runSequentialEM(inputData, opts);
    } else if (opts.mode == "b") {
        result = runBasicGPU(inputData, opts);
    } else if (opts.mode == "f") {
        result = runFineGPU(inputData, opts);
    } else if (opts.mode == "cs") {
        result = runServerClientGPU(inputData, opts);
    } else {
        std::cerr << "Error: Unknown mode. Exiting." << std::endl;
        return EXIT_FAILURE;
    }
    
    // print results
    std::cout << "EM completed in " << result.timeElapsed << " seconds after " 
              << result.iterations << " iterations." << std::endl;
    std::cout << "Final log-likelihood: " << result.logLikelihood << std::endl;
    
    float error = compareWithGroundTruth(result, inputData.trueComponents);
    std::cout << "Error compared to ground truth: " << error << std::endl;
    
    // save results
    saveResults(result, inputData, opts.outputFilePath);

    auto end = std::chrono::steady_clock::now();
    auto duration = end - start;
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
    std::cout << duration_ms.count() << " milliseconds" << std::endl;
    
    return EXIT_SUCCESS;
}

// --- Command line parsing functions ---
ArgOpts parseCommandLine(int argc, char* argv[]) {
    ArgOpts opts;
    
    int opt;
    while ((opt = getopt(argc, argv, "i:o:m:n:t:s:r:")) != -1) {
        switch (opt) {
            case 'i':
                opts.inputFilePath = optarg;
                break;
            case 'o':
                opts.outputFilePath = optarg;
                break;
            case 'm':
                opts.mode = optarg;
                if (opts.mode != "s" && opts.mode != "b" && opts.mode != "f" && opts.mode != "cs") {
                    std::cerr << "Error: Invalid mode." << std::endl;
                    exit(EXIT_FAILURE);
                }
                break;
            case 'n':
                opts.maxIterations = parseOptionToInt(optarg, 'n');
                break;
            case 't':
                opts.tolerance = parseOptionToFloat(optarg, 't');
                break;
            case 's':
                opts.seed = parseOptionToInt(optarg, 's');
                break;
            case 'r':
                opts.workloadRatio = parseOptionToFloat(optarg, 'r');
                break;
            default:
                exit(EXIT_FAILURE);
        }
    }
    
    return opts;
}

int parseOptionToInt(const char* optarg, char option) {
    try {
        int value = std::stoi(optarg);
        if (value <= 0) {
            throw std::invalid_argument("Value must be positive");
        }
        return value;
    } catch (const std::exception& e) {
        std::cerr << "Error: Invalid value for option -" << option << ". " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }
}

float parseOptionToFloat(const char* optarg, char option) {
    try {
        float value = std::stof(optarg);
        if (value <= 0) {
            throw std::invalid_argument("Value must be positive");
        }
        return value;
    } catch (const std::exception& e) {
        std::cerr << "Error: Invalid value for option -" << option << ". " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }
}

// --- Input/Output functions ---
InputData inputReader(const std::string& filePath) {
    InputData inputData;
    std::ifstream file(filePath);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filePath << std::endl;
        return inputData;
    }
    
    // read number of data points (first line)
    std::string line;
    if (!std::getline(file, line)) {
        std::cerr << "Error: Empty file or could not read first line." << std::endl;
        return inputData;
    }
    
    int numDataPoints;
    try {
        numDataPoints = std::stoi(line);
    } catch (const std::exception& e) {
        std::cerr << "Error parsing number of data points: " << e.what() << std::endl;
        return inputData;
    }
    
    // read dimensions (second line)
    if (!std::getline(file, line)) {
        std::cerr << "Error: Could not read dimensions line." << std::endl;
        return inputData;
    }
    
    int dimensions;
    try {
        dimensions = std::stoi(line);
    } catch (const std::exception& e) {
        std::cerr << "Error parsing dimensions: " << e.what() << std::endl;
        return inputData;
    }
    
    // read number of true components (third line)
    if (!std::getline(file, line)) {
        std::cerr << "Error: Could not read number of true components line." << std::endl;
        return inputData;
    }
    
    int numTrueComponents;
    try {
        numTrueComponents = std::stoi(line);
    } catch (const std::exception& e) {
        std::cerr << "Error parsing number of true components: " << e.what() << std::endl;
        return inputData;
    }
    
    // read true component parameters
    inputData.trueComponents.resize(numTrueComponents);
    
    for (int k = 0; k < numTrueComponents; k++) {
        // read weight
        if (!std::getline(file, line)) {
            std::cerr << "Error: Could not read weight for component " << k << std::endl;
            return inputData;
        }
        
        try {
            inputData.trueComponents[k].weight = std::stof(line);
        } catch (const std::exception& e) {
            std::cerr << "Error parsing weight for component " << k << ": " << e.what() << std::endl;
            return inputData;
        }
        
        // read mean
        if (!std::getline(file, line)) {
            std::cerr << "Error: Could not read mean for component " << k << std::endl;
            return inputData;
        }
        
        std::vector<float> mean;
        std::stringstream ssMean(line);
        std::string value;
        
        while (std::getline(ssMean, value, ',')) {
            try {
                mean.push_back(std::stof(value));
            } catch (const std::exception& e) {
                std::cerr << "Error parsing mean value: " << value << " - " << e.what() << std::endl;
                return inputData;
            }
        }
        
        if (mean.size() != dimensions) {
            std::cerr << "Error: Mean vector for component " << k << " has " << mean.size() 
                      << " dimensions, expected " << dimensions << std::endl;
            return inputData;
        }
        
        inputData.trueComponents[k].mean = mean;
        
        // read covariance matrix (one row per line)
        inputData.trueComponents[k].covariance.resize(dimensions, std::vector<float>(dimensions));
        
        for (int i = 0; i < dimensions; i++) {
            if (!std::getline(file, line)) {
                std::cerr << "Error: Could not read covariance row " << i 
                          << " for component " << k << std::endl;
                return inputData;
            }
            
            std::vector<float> covRow;
            std::stringstream ssCov(line);
            
            while (std::getline(ssCov, value, ',')) {
                try {
                    covRow.push_back(std::stof(value));
                } catch (const std::exception& e) {
                    std::cerr << "Error parsing covariance value: " << value << " - " << e.what() << std::endl;
                    return inputData;
                }
            }
            
            if (covRow.size() != dimensions) {
                std::cerr << "Error: Covariance row " << i << " for component " << k 
                          << " has " << covRow.size() << " values, expected " << dimensions << std::endl;
                return inputData;
            }
            
            inputData.trueComponents[k].covariance[i] = covRow;
        }
    }
    
    // read data points
    for (int i = 0; i < numDataPoints; i++) {
        if (!std::getline(file, line)) {
            std::cerr << "Error: Not enough data points in file. Expected " << numDataPoints 
                      << " but found " << i << "." << std::endl;
            break;
        }
        
        std::vector<float> point;
        std::stringstream ss(line);
        std::string value;
        
        while (std::getline(ss, value, ',')) {
            try {
                point.push_back(std::stof(value));
            } catch (const std::exception& e) {
                std::cerr << "Error parsing value: " << value << " - " << e.what() << std::endl;
                // Continue to parse remaining values
            }
        }
        
        if (point.size() != dimensions) {
            std::cerr << "Warning: Data point at line " << (i + 3 + numTrueComponents * (dimensions + 1)) 
                      << " has " << point.size() << " dimensions, expected " << dimensions 
                      << ". Skipping." << std::endl;
            continue;
        }
        
        inputData.points.push_back(point);
    }
    
    file.close();
    
    if (inputData.points.empty()) {
        std::cerr << "Error: No valid data points read from file." << std::endl;
    } else if (inputData.points.size() < numDataPoints) {
        std::cerr << "Warning: Expected " << numDataPoints << " data points but read " 
                  << inputData.points.size() << "." << std::endl;
    }
    
    return inputData;
}

// compare estimated model with ground truth
float compareWithGroundTruth(const EMModel& model, const std::vector<GaussianComponent>& trueComponents) {
    if (model.numComponents != trueComponents.size()) {
        std::cerr << "Warning: Number of components in model (" << model.numComponents 
                  << ") doesn't match number of true components (" << trueComponents.size() << ")" << std::endl;
    }
    
    const int numComponents = std::min(model.numComponents, (int)trueComponents.size());
    
    if (numComponents > 8) {
        std::cerr << "Warning: Too many components - not supported." << std::endl;
        float totalError = 0.0f;
        return totalError / numComponents;
    } else {
        // for small number of components, try all permutations
        std::vector<int> indices(numComponents);
        for (int i = 0; i < numComponents; i++) {
            indices[i] = i;
        }
        
        float minError = std::numeric_limits<float>::max();
        
        do {
            float error = 0.0f;
            
            for (int i = 0; i < numComponents; i++) {
                // calculate mean error
                float meanError = 0.0f;
                for (int d = 0; d < model.numDimensions; d++) {
                    float diff = model.components[i].mean[d] - trueComponents[indices[i]].mean[d];
                    meanError += diff * diff;
                }
                meanError = std::sqrt(meanError);
                error += meanError;
                
                // calculate weight error
                float weightError = std::abs(model.components[i].weight - trueComponents[indices[i]].weight);
                error += weightError;
                
                // calculate covariance error (Frobenius norm!!)
                float covError = 0.0f;
                for (int d1 = 0; d1 < model.numDimensions; d1++) {
                    for (int d2 = 0; d2 < model.numDimensions; d2++) {
                        float diff = model.components[i].covariance[d1][d2] - trueComponents[indices[i]].covariance[d1][d2];
                        covError += diff * diff;
                    }
                }
                covError = std::sqrt(covError);
                error += covError;
            }
            
            if (error < minError) {
                minError = error;
            }
            
        } while (std::next_permutation(indices.begin(), indices.end()));
        
        return minError / numComponents;
    }
}

void saveResults(const EMModel& model, const InputData& inputData, const std::string& filePath) {
    std::ofstream file(filePath);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open output file " << filePath << std::endl;
        return;
    }
    
    // header
    file << "# EM Algorithm Results\n";
    file << "# Number of components: " << model.numComponents << "\n";
    file << "# Number of dimensions: " << model.numDimensions << "\n";
    file << "# Final log-likelihood: " << model.logLikelihood << "\n";
    file << "# Number of iterations: " << model.iterations << "\n";
    file << "# Time elapsed: " << model.timeElapsed << " seconds\n";
    
    // add ground truth comparison
    float error = compareWithGroundTruth(model, inputData.trueComponents);
    file << "# Error compared to ground truth: " << error << "\n\n";
    
    // write components
    for (int k = 0; k < model.numComponents; k++) {
        const auto& comp = model.components[k];
        
        file << "Component " << k << ":\n";
        file << "Weight: " << comp.weight << "\n";
        
        file << "Mean: ";
        for (size_t d = 0; d < comp.mean.size(); d++) {
            file << comp.mean[d];
            if (d < comp.mean.size() - 1) file << ", ";
        }
        file << "\n";
        
        file << "Covariance:\n";
        for (size_t i = 0; i < comp.covariance.size(); i++) {
            for (size_t j = 0; j < comp.covariance[i].size(); j++) {
                file << comp.covariance[i][j];
                if (j < comp.covariance[i].size() - 1) file << ", ";
            }
            file << "\n";
        }
        file << "\n";
    }
    
    // write ground truth components
    file << "# Ground Truth Components\n";
    
    for (size_t k = 0; k < inputData.trueComponents.size(); k++) {
        const auto& comp = inputData.trueComponents[k];
        
        file << "True Component " << k << ":\n";
        file << "Weight: " << comp.weight << "\n";
        
        file << "Mean: ";
        for (size_t d = 0; d < comp.mean.size(); d++) {
            file << comp.mean[d];
            if (d < comp.mean.size() - 1) file << ", ";
        }
        file << "\n";
        
        file << "Covariance:\n";
        for (size_t i = 0; i < comp.covariance.size(); i++) {
            for (size_t j = 0; j < comp.covariance[i].size(); j++) {
                file << comp.covariance[i][j];
                if (j < comp.covariance[i].size() - 1) file << ", ";
            }
            file << "\n";
        }
        file << "\n";
    }
    
    file.close();
}

EMModel runSequentialEM(const InputData& inputData, const ArgOpts& opts) {
    EMModel model;
    model.numComponents = inputData.trueComponents.size();
    model.numDimensions = inputData.points.at(0).size();

    // initialize parameters
    initializeModel(model, inputData.points, opts.seed);

    // compute initial log-likelihood
    model.logLikelihood = computeLogLikelihood(model, inputData.points);
    float prevLogLikelihood = model.logLikelihood;

    std::vector<std::vector<float>> responsibilities(
        inputData.points.size(), std::vector<float>(model.numComponents));

    model.iterations = 0;
    auto startTime = std::chrono::high_resolution_clock::now();
    bool converged = false;

    while (model.iterations < opts.maxIterations && !converged) {
        eStep(model, inputData.points, responsibilities);

        mStep(model, inputData.points, responsibilities);

        model.logLikelihood = computeLogLikelihood(model, inputData.points);

        converged = std::abs(model.logLikelihood - prevLogLikelihood)
                    < opts.tolerance * std::abs(prevLogLikelihood);
        prevLogLikelihood = model.logLikelihood;
        ++model.iterations;

        std::cout << "Iteration " << model.iterations
                  << ", Log-likelihood: " << model.logLikelihood << std::endl;
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    model.timeElapsed = std::chrono::duration<double>(endTime - startTime).count();

    std::cout << (converged ? "EM converged" : "Reached max iterations")
              << " after " << model.iterations << " iterations." << std::endl;
    return model;
}

void initializeModel(EMModel& model, const std::vector<std::vector<float>>& data, int seed) {
    std::mt19937 rng(seed);
    int numData = data.size();
    
    // initialize components
    model.components.resize(model.numComponents);
    
    // equal weights initially
    float weight = 1.0f / model.numComponents;
    
    for (int k = 0; k < model.numComponents; k++) {
        // initialize weights
        model.components[k].weight = weight;
        
        // random mean initialization
        // pick a random data point as the initial mean
        int randomIdx = rng() % numData;
        model.components[k].mean = data[randomIdx];
        
        // initialize covariance matrix to identity * small value for pos definiteness
        model.components[k].covariance.resize(model.numDimensions, 
                                            std::vector<float>(model.numDimensions, 0.0f));
        
        // set diagonal elements to a small value
        for (int d = 0; d < model.numDimensions; d++) {
            model.components[k].covariance[d][d] = 1.0f;
        }
    }
}

// E-step: compute responsibilities
void eStep(EMModel& model,
            const std::vector<std::vector<float>>& data,
            std::vector<std::vector<float>>& responsibilities) {
    int N = data.size();
    int K = model.numComponents;

    for (int i = 0; i < N; ++i) {
        // compute log-weights + log-pdfs
        std::vector<float> logT(K);
        float maxLog = -std::numeric_limits<float>::infinity();
        for (int k = 0; k < K; ++k) {
            logT[k] = std::log(model.components[k].weight)
                    + logMultivariatePDF(data[i], model.components[k].mean,
                                            model.components[k].covariance);
            maxLog = std::max(maxLog, logT[k]);
        }
        // log-sum-exp denominator
        float sumExp = 0.0f;
        for (int k = 0; k < K; ++k) sumExp += std::exp(logT[k] - maxLog);
        float logDenom = maxLog + std::log(sumExp);
        // fill responsibilities
        for (int k = 0; k < K; ++k) {
            responsibilities[i][k] = std::exp(logT[k] - logDenom);
        }
    }
}

// M-step: update parameters
void mStep(EMModel& model, const std::vector<std::vector<float>>& data, const std::vector<std::vector<float>>& responsibilities) {
    const int numData = data.size();
    
    for (int k = 0; k < model.numComponents; k++) {
        float Nk = 0.0f;
        
        // sum of responsibilities for component k
        for (int i = 0; i < numData; i++) {
            Nk += responsibilities[i][k];
        }
        
        if (Nk > 0) {
            // update weight
            model.components[k].weight = Nk / numData;
            
            // update mean
            std::vector<float> newMean(model.numDimensions, 0.0f);
            for (int i = 0; i < numData; i++) {
                for (int d = 0; d < model.numDimensions; d++) {
                    newMean[d] += responsibilities[i][k] * data[i][d];
                }
            }
            for (int d = 0; d < model.numDimensions; d++) {
                model.components[k].mean[d] = newMean[d] / Nk;
            }
            
            // update covariance
            for (int d1 = 0; d1 < model.numDimensions; d1++) {
                for (int d2 = 0; d2 < model.numDimensions; d2++) {
                    float cov = 0.0f;
                    for (int i = 0; i < numData; i++) {
                        cov += responsibilities[i][k] * 
                              (data[i][d1] - model.components[k].mean[d1]) * 
                              (data[i][d2] - model.components[k].mean[d2]);
                    }
                    model.components[k].covariance[d1][d2] = cov / Nk;
                }
            }
            
            // add a small regularization term to diagonal to ensure covariance mat stays positive definite
            for (int d = 0; d < model.numDimensions; d++) {
                model.components[k].covariance[d][d] += 1e-6f;
            }
        }
    }
}

float computeLogLikelihood(const EMModel& model,
                            const std::vector<std::vector<float>>& data) {
    float ll = 0.0f;
    int N = data.size();
    int K = model.numComponents;
    for (int i = 0; i < N; ++i) {
        std::vector<float> logT(K);
        float maxLog = -std::numeric_limits<float>::infinity();
        for (int k = 0; k < K; ++k) {
            logT[k] = std::log(model.components[k].weight)
                    + logMultivariatePDF(data[i], model.components[k].mean,
                                model.components[k].covariance);
            maxLog = std::max(maxLog, logT[k]);
        }
        float sumExp = 0.0f;
        for (int k = 0; k < K; ++k) sumExp += std::exp(logT[k] - maxLog);
        ll += maxLog + std::log(sumExp);
    }
    return ll;
}

bool checkConvergence(const EMModel& model, float prevLogLikelihood, float tolerance) {
    return std::abs(model.logLikelihood - prevLogLikelihood) < tolerance * std::abs(prevLogLikelihood);
}

float logMultivariatePDF(const std::vector<float>& x,
                        const std::vector<float>& mean,
                        const std::vector<std::vector<float>>& cov) {
    int d = x.size();
    // diff = x - mean
    std::vector<float> diff(d);
    for (int i = 0; i < d; ++i) diff[i] = x[i] - mean[i];

    // invert covariance and compute quadratic form
    auto invCov = inverse(cov);
    float quad = 0.0f;
    for (int i = 0; i < d; ++i)
    for (int j = 0; j < d; ++j)
    quad += diff[i] * invCov[i][j] * diff[j];

    // Determinant
    float detCov = determinant(cov);
    if (detCov <= 0) detCov = 1e-6f;

    // log normalization constant
    float logNorm = -0.5f * d * std::log(2 * M_PI) - 0.5f * std::log(detCov);
    return logNorm - 0.5f * quad;
}

static bool luDecompose(const std::vector<std::vector<float>>& A,
                        std::vector<std::vector<float>>& LU,
                        std::vector<int>& pivots,
                        int& pivSign) {
    int n = A.size();
    LU = A;
    pivots.resize(n);
    for (int i = 0; i < n; ++i) pivots[i] = i;
    pivSign = 1;

    for (int j = 0; j < n; ++j) {
        // apply previous transformations
        for (int i = 0; i < n; ++i) {
            int kmax = std::min(i, j);
            float s = 0.0f;
            for (int k = 0; k < kmax; ++k) s += LU[i][k] * LU[k][j];
            LU[i][j] -= s;
        }
        // pivot search
        int p = j;
        float maxA = std::fabs(LU[j][j]);
        for (int i = j + 1; i < n; ++i) {
            float absA = std::fabs(LU[i][j]);
            if (absA > maxA) { maxA = absA; p = i; }
        }
        if (maxA < 1e-12f) return false;  // singular or nearly
        // row swap if needed
        if (p != j) {
            std::swap(LU[p], LU[j]);
            std::swap(pivots[p], pivots[j]);
            pivSign = -pivSign;
        }
        // compute multipliers
        float diag = LU[j][j];
        for (int i = j + 1; i < n; ++i) {
            LU[i][j] /= diag;
        }
    }
    return true;
}

float determinant(const std::vector<std::vector<float>>& matrix) {
    int n = matrix.size();
    std::vector<std::vector<float>> LU;
    std::vector<int> pivots;
    int pivSign;
    if (!luDecompose(matrix, LU, pivots, pivSign)) {
        return 0.0f;
    }
    float det = static_cast<float>(pivSign);
    for (int i = 0; i < n; ++i) {
        det *= LU[i][i];
    }
    return det;
}

std::vector<std::vector<float>> inverse(const std::vector<std::vector<float>>& matrix) {
    int n = matrix.size();
    std::vector<std::vector<float>> inv(n, std::vector<float>(n, 0.0f));
    std::vector<std::vector<float>> LU;
    std::vector<int> pivots;
    int pivSign;
    if (!luDecompose(matrix, LU, pivots, pivSign)) {
        // return identity on singular
        for (int i = 0; i < n; ++i) inv[i][i] = 1.0f;
        return inv;
    }

    std::vector<float> col(n), x(n);
    for (int j = 0; j < n; ++j) {
        // Permuted RHS for e_j
        for (int i = 0; i < n; ++i) {
            col[i] = (pivots[i] == j ? 1.0f : 0.0f);
        }
        // forward solve L*y = P*e_j
        for (int i = 0; i < n; ++i) {
            float s = col[i];
            for (int k = 0; k < i; ++k) s -= LU[i][k] * x[k];
            x[i] = s;
        }
        // Backward solve U*x = y
        for (int i = n - 1; i >= 0; --i) {
            float s = x[i];
            for (int k = i + 1; k < n; ++k) s -= LU[i][k] * x[k];
            x[i] = s / LU[i][i];
        }
        // Write inverse column
        for (int i = 0; i < n; ++i) {
            inv[i][j] = x[i];
        }
    }
    return inv;
}