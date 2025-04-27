#!/usr/bin/env python3
# used chat to generate script to create test inputs-- prob delete later
"""
Gaussian Mixture Model (GMM) Data Generator

This script generates synthetic data from Gaussian mixture models and
saves it in the format expected by the EM-GMM C++ implementation.

The script can generate specific workload patterns that stress lock contention:
1. Few components with many data points
2. Skewed distributions where most points belong to a small subset of components

Usage:
    python generate_gmm_data.py [options]

Options:
    -k, --components INT    Number of Gaussian components [default: 3]
    -d, --dimensions INT    Dimensionality of the data [default: 2]
    -n, --num-points INT    Number of data points to generate [default: 1000]
    -o, --output STRING     Output file name [default: gmm_data.txt]
    -s, --seed INT          Random seed for reproducibility [default: 42]
    --cov-scale FLOAT       Scale factor for covariance matrices [default: 1.0]
    --separation FLOAT      Minimum separation between means [default: 3.0]
    --workload STRING       Workload type (balanced, skewed) [default: balanced]
    --skew-factor FLOAT     For skewed workload: how much to concentrate weights [default: 0.8]
    --dominant-comps INT    For skewed workload: number of dominant components [default: 2]
    --weights STRING        Comma-separated list of custom weights (overrides workload)
    --help                  Show this help message and exit
"""

import numpy as np
from scipy.stats import multivariate_normal
import argparse
import sys


def generate_random_means(n_components, n_dimensions, min_separation, rng):
    """
    Generate random means with a minimum separation between them.
    
    Args:
        n_components: Number of components
        n_dimensions: Number of dimensions
        min_separation: Minimum Euclidean distance between any two means
        rng: NumPy random number generator
        
    Returns:
        Array of means with shape (n_components, n_dimensions)
    """
    means = []
    attempts = 0
    max_attempts = 1000
    
    while len(means) < n_components and attempts < max_attempts:
        attempts += 1
        # Generate a random point in the hypercube [0, 10*n_components]^n_dimensions
        # This scaling ensures enough space for separation as n_components increases
        new_mean = rng.uniform(0, 10 * n_components, n_dimensions)
        
        # Check if it's far enough from existing means
        if all(np.linalg.norm(new_mean - existing_mean) >= min_separation 
               for existing_mean in means):
            means.append(new_mean)
            attempts = 0  # Reset attempts counter after successful addition
    
    if len(means) < n_components:
        print(f"Warning: Could only generate {len(means)} means with the required separation.")
        print("Try reducing the separation parameter or the number of components.")
    
    return np.array(means)


def generate_random_covariances(n_components, n_dimensions, scale, rng):
    """
    Generate random positive definite covariance matrices.
    
    Args:
        n_components: Number of components
        n_dimensions: Number of dimensions
        scale: Scale factor for covariance matrices
        rng: NumPy random number generator
        
    Returns:
        List of covariance matrices
    """
    covariances = []
    
    for _ in range(n_components):
        # Generate a random matrix
        A = rng.normal(0, 1, (n_dimensions, n_dimensions))
        # Make it positive definite by multiplying with its transpose and adding a small diagonal
        cov = np.dot(A, A.T) + 0.01 * np.eye(n_dimensions)
        # Scale it
        cov = scale * cov
        covariances.append(cov)
    
    return covariances


def generate_weights(n_components, workload_type, skew_factor, dominant_comps, custom_weights, rng):
    """
    Generate component weights according to the specified workload type.
    
    Args:
        n_components: Number of components
        workload_type: Type of workload ('balanced' or 'skewed')
        skew_factor: For skewed workload, how much weight to put in dominant components
        dominant_comps: For skewed workload, number of dominant components
        custom_weights: Custom weights provided by the user
        rng: NumPy random number generator
        
    Returns:
        Array of weights summing to 1
    """
    if custom_weights is not None:
        # Use custom weights
        weights = np.array(custom_weights)
        # Normalize to sum to 1
        weights = weights / weights.sum()
        # Check if we have the right number of weights
        if len(weights) != n_components:
            print(f"Warning: {len(weights)} weights provided, but {n_components} components requested.")
            if len(weights) < n_components:
                # Pad with small weights
                padding = np.ones(n_components - len(weights)) * 0.001
                weights = np.append(weights, padding)
                weights = weights / weights.sum()  # Renormalize
            else:
                # Truncate
                weights = weights[:n_components]
                weights = weights / weights.sum()  # Renormalize
        return weights
    
    if workload_type == 'balanced':
        # Approximately uniform weights with some randomness
        weights = rng.dirichlet(np.ones(n_components) * 5.0)
    else:  # skewed
        # Ensure dominant_comps doesn't exceed n_components
        dominant_comps = min(dominant_comps, n_components)
        
        # Create a concentration parameter that heavily weights the first dominant_comps components
        alpha = np.ones(n_components)
        # Make dominant components have higher concentration
        alpha[:dominant_comps] = 20.0
        # Make non-dominant components have lower concentration
        alpha[dominant_comps:] = 1.0
        
        # Generate weights from Dirichlet distribution
        raw_weights = rng.dirichlet(alpha)
        
        # Apply additional skewing
        weights = np.zeros(n_components)
        # Dominant components get skew_factor of the total weight
        dominant_sum = raw_weights[:dominant_comps].sum()
        if dominant_sum > 0:  # Avoid division by zero
            weights[:dominant_comps] = raw_weights[:dominant_comps] / dominant_sum * skew_factor
        
        # Non-dominant components share the remaining weight
        non_dominant_sum = raw_weights[dominant_comps:].sum()
        if non_dominant_sum > 0:  # Avoid division by zero
            weights[dominant_comps:] = raw_weights[dominant_comps:] / non_dominant_sum * (1 - skew_factor)
        else:
            # If non-dominant weights sum to zero, distribute remaining weight evenly
            weights[dominant_comps:] = (1 - skew_factor) / (n_components - dominant_comps)
    
    return weights


def generate_gmm_data(n_components, n_dimensions, n_samples, cov_scale, min_separation, 
                     workload_type, skew_factor, dominant_comps, custom_weights, seed):
    """
    Generate data from a Gaussian mixture model.
    
    Args:
        n_components: Number of components in the mixture
        n_dimensions: Dimensionality of the data
        n_samples: Number of samples to generate
        cov_scale: Scale factor for covariance matrices
        min_separation: Minimum separation between means
        workload_type: Type of workload ('balanced' or 'skewed')
        skew_factor: For skewed workload, how much weight to put in dominant components
        dominant_comps: For skewed workload, number of dominant components
        custom_weights: Custom weights provided by the user
        seed: Random seed
        
    Returns:
        data: Generated data points
        weights: Component weights
        means: Component means
        covariances: Component covariance matrices
    """
    rng = np.random.RandomState(seed)
    
    # Generate weights according to workload type
    weights = generate_weights(n_components, workload_type, skew_factor, dominant_comps, custom_weights, rng)
    
    # Generate random means with minimum separation
    means = generate_random_means(n_components, n_dimensions, min_separation, rng)
    
    # Generate random covariance matrices
    covariances = generate_random_covariances(n_components, n_dimensions, cov_scale, rng)
    
    # Generate data
    data = []
    component_indices = rng.choice(n_components, size=n_samples, p=weights)
    
    for i in range(n_samples):
        component = component_indices[i]
        sample = rng.multivariate_normal(means[component], covariances[component])
        data.append(sample)
    
    return np.array(data), weights, means, covariances


def save_gmm_data(filename, data, weights, means, covariances):
    """
    Save GMM data to a file in the format expected by the EM-GMM C++ implementation.
    
    Format:
    - First line: Number of data points
    - Second line: Number of dimensions
    - Third line: Number of true components
    - Then for each component:
      - One line for weight
      - One line for mean (comma-separated)
      - Multiple lines for covariance matrix (one per row, comma-separated)
    - Finally, all data points (one per line, comma-separated)
    
    Args:
        filename: Name of the output file
        data: Generated data points
        weights: Component weights
        means: Component means
        covariances: Component covariance matrices
    """
    n_samples, n_dimensions = data.shape
    n_components = len(weights)
    
    with open(filename, 'w') as f:
        # Write number of data points
        f.write(f"{n_samples}\n")
        
        # Write dimensionality
        f.write(f"{n_dimensions}\n")
        
        # Write number of true components
        f.write(f"{n_components}\n")
        
        # Write component parameters
        for k in range(n_components):
            # Write weight
            f.write(f"{weights[k]}\n")
            
            # Write mean
            f.write(",".join(f"{x}" for x in means[k]) + "\n")
            
            # Write covariance matrix
            for i in range(n_dimensions):
                f.write(",".join(f"{covariances[k][i, j]}" for j in range(n_dimensions)) + "\n")
        
        # Write data points
        for i in range(n_samples):
            f.write(",".join(f"{x}" for x in data[i]) + "\n")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate synthetic GMM data")
    
    parser.add_argument('-k', '--components', type=int, default=3,
                        help='Number of Gaussian components (default: 3)')
    parser.add_argument('-d', '--dimensions', type=int, default=2,
                        help='Dimensionality of the data (default: 2)')
    parser.add_argument('-n', '--num-points', type=int, default=1000,
                        help='Number of data points to generate (default: 1000)')
    parser.add_argument('-o', '--output', type=str, default='gmm_data.txt',
                        help='Output file name (default: gmm_data.txt)')
    parser.add_argument('-s', '--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--cov-scale', type=float, default=1.0,
                        help='Scale factor for covariance matrices (default: 1.0)')
    parser.add_argument('--separation', type=float, default=3.0,
                        help='Minimum separation between means (default: 3.0)')
    parser.add_argument('--workload', type=str, default='balanced',
                        choices=['balanced', 'skewed'],
                        help='Workload type: balanced or skewed (default: balanced)')
    parser.add_argument('--skew-factor', type=float, default=0.8,
                        help='For skewed workload: proportion of weight in dominant components (default: 0.8)')
    parser.add_argument('--dominant-comps', type=int, default=2,
                        help='For skewed workload: number of dominant components (default: 2)')
    parser.add_argument('--weights', type=str, default=None,
                        help='Custom weights as comma-separated values (overrides workload type)')
    
    args = parser.parse_args()
    
    # Process custom weights if provided
    if args.weights:
        try:
            args.custom_weights = [float(w) for w in args.weights.split(',')]
        except ValueError:
            print("Error: Custom weights must be comma-separated floating-point numbers")
            sys.exit(1)
    else:
        args.custom_weights = None
    
    return args


def main():
    """Main function"""
    args = parse_arguments()
    
    print(f"Generating {args.num_points} data points with {args.components} components in {args.dimensions} dimensions...")
    print(f"Workload type: {args.workload}")
    
    if args.custom_weights:
        print(f"Using custom weights: {args.custom_weights}")
    elif args.workload == 'skewed':
        print(f"Skew factor: {args.skew_factor}, Dominant components: {args.dominant_comps}")
    
    data, weights, means, covariances = generate_gmm_data(
        args.components, args.dimensions, args.num_points,
        args.cov_scale, args.separation, args.workload,
        args.skew_factor, args.dominant_comps, args.custom_weights, args.seed
    )
    
    print(f"Saving data to {args.output}...")
    save_gmm_data(args.output, data, weights, means, covariances)
    
    print("Component weights:")
    for i, w in enumerate(weights):
        print(f"  Component {i}: {w:.4f}")
    
    print("\nComponent means:")
    for i, mean in enumerate(means):
        print(f"  Component {i}: {mean}")
    
    # Calculate approximate number of points per component
    expected_counts = weights * args.num_points
    print("\nExpected points per component:")
    for i, count in enumerate(expected_counts):
        print(f"  Component {i}: {count:.1f}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()