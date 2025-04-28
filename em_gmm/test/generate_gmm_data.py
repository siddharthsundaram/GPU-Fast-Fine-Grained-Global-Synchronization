"""
Options:
    -k, INT    Number of Gaussian components [default: 3]
    -d, INT    Dimensionality of the data [default: 2]
    -n, INT    Number of data points to generate [default: 1000]
    -o, STRING     Output file name [default: gmm_data.txt]
    -s, INT          Random seed for reproducibility [default: 42]
    --cov-scale FLOAT       Scale factor for covariance matrices [default: 1.0]
    --separation FLOAT      Minimum separation between means [default: 3.0]
    --workload STRING       Workload type (balanced, skewed) [default: balanced]
    --skew-factor FLOAT     For skewed workload: how much to concentrate weights [default: 0.8]
    --dominant-comps INT    For skewed workload: number of dominant components [default: 2]
    --weights STRING        Comma-separated list of custom weights (overrides workload)
    --help                  Show this help message and exit
"""

import numpy as np
import argparse

def generate_random_means(n_components, n_dimensions, min_separation, rng):
    means = []
    while len(means) < n_components:
        new_mean = rng.uniform(0, 10 * n_components, n_dimensions)
        if all(np.linalg.norm(new_mean - m) >= min_separation for m in means):
            means.append(new_mean)
    return np.array(means)

def generate_random_covariances(n_components, n_dimensions, scale, rng):
    covariances = []
    for _ in range(n_components):
        A = rng.normal(0, 1, (n_dimensions, n_dimensions))
        cov = np.dot(A, A.T) + 0.01 * np.eye(n_dimensions)
        cov = scale * cov
        covariances.append(cov)
    return covariances

def generate_weights(n_components, workload_type, skew_factor, dominant_comps, custom_weights, rng):
    if custom_weights is not None:
        weights = np.array(custom_weights)
        weights = weights / weights.sum()
        return weights[:n_components]
    if workload_type == 'balanced':
        return rng.dirichlet(np.ones(n_components))
    alpha = np.ones(n_components)
    alpha[:dominant_comps] = 20.0
    raw_weights = rng.dirichlet(alpha)
    weights = np.zeros(n_components)
    weights[:dominant_comps] = raw_weights[:dominant_comps] / raw_weights[:dominant_comps].sum() * skew_factor
    weights[dominant_comps:] = raw_weights[dominant_comps:] / raw_weights[dominant_comps:].sum() * (1 - skew_factor)
    return weights

def generate_gmm_data(n_components, n_dimensions, n_samples, cov_scale, min_separation, 
                      workload_type, skew_factor, dominant_comps, custom_weights, seed):
    rng = np.random.RandomState(seed)
    weights = generate_weights(n_components, workload_type, skew_factor, dominant_comps, custom_weights, rng)
    means = generate_random_means(n_components, n_dimensions, min_separation, rng)
    covariances = generate_random_covariances(n_components, n_dimensions, cov_scale, rng)
    data = []
    component_indices = rng.choice(n_components, size=n_samples, p=weights)
    for i in range(n_samples):
        c = component_indices[i]
        sample = rng.multivariate_normal(means[c], covariances[c])
        data.append(sample)
    return np.array(data), weights, means, covariances

def save_gmm_data(filename, data, weights, means, covariances):
    n_samples, n_dimensions = data.shape
    n_components = len(weights)
    with open(filename, 'w') as f:
        f.write(f"{n_samples}\n{n_dimensions}\n{n_components}\n")
        for k in range(n_components):
            f.write(f"{weights[k]}\n")
            f.write(",".join(f"{x}" for x in means[k]) + "\n")
            for i in range(n_dimensions):
                f.write(",".join(f"{covariances[k][i, j]}" for j in range(n_dimensions)) + "\n")
        for i in range(n_samples):
            f.write(",".join(f"{x}" for x in data[i]) + "\n")

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--components', type=int, default=3)
    parser.add_argument('-d', '--dimensions', type=int, default=2)
    parser.add_argument('-n', '--num-points', type=int, default=1000)
    parser.add_argument('-o', '--output', type=str, default='gmm_data.txt')
    parser.add_argument('-s', '--seed', type=int, default=42)
    parser.add_argument('--cov-scale', type=float, default=1.0)
    parser.add_argument('--separation', type=float, default=3.0)
    parser.add_argument('--workload', type=str, default='balanced', choices=['balanced', 'skewed'])
    parser.add_argument('--skew-factor', type=float, default=0.8)
    parser.add_argument('--dominant-comps', type=int, default=2)
    parser.add_argument('--weights', type=str, default=None)
    args = parser.parse_args()
    args.custom_weights = [float(w) for w in args.weights.split(',')] if args.weights else None
    return args

def main():
    args = parse_arguments()
    data, weights, means, covariances = generate_gmm_data(
        args.components, args.dimensions, args.num_points,
        args.cov_scale, args.separation, args.workload,
        args.skew_factor, args.dominant_comps, args.custom_weights, args.seed
    )
    save_gmm_data(args.output, data, weights, means, covariances)

if __name__ == "__main__":
    main()