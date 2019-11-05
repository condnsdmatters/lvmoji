"""Generate some toy datasets to play around with."""

import numpy as np

def generate_mog_datasets(points: int, latent_dims: int, output_dims: int):
    '''Create data from a Mixture of Gaussians generative model.

    Args:
        points (int): number of points to generate
        latent_dims (int): latent dimensions
        output_dims (int): output dimensionality
    '''
    latent = list(range(latent_dims))

    spacing = 20
    scale = 5
    means = np.random.multivariate_normal(
        mean=np.zeros(output_dims),
        cov=np.eye(output_dims) * spacing,
        size=latent_dims)
    covariances = [np.eye(output_dims) * s * scale for s in np.random.random(latent_dims)]
    noise = 1

    # TO DO: speed up by sampling same Gaussians together
    latent_samples = np.random.choice(latent, points)
    noise_samples = np.random.multivariate_normal(
        np.zeros(output_dims), np.eye(output_dims) * noise, points)
    gaussian_samples = []
    for latent_i, noise in zip(latent_samples, noise_samples):
        sample = np.random.multivariate_normal(means[latent_i], covariances[latent_i]) + noise
        gaussian_samples.append(sample)
    return np.array(gaussian_samples), latent_samples

def generate_nonlinear_datasets(points: int, clusters: int, latent_dims: int, output_dims: int):
    '''Create data from mixture of Gaussians, through a non-linearity, and add observation noise.

    Args:
        points (int): number of points to generate
        latent_dims (int): latent dimensions
        output_dims (int): output dimensionality
    '''
    unnormalized_cluster_means = np.random.uniform(low=-1.0, high=1.0, size=(clusters, latent_dims))
    print(unnormalized_cluster_means)
    norms = np.linalg.norm(unnormalized_cluster_means, axis=1)
    cluster_means = (unnormalized_cluster_means.T / norms).T


    labels = np.sort(np.random.choice(clusters, size=points))
    latent_samples = []
    for c in range(clusters):
        count = np.sum(labels == c) # ToDo: unefficient
        latent_noise = 0.1
        latent_samples.append(np.random.multivariate_normal(
            cluster_means[c], np.eye(latent_dims) * latent_noise, count))
    latent_samples = np.vstack(latent_samples)

    relu = lambda x: x * (x > 0)
    W1 = np.random.normal(size=(latent_dims, output_dims))
    f1 = lambda X: np.tanh(X @ W1)
    W2 = np.random.normal(size=(output_dims, output_dims))
    f2 = lambda X: relu(X @ W2)
    W3 = np.random.normal(size=(output_dims, output_dims))
    f3 = lambda X: X @ W3

    noise_scale = 0.01
    noise = np.random.multivariate_normal(
        np.zeros(output_dims), np.eye(output_dims)*noise_scale, points)
    samples = f3(f2(f1(latent_samples))) + noise
    return samples, latent_samples, labels