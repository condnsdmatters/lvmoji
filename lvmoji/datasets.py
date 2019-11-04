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

def generate_nonlinear_datasets(points: int, latent_dims: int, output_dims: int):
    '''Create data from a Gaussian, through a non-linearity, and add observation noise.

    Args:
        points (int): number of points to generate
        latent_dims (int): latent dimensions
        output_dims (int): output dimensionality
    '''
    latent_noise = 10
    latent_samples = np.random.multivariate_normal(
        np.zeros(latent_dims), np.eye(latent_dims) * latent_noise, points)

    W1 = np.random.normal(size=(latent_dims, output_dims))
    f1 = lambda X: np.tanh(X @ W1)
    W2 = np.random.normal(size=(output_dims, output_dims))
    f2 = lambda X: X @ W2

    noise_scale = 0.1
    noise = np.random.multivariate_normal(
        np.zeros(output_dims), np.eye(output_dims)*noise_scale, points)
    samples = f2(f1(latent_samples) + noise)
    return samples, latent_samples