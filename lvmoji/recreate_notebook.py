'''Recreate the notebook for gplvm in gpflow'''

import gpflow
from gpflow.config import set_default_float, default_float, set_summary_fmt

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import datasets



def fit_gplvm(ydata):
    latent_dim = 2
    num_inducing_points = 10

    # Z = np.random.random(size=(inducing_points, latent))



    x_mean_init = tf.convert_to_tensor(gpflow.utilities.ops.pca_reduce(ydata, latent_dim), dtype=default_float())
    x_var_init = tf.convert_to_tensor(np.ones((ydata.shape[0], latent_dim)), dtype=default_float())

    inducing_variable = tf.convert_to_tensor(np.random.permutation(x_mean_init.numpy())[:num_inducing_points], dtype=default_float())
    lengthscale = tf.convert_to_tensor([1.0] * latent_dim, dtype=default_float())
    kernel = gpflow.kernels.RBF(lengthscale=lengthscale, ard=True)

    model = gpflow.models.BayesianGPLVM(ydata,
            x_data_mean=x_mean_init,
            x_data_var=x_var_init,
            kernel=kernel,
            inducing_variable=inducing_variable)

    model.likelihood.variance.assign(0.01)

    opt = gpflow.optimizers.Scipy()

    gpflow.utilities.print_summary(model)

    # @tf.function(autograph=False)
    def optimization_step():
        return model.neg_log_marginal_likelihood()

    _ = opt.minimize(optimization_step, variables=model.trainable_variables, options=dict(maxiter=10000))

    gpflow.utilities.print_summary(model)





def main():
    '''run the whole notebook'''
    clusters = 3
    points = 500
    true_latent_dim = 2
    output_dim = 10
    data, latent, labels = datasets.generate_nonlinear_datasets(points, clusters, true_latent_dim, output_dim)
    fig = plt.figure(figsize=(6, 12))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.scatter(data[:, 2], data[:, 4], s=1, c=labels)
    ax2.scatter(latent[:, 0], latent[:, 1], s=1, c=labels)
    plt.show()

    fit_gplvm(data)




if __name__ == '__main__':
    main()
