'''Recreate the notebook for gplvm in gpflow'''

import gpflow
import numpy as np
import matplotlib.pyplot as plt

import datasets



def main():
    '''run the whole notebook'''
    data, latent = datasets.generate_nonlinear_datasets(5000, 2, 10)
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.scatter(data[:, 2], data[:, 4], s=1, c=latent[:, 0])
    ax2.scatter(data[:, 2], data[:, 4], s=1, c=latent[:, 1])
    plt.show()



if __name__ == '__main__':
    main()
