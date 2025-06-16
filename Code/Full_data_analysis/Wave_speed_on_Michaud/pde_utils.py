import numpy as np
from scipy.ndimage import gaussian_filter

# approximates the laplacian with periodic boundaries
def laplacian(Z, DX=1.0):
    return (np.roll(Z, 1, axis=1) + np.roll(Z, -1, axis=1) + np.roll(Z, 1, axis=0) + np.roll(Z, -1, axis=0) - 4*Z) / DX**2

# approximates the double derivative
def double_derivative(Z, DX=1.0):
    return (np.roll(Z, 1) + np.roll(Z, -1) - 2*Z) / DX**2

# generate correlated gaussian field
def correlated_gaussian_field(SIGMA, S, SHAPE, MEAN=1.0):
    # apply Gaussian filter to random noise
    field = gaussian_filter(np.random.normal(0, 1, SHAPE), sigma=S, mode='wrap')
    # scale and shift for desired standard deviation and mean
    return MEAN + SIGMA * field / np.std(field)