import numpy as np
from scipy.ndimage import gaussian_filter

# laplacian function for periodic boundary conditions
def periodic_laplacian(Z, DX=1.0):
    return (np.roll(Z, 1, axis=1) + np.roll(Z, -1, axis=1) + np.roll(Z, 1, axis=0) + np.roll(Z, -1, axis=0) - 4*Z) / DX**2

# laplacian function for neumann boundary conditions
def neumann_laplacian(Z, DX=1.0):
    return (Z[0:-2, 1:-1] + Z[1:-1, 0:-2] + Z[2:, 1:-1] + Z[1:-1, 2:] - 4*Z[1:-1, 1:-1]) / DX**2

# 1D laplacian function for periodic boundary conditions
def periodic_laplacian_1D(Z):
    return np.roll(Z, 1) + np.roll(Z, -1) - 2*Z

# generate correlated gaussian field
def correlated_gaussian_field(SIGMA, S, SHAPE, MEAN=1.0):
    # apply Gaussian filter to random noise
    field = gaussian_filter(np.random.normal(0, 1, SHAPE), sigma=S, mode='wrap')
    # scale and shift for desired standard deviation and mean
    return MEAN + SIGMA * field / np.std(field)