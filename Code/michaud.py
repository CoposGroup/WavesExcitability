import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from pde_utils import laplacian, correlated_gaussian_field

def run(params, output_name, mode="gif"):

    # get parameter values from JSON
    try: 
        (k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, Drt, Drd, Df, sigma, s, f, alpha, beta, size, dt, t, frame_int, random_seed)\
        = (params[k] for k in ["k0", "k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8", "k9", "k10","Drt", "Drd", "Df", "sigma", "s", "f", "alpha", "beta","size", "dt", "t", "frame_int", "random_seed"])
    except KeyError as e:
        raise ValueError(f"Missing parameter: {e}")
    
    # set random seed
    np.random.seed(random_seed)

    # define reaction function
    def R(A,B,C):
        return (k0 + alpha*k1*A**3/(1 + k2*A**2))*B - (k3 + k4*(1 + beta)*C)*A

    # set initial concentrations
    RT = 0.1 + 0.9 * np.random.rand(size,size)
    RD = np.full((size, size),0.1)
    F = np.full((size, size),0)

    # set initial stochastic noise term
    dW = correlated_gaussian_field(sigma,s,(size,size),1.0)

    # set up figure
    fig, ax = plt.subplots(figsize=(6,6))
    ax.axis('off')
    frames = []

    # create colorbar
    im = ax.imshow(RT, cmap='cividis', vmin=0, vmax=1, animated=True)
    fig.colorbar(im, ax=ax, shrink=0.8)

    steps = int(t / dt)

    # simulate the PDE with finite difference method
    for i in range(steps):

        # update concentrations
        RT = RT + dt * (R(RT,RD,F) + Drt*laplacian(RT))
        RD = RD + dt * (k5 - k6*RD - R(RT,RD,F) + Drd*laplacian(RD))
        F = F + dt * (k7 + k8*RT**2 / (1 + k9*RT**2) - k10*dW*F + Df*laplacian(F))

        # update stocastic noise term every f seconds
        if i % int(f/dt) == 0:
            dW = correlated_gaussian_field(sigma,s,(size,size),1.0)

        # capture a frame every frame_int seconds
        if mode == "gif" and i % int(frame_int/dt) == 0:
            frames.append([ax.imshow(RT, cmap='cividis', animated=True)])

    # save animation
    if mode == "gif":
        animation.ArtistAnimation(fig, frames, interval=50, blit=True).save(output_name, writer=PillowWriter(fps=10))
    elif mode == "png":
        ax.imshow(RT, cmap='cividis')
        plt.savefig(output_name, dpi=150, bbox_inches='tight')
    plt.close(fig)