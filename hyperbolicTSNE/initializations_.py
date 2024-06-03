""" Methods for initializing embedding.
"""
import ctypes
import numpy as np
from sklearn.decomposition import PCA

from sklearn.utils import check_random_state
from .hyperbolic_barnes_hut.lotsne import poincare_to_lorentz, lorentz_to_poincare


def initialization(n_samples, n_components, X=None, method="random", random_state=None):
    """
    Generates an initial embedding.

    Parameters
    ----------
    n_samples : int
        Number of samples (points) of the embedding.
    n_components : int
        Number of components (dimensions) of the embedding.
    X : ndarray, optional
        High-dimensional points if using method=`pca`.
    method : string, optional
        Method to use for generating the initial embedding.
        Should be a string in [random, pca]
    random_state : int
        To ensure reproducibility (used in sklearn `check_random_state` function.

    Returns
    -------
    X_embedded : ndarray
        array of shape (n_samples, n_components)
    """

    random_state = check_random_state(random_state)

    if method in ["pca"] and X is None:
        raise ValueError("The pca initialization requires the data X")

    if method == "random":
        X_embedded = 1e-4 * random_state.randn(n_samples, n_components).astype(np.float32)
    elif method == "pca":
        pca = PCA(n_components=n_components, svd_solver='randomized', random_state=random_state)
        X_embedded = pca.fit_transform(X).astype(np.float32, copy=False)
        X_embedded /= np.std(X_embedded[:, 0]) * 10000  # Need to rescale to avoid convergence issues
    else:
        raise ValueError(f"Method of initialization `{method}` not supported. init' must be 'pca', 'random', or a numpy array")

    return X_embedded

def to_lorentz(X):
    n_components = X.shape[1]
    n_samples = X.shape[0]

    if n_components != 2:
        raise ValueError("Data has to have exactly 2 components!")
    
    X_lorentz = np.zeros([n_samples, n_components + 1], dtype=ctypes.c_double)
    X_li = np.zeros([n_components + 1], dtype=ctypes.c_double)

    for i in range(n_samples):
        poincare_to_lorentz(X[i, 0], X[i, 1], X_li)
        X_lorentz[i, :] = X_li.copy()

    return X_lorentz

def from_lorentz(X):
    n_components = X.shape[1]
    n_samples = X.shape[0]

    if n_components != 3:
        raise ValueError("Data has to have exactly 3 components!")

    X_poincare = np.zeros([n_samples, n_components - 1], dtype=ctypes.c_double)
    X_li = np.zeros([n_components - 1], dtype=ctypes.c_double)

    for i in range(n_samples):
        lorentz_to_poincare(X[i, :], X_li)
        X_poincare[i, :] = X_li.copy()

    return X_poincare

