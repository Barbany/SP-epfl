# ##############################################################################
# helpers.py
# ==========
# Author : Sepand KASHANI [sepand.kashani@epfl.ch]
# ##############################################################################

"""
Helper functions.
"""

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import scipy.sparse as sparse

def plot_image(I, title=None, ax=None):
    """
    Plot a 2D mono-chromatic image.

    Parameters
    ----------
    I : :py:class:`~numpy.ndarray`
        (N_height, N_width) image.
    title : str
        Optional title to add to figure.
    ax : :py:class:`~matplotlib.axes.Axes`
        Optional axes on which to draw figure.
    """
    if ax is None:
        _, ax = plt.subplots()

    ax.imshow(I, cmap='bone')

    if title is not None:
        ax.set_title(title)

    return ax


def backprojection(S, N):
    """
    Form image via backprojection.

    Parameters
    ----------
    S : :py:class:`~scipy.sparse.csr_matrix`
        (N_tube, N_px) sampling operator.
    N : :py:class:`~numpy.ndarray`
        (N_tube,) PET samples

    Returns
    -------
    I : :py:class:`~numpy.sparse.csr_matrix`
        (N_px,) vectorized image
    """
    P = S.T  # (N_px, N_tube) synthesis op
    I = P @ N

    return I


def least_squares(S, N, regularize=False):
    """
    Form image via least-squares.

    Parameters
    ----------
    S : :py:class:`~scipy.sparse.csr_matrix`
        (N_tube, N_px) sampling operator.
    N : :py:class:`~numpy.ndarray`
        (N_tube,) PET samples
    regularize : bool
        If `True`, then drop least-significant eigenpairs from Gram matrix.

    Returns
    -------
    I : :py:class:`~numpy.sparse.csr_matrix`
        (N_px,) vectorized image
    """
    P = S.T  # (N_px, N_tube)  # sampling operator adjoint
    G = (S @ P).toarray()
    D, V = linalg.eigh(G)

    if regularize:
        # Careful, G is not easily invertible due to eigenspaces with almost-zero eigenvalues.
        # Inversion must be done as such.
        mask = np.isclose(D, 0)
        D, V = D[~mask], V[:, ~mask]
        
        # In addition, we will only keep the spectral components that account for 95% of \norm{G}{F}
        idx = np.argsort(D)[::-1]
        D, V = D[idx], V[:, idx]
        mask = np.clip(np.cumsum(D) / np.sum(D), 0, 1) <= 0.95
        D, V = D[mask], V[:, mask]

    # The division by D may blow in your face depending on the size of G. 
    # Always regularize your inversions (as [optionally] done above)!
    I = P @ (V / D) @ V.T @ N  

    return I
