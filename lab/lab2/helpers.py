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
import skimage.color as color
import tqdm


def rotation_matrix(angle):
    """
    Parameters
    ----------
    angle : float
        Counter-clockwise rotation angle [rad].

    Returns
    -------
    R : :py:class:`~numpy.ndarray`
        (2, 2) rotation matrix.
    """
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    return R


def tube_decomposition(N_detector):
    """
    Compute unique characterisation (d, xi, p, w) of detector tubes.

    Parameters
    ----------
    N_detector: int
        Number of detectors.
    
    Returns
    -------
    detector : :py:class:`~numpy.ndarray'
        (N_detector, 2) detector locations.
    xi : :py:class:`~numpy.ndarray`
        (N_tube, 2) normal vector to tube. (Must lie in quadrant I or II.)
    p : :py:class:`~numpy.ndarray`
        (N_tube,) tube distance from origin. (Can be negative.)
    w : :py:class:`~numpy.ndarray`
        (N_tube,) tube width.
    """
    # Uniformly distribute detectors on ring.
    detector_angle = np.linspace(0, 2 * np.pi, N_detector, endpoint=False)
    detector = np.stack((np.cos(detector_angle), np.sin(detector_angle)), axis=-1)
    dA, dB = np.triu_indices(N_detector, k=1)
    N_tube = len(dA)

    # Normal vector to detector tube, for all detector pairs.
    # This vector is always located in quadrant I or II.
    xi = detector[dA] + detector[dB]
    xi /= linalg.norm(xi, axis=-1, keepdims=True)
    xi[xi[:, 1] < 0] *= -1

    # Tube offset from origin such that xi*p points to the tube's mid-point.
    p = np.squeeze(xi.reshape((N_tube, 1, 2)) @ 
                   detector[dA].reshape((N_tube, 2, 1)))

    # Tube width.
    intra_detector_angle = np.mean(detector_angle[1:] - detector_angle[:-1])
    M = rotation_matrix(intra_detector_angle)
    intra_detector = np.dot(detector, M.T)

    diff_vector = intra_detector[dA] - intra_detector[dA - 1]
    w = np.squeeze(diff_vector.reshape((N_tube, 1, 2)) @ 
                   xi.reshape((N_tube, 2, 1)))
    w = np.abs(w) 

    # `w` can be very close to 0 and cause problem later on. 
    # We discard these tubes for practical purposes.
    mask = ~np.isclose(w, 0)
    xi, p, w = xi[mask], p[mask], w[mask]

    return detector, xi, p, w


def draw_tubes(S, N_height, N_width, idx, ax):
    """
    Draw detectors and detector tubes.
    
    Parameters
    ----------
    S : :py:class:`~scipy.sparse.csr_matrix`
        (N_tube, N_px) sampling operator.
    N_height : int
        Number of uniform vertical spatial samples in [-1, 1].
    N_width : int
        Number of uniform horizontal spatial samples in [-1, 1].
    idx : :py:class:`~numpy.ndarray`
        Tube indices to plot.
    ax : :py:class:`~matplotlib.axes.Axes`
    
    Returns
    -------
    ax : :py:class:`~matplotlib.axes.Axes`
    """
    N_tube_wanted = len(idx)

    tubes = S[idx, :]
    if sparse.issparse(tubes):
        tubes = tubes.toarray()
        
    tubes = (tubes
             .reshape((N_tube_wanted, N_height, N_width))
             .sum(axis=0))

    ax.get_xaxis().set_visible(False) 
    ax.get_yaxis().set_visible(False)
    ax.set_title('Detector Tubes')
    cmap = cm.gray

    ax.imshow(tubes, cmap=cmap)
    return ax


def sinogram(xi, p, N, ax):
    r"""
    Plot Sinogram scatterplot, with x-axis representing \angle(xi)
    
    Parameters
    ----------
    xi : :py:class:`~numpy.ndarray`
        (N_tube, 2) normal vector to tube. (Must lie in quadrant I or II.)
    p : :py:class:`~numpy.ndarray`
        (N_tube,) tube distance from origin. (Can be negative.)
    N : :py:class:`~numpy.ndarray`
        (N_tube,) PET measurements.
    ax : :py:class:`~matplotlib.axes.Axes`
    
    Returns
    -------
    ax : :py:class:`~matplotlib.axes.Axes`
    """
    N_tube = len(xi)
    theta = np.around(np.arctan2(*xi.T), 3)
    N = N.astype(float) / N.max()

    cmap = cm.RdBu_r
    ax.scatter(theta, p, s=N*20, c=N, cmap=cmap)

    ax.set_xlabel('theta [rad]')
    ax.set_xlim(-np.pi / 2, np.pi / 2)
    ax.set_ylabel('p')
    ax.set_ylim(-1, 1)

    ax.set_title('Sinogram')
    ax.axis(aspect='equal')

    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array(N)
    ax.get_figure().colorbar(mappable)
    
    return ax


def plot_image(I, title=None, ax=None):
    if ax is None:
        _, ax = plt.subplots()

    ax.imshow(I, cmap='bone')

    if title is not None:
        ax.set_title(title)
