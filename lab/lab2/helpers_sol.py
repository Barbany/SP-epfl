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
import skimage.transform as transform
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


def sampling_op(xi, p, w, N_height, N_width, window_profile='raised-cosine'):
    """
    Numerical approximation of continuous-domain sampling operator.
    
    Parameters
    ----------
    xi : :py:class:`~numpy.ndarray`
        (N_tube, 2) normal vector to tube. (Must lie in quadrant I or II.)
    p : :py:class:`~numpy.ndarray`
        (N_tube,) tube distance from origin. (Can be negative.)
    w : :py:class:`~numpy.ndarray`
        (N_tube,) tube width.
    N_height : int
        Number of uniform vertical spatial samples in [-1, 1].
    N_width : int
        Number of uniform horizontal spatial samples in [-1, 1].
    window_profile : str
        Shape of the window. 
        Must be one of ['raised-cosine', 'rect', 'tri']

    Returns
    -------
    S : :py:class:`~scipy.sparse.csr_matrix`
        (N_tube, N_height*N_width) sampling operator, where each row contains the basis function of
        the instrument (vectorized row-by-row).
    """
    ### Generate grid
    Y, X = np.meshgrid(np.linspace(-1, 1, N_height), 
                       np.linspace(-1, 1, N_width), indexing='ij')
    V = np.stack((X, Y), axis=-1).reshape((N_height * N_width, 2))
    
    # We only want a regular grid on the circumcircle, hence we throw away
    # all vectors that lie outside the unit circle.
    V[linalg.norm(V, axis=-1) >= 1] = 0

    def window(x):
        """
        Parameters
        ----------
        x : :py:class:`~numpy.ndarray`
            (N_sample,) evaluation points.

        Returns
        -------
        y : :py:class:`~numpy.ndarray`
            (N_sample,) values of window at `x`.
        """
        y = np.zeros_like(x, dtype=float)
        
        if window_profile == 'rect':
            # rect(x) = 1 if (-0.5 <= x <= 0.5)
            mask = (-0.5 <= x) & (x <= 0.5)
            y[mask] = 1
        elif window_profile == 'tri':
            # tri(x) = 1 - abs(x) if (-1 <= x <= 1)
            mask = (-1 <= x) & (x <= 1)
            y[mask] = 1 - np.abs(x[mask])
        elif window_profile == 'raised-cosine':
            # rcos(x) = cos(0.5 * \pi * x) if (-1 <= x <= 1)
            mask = (-1 <= x) & (x <= 1)
            y[mask] = np.cos(0.5 * np.pi * x[mask])            
        else:
            raise ValueError('Parameter[window_profile] is not recognized.')

        return y


    N_tube = len(xi)
    S = sparse.lil_matrix((N_tube, N_height * N_width), dtype=float)
    for i in tqdm.tqdm(np.arange(N_tube)):
        projection = V @ xi[i]
        x = (projection - p[i]) * (2 / w[i])
        mask = ((np.abs(x) <= 1)         &  # inner/outer boundary
                ~np.isclose(projection, 0))    # circular boundary
        S[i, mask.nonzero()] = window(x[mask])
        
    S = S.tocsr(copy=True)
    return S


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


def get_intensity(path_img, N_height, N_width, pad_size, max_rate=1e6):
    """
    Parameters
    ----------
    path_img : str
        Path to RGB image (PNG format).
    N_height : int
        Number of vertical pixels the image should have at the output.
    N_width : int
        Number of horizontal pixels the image should have at the output.
    pad_size : tuple(int, int)
        Symmetric padding [px] around (vertical, horizontal) image dimensions.
    max_rate : float
        Scale factor such that all image intensities lie in [0, `max_rate`].

    Returns
    -------
    lambda_ : :py:class:`~numpy.ndarray`
        (N_height, N_width) intensity.
    """
    lambda_rgb = plt.imread(path_img).astype(float)
    lambda_ = color.rgb2gray(lambda_rgb)
    
    # We pad the image with zeros so that the mask does not touch the detector ring.
    lambda_ = np.pad(lambda_, pad_size, mode='constant')
    lambda_ = transform.resize(lambda_, (N_height, N_width), order=1, mode='constant')
    lambda_ *= max_rate / lambda_.max()  # (N_height, N_width)

    return lambda_


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


def kaczmarz(S, N, N_iter, permute=False, I0=None):
    """
    Form image via Kaczmarz's algorithm.

    Parameters
    ----------
    S : :py:class:`~scipy.sparse.csr_matrix`
        (N_tube, N_px) sampling operator.
    N : :py:class:`~numpy.ndarray`
        (N_tube,) PET samples
    N_iter : int
        Number of iterations to perform.
    I0 : :py:class:`~numpy.ndarray`
        (N_px,) initial point of the optimization.
        If unspecified, the initial point is set to 0.

    Returns
    -------
    I : :py:class:`~numpy.ndarray`
        (N_px,) vectorized image
    """
    N_tube, N_px = S.shape

    I = np.zeros((N_px,), dtype=float) if (I0 is None) else I0.copy()

    if permute:
        index = np.random.permutation(N_iter)
    else:
        index = np.arange(N_iter)
        
    for k in tqdm.tqdm(index):
        idx = k % N_tube
        n, s = N[idx], S[idx].toarray()[0]
        l = s @ s

        if ~np.isclose(l, 0):
            # `l` can be very small, in which case it is dangerous to do the rescale. 
            # We'll simply drop these degenerate basis vectors.
            scale = (n - s @ I) / l
            I += scale * s

    return I
