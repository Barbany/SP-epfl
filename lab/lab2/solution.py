# ##############################################################################
# solution.py
# ===========
# Author : Sepand KASHANI [sepand.kashani@epfl.ch]
# ##############################################################################

"""
Running example of simulated PET imaging.
"""

import matplotlib.pyplot as plt
import numpy as np

import helpers


# Tube/image parameters
N_detector = 80
N_h = N_w = 256
N_px = N_h * N_w


# Building the sampling operator.
d_x, d_xi, d_p, d_w = helpers.tube_decomposition(N_detector)
S = helpers.sampling_op(d_xi, d_p, d_w, N_h, N_w, window_profile='raised-cosine')
N_tube = S.shape[0]


# Draw some tubes to see what the basis functions look like.
fig, ax = plt.subplots()
helpers.draw_tubes(S, N_h, N_w, idx=[132, 1800], ax=ax)
fig.show()


# Generate some PET measurements from an image (i.e. Poisson \lambda parameter per pixel.)
path_img = './img/phantom_3.png' # check the img/ directory for alternatives
lambda_ = helpers.get_intensity(path_img, N_h, N_w, pad_size=np.r_[N_h, N_w] // 3)  # (N_h, N_w)
sample = S @ lambda_.reshape((N_h * N_w,))


# Plot Sinogram to see if the sinusoidal patterns are visible.
fig, (ax_0, ax_1) = plt.subplots(nrows=1, ncols=2)
helpers.plot_image(lambda_, 'ground truth', ax_0)
helpers.sinogram(d_xi, d_p, sample, ax_1)
fig.show()


# Probably hard to see the sinusoidal patterns above because the image is large.
# Let's try the same on an image that consists only of a few point sources.
lambda_point = np.zeros((N_h, N_w), dtype=float)
lambda_point[np.random.randint(N_h, size=5), np.random.randint(N_w, size=5)] = 1
sample_point = S @ lambda_point.reshape((N_h * N_w,))
fig, (ax_0, ax_1) = plt.subplots(nrows=1, ncols=2)
helpers.plot_image(lambda_point, 'ground truth', ax_0)
helpers.sinogram(d_xi, d_p, sample_point, ax_1)
fig.show()


# Image the data using different reconstruction algorithms.
N_iter = 5 * N_tube  # for Kaczmarz's algorithm
I_bp        = helpers.backprojection(S, sample).reshape((N_h, N_w))
I_lsq       = helpers.least_squares(S, sample).reshape((N_h, N_w))                   # Might take 30[s]
I_lsq_r     = helpers.least_squares(S, sample, regularize=True).reshape((N_h, N_w))  # Might take 30[s]
I_kacz      = helpers.kaczmarz(S, sample, N_iter).reshape((N_h, N_w))
I_kacz_perm = helpers.kaczmarz(S, sample, N_iter, permute=True).reshape((N_h, N_w))


# And finally plot them all to see the differences.
fig, ax = plt.subplots(nrows=2, ncols=3)
helpers.plot_image(lambda_, 'ground truth', ax[0, 0])
helpers.plot_image(I_bp, 'backprojection', ax[1, 0])
helpers.plot_image(I_lsq, 'least-squares', ax[0, 1])
helpers.plot_image(I_lsq_r, 'least-squares (regularized)', ax[1, 1])
helpers.plot_image(I_kacz, f'Kaczmarz (in-order, {N_iter} iter)', ax[0, 2])
helpers.plot_image(I_kacz_perm, f'Kaczmarz (out-of-order, {N_iter} iter)', ax[1, 2])
fig.show()
