#!/usr/bin/env python
###############################################################################
# metrics.py - File containing additonal distance metrics for use with UMAP
###############################################################################
#                                                                             #
# This program is free software: you can redistribute it and/or modify        #
# it under the terms of the GNU General Public License as published by        #
# the Free Software Foundation, either version 3 of the License, or           #
# (at your option) any later version.                                         #
#                                                                             #
# This program is distributed in the hope that it will be useful,             #
# but WITHOUT ANY WARRANTY; without even the implied warranty of              #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the                #
# GNU General Public License for more details.                                #
#                                                                             #
# You should have received a copy of the GNU General Public License           #
# along with this program. If not, see <http://www.gnu.org/licenses/>.        #
#                                                                             #
###############################################################################
__author__ = "Rhys Newell"
__copyright__ = "Copyright 2020"
__credits__ = ["Rhys Newell"]
__license__ = "GPL3"
__maintainer__ = "Rhys Newell"
__email__ = "rhys.newell near hdr.qut.edu.au"
__status__ = "Development"

###############################################################################
# System imports
import sys
import argparse
import logging
import os
import shutil
import datetime

# Function imports
import numba
import numpy as np
import math

###############################################################################                                                                                                                      [44/1010]
################################ - Functions - ################################

@numba.njit()
def tnf(a, b, n_samples):
    cov_mat = np.cov(a[n_samples:], b[n_samples:])
    cov = cov_mat[0, 1]
    a_sd = np.sqrt(cov_mat[0,0])
    b_sd = np.sqrt(cov_mat[1,1])
    rho = cov / (a_sd * b_sd)
    rho += 1
    rho = 2 - rho
    return rho
    # L2 norm is equivalent to euclidean distance
    # euc_dist = np.linalg.norm(a[n_samples:] - b[n_samples:])

    # return euc_dist

@numba.njit()
def kl_divergence(a, b, n_samples):
    """
    a - The mean and variance vec for contig a over n_samples
    b - The mean and variance vec for contig b over n_samples

    returns the geometric mean of the KL divergences for contigs a and b over n_samples
    """
    a = iter(a)
    b = iter(b)
    # generate distirbutions for each sample
    # and calculate divergence between them
    # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    kl_vec = []
    for a_mean, b_mean in zip(a, b):
        a_var = next(a)
        b_var = next(b)
        kl = np.log(np.sqrt(b_var) / np.sqrt(a_var)) + ((a_var + (a_mean - b_mean)**2) / (2*b_var)) - 1/2
        kl_vec.append(kl)

    # convert to log space to avoid overflow errors
    kl_vec = np.log(kl_vec)
    # return the geometric mean
    return np.exp(kl_vec.sum() / len(kl_vec))

@numba.njit()
def rho(a, b, n_samples):
    """
    a - CLR transformed coverage distribution vector a
    b - CLR transformed coverage distribution vector b

    return - This is a transformed, inversed version of rho. Normal those -1 <= rho <= 1
    transformed rho: 0 <= rho <= 2, where 0 is perfect concordance
    """

    covariance_mat = np.cov(a[:n_samples], b[:n_samples], rowvar=True)
    covariance = covariance_mat[0, 1]
    var_a = covariance_mat[0, 0]
    var_b = covariance_mat[1, 1]
    vlr = -2 * covariance + var_a + var_b
    rho = 1 - vlr / (var_a + var_b)
    rho += 1
    rho = 2 - rho
    # Since these compositonal arrays are CLR transformed
    # This is the equivalent to the aitchinson distance but we calculat the l2 norm
    euc_dist = np.linalg.norm(a[:n_samples] - b[:n_samples])

    dist = min(euc_dist, rho)
    
    return dist

@numba.njit()
def aggregate_tnf(a, b, n_samples):
    """
    a, b - concatenated contig depth, variance, and TNF info with contig length at index 0
    n_samples - the number of samples

    returns - an aggregate distance metric between KL divergence and TNF
    """
    w = n_samples / (n_samples + 1) # weighting by number of samples same as in metabat2
    l = min(a[0], b[0]) / (max(a[0], b[0]) + 1)

    tnf_dist = tnf(a[1:], b[1:], n_samples*2)
    kl = kl_divergence(a[1:n_samples*2 + 1], b[1:n_samples*2 + 1], n_samples)
    agg = (tnf_dist**w) * kl**(1-w)

    return agg
