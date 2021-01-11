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

# Function imports
from numba import njit, float64
from numba.experimental import jitclass
import numpy as np
import math

###############################################################################                                                                                                                      [44/1010]
################################ - Globals - ##################################
spec = [
    ('loc', float64),               # a simple scalar field
    ('scale', float64),          # an array field
]

###############################################################################                                                                                                                      [44/1010]
################################ - Functions - ################################

@jitclass(spec)
class NormalDist:
    def __init__(self, loc=0.0, scale=1.0):
        self.loc = loc
        self.scale = scale

    def cdf(self, x):
        # https: // stackoverflow.com / questions / 809362 / how - to - calculate - cumulative - normal - distribution
        # 'Cumulative distribution function for the standard normal distribution'
        # Scale and shift the x value
        x = (x - self.loc) / self.scale
        return (1.0 + math.erf(x / np.sqrt(2.0))) / 2.0

@njit()
def tnf(a, b, n_samples):
    # cov_mat = np.cov(a[n_samples:], b[n_samples:])
    # cov = cov_mat[0, 1]
    # a_sd = np.sqrt(cov_mat[0,0])
    # b_sd = np.sqrt(cov_mat[1,1])
    # rho = cov / (a_sd * b_sd)
    # rho += 1
    # rho = 2 - rho
    # return rho
    # L2 norm is equivalent to euclidean distance
    # euc_dist = np.linalg.norm(a[n_samples:] - b[n_samples:])

    # return euc_dist
    # Cosine distance
    a_np = a[n_samples:]
    b_np = b[n_samples:]

    cosine = np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np))
    cosine = 1 - cosine
    return cosine

@njit()
def metabat_distance(a, b, n_samples):
    """
    a - The mean and variance vec for contig a over n_samples
    b - The mean and variance vec for contig b over n_samples

    returns distance as defined in metabat1 paper
    """

    # generate distirbutions for each sample
    # and calculate divergence between them
    # 
    mb_vec = []

    # Get the means and variances for each contig
    a_means = a[::2]
    a_vars = a[1::2]
    b_means = b[::2]
    b_vars = b[1::2]

    for i in range(0, n_samples):
        k1, k2, tmp, d = 0, 0, 0, 0
        # Use this indexing method as zip does not seem to work so well in njit
        # Add tiny value to each to avoid division by zero
        a_mean = a_means[i] + 1e-6
        a_var = a_vars[i] + 1e-6
        b_mean = b_means[i] + 1e-6
        b_var = b_vars[i] + 1e-6

        if abs(a_var - b_var) < 1e-4:
            k1 = k2 = (a_mean + b_mean) / 2
        else:
            tmp = np.sqrt(a_var * b_var * ((a_mean - b_mean) * (a_mean - b_mean) - 2 * (a_var - b_var) * np.log(np.sqrt(b_var / a_var))))
            k1 = (tmp - a_mean * b_var + b_mean * a_var) / (a_var - b_var)
            k2 = (tmp + a_mean * b_var - b_mean * a_var) / (b_var - a_var)

        if k1 > k2:
            tmp = k1
            k1 = k2
            k2 = tmp

        if a_var > b_var:
            p1 = NormalDist(b_mean, np.sqrt(b_var))
            p2 = NormalDist(a_mean, np.sqrt(a_var))
        else:
            p1 = NormalDist(a_mean, np.sqrt(a_var))
            p2 = NormalDist(b_mean, np.sqrt(b_var))

        if k1 == k2:
            mb_vec.append((abs(p1.cdf(k1) - p2.cdf(k1))))
        else:
            mb_vec.append((p1.cdf(k2) - p1.cdf(k1) + p2.cdf(k1) - p2.cdf(k2)))

    # convert to log space to avoid overflow errors
    mb_vec = np.log(np.array(mb_vec))
    # return the geometric mean
    return np.exp(mb_vec.sum() / len(mb_vec))
    

@njit()
def kl_divergence(a, b, n_samples):
    """
    a - The mean and variance vec for contig a over n_samples
    b - The mean and variance vec for contig b over n_samples

    returns the geometric mean of the KL divergences for contigs a and b over n_samples
    """

    # generate distirbutions for each sample
    # and calculate divergence between them
    # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    kl_vec = []

    # Get the means and variances for each contig
    a_means = a[::2]
    a_vars = a[1::2]
    b_means = b[::2]
    b_vars = b[1::2]

    for i in range(0, n_samples):
        # Use this indexing method as zip does not seem to work so well in njit
        # Add tiny value to each to avoid division by zero
        a_mean = a_means[i] + 1e-6
        a_var = a_vars[i] + 1e-6
        b_mean = b_means[i] + 1e-6
        b_var = b_vars[i] + 1e-6
        
        kl_1 = np.log(np.sqrt(b_var) / np.sqrt(a_var)) + ((a_var + (a_mean - b_mean)**2) / (2*b_var)) - 1/2
        kl_2 = np.log(np.sqrt(a_var) / np.sqrt(b_var)) + ((b_var + (b_mean - a_mean)**2) / (2*a_var)) - 1/2
        kl_sym = (kl_1 + kl_2)/2
        
        kl_vec.append(kl_sym)

    # convert to log space to avoid overflow errors
    kl_vec = np.log(np.array(kl_vec))
    # return the geometric mean
    return np.exp(kl_vec.sum() / len(kl_vec))

@njit()
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

@njit()
def pearson(a, b):
    return np.corrcoef(a, b)[0, 1]

@njit()
def aggregate_tnf(a, b, n_samples):
    """
    a, b - concatenated contig depth, variance, and TNF info with contig length at index 0
    n_samples - the number of samples

    returns - an aggregate distance metric between KL divergence and TNF
    """
    w = n_samples / (n_samples + 1) # weighting by number of samples same as in metabat2
    l = min(a[0], b[0]) / (max(a[0], b[0]) + 1)

    tnf_dist = tnf(a[1:], b[1:], n_samples*2)
    kl = metabat_distance(a[1:n_samples*2 + 1], b[1:n_samples*2 + 1], n_samples)
    if n_samples < 3:
        agg = (tnf_dist**(1-w)) * kl**(w)
    else:
        agg = np.sqrt((tnf_dist**(1-w)) * (kl**(w)) * pearson(a[1:n_samples*2 + 1][0::2], b[1:n_samples*2 + 1][0::2]))

    return agg
