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
from numba import njit, float64, prange
from numba.typed import List
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
        # https://stackoverflow.com/questions/809362/how-to-calculate-cumulative-normal-distribution
        # 'Cumulative distribution function for the standard normal distribution'
        # Scale and shift the x value
        # https: // www.boost.org / doc / libs / 1_38_0 / libs / math / doc / sf_and_dist / html / math_toolkit / dist / dist_ref / dists / normal_dist.html
        # x = (x - self.loc) / self.scale
        return (math.erfc(-(x - self.loc) / (self.scale * np.sqrt(2.0)))) / 2.0

@njit(fastmath=True)
def tnf_euclidean(a, b):

    # l = length_weighting(a[0], b[0])
    # rp = max(a[0], b[0], 1)

    result = 0.0
    for i in range(a.shape[0]):
        result += (a[i] - b[i]) ** 2
        
    d = result ** (1/2)
    # if rp > 1:
    d = d
    
    return d

@njit(fastmath=True)
def euclidean(a, b):

    result = 0.0
    for i in range(a.shape[0] - 1):
        result += (a[i + 1] - b[i + 1]) ** 2

    return np.sqrt(result)


@njit(fastmath=True)
def length_weighting(a, b):
    return min(np.log(a), np.log(b)) / (max(np.log(a), np.log(b)))

@njit(fastmath=True)
def tnf_correlation(a, b):
    rp =  max(a[0], b[0])
    # l = 0
    x = a[1:]
    y = b[1:]
    mu_x = 0.0
    mu_y = 0.0
    norm_x = 0.0
    norm_y = 0.0
    dot_product = 0.0

    for i in range(x.shape[0]):
        mu_x += x[i]
        mu_y += y[i]

    mu_x /= x.shape[0]
    mu_y /= x.shape[0]

    for i in range(x.shape[0]):
        shifted_x = x[i] - mu_x
        shifted_y = y[i] - mu_y
        norm_x += shifted_x ** 2
        norm_y += shifted_y ** 2
        dot_product += shifted_x * shifted_y

    if norm_x == 0.0 and norm_y == 0.0:
        return 0.0
    elif dot_product == 0.0:
        return 1.0
    else:
        return rp * (1.0 - (dot_product / np.sqrt(norm_x * norm_y)))


@njit(fastmath=True)
def inverse_correlation(a, b):
    rp =  max(a[0], b[0]) ** min(a[0], b[0])
    # l = 0
    x = a[1:]
    y = b[1:]
    mu_x = 0.0
    mu_y = 0.0
    norm_x = 0.0
    norm_y = 0.0
    dot_product = 0.0

    for i in range(x.shape[0]):
        mu_x += x[i]
        mu_y += y[i]

    mu_x /= x.shape[0]
    mu_y /= x.shape[0]

    for i in range(x.shape[0]):
        shifted_x = x[i] - mu_x
        shifted_y = y[i] - mu_y
        norm_x += shifted_x ** 2
        norm_y += shifted_y ** 2
        dot_product += shifted_x * shifted_y

    if norm_x == 0.0 and norm_y == 0.0:
        return 1.0
    elif dot_product == 0.0:
        return 0.0
    else:
        return (dot_product / np.sqrt(norm_x * norm_y) + 1) / rp

@njit(fastmath=True)
def tnf_spearman(a, b):
    rp = max(a[0], b[0])

    # Sum of sqaured difference between ranks
    result = 0.0
    for i in range(a.shape[0] - 1):
        result += (a[i + 1] - b[i + 1]) ** 2

    n = len(a[1:])
    
    spman = 1 -  6 * result / (n*(n**2 - 1))
    # we would use spman = 1 - spman for actual correlation
    spman += 1
    spman = 2 - spman
    
    if a[0] > 1 and b[0] > 1:
       spman = spman * a[0] * b[0]
    elif rp > 1:
        spman = spman * rp

    return spman
    
@njit(fastmath=True)
def tnf_kendall_tau(x, y):
    rp = max(y[0], x[0])
    values1 = x[1:]
    values2 = y[1:]
    
    n = len(values1)
    d = 0
    for i in range(0, n):
        a = values1[i] - values2[i] # 
        b = values2[i] - values1[i]
        if a * b < 0:
            d += 1

    d = d / (n * (n - 1) / 2)
    if rp > 1:
        d = d * rp
    return d
      


@njit()
def hellinger_distance_normal(a, b, n_samples, sample_distances):
    """
    a - The mean and variance vec for contig a over n_samples
    b - The mean and variance vec for contig b over n_samples

    returns average hellinger distance of multiple normal distributions
    """

    # generate distirbutions for each sample
    # and calculate divergence between them
  
    # Get the means and variances for each contig
    a_means = a[::2]
    a_vars = a[1::2]
    b_means = b[::2]
    b_vars = b[1::2]
    h_geom_mean = []
    both_present = []

    for i in range(0, n_samples):
        # Use this indexing method as zip does not seem to work so well in njit
        # Add tiny value to each to avoid division by zero
        a_mean = a_means[i] + 1e-6
        a_var = a_vars[i] + 1e-6
        b_mean = b_means[i] + 1e-6
        b_var = b_vars[i] + 1e-6
        if a_mean > 1e-6 and b_mean > 1e-6:
            both_present.append(i)
        # First component of hellinger distance
        h1 = np.sqrt(((2*np.sqrt(a_var)*np.sqrt(b_var)) / (a_var + b_var)))

        h2 = math.exp(-0.25 * ((a_mean - b_mean)**2) / (a_var + b_var))

        h_geom_mean.append(1 - h1 * h2)

    if len(h_geom_mean) >= 1:
    
        # convert to log space to avoid overflow errors
        d = np.log(np.array(h_geom_mean))
        # return the geometric mean
        d = np.exp(d.sum() / len(d))
        geom_sim = geom_sim_calc(both_present, sample_distances)
        d = d ** (1/geom_sim)
    else:
        d = 1
    
    return d


@njit()
def hellinger_distance_poisson(a, b, n_samples, sample_distances):
    """
    a - The mean and variance vec for contig a over n_samples
    b - The mean and variance vec for contig b over n_samples

    returns average hellinger distance of multiple normal distributions
    """

    # generate distirbutions for each sample
    # and calculate divergence between them

    # Get the means for each contig
    a_means = a[::2]
    b_means = b[::2]
    h_geom_mean = []
    both_present = []
    
    for i in range(0, n_samples):
        # Use this indexing method as zip does not seem to work so well in njit
        # Add tiny value to each to avoid division by zero
        a_mean = a_means[i] + 1e-6
        b_mean = b_means[i] + 1e-6
        
        if a_mean > 1e-6 and b_mean > 1e-6:
            both_present.append(i)

        if a_mean > 1e-6 or b_mean > 1e-6:
            # First component of hellinger distance
            h1 = math.exp(-0.5 * ((np.sqrt(a_mean) - np.sqrt(b_mean))**2))

            h_geom_mean.append(1 - h1)
        
    if len(h_geom_mean) >= 1:
        # convert to log space to avoid overflow errors
        d = np.log(np.array(h_geom_mean))
        # return the geometric mean
        d = np.exp(d.sum() / len(d))
        geom_sim = geom_sim_calc(both_present, sample_distances)
        d = d ** (1/geom_sim)
    else:
        d = 1
    
    return d

@njit(fastmath=True)
def hellinger_distance_poisson_variants(a_means, b_means, n_samples, sample_distances):
    """
    a - The coverage vec for a variant over n_samples
    b - The coverage vec for a variant over n_samples

    returns average hellinger distance of multiple poisson distributions
    """

    # generate distirbutions for each sample
    # and calculate divergence between them

    # Get the means for each contig
    h_geom_mean = []
    both_present = []

    for i in range(0, n_samples):
        # Use this indexing method as zip does not seem to work so well in njit
        # Add tiny value to each to avoid division by zero
        a_mean = a_means[i] + 1e-6
        b_mean = b_means[i] + 1e-6

        if a_mean > 1e-6 and b_mean > 1e-6:
            both_present.append(i)

        if a_mean > 1e-6 or b_mean > 1e-6:
            # First component of hellinger distance
            h1 = math.exp(-0.5 * ((np.sqrt(a_mean) - np.sqrt(b_mean))**2))

            h_geom_mean.append(1 - h1)

    if len(h_geom_mean) >= 1:
        # convert to log space to avoid overflow errors
        d = np.log(np.array(h_geom_mean))
        # return the geometric mean
        d = np.exp(d.sum() / len(d))
        geom_sim = geom_sim_calc(both_present, sample_distances)
        d = d ** (1/geom_sim)
    else:
        d = 1

    return d

@njit(fastmath=True)
def metabat_distance(a, b):
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

    both_present = []  # sample indices where both a and b were present
    # only_a = []
    # only_b = []
    n_samples = len(a_means)
    for i in range(0, n_samples):
        # Use this indexing method as zip does not seem to work so well in njit
        # Add tiny value to each to avoid division by zero
        a_mean = a_means[i] + 1e-6
        a_var = a_vars[i] + 1e-6
        b_mean = b_means[i] + 1e-6
        b_var = b_vars[i] + 1e-6

        if a_mean > 1e-6 and b_mean > 1e-6:
            # both_present[i] = True
            both_present.append(i)

        if (a_mean > 1e-6 or b_mean > 1e-6) and a_mean != b_mean:
            if a_var < 1:
                a_var = 1
            if b_var < 1:
                b_var = 1

            if abs(a_var - b_var) < 1e-4:
                k1 = k2 = (a_mean + b_mean) / 2
            else:
                tmp = np.sqrt(a_var * b_var * ((a_mean - b_mean) * (a_mean - b_mean) - 2 * (a_var - b_var) * np.log(
                    np.sqrt(b_var / a_var))))
                k1 = (tmp - a_mean * b_var + b_mean * a_var) / (a_var - b_var)
                k2 = (tmp + a_mean * b_var - b_mean * a_var) / (b_var - a_var)

            if k1 > k2:
                tmp = k1
                k1 = k2
                k2 = tmp

            if a_var > b_var:
                # p1 = NormalDist(b_mean, np.sqrt(b_var))
                p1 = (b_mean, np.sqrt(b_var))
                # p2 = NormalDist(a_mean, np.sqrt(a_var))
                p2 = (a_mean, np.sqrt(a_var))
            else:
                # p1 = NormalDist(a_mean, np.sqrt(a_var))
                p1 = (a_mean, np.sqrt(a_var))
                # p2 = NormalDist(b_mean, np.sqrt(b_var))
                p2 = (b_mean, np.sqrt(b_var))

            if k1 == k2:
                d = abs(cdf(p1, k1) - cdf(p2, k1))
                # mb_vec[i] = min(max(d, 1e-6), 1 - 1e-6)
                mb_vec.append(min(max(d, 1e-6), 1 - 1e-6))
                # mb_vec.append(d)
            else:
                d = abs(cdf(p1, k2) - cdf(p1, k1) + cdf(p2, k1) - cdf(p2, k2))
                # mb_vec[i] = min(max(d, 1e-6), 1 - 1e-6)
                mb_vec.append(min(max(d, 1e-6), 1 - 1e-6))
                # mb_vec.append(d)
        # elif a_mean == b_mean:
        #     # mb_vec[i] = min(max(d, 1e-6), 1 - 1e-6)
        #     mb_vec.append(1e-6)
        #     # pass
        else:
            pass

    return mb_vec


def coverage_distance(a, b):
    if np.array_equal(a, b):
        return 0
    try:
        mb_vec = metabat_distance(a, b)
        if len(mb_vec) >= 1:
            # convert to log space to avoid overflow errors
            d = np.log(np.array(mb_vec))
            # return the geometric mean
            d = np.exp(d.sum() / len(mb_vec))
            return d
        else:
            return 1
    except ZeroDivisionError:
        return 1

@njit(fastmath=True)
def metabat_distance_nn(a, b):
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

    both_present = []  # sample indices where both a and b were present
    # only_a = []
    # only_b = []
    n_samples = len(a_means)
    for i in range(0, n_samples):
        # Use this indexing method as zip does not seem to work so well in njit
        # Add tiny value to each to avoid division by zero
        a_mean = a_means[i] + 1e-6
        a_var = a_vars[i] + 1e-6
        b_mean = b_means[i] + 1e-6
        b_var = b_vars[i] + 1e-6

        if a_mean > 1e-6 and b_mean > 1e-6:
            # both_present[i] = True
            both_present.append(i)

        if (a_mean > 1e-6 or b_mean > 1e-6) and a_mean != b_mean:
            if a_var < 1:
                a_var = 1
            if b_var < 1:
                b_var = 1

            if abs(b_var - a_var) < 1e-4:
                k1 = k2 = (a_mean + b_mean) / 2
            else:
                tmp = np.sqrt(a_var * b_var * ((a_mean - b_mean) * (a_mean - b_mean) - 2 * (a_var - b_var) * np.log(
                    np.sqrt(b_var / a_var))))
                k1 = (tmp - a_mean * b_var + b_mean * a_var) / (a_var - b_var)
                k2 = (tmp + a_mean * b_var - b_mean * a_var) / (b_var - a_var)

            if k1 > k2:
                tmp = k1
                k1 = k2
                k2 = tmp

            if a_var > b_var:
                # p1 = NormalDist(b_mean, np.sqrt(b_var))
                p1 = (b_mean, np.sqrt(b_var))
                # p2 = NormalDist(a_mean, np.sqrt(a_var))
                p2 = (a_mean, np.sqrt(a_var))
            else:
                # p1 = NormalDist(a_mean, np.sqrt(a_var))
                p1 = (a_mean, np.sqrt(a_var))
                # p2 = NormalDist(b_mean, np.sqrt(b_var))
                p2 = (b_mean, np.sqrt(b_var))

            if k1 == k2:
                d = abs(cdf(p1, k1) - cdf(p2, k1))
                # mb_vec[i] = min(max(d, 1e-6), 1 - 1e-6)
                mb_vec.append(min(max(d, 1e-6), 1 - 1e-6))
                # mb_vec.append(d)
            else:
                d = abs(cdf(p1, k2) - cdf(p1, k1) + cdf(p2, k1) - cdf(p2, k2))
                # mb_vec[i] = min(max(d, 1e-6), 1 - 1e-6)
                mb_vec.append(min(max(d, 1e-6), 1 - 1e-6))
                # mb_vec.append(d)
        # elif a_mean == b_mean:
        #     # mb_vec[i] = min(max(d, 1e-6), 1 - 1e-6)
        #     mb_vec.append(1e-6)
        #     # pass
        else:
            pass

    if len(mb_vec) >= 1:
        # convert to log space to avoid overflow errors
        d = np.log(np.array(mb_vec))
        # return the geometric mean
        d = np.exp(d.sum() / len(mb_vec))

    else:
        d = 1

    return d

#function only method for getting CDF of normal distribution
# since jitclass objects aren't pickleable
@njit(fastmath=True)
def cdf(dist, x):
    """
    @ param: dist is a tuple object of (f64, f64). The left index represents the loc (mean) and the right
             index represents the scale (standard deviation)
    @ param: x is the value being sampled from the distribution

    @ return: cdf value f64
    """
    t = x - dist[0]
    sigma = dist[1]

    y = 0.5 * (1.0 + math.erf(t / (sigma * np.sqrt(2.0))))
    if y > 1.0:
        y = 1.0

    return y

@njit(fastmath=True)
def geom_sim_calc(both_present, sample_distances):
    similarity_vec = []
    result = combinations(both_present, 2)
    for i in result:
        similarity_vec.append(sample_distances[i[0], i[1]])
        
    if len(similarity_vec) >= 1:
        geom_sim = np.log(np.array(similarity_vec))
        geom_sim = np.exp(geom_sim.sum() / len(geom_sim))
    else:
        geom_sim = 1
    return geom_sim


@njit(fastmath=True)
def combinations(pool, r):
    n = len(pool)
    indices = list(range(r))
    empty = not(n and (0 < r <= n))
    results = []
    if not empty:
        result = [pool[i] for i in indices]
        results.append(result)

    while not empty:
        i = r - 1
        while i >= 0 and indices[i] == i + n - r:
            i -= 1
        if i < 0:
            empty = True
        else:
            indices[i] += 1
            for j in range(i+1, r):
                indices[j] = indices[j-1] + 1

            result = [pool[i] for i in indices]
            results.append(result)
    return results


@njit(fastmath=True)
def rho(x, y):
    """
    a - CLR transformed coverage distribution vector a
    b - CLR transformed coverage distribution vector b

    return - This is a transformed, inversed version of rho. Normal rho -1 <= rho <= 1
    transformed rho: 0 <= rho <= 2, where 0 is perfect concordance
    """

    # rp = max(a[0], b[0], 1)
    # rp = 1
    # # l = 0
    # x = a[1:]
    # y = b[1:]
    mu_x = 0.0
    mu_y = 0.0
    norm_x = 0.0
    norm_y = 0.0
    dot_product = 0.0
    total_shared = 0

    for i in range(x.shape[0]):
        mu_x += x[i]
        mu_y += y[i]

    mu_x /= x.shape[0]
    mu_y /= x.shape[0]

    for i in range(x.shape[0]):
        shifted_x = x[i] - mu_x
        shifted_y = y[i] - mu_y
        norm_x += shifted_x ** 2
        norm_y += shifted_y ** 2
        dot_product += shifted_x * shifted_y
    
    norm_x = norm_x / (x.shape[0] - 1)
    norm_y = norm_y / (x.shape[0] - 1)
    dot_product = dot_product / (x.shape[0] - 1)
    vlr = -2 * dot_product + norm_x + norm_y
    rho = 1 - vlr / (norm_x + norm_y)
    rho += 1
    rho = 2 - rho
    
    return rho

@njit(fastmath=True)
def rho_variants(x, y):
    """
    x - CLR transformed coverage distribution vector x
    y - CLR transformed coverage distribution vector y

    return - This is a transformed, inversed version of rho. Normal those -1 <= rho <= 1
    transformed rho: 0 <= rho <= 2, where 0 is perfect concordance
    """

    # if x[0] == y[0] and x[1] == y[1]:
    #     return 2
    #
    # x = x[2:]
    # y = y[2:]

    mu_x = 0.0
    mu_y = 0.0
    norm_x = 0.0
    norm_y = 0.0
    dot_product = 0.0

    for i in range(x.shape[0]):
        mu_x += x[i]
        mu_y += y[i]

    mu_x /= x.shape[0]
    mu_y /= x.shape[0]

    for i in range(x.shape[0]):
        shifted_x = x[i] - mu_x
        shifted_y = y[i] - mu_y
        norm_x += shifted_x ** 2
        norm_y += shifted_y ** 2
        dot_product += shifted_x * shifted_y
    
    norm_x = norm_x / (x.shape[0] - 1)
    norm_y = norm_y / (x.shape[0] - 1)
    dot_product = dot_product / (x.shape[0] - 1)
    vlr = -2 * dot_product + norm_x + norm_y
    rho = 1 - vlr / (norm_x + norm_y)
    rho += 1
    rho = 2 - rho
    
    return rho


@njit(fastmath=True)
def euclidean_variant(a, b):
    # if a[0] == b[0] and a[1] == b[1]:
    #     return 100000000000000000

    # a = a[2:]
    # b = b[2:]

    result = 0.0
    for i in range(a.shape[0] - 1):
        result += (a[i + 1] - b[i + 1]) ** 2

    return np.sqrt(result)



@njit(fastmath=True)
def rho_coverage(a, b):
    """
    a - CLR transformed coverage distribution vector a
    b - CLR transformed coverage distribution vector b

    return - This is a transformed, inversed version of rho. Normal those -1 <= rho <= 1
    transformed rho: 0 <= rho <= 2, where 0 is perfect concordance
    """
    covariance_mat = np.cov(a, b, rowvar=True)
    covariance = covariance_mat[0, 1]
    var_a = covariance_mat[0, 0]
    var_b = covariance_mat[1, 1]
    vlr = -2 * covariance + var_a + var_b
    rho = 1 - vlr / (var_a + var_b)
    rho += 1
    rho = 2 - rho

    return rho


@njit(fastmath=True)
def aggregate_tnf(a, b):
    """
    a, b - concatenated contig depth, variance, and TNF info with contig length at index 0
    n_samples - the number of samples

    returns - an aggregate distance metric between MetaBAT ADP divergence and TNF
    """
    # w = (n_samples) / (n_samples + 1) # weighting by number of samples same as in metabat2

    
    md = metabat_distance(a, b)
    # tnf_dist = rho(a[n_samples*2:], b[n_samples*2:])
    # agg = np.sqrt((md**w) * (tnf_dist**(1-w)))
       
    return md


@njit(fastmath=True)
def aggregate_md(a, b, n_samples, sample_distances):
    """
    a, b - concatenated contig depth, variance, and TNF info with contig length at index 0
    n_samples - the number of samples

    returns - an aggregate distance metric between MetaBAT ADP divergence and TNF
    """
    w = (n_samples) / (n_samples + 1)  # weighting by number of samples same as in metabat2

    md = metabat_distance(a[0:n_samples * 2], b[0:n_samples * 2], n_samples, sample_distances)
    tnf_dist = rho(a[n_samples * 2:], b[n_samples * 2:])
    agg = np.sqrt((md ** w) * (tnf_dist ** (1 - w)))

    return agg


@njit(fastmath=True)
def rho_variants(x, y):
    """
    x - CLR transformed coverage distribution vector x
    y - CLR transformed coverage distribution vector y

    return - This is a transformed, inversed version of rho. Normal those -1 <= rho <= 1
    transformed rho: 0 <= rho <= 2, where 0 is perfect concordance
    """

    mu_x = 0.0
    mu_y = 0.0
    norm_x = 0.0
    norm_y = 0.0
    dot_product = 0.0

    for i in range(x.shape[0]):
        mu_x += x[i]
        mu_y += y[i]

    mu_x /= x.shape[0]
    mu_y /= x.shape[0]

    for i in range(x.shape[0]):
        shifted_x = x[i] - mu_x
        shifted_y = y[i] - mu_y
        norm_x += shifted_x ** 2
        norm_y += shifted_y ** 2
        dot_product += shifted_x * shifted_y

    norm_x = norm_x / (x.shape[0] - 1)
    norm_y = norm_y / (x.shape[0] - 1)
    dot_product = dot_product / (x.shape[0] - 1)
    vlr = -2 * dot_product + norm_x + norm_y
    rho = 1 - vlr / (norm_x + norm_y)
    rho += 1
    rho = 2 - rho

    return rho

@njit(fastmath=True, parallel=False)
def check_connections(current, others, n_samples, sample_distances, rho_threshold=0.05, euc_threshold=3, dep_threshold=0.05):
    rho_connected = False
    euc_connected = False
    dep_connected = False
    columns = n_samples * 2
    for contig_idx in prange(others.shape[0]):
        other = others[contig_idx]
        if not rho_connected:
            rho_value = rho(current[columns + 1:], other[columns + 1:])
            if rho_value <= rho_threshold:
                rho_connected = True

        if not euc_connected:
            euc_value = tnf_euclidean(current[columns + 1:], other[columns + 1:])
            if euc_value <= euc_threshold:
                euc_connected = True

        if not dep_connected:
            dep_value = metabat_distance_nn(current[:n_samples * 2],
                                         other[:n_samples * 2])
            if dep_value <= dep_threshold:
                dep_connected = True

        if euc_connected and rho_connected and dep_connected:
            break

    # rho_bool = rho_connected > 0
    # euc_bool = euc_connected > 0
    # dep_bool = dep_connected > 0

    return rho_connected, euc_connected, dep_connected

@njit(fastmath=True)
def average_values_and_min_values(current, others, n_samples, sample_distances):
    """
    Computes the average distances for a given contig values compared to all other contigs
    """
    rho_sum = 0
    euc_sum = 0
    dep_sum = 0

    rho_min = None
    euc_min = None
    dep_min = None

    for contig_idx in range(others.shape[0]):
        rho_value = rho(current[0, n_samples * 2:],
                        others[contig_idx, n_samples * 2:])
        rho_sum += rho_value
        if rho_min is not None:
            if rho_min > rho_value:
                rho_min = rho_value
        else:
            rho_min = rho_value

        euc_value = tnf_euclidean(current[0, n_samples * 2:],
                                  others[contig_idx, n_samples * 2:])
        euc_sum += euc_value
        if euc_min is not None:
            if euc_min > euc_value:
                euc_min = euc_value
        else:
            euc_min = euc_value

        mb_vec = metabat_distance(current[0, :n_samples * 2],
                                     others[contig_idx, :n_samples * 2])
        if len(mb_vec) >= 1:
            # convert to log space to avoid overflow errors
            dep_value = np.log(np.array(mb_vec))
            # return the geometric mean
            dep_value = np.exp(dep_value.sum() / len(mb_vec))
        else:
            dep_value = 1
        dep_sum += dep_value
        if dep_min is not None:
            if dep_min > dep_value:
                dep_min = dep_value
        else:
            dep_min = dep_value

    rho_mean = rho_sum / others.shape[0]
    euc_mean = euc_sum / others.shape[0]
    dep_mean = dep_sum / others.shape[0]

    return (rho_mean, euc_mean, dep_mean), (rho_min, euc_min, dep_min)

@njit(fastmath=True)
def get_single_contig_averages(contig_depth, depths, n_samples, sample_distances):
    values = List([0.0, 0.0, 0.0, 0.0])
    # distances = np.zeros((depths.shape[0], depths.shape[0]))
    w = (n_samples) / (n_samples + 1)  # weighting by number of samples same as in metabat2

    for i in range(depths.shape[0]):
        mb_vec = metabat_distance(depths[i[0], :n_samples * 2], depths[i[1], :n_samples * 2])
        if len(mb_vec) >= 1:
            # convert to log space to avoid overflow errors
            md = np.log(np.array(mb_vec))
            # return the geometric mean
            md = np.exp(md.sum() / len(mb_vec))
        else:
            md = 1
        tnf_dist = rho(contig_depth[0, n_samples * 2:], depths[i, n_samples * 2:])
        tnf_euc = tnf_euclidean(contig_depth[0, n_samples * 2:], depths[i, n_samples * 2:])
        agg = np.sqrt((md ** w) * (tnf_dist ** (1 - w)))

        values[0] += md
        values[1] += tnf_dist
        values[2] += tnf_euc
        values[3] += agg

    values[0] /= depths.shape[0]
    values[1] /= depths.shape[0]
    values[2] /= depths.shape[0]
    values[3] /= depths.shape[0]

    return values

@njit(fastmath=True)
def get_averages(depths, n_samples, sample_distances):
    contigs = List()
    tids = List()
    # distances = np.zeros((depths.shape[0], depths.shape[0]))
    w = (n_samples) / (n_samples + 1) # weighting by number of samples same as in metabat2

    [(contigs.append(List([0.0, 0.0, 0.0, 0.0])), tids.append(x)) for x in range(depths.shape[0])]

    pairs = combinations(tids, 2)
    mean_md = 0
    mean_tnf = 0
    mean_euc = 0
    mean_agg = 0

    col_start = n_samples * 2
    
    for i in pairs:
        mb_vec = metabat_distance(depths[i[0], :col_start], depths[i[1], :col_start])
        if len(mb_vec) >= 1:
            # convert to log space to avoid overflow errors
            md = np.log(np.array(mb_vec))
            # return the geometric mean
            md = np.exp(md.sum() / len(mb_vec))
        else:
            md = 1
        tnf_dist = rho(depths[i[0], col_start + 1:], depths[i[1], col_start + 1:])
        tnf_euc = tnf_euclidean(depths[i[0], col_start + 1:], depths[i[1], col_start + 1:])

        agg = np.sqrt((md**w) * (tnf_dist**(1-w)))

        mean_md += md
        mean_tnf += tnf_dist
        mean_euc += tnf_euc
        mean_agg += agg

        contigs[i[0]][0] += md
        contigs[i[0]][1] += tnf_dist
        contigs[i[0]][2] += tnf_euc
        contigs[i[0]][3] += agg


        contigs[i[1]][0] += md
        contigs[i[1]][1] += tnf_dist
        contigs[i[1]][2] += tnf_euc
        contigs[i[1]][3] += agg


    for i in range(len(contigs)):
        contigs[i][0] /= (len(contigs) - 1)
        contigs[i][1] /= (len(contigs) - 1)
        contigs[i][2] /= (len(contigs) - 1)
        contigs[i][3] /= (len(contigs) - 1)
                
    mean_md = mean_md / len(pairs)
    mean_tnf = mean_tnf / len(pairs)
    mean_euc = mean_euc / len(pairs)
    mean_agg = mean_agg / len(pairs)
    
    return mean_md, mean_tnf, mean_euc, mean_agg, contigs


@njit(fastmath=True)
def distance_matrix(depths, n_samples, sample_distances):
    w = n_samples / (n_samples + 1) # weighting by number of samples same as in metabat2
    distances = np.zeros((depths.shape[0], depths.shape[0]))
    tids = List()
    [tids.append(x) for x in range(depths.shape[0])]

    pairs = combinations(tids, 2)
    mean_md = 0.0
    mean_tnf = 0.0
    mean_agg = 0.0
    
    for i in pairs:
        mb_vec = metabat_distance(depths[i[0], :n_samples * 2], depths[i[1], :n_samples * 2])
        if len(mb_vec) >= 1:
            # convert to log space to avoid overflow errors
            md = np.log(np.array(mb_vec))
            # return the geometric mean
            md = np.exp(md.sum() / len(mb_vec))
        else:
            md = 1
        tnf_dist = rho(depths[i[0], n_samples*2:], depths[i[1], n_samples*2:])

        agg = np.sqrt((md**w) * (tnf_dist**(1-w)))

        mean_md += md
        mean_tnf += tnf_dist
        mean_agg += agg

        distances[i[0]][i[1]] = agg
        distances[i[1]][i[0]] = agg
    
    return distances


