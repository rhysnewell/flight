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
def tnf_euclidean(a, b):
    # cov_mat = np.cov(a[n_samples:], b[n_samples:])
    # cov = cov_mat[0, 1]
    # a_sd = np.sqrt(cov_mat[0,0])
    # b_sd = np.sqrt(cov_mat[1,1])
    # rho = cov / (a_sd * b_sd)
    # rho += 1
    # rho = 2 - rho
    # return rho
    l = length_weighting(a[0], b[0])
    # L2 norm is equivalent to euclidean distance
    result = 0.0
    for i in range(a.shape[0] - 1):
        result += (a[i + 1] - b[i + 1]) ** 2

    return np.sqrt(result)


@njit()
def length_weighting(a, b):
    return min(np.log(a), np.log(b)) / (max(np.log(a), np.log(b)))

@njit()
def tnf_correlation(a, b):
    l = length_weighting(a[0], b[0])
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
        return (1.0 - (dot_product / np.sqrt(norm_x * norm_y)))**(1-l)

@njit()
def hellinger_distance_normal(a, b, n_samples):
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
    
    for i in range(0, n_samples):
        # Use this indexing method as zip does not seem to work so well in njit
        # Add tiny value to each to avoid division by zero
        a_mean = a_means[i] + 1e-6
        a_var = a_vars[i] + 1e-6
        b_mean = b_means[i] + 1e-6
        b_var = b_vars[i] + 1e-6

        # First component of hellinger distance
        h1 = np.sqrt(((2*np.sqrt(a_var)*np.sqrt(b_var)) / (a_var + b_var)))

        h2 = math.exp(-0.25 * ((a_mean - b_mean)**2) / (a_var + b_var))

        h_geom_mean.append(1 - h1 * h2)
        
    # convert to log space to avoid overflow errors
    h_geom_mean = np.log(np.array(h_geom_mean))
    # return the geometric mean
    return np.exp(h_geom_mean.sum() / len(h_geom_mean))


@njit()
def hellinger_distance_poisson(a, b, n_samples):
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
    
    for i in range(0, n_samples):
        # Use this indexing method as zip does not seem to work so well in njit
        # Add tiny value to each to avoid division by zero
        a_mean = a_means[i] + 1e-6
        b_mean = b_means[i] + 1e-6

        # First component of hellinger distance
        h1 = math.exp(-0.5 * ((np.sqrt(a_mean) - np.sqrt(b_mean))**2))

        h_geom_mean.append(1 - h1)
        
    # convert to log space to avoid overflow errors
    h_geom_mean = np.log(np.array(h_geom_mean))
    # return the geometric mean
    return np.exp(h_geom_mean.sum() / len(h_geom_mean)) 

@njit()
def metabat_distance(a, b, n_samples, sample_distances):
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

    both_present = [] # sample indices where both a and b were present
    # only_a = []
    # only_b = []

    for i in range(0, n_samples):
        # Use this indexing method as zip does not seem to work so well in njit
        # Add tiny value to each to avoid division by zero
        a_mean = a_means[i] + 1e-6
        a_var = a_vars[i] + 1e-6
        b_mean = b_means[i] + 1e-6
        b_var = b_vars[i] + 1e-6
        if a_mean > 1e-6 and b_mean > 1e-6:
            both_present.append(i)

        if a_mean > 1e-6 or b_mean > 1e-6:
            if a_var < 1:
                a_var = 1
            if b_var < 1:
                b_var = 1


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
    
    if len(mb_vec) >= 1:
        # convert to log space to avoid overflow errors
        mb_vec = np.log(np.array(mb_vec))
        # return the geometric mean
        d = np.exp(mb_vec.sum() / len(mb_vec))

        # Calculate geometric mean of sample distances
        geom_sim = geom_sim_calc(both_present, sample_distances)
        # geom_sim = 1
        d = d ** (1/geom_sim)
    else:
        d = 1

    return d

@njit()
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

@njit()
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
def rho(a, b):
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
    # Since these compositonal arrays are CLR transformed
    # This is the equivalent to the aitchinson distance but we calculat the l2 norm
    # euc_dist = np.linalg.norm(a[:n_samples] - b[:n_samples])

    # dist = min(euc_dist, rho)
    
    return rho

@njit()
def pearson(a, b):
    return np.corrcoef(a, b)[0, 1]

@njit()
def aggregate_tnf(a, b, n_samples, sample_distances):
    """
    a, b - concatenated contig depth, variance, and TNF info with contig length at index 0
    n_samples - the number of samples

    returns - an aggregate distance metric between KL divergence and TNF
    """
    w = n_samples / (n_samples + 1) # weighting by number of samples same as in metabat2
    # l = min(a[0], b[0]) / (max(a[0], b[0]) + 1)

    # tnf_dist = tnf_correlation(a[1:], b[1:], n_samples*2)
    tnf_dist = tnf_euclidean(a[n_samples*2:], b[n_samples*2:])
    kl = metabat_distance(a[0:n_samples*2], b[0:n_samples*2], n_samples, sample_distances)
    # if n_samples < 3:
    agg = ((tnf_dist**(1-w))) * kl**(w)
    # else:
        # agg = np.sqrt((tnf_dist**(1-w)) * (kl**(w)) * pearson(a[1:n_samples*2 + 1][0::2], b[1:n_samples*2 + 1][0::2]))

    return agg


@njit()
def metabat_tdp(a, b):
    """
    a, b - concatenated TNF info with contig length at index 0    
    returns - A correlation distance weighted by contig lengths
    """
    lw11 = min(a[0], b[0])
    lw21 = max(a[0], b[0])
    lw12 = lw11 * lw11
    lw13 = lw12 * lw11
    lw14 = lw13 * lw11
    lw15 = lw14 * lw11
    lw16 = lw15 * lw11
    lw17 = lw16 * lw11
    lw22 = lw21 * lw21
    lw23 = lw22 * lw21
    lw24 = lw23 * lw21
    lw25 = lw24 * lw21
    lw26 = lw25 * lw21

    param1 = 46349.1624324381 + -76092.3748553155 * lw11 + -639.918334183 * lw21 + 53873.3933743949 * lw12 + -156.6547554844 * lw22 + -21263.6010657275 * lw13 + 64.7719132839 * lw23 + 5003.2646455284 * lw14 + -8.5014386744 * lw24 + -700.5825500292 * lw15 + 0.3968284526 * lw25 + 54.037542743 * lw16 + -1.7713972342 * lw17 + 474.0850141891 * lw11 * lw21 + -23.966597785 * lw12 * lw22 + 0.7800219061 * lw13 * lw23 + -0.0138723693 * lw14 * lw24 + 0.0001027543 * lw15 * lw25
    param2 = -443565.465710869 + 718862.10804858 * lw11 + 5114.1630934534 * lw21 + -501588.206183097 * lw12 + 784.4442123743 * lw22 + 194712.394138513 * lw13 + -377.9645994741 * lw23 + -45088.7863182741 * lw14 + 50.5960513287 * lw24 + 6220.3310639927 * lw15 + -2.3670776453 * lw25 + -473.269785487 * lw16 + 15.3213264134 * lw17 + -3282.8510348085 * lw11 * lw21 + 164.0438603974 * lw12 * lw22 + -5.2778800755 * lw13 * lw23 + 0.0929379305 * lw14 * lw24 + -0.0006826817 * lw15 * lw25

    
    # l1 = ((a[0] + b[0]) / 2 )/ max(a[0], b[0])
    # l2 = min(a[0], b[0]) / (max(a[0], b[0]) + 1)

    
    # tnf_dist = tnf_correlation(a[1:], b[1:], 0)
    # L2 norm is equivalent to euclidean distance
    euc_dist = np.linalg.norm(a[1:] - b[1:])
    
    tnf_dist = -(param1 + param2 * euc_dist)

    prob = 1 / (1 + math.exp(tnf_dist))

    floor_preProb = np.log((1.0 / 0.1) - 1.0)
    
    if prob >= 0.1:
        param1 = 6770.9351457442 + -5933.7589419767 * lw11 + -2976.2879986855 * lw21 + 3279.7524685865 * lw12 + 1602.7544794819 * lw22 + -967.2906583423 * lw13 + -462.0149190219 * lw23 + 159.8317289682 * lw14 + 74.4884405822 * lw24 + -14.0267151808 * lw15 + -6.3644917671 * lw25 + 0.5108811613 * lw16  + 0.2252455343 * lw26 + 0.965040193 * lw12 * lw22 + -0.0546309127 * lw13 * lw23 + 0.0012917084 * lw14 * lw24 + -1.14383e-05 * lw15 * lw25
        param2 = 39406.5712626297 + -77863.1741143294 * lw11 + 9586.8761567725 * lw21 + 55360.1701572325 * lw12 + -5825.2491611377 * lw22 + -21887.8400068324 * lw13 + 1751.6803621934 * lw23 + 5158.3764225203 * lw14 + -290.1765894829 * lw24 + -724.0348081819 * lw15 + 25.364646181 * lw25 + 56.0522105105 * lw16  + -0.9172073892 * lw26 + -1.8470088417 * lw17 + 449.4660736502 * lw11 * lw21 + -24.4141920625 * lw12 * lw22 + 0.8465834103 * lw13 * lw23 + -0.0158943762 * lw14 * lw24 + 0.0001235384 * lw15 * lw25
        preProb = -(param1 + param2 * euc_dist)
        if preProb <= floor_preProb:
            prob = 1.0 / (1 + math.exp(preProb))
        else:
            prob = 0.1    

    return prob
    
