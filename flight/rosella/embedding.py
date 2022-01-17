#!/usr/bin/env python
###############################################################################
# binning.py - A binning algorithm spinning off of the methodology of
#              Lorikeet
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
import logging

# Function imports
import numpy as np
from numba import njit, set_num_threads
import pandas as pd
import seaborn as sns
import matplotlib
import umap
import scipy.stats as sp_stats
import scipy.spatial.distance as sp_distance
from sklearn.mixture import GaussianMixture
import pebble
import multiprocessing
import random
from pynndescent import NNDescent
import warnings


# self imports
import flight.metrics as metrics
from flight.rosella.binning import Binner

# Set plotting style
sns.set(style='white', context='notebook', rc={'figure.figsize': (14, 10)})
matplotlib.use('pdf')

# Debug
debug = {
    1: logging.CRITICAL,
    2: logging.ERROR,
    3: logging.WARNING,
    4: logging.INFO,
    5: logging.DEBUG
}

###############################################################################
############################### - Exceptions - ################################


class BadTreeFileException(Exception):
    pass


###############################################################################                                                                                                                      [44/1010]
################################ - Functions - ################################

@njit
def index(array, item):
    for idx, val in np.ndenumerate(array):
        if val == item:
            return idx

###############################################################################
################################ - Classes - ##################################

class Embedder(Binner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def filter(self, minimum_connections=1, close_check=None):
        """
        Performs quick UMAP embeddings with stringent disconnection distances
        """
        try:

            initial_disconnections = self.check_contigs(self.large_contigs['tid'], minimum_connections, close_check)

            disconnected = initial_disconnections #+ np.array(self.large_contigs['tid'].isin(
        except ValueError: # Everything was disconnected
            disconnected = np.array([True for i in range(self.large_contigs.values.shape[0])])

        self.disconnected = disconnected

    def check_contigs(self, tids, minimum_connections=1, close_check=None):
        """
        The weight length distribution of contigs is not normal, instead it kind of looks like two normal distributions
        akin to camel with two humps. We can hastily model this by using the n75 of the contig lengths and create a skewed
        normal distribution.

        Contigs that fall below n75 use the standard normal distribution, contigs that fall above use the skewed distribution

        The pdf value is used as the thresholds for the different metrics for that contig
        """
        _, n25 = self.nX(25)
        # n25 = min(n25, 150000) # we want to limit these values as some assemblies may be ungodly large which causes this method to breakdown
        _, n75 = self.nX(75)
        # n75 = min(n75, 1000000)
        lengths = self.large_contigs[self.large_contigs['tid'].isin(tids)]['contigLen']
        log_lengths = np.log10(lengths.values)

        # Account for stochasticity by running a few times
        max_sum = 0
        min_sum = 0
        max_covar = 0
        min_covar = 0
        count = 0
        for i in range(10):
            gm = GaussianMixture(
                n_components=2,
                covariance_type='full'
            ).fit(log_lengths.reshape(-1, 1))
            max_sum += max(gm.means_)[0]
            max_covar += gm.covariances_[gm.means_.argmax()][0][0]
            min_sum += min(gm.means_)[0]
            min_covar += gm.covariances_[gm.means_.argmin()][0][0]

            count += 1

        max_mean = max_sum / count
        max_covar = max_covar / count
        min_mean = min_sum / count
        min_covar = min_covar / count
        logging.info("Filtering Params - Max: %f, %f, %d Min: %f, %f, %d"
                     % (max_mean, max_covar, n75, max(min_mean, np.log10(5000)), min_covar, n25))

        skew_dist1 = sp_stats.skewnorm(np.log10(n25), max(min_mean, np.log10(5000)), 1 + min(max(min_covar, 0.01), 0.2))
        skew_dist2 = sp_stats.skewnorm(np.log10(n75), min(max_mean, 4.5), 1 + min(max(max_covar, 0.1), 0.2))
        disconnected_tids = []
        if self.n_samples > 0:
            n_samples = self.n_samples
            sample_distances = self.short_sample_distance
        else:
            n_samples = self.long_samples
            sample_distances = self.long_sample_distance

        contigs, ll, tnfs = \
            self.extract_contigs(self.large_contigs['tid'])
        current_tnfs = tnfs.iloc[:, 2:].values

        index_rho = NNDescent(current_tnfs, metric=metrics.rho, n_neighbors=30)
        index_euc = NNDescent(current_tnfs, metric=metrics.tnf_euclidean, n_neighbors=30)
        index_dep = NNDescent(
            contigs.values[:, 3:],
            metric=metrics.metabat_distance_nn,
            n_neighbors=30)

        for idx, tid in enumerate(tids):

            disconnected = False
            # Scale the thresholds based on the probability distribution
            # Use skew dist to get PDF value of current contig
            prob1 = min(skew_dist1.pdf(log_lengths[idx]), 1.0)
            prob2 = min(skew_dist2.pdf(log_lengths[idx]), 1.0)
            prob = max(prob1, prob2)
            rho_thresh = min(max(prob * 0.5, 0.05), 1.0)
            euc_thresh = min(max(prob * 10, 1.0), 10)
            dep_thresh = min(max((prob / 2) + 0.05, 0.1), 1.0)

            dep_connected = sum(x <= dep_thresh
                                for x in index_dep.neighbor_graph[1][idx, 1:(minimum_connections + 1)]) # exclude first index since it is to itself
            rho_connected = sum(x <= rho_thresh
                                for x in index_rho.neighbor_graph[1][idx, 1:(minimum_connections + 1)]) # exclude first index since it is to itself
            euc_connected = sum(x <= euc_thresh
                                for x in index_euc.neighbor_graph[1][idx, 1:(minimum_connections + 1)]) # exclude first index since it is to itself


            if sum(x < minimum_connections for x in [rho_connected, euc_connected, dep_connected]) >= 1:
                disconnected_tids.append(tid)
                disconnected = True

            if close_check is not None:
                if close_check == tid or tid in close_check:
                    print("tid: ", tid, disconnected)
                    print(dep_thresh, rho_thresh, euc_thresh)
                    print(dep_connected, rho_connected, euc_connected)
                    print(index_dep.neighbor_graph[1][idx, 1:(minimum_connections + 1)], index_dep.neighbor_graph[0][idx, 1:(minimum_connections + 1)])
                    print(index_rho.neighbor_graph[1][idx, 1:(minimum_connections + 1)], index_rho.neighbor_graph[0][idx, 1:(minimum_connections + 1)])
                    print(index_euc.neighbor_graph[1][idx, 1:(minimum_connections + 1)], index_euc.neighbor_graph[0][idx, 1:(minimum_connections + 1)])


        disconnections = np.array(self.large_contigs['tid'].isin(disconnected_tids))

        return disconnections

    def check_contigs_inside_bin(self, tids, close_check=None):
        """
        The weight length distribution of contigs is not normal, instead it kind of looks like two normal distributions
        akin to camel with two humps. We can hastily model this by using the n75 of the contig lengths and create a skewed
        normal distribution.

        Contigs that fall below n75 use the standard normal distribution, contigs that fall above use the skewed distribution

        The pdf value is used as the thresholds for the different metrics for that contig
        """
        _, n25 = self.nX(25)
        _, n75 = self.nX(75)
        lengths = self.large_contigs[self.large_contigs['tid'].isin(tids)]['contigLen']
        log_lengths = np.log10(lengths.values)

        # Account for stochasticity by running a few times
        max_sum = 0
        min_sum = 0
        max_covar = 0
        min_covar = 0
        count = 0
        for i in range(10):
            gm = GaussianMixture(n_components=2).fit(log_lengths.reshape(-1, 1))
            max_sum += max(gm.means_)[0]
            max_covar += gm.covariances_[gm.means_.argmax()][0][0]
            min_sum += min(gm.means_)[0]
            min_covar += gm.covariances_[gm.means_.argmin()][0][0]

            count += 1

        max_mean = max_sum / count
        max_covar = max_covar / count
        min_mean = min_sum / count
        min_covar = min_covar / count
        logging.info("Filtering Params - Max: %f, %f, %d Min: %f, %f, %d"
                     % (max_mean, max_covar, n75, max(min_mean, np.log10(5000)), min_covar, n25))

        skew_dist1 = sp_stats.skewnorm(np.log10(n25), max(min_mean, np.log10(5000)), 1 + max(min_covar / count, 0.01))
        skew_dist2 = sp_stats.skewnorm(np.log10(n75), max_mean, 1 + max(max_covar / count, 0.1))
        disconnected_tids = []
        if self.n_samples > 0:
            n_samples = self.n_samples
            sample_distances = self.short_sample_distance
        else:
            n_samples = self.long_samples
            sample_distances = self.long_sample_distance

        for idx, tid in enumerate(tids):
            current_contigs, current_log_lengths, current_tnfs = \
                self.extract_contigs([tid])

            other_contigs, other_log_lengths, other_tnfs = \
                self.extract_contigs([t for t in tids if t != tid])

            current = np.concatenate((current_contigs.iloc[:, 3:].values,
                                      current_log_lengths.values[:, None],
                                      current_tnfs.iloc[:, 2:].values), axis=1)

            others = np.concatenate((other_contigs.iloc[:, 3:].values,
                                     other_log_lengths.values[:, None],
                                     other_tnfs.iloc[:, 2:].values), axis=1)


            # Scale the thresholds based on the probability distribution
            # Use skew dist to get PDF value of current contig
            prob1 = min(skew_dist1.pdf(log_lengths[idx]), 1.0)
            prob2 = min(skew_dist2.pdf(log_lengths[idx]), 1.0)
            prob = max(prob1, prob2)
            rho_thresh = min(max(prob / 2, 0.05), 0.5)
            euc_thresh = min(max(prob * 10, 1.0), 6)
            dep_thresh = min(max(prob / 2, 0.05), 0.5)


            connections = metrics.check_connections(
                current, others, n_samples, sample_distances,
                rho_threshold=rho_thresh, euc_threshold=euc_thresh, dep_threshold=dep_thresh
            )

            if sum(connections) <= 2:
                disconnected_tids.append(True)
            else:
                disconnected_tids.append(False)

            if close_check is not None:
                if close_check == tid:
                    print(rho_thresh, euc_thresh, dep_thresh)
                    print(connections)

        logging.info("Preemptively filtered %d contigs" % len(disconnected_tids))
        disconnections = np.array(disconnected_tids)

        return disconnections


    def check_contig(self, tid, rho_threshold=0.05, euc_threshold=2.0, dep_threshold=0.05):
        if self.n_samples > 0:
            n_samples = self.n_samples
            sample_distances = self.short_sample_distance
        else:
            n_samples = self.long_samples
            sample_distances = self.long_sample_distance

        current_contigs, current_log_lengths, current_tnfs = \
            self.extract_contigs([tid])
        other_contigs, other_log_lengths, other_tnfs = \
            self.extract_contigs(self.large_contigs[self.large_contigs['tid'] != tid]['tid'])

        current = np.concatenate((current_contigs.iloc[:, 3:].values,
                                  current_log_lengths.values[:, None],
                                  current_tnfs.iloc[:, 2:].values), axis=1)[0]

        others = np.concatenate((other_contigs.iloc[:, 3:].values,
                                 other_log_lengths.values[:, None],
                                 other_tnfs.iloc[:, 2:].values), axis=1)

        return metrics.check_connections(
            current, others, n_samples, sample_distances,
            rho_threshold=rho_threshold, euc_threshold=euc_threshold, dep_threshold=dep_threshold
        )


    def fit_disconnect(self):
        """
        Filter contigs based on ADP connections
        """
        disconnected_intersected = np.array([False for i in range(self.large_contigs[~self.disconnected].values.shape[0])])
        logging.info("Found %d disconnected points. %d TNF disconnected and %d ADP disconnected..." %
                     (sum(self.disconnected) + sum(disconnected_intersected), sum(self.disconnected),
                      sum(disconnected_intersected)))
        # logging.debug out all disconnected contigs
        pd.concat([self.large_contigs[self.disconnected],
                   self.large_contigs[~self.disconnected][disconnected_intersected]]) \
            .to_csv(self.path + "/disconnected_contigs.tsv", sep="\t", header=True)

        self.disconnected_intersected = disconnected_intersected

    def fit_transform_precomputed(self, stat, set_embedding=False):

        embedders = [
            self.precomputed_reducer_low,
            self.precomputed_reducer_mid,
            self.precomputed_reducer_high
        ]

        with pebble.ProcessPool(max_workers=3, context=multiprocessing.get_context('forkserver')) as executor:
            futures = [
                executor.schedule(
                    multi_transform_static,
                    (
                        stat,
                        embedder,
                        self.random_seed
                    )
                ) for embedder in embedders
            ]

            results = []
            # executor.close()
            for future in futures:
                result = future.result()
                if result is not None:
                    results.append(result)

            if set_embedding:
                self.embeddings = results[0]
                # self.embeddings2 = results[1]
                # self.embeddings3 = results[2]

            return results

def multi_transform_static(
        stat,
        reducer=None,
        random_seed=42069
):
    """
    Main function for performing UMAP embeddings and intersections
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.random.seed(random_seed)
        random.seed(random_seed)
        # update parameters to artificially high values to avoid disconnected vertices in the final manifold
        if reducer is None:
            warnings.warn("No reducers provided")
            return None

        # reducer.random_state = random_seed
        try:
            embedding = reducer.fit_transform(sp_distance.squareform(stat))

            return embedding

        except TypeError:
            return None

def switch_intersector_static(depth_reducer, tnf_reducer, euc_reducer, switch=None):
    if switch is None:
        switch = [0, 1, 2]
    if 0 in switch and 1 in switch and 2 in switch:
        # All
        # print("All: Switch", switch)
        return depth_reducer * tnf_reducer * euc_reducer
    elif 0 in switch and 1 in switch:
        # Rho and MD
        # print("MD and TNF: Switch", switch)
        return depth_reducer * tnf_reducer
    elif 0 in switch and 2 in switch:
        # print("MD and EUC: Switch", switch)
        return depth_reducer * euc_reducer
    elif 1 in switch and 2 in switch:
        # print("EUC and TNF: Switch", switch)
        return tnf_reducer * euc_reducer
    elif 0 in switch:
        # print("MD: Switch", switch)
        return depth_reducer
    elif 1 in switch:
        # print("TNF: Switch", switch)
        return tnf_reducer
    elif 2 in switch:
        # print("EUC: Switch", switch)
        return euc_reducer


def fit_transform_static(
        contigs, log_lengths, tnfs,
        n_neighbours, n_samples, sample_distances,
        a, b, random_seed,
        switch=0):
    np.random.seed(random_seed)
    random.seed(random_seed)
    if switch == 0:
        depth_reducer = umap.UMAP(
            metric=metrics.aggregate_tnf,
            # disconnection_distance=2,
            metric_kwds={"n_samples": n_samples,
                         "sample_distances": sample_distances},
            n_neighbors=n_neighbours,
            n_components=2,
            min_dist=0,
            set_op_mix_ratio=1,
            a=a,
            b=b,
            random_state=random_seed
        )

        return depth_reducer.fit(
            np.concatenate(
                (contigs.iloc[:, 3:], log_lengths.values[:, None], tnfs.iloc[:, 2:]), axis=1)
        )
    elif switch == 1:
        tnf_reducer = umap.UMAP(
            metric=metrics.rho,
            # disconnection_distance=2,
            n_neighbors=n_neighbours,
            n_components=2,
            min_dist=0,
            set_op_mix_ratio=1,
            a=a,
            b=b,
            random_state=random_seed
        )

        return tnf_reducer.fit(
            np.concatenate(
                (log_lengths.values[:, None],
                 tnfs.iloc[:, 2:]),
                axis=1)
        )
    elif switch == 2:
        euc_reducer = umap.UMAP(
            metric=metrics.tnf_euclidean,
            # disconnection_distance=10,
            n_neighbors=n_neighbours,
            n_components=2,
            min_dist=0,
            set_op_mix_ratio=1,
            a=a,
            b=b,
            random_state=random_seed
        )

        return euc_reducer.fit(
            np.concatenate(
                (log_lengths.values[:, None],
                 tnfs.iloc[:, 2:]),
                axis=1)
        )