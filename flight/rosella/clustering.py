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
import copy
import sys
import argparse
import logging
import warnings

# Function imports
import math
import numpy as np
from numba import njit
import hdbscan
import seaborn as sns
import matplotlib
import sklearn.cluster as sk_cluster
import sklearn.metrics as sk_metrics
import ClusterEnsembles as CE
from flight.DBCV import DBCV
from scipy.spatial.distance import euclidean

# self imports
import flight.metrics as metrics
import flight.utils as utils
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

class Clusterer(Binner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def cluster(self, distances, metric='euclidean',
                allow_single_cluster=False, prediction_data=False, min_cluster_size=2):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ## Cluster on the UMAP embeddings and return soft clusters
            tuned_eom = utils.hyperparameter_selection(distances, self.threads,
                                                   metric=metric,
                                                   method="eom",
                                                   allow_single_cluster=allow_single_cluster,
                                                   starting_size = min_cluster_size)
            tuned_leaf = utils.hyperparameter_selection(distances, self.threads,
                                                   metric=metric,
                                                   method="leaf",
                                                   allow_single_cluster=allow_single_cluster,
                                                   starting_size = min_cluster_size)
            best_eom = utils.best_validity(tuned_eom)
            best_leaf = utils.best_validity(tuned_leaf)

            if int(best_eom["validity_score"]) >= int(best_leaf["validity_score"]):
                best = best_eom
                binning_method = "eom"
            else:
                best = best_leaf
                binning_method = "leaf"

            if metric == 'precomputed':
                clusterer = hdbscan.HDBSCAN(
                    algorithm='best',
                    alpha=1.0,
                    cluster_selection_method=binning_method,
                    metric=metric,
                    min_cluster_size=int(best['min_cluster_size']),
                    min_samples=int(best['min_samples']),
                    allow_single_cluster=allow_single_cluster,
                    core_dist_n_jobs=self.threads,
                    approx_min_span_tree=False
                )
                clusterer.fit(distances)
                if prediction_data:
                    self.soft_clusters = None

            else:
                clusterer = hdbscan.HDBSCAN(
                    algorithm='best',
                    alpha=1.0,
                    approx_min_span_tree=True,
                    gen_min_span_tree=True,
                    leaf_size=40,
                    cluster_selection_method=binning_method,
                    metric=metric,
                    min_cluster_size=int(best['min_cluster_size']),
                    min_samples=int(best['min_samples']),
                    allow_single_cluster=allow_single_cluster,
                    core_dist_n_jobs=self.threads,
                    prediction_data=prediction_data
                )
                clusterer.fit(distances)
                if prediction_data:
                    self.soft_clusters = hdbscan.all_points_membership_vectors(clusterer)
            return clusterer.labels_

    @staticmethod
    def get_cluster_labels_array(
            distances,
            metric="euclidean",
            cluster_selection_methods=["eom"],
            top_n=3,
            # min_sizes = [2, 4, 6, 8, 10],
            # min_samples = [3, 6, 9, 12],
            solver="hbgf",
            threads=16,
            embeddings_for_precomputed=None
    ):
        """
        Uses cluster ensembling with ClusterEnsembles package to produce partitioned set of
        high quality clusters from multiple HDBSCAN runs
        Takes top N clustering results and combines them
        solver - one of {'cspa', 'hgpa', 'mcla', 'hbgf', 'nmf', 'all'}, default='hbgf'
        """
        label_array = np.array([np.array([-1 for _ in range(distances.shape[0])]) for _ in range(top_n)])
        best_min_size = np.array([None for _ in range(top_n)])
        best_min_sample = np.array([None for _ in range(top_n)])
        best_selection_method = np.array([None for _ in range(top_n)])
        best_validity = np.array([None for _ in range(top_n)])
        index = 0
        for min_size in range(3, 15):
            for min_sample in range(5, 20):
                for selection_method in cluster_selection_methods:
                    clusterer = hdbscan.HDBSCAN(
                        algorithm='best',
                        alpha=1.0,
                        cluster_selection_method=selection_method,
                        metric=metric,
                        min_cluster_size=min_size,
                        min_samples=min_sample,
                        core_dist_n_jobs=threads,
                        approx_min_span_tree=False
                    )
                    clusterer.fit(distances)

                    try:
                        if metric != "precomputed":
                            cluster_validity = Clusterer.validity(
                                clusterer.labels_, distances, quick=True
                            )
                        else:
                            cluster_validity = Clusterer.validity(
                                clusterer.labels_, embeddings_for_precomputed, quick=True
                            )
                    except (ValueError, FloatingPointError):
                        try:
                            if metric != "precomputed":
                                cluster_validity = Clusterer.validity(
                                    clusterer.labels_, distances, quick=False
                                )
                            else:
                                cluster_validity = Clusterer.validity(
                                    clusterer.labels_, embeddings_for_precomputed, quick=False
                                )
                        except (ValueError, FloatingPointError):
                            cluster_validity = -1

                    if np.any(best_validity == None):
                        best_min_size[index] = min_size
                        best_min_sample[index] = min_sample
                        best_selection_method[index] = selection_method
                        best_validity[index] = cluster_validity
                        label_array[index] = clusterer.labels_
                        index += 1

                        if index == top_n:

                            # sort the current top by ascending validity order
                            ranks = np.argsort(best_validity)
                            best_validity = best_validity[ranks]
                            best_min_sample = best_min_sample[ranks]
                            best_min_size = best_min_size[ranks]
                            best_selection_method = best_selection_method[ranks]
                            label_array = label_array[ranks]
                            # with np.set_printoptions(precision=3, suppress=True, formatter={'float': '{: 0.3f}'.format}):
                            # print(best_validity)
                            # print(label_array)

                    elif np.any(best_validity < cluster_validity):
                        # insert the new result and remove the worst result
                        ind = np.searchsorted(best_validity, cluster_validity)
                        best_validity = np.insert(best_validity, ind, cluster_validity)[1:]
                        best_min_size = np.insert(best_min_size, ind, min_size)[1:]
                        best_min_sample = np.insert(best_min_sample, ind, min_sample)[1:]
                        best_selection_method = np.insert(best_selection_method, ind, selection_method)[1:]
                        label_array = np.insert(label_array, ind, clusterer.labels_, axis=0)[1:]


        return label_array

    @staticmethod
    def ensemble_cluster_multiple_embeddings(
            embeddings_array,
            metric="euclidean",
            cluster_selection_methods=["eom"],
            top_n=3,
            # min_sizes = [2, 4, 6, 8, 10],
            # min_samples = [3, 6, 9, 12],
            solver="hbgf",
            threads=16,
            embeddings_for_precomputed=None
    ):
        """
        Uses cluster ensembles to find best results across multiple different embeddings
        and clustering results.
        embeddings_array - an array of n different embeddings of distance matrix
        """
        if metric == "precomputed":
            if len(embeddings_for_precomputed) != len(embeddings_array):
                sys.exit("Require reduced embeddings via UMAP or other method in addition to precomputed distance matrix")
        else:
            embeddings_for_precomputed = [None for _ in range(len(embeddings_array))]

        best_clusters = []
        for idx, embeddings in enumerate(embeddings_array):
            #         print(embeddings)
            results = Clusterer.get_cluster_labels_array(
                embeddings,
                metric,
                cluster_selection_methods,
                top_n,
                solver,
                threads,
                embeddings_for_precomputed[idx]
            )
            for result in range(results.shape[0]):
                best_clusters.append(results[result, :])

        label_ensemble = CE.cluster_ensembles(np.array(best_clusters))
        return label_ensemble

    def iterative_clustering(self,
                             distances,
                             metric='euclidean',
                             allow_single_cluster=False,
                             prediction_data=False,
                             double=True,
                             min_cluster_size = 2, min_samples = 1):
        if metric != "precomputed":
            try:
                first_labels = self.cluster(distances, metric=metric,
                                            allow_single_cluster=allow_single_cluster,
                                            prediction_data=prediction_data,
                                            min_cluster_size=min_cluster_size)
            except TypeError:
                first_labels = np.array([-1 for i in range(distances.shape[0])])
            # if prediction_data is False:
            bool_arr = np.array([True if i == -1 else False for i in first_labels])
            if len(distances[bool_arr]) >= 10 and double: # try as it might fail with low numbers of leftover contigs
                try:
                    # Try to get unbinned super small clusters if they weren't binned
                    second_labels = self.cluster(distances[bool_arr],
                                                 metric=metric,
                                                 allow_single_cluster=False,
                                                prediction_data=prediction_data,
                                            min_cluster_size=min_cluster_size)

                except TypeError:
                    bool_arr = np.array([False for _ in first_labels])
            else:
                bool_arr = np.array([False for _ in first_labels])
        else:
            try:
                first_labels = self.cluster(distances, metric=metric,
                                            allow_single_cluster=allow_single_cluster,
                                            prediction_data=prediction_data,
                                            min_cluster_size=min_cluster_size)
            except TypeError:
                first_labels = np.array([-1 for i in range(distances.shape[0])])
                # first_labels = np.array([-1 for i in range(distances.shape[0])])
            bool_arr = np.array([False for _ in first_labels])

        #
        #
        main_labels = []  # container for complete clustering
        max_label = max(first_labels) + 1  # value to shift second labels by
        second_idx = 0  # current index in second labels
        #
        for first, second_bool in zip(first_labels, bool_arr):
            if first != -1:  # use main label
                main_labels.append(first)
            elif second_bool:
                second_label = second_labels[second_idx]
                if second_label != -1:
                    main_labels.append(max_label + second_labels[second_idx] + 1)
                else:
                    main_labels.append(-1)
                second_idx += 1
            else:  # Should never get here but just have it here in case
                main_labels.append(-1)

        return main_labels

    @staticmethod
    def validity(labels, distances, quick=False):
        """
        Calculates cluster validity using Density Based Cluster Validity from HDBSCAN

        Params:
            :labels:
                Cluster labels to test
            :distances:
                Either a pairwise distance matrix or UMAP embedding values for the provided contig labels
        """
        if len(set(labels)) > 1:
            # cluster_validity = DBCV(distances, np.array(labels), dist_function=euclidean)
            try:
                if not quick:
                    cluster_validity = DBCV(distances, np.array(labels), dist_function=euclidean)
                else:
                    cluster_validity = hdbscan.validity.validity_index(distances.astype(np.float64), np.array(labels), per_cluster_scores=False)
            except (ValueError, FloatingPointError):
                # cluster_validity = DBCV(distances, np.array(labels), dist_function=euclidean)
                # cluster_validity = hdbscan.validity.validity_index(distances.astype(np.float64), np.array(labels), per_cluster_scores=False)
                cluster_validity = -1
        else:
            return -1

        if math.isnan(cluster_validity):
            # cluster_validity = DBCV(distances, np.array(labels), dist_function=euclidean)
            return -1

        return cluster_validity

def kmeans_cluster(distances, n_clusters=2, random_seed=42, n_jobs=10):
    """
    Takes a set of precomputed distances and performs kmeans clustering on them
    returns the calculated labels and silhouette score
    """
    # n_jobs was deprecated in sklearn 0.23 but seems to prevent hanging still?
    kmeans = sk_cluster.KMeans(n_clusters=n_clusters, random_state=random_seed).fit(distances)
    score = sk_metrics.silhouette_score(distances, kmeans.labels_)
    return (kmeans.labels_, score)

def cluster_static(
        distances, metric='euclidean', binning_method='eom',
        allow_single_cluster=False, threads=1, min_cluster_size=2,
        use_multi_processing=True
):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ## Cluster on the UMAP embeddings and return soft clusters
            tuned = utils.hyperparameter_selection(distances, threads,
                                                   metric=metric,
                                                   method=binning_method,
                                                   allow_single_cluster=allow_single_cluster,
                                                   starting_size = min_cluster_size,
                                                   use_multi_processing = use_multi_processing)
            best = utils.best_validity(tuned)

            if metric == 'precomputed':
                clusterer = hdbscan.HDBSCAN(
                    algorithm='best',
                    alpha=1.0,
                    cluster_selection_method=binning_method,
                    metric=metric,
                    min_cluster_size=int(best['min_cluster_size']),
                    min_samples=int(best['min_samples']),
                    allow_single_cluster=allow_single_cluster,
                    core_dist_n_jobs=threads,
                )
                clusterer.fit(distances)

            else:
                clusterer = hdbscan.HDBSCAN(
                    algorithm='best',
                    alpha=1.0,
                    approx_min_span_tree=True,
                    gen_min_span_tree=True,
                    leaf_size=40,
                    cluster_selection_method=binning_method,
                    metric=metric,
                    min_cluster_size=int(best['min_cluster_size']),
                    min_samples=int(best['min_samples']),
                    allow_single_cluster=allow_single_cluster,
                    core_dist_n_jobs=threads,
                )
                clusterer.fit(distances)
            return clusterer.labels_

def iterative_clustering_static(
         distances,
         metric='euclidean',
         allow_single_cluster=False,
         double=True,
         threads=1,
         use_multi_processing=True,
         cluster_selection_method="eom"
):
    if metric != "precomputed":
        try:
            first_labels = cluster_static(
                distances,
                metric=metric,
                allow_single_cluster=allow_single_cluster,
                threads=threads,
                use_multi_processing=use_multi_processing,
                binning_method=cluster_selection_method
            )
        except TypeError:
            first_labels = np.array([-1 for i in range(distances.shape[0])])
        # if prediction_data is False:
        bool_arr = np.array([True if i == -1 else False for i in first_labels])
        if len(distances[bool_arr]) >= 10 and double: # try as it might fail with low numbers of leftover contigs
            try:
                # Try to get unbinned super small clusters if they weren't binned
                second_labels = cluster_static(
                    distances[bool_arr],
                    metric=metric,
                    allow_single_cluster=False,
                    threads=threads,
                    use_multi_processing=use_multi_processing,
                    binning_method=cluster_selection_method
                )

            except TypeError:
                bool_arr = np.array([False for _ in first_labels])
        else:
            bool_arr = np.array([False for _ in first_labels])
    else:
        try:
            first_labels = cluster_static(
                distances,
                metric=metric,
                allow_single_cluster=allow_single_cluster,
                use_multi_processing=use_multi_processing,
                binning_method=cluster_selection_method
            )
        except TypeError:
            first_labels = np.array([-1 for i in range(distances.shape[0])])
            # first_labels = np.array([-1 for i in range(distances.shape[0])])
        bool_arr = np.array([False for _ in first_labels])

    #
    #
    main_labels = []  # container for complete clustering
    max_label = max(first_labels) + 1  # value to shift second labels by
    second_idx = 0  # current index in second labels
    #
    for first, second_bool in zip(first_labels, bool_arr):
        if first != -1:  # use main label
            main_labels.append(first)
        elif second_bool:
            second_label = second_labels[second_idx]
            if second_label != -1:
                main_labels.append(max_label + second_labels[second_idx] + 1)
            else:
                main_labels.append(-1)
            second_idx += 1
        else:  # Should never get here but just have it here in case
            main_labels.append(-1)

    return main_labels