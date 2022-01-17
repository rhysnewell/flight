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
from flight.DBCV import DBCV
from scipy.spatial.distance import euclidean
import pebble
import multiprocessing
import itertools
import threadpoolctl

# self imports
import flight.metrics as metrics
import flight.utils as utils
from flight.rosella.binning import Binner

# Set plotting style
sns.set(style='white', context='notebook', rc={'figure.figsize': (14, 10)})
matplotlib.use('pdf')

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
    def generate_cluster(
            distances,
            embeddings_for_precomputed=None,
            selection_method="eom",
            metric="euclidean",
            min_size=2,
            min_sample=2,
            threads=1
    ):
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

        return (cluster_validity, min_size, min_sample, clusterer.labels_)

    @staticmethod
    def get_cluster_labels_array(
            distances,
            metric="euclidean",
            selection_method="eom",
            top_n=3,
            # min_sizes = [2, 4, 6, 8, 10],
            # min_samples = [3, 6, 9, 12],
            solver="hbgf",
            threads=16,
            embeddings_for_precomputed=None,
            use_multiple_processes=True,
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
        best_validity = np.array([None for _ in range(top_n)])
        best_unbinned = np.array([None for _ in range(top_n)])
        best_n_bins = np.array([None for _ in range(top_n)])
        index = 0


        if use_multiple_processes:

            worker_limit = threads // 5
            # thread_limit = worker_limit // 5
            # with threadpoolctl.threadpool_limits(limits=max(threads // 5, 1), user_api='blas'):
            with pebble.ProcessPool(max_workers=threads // 5, context=multiprocessing.get_context('forkserver')) as executor:
                futures = [
                    executor.schedule(
                        Clusterer.generate_cluster,
                        (
                            distances,
                            embeddings_for_precomputed,
                            selection_method,
                            metric,
                            min_size,
                            min_sample,
                            threads
                        ),
                        timeout=1800,
                    ) for (min_size, min_sample) in itertools.combinations(range(1, 10), 2) if min_size != 1
                ]
                # executor.close()
                for future in futures:
                    try:
                        (cluster_validity, min_size, min_sample, labels) = future.result()
                        if np.any(best_validity == None):
                            best_min_size[index] = min_size
                            best_min_sample[index] = min_sample
                            best_validity[index] = cluster_validity
                            label_array[index] = labels
                            best_n_bins[index] = np.unique(labels).shape[0]
                            best_unbinned[index] = (labels == -1).sum()
                            index += 1

                            if index == top_n:

                                # sort the current top by ascending validity order
                                ranks = np.argsort(best_validity)
                                best_validity = best_validity[ranks]
                                best_min_sample = best_min_sample[ranks]
                                best_min_size = best_min_size[ranks]
                                label_array = label_array[ranks]
                                best_n_bins = best_n_bins[ranks]
                                best_unbinned = best_unbinned[ranks]

                        elif np.any(best_validity < cluster_validity):
                            # insert the new result and remove the worst result
                            ind = np.searchsorted(best_validity, cluster_validity)
                            best_validity = np.insert(best_validity, ind, cluster_validity)[1:]
                            best_min_size = np.insert(best_min_size, ind, min_size)[1:]
                            best_min_sample = np.insert(best_min_sample, ind, min_sample)[1:]
                            label_array = np.insert(label_array, ind, labels, axis=0)[1:]
                            best_n_bins = np.insert(best_n_bins, ind, np.unique(labels).shape[0])[1:]
                            best_unbinned = np.insert(best_unbinned, ind, (labels == -1).sum())[1:]
                    except TimeoutError:
                        continue

        else:
            results = [
                    Clusterer.generate_cluster
                    (
                        distances,
                        embeddings_for_precomputed,
                        selection_method,
                        metric,
                        min_size,
                        min_sample,
                        threads
                ) for (min_size, min_sample) in itertools.combinations(range(1, 10), 2) if min_size != 1
            ]

            for result in results:
                (cluster_validity, min_size, min_sample, labels) = result
                if np.any(best_validity == None):
                    best_min_size[index] = min_size
                    best_min_sample[index] = min_sample
                    best_validity[index] = cluster_validity
                    label_array[index] = labels
                    best_n_bins[index] = np.unique(labels).shape[0]
                    best_unbinned[index] = (labels == -1).sum()
                    index += 1

                    if index == top_n:
                        # sort the current top by ascending validity order
                        ranks = np.argsort(best_validity)
                        best_validity = best_validity[ranks]
                        best_min_sample = best_min_sample[ranks]
                        best_min_size = best_min_size[ranks]
                        label_array = label_array[ranks]
                        best_n_bins = best_n_bins[ranks]
                        best_unbinned = best_unbinned[ranks]

                elif np.any(best_validity < cluster_validity):
                    # insert the new result and remove the worst result
                    ind = np.searchsorted(best_validity, cluster_validity)
                    best_validity = np.insert(best_validity, ind, cluster_validity)[1:]
                    best_min_size = np.insert(best_min_size, ind, min_size)[1:]
                    best_min_sample = np.insert(best_min_sample, ind, min_sample)[1:]
                    label_array = np.insert(label_array, ind, labels, axis=0)[1:]
                    best_n_bins = np.insert(best_n_bins, ind, np.unique(labels).shape[0])[1:]
                    best_unbinned = np.insert(best_unbinned, ind, (labels == -1).sum())[1:]

        return label_array, best_validity, best_n_bins, best_unbinned

    @staticmethod
    def ensemble_cluster_multiple_embeddings(
            embeddings_array,
            metric="euclidean",
            cluster_selection_methods="eom",
            top_n=3,
            # min_sizes = [2, 4, 6, 8, 10],
            # min_samples = [3, 6, 9, 12],
            solver="hbgf",
            threads=16,
            embeddings_for_precomputed=None,
            use_multiple_processes=True,
    ):
        """
        Uses cluster ensembles to find best results across multiple different embeddings
        and clustering results.
        embeddings_array - an array of n different embeddings of distance matrix
                         - The length of each array of embeddings must be equal
        """
        if metric == "precomputed":
            if len(embeddings_for_precomputed) != len(embeddings_array):
                sys.exit("Require reduced embeddings via UMAP or other method in addition to precomputed distance matrix")
        else:
            embeddings_for_precomputed = [None for _ in range(len(embeddings_array))]

        best_clusters = np.array([np.array([-1 for _ in range(embeddings_array[0].shape[0])]) for _ in range(top_n * len(embeddings_array))])
        best_validities = np.array([np.nan for _ in range(top_n * len(embeddings_array))])
        best_n_bins = np.array([None for _ in range(top_n * len(embeddings_array))])
        best_unbinned = np.array([None for _ in range(top_n * len(embeddings_array))])

        stored_index = 0
        for idx, embeddings in enumerate(embeddings_array):

            label_results, label_validities, label_n_bins, label_unbinned = Clusterer.get_cluster_labels_array(
                embeddings,
                metric,
                cluster_selection_methods,
                top_n,
                solver,
                threads,
                embeddings_for_precomputed[idx],
                use_multiple_processes=use_multiple_processes
            )
            for result_index in range(label_results.shape[0]):
                best_clusters[stored_index] = label_results[result_index]
                best_validities[stored_index] = label_validities[result_index]
                best_n_bins[stored_index] = label_n_bins[result_index]
                best_unbinned[stored_index] = label_unbinned[result_index]
                stored_index += 1


        # print("validities: ", str(best_validities))
        # print("unbinned: ", str(best_unbinned))
        # print("n_bins: ", str(best_n_bins))
        ranks_val = np.argsort(-best_validities)
        ranks_val = ranks_val / max(ranks_val)
        ranks_unbinned = np.argsort(best_unbinned)
        ranks_unbinned = ranks_unbinned / max(ranks_unbinned)
        ranks_n_bins = np.argsort(best_n_bins)
        ranks_n_bins = ranks_n_bins / max(ranks_n_bins)

        ranks = np.argsort(ranks_val + ranks_unbinned + ranks_n_bins)
        best_validities = best_validities[ranks]
        best_clusters = best_clusters[ranks]
        best_n_bins = best_n_bins[ranks]
        best_unbinned = best_unbinned[ranks]

        return best_clusters, best_validities, best_n_bins, best_unbinned

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