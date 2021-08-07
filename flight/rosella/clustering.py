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
import sklearn.metrics as sk_metrics
from scipy.spatial.distance import euclidean

# self imports
import flight.metrics as metrics
import flight.utils as utils
from flight.rosella.binning import Binner
from flight.DBCV import DBCV

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

    def cluster(self, distances, metric='euclidean', binning_method='eom',
                allow_single_cluster=False, prediction_data=False):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ## Cluster on the UMAP embeddings and return soft clusters
            tuned = utils.hyperparameter_selection(distances, self.threads,
                                                   metric=metric,
                                                   method=binning_method,
                                                   allow_single_cluster=allow_single_cluster,
                                                   starting_size = self.min_cluster_size)
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
                    core_dist_n_jobs=self.threads,
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
                                            prediction_data=prediction_data)
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
                                                prediction_data=prediction_data)

                except TypeError:
                    bool_arr = np.array([False for _ in first_labels])
            else:
                bool_arr = np.array([False for _ in first_labels])
        else:
            try:
                first_labels = self.cluster(distances, metric=metric,
                                            allow_single_cluster=allow_single_cluster,
                                            prediction_data=prediction_data)
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
    def validity(labels, distances):
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
                cluster_validity = DBCV(distances, np.array(labels), dist_function=euclidean)
                # cluster_validity = hdbscan.validity.validity_index(distances.astype(np.float64), np.array(labels), per_cluster_scores=False)
            except ValueError:
                # cluster_validity = DBCV(distances, np.array(labels), dist_function=euclidean)
                # cluster_validity = hdbscan.validity.validity_index(distances.astype(np.float64), np.array(labels), per_cluster_scores=False)
                cluster_validity = -1
        else:
            return -1

        if math.isnan(cluster_validity):
            # cluster_validity = DBCV(distances, np.array(labels), dist_function=euclidean)
            return -1

        return cluster_validity


def cluster_static(
        distances, metric='euclidean', binning_method='eom',
        allow_single_cluster=False, threads=1, min_cluster_size=2
):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ## Cluster on the UMAP embeddings and return soft clusters
            tuned = utils.hyperparameter_selection(distances, threads,
                                                   metric=metric,
                                                   method=binning_method,
                                                   allow_single_cluster=allow_single_cluster,
                                                   starting_size = min_cluster_size)
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
         threads=1
):
    if metric != "precomputed":
        try:
            first_labels = cluster_static(distances, metric=metric,
                                        allow_single_cluster=allow_single_cluster, threads=threads)
        except TypeError:
            first_labels = np.array([-1 for i in range(distances.shape[0])])
        # if prediction_data is False:
        bool_arr = np.array([True if i == -1 else False for i in first_labels])
        if len(distances[bool_arr]) >= 10 and double: # try as it might fail with low numbers of leftover contigs
            try:
                # Try to get unbinned super small clusters if they weren't binned
                second_labels = cluster_static(distances[bool_arr],
                                             metric=metric,
                                             allow_single_cluster=False, threads=threads)

            except TypeError:
                bool_arr = np.array([False for _ in first_labels])
        else:
            bool_arr = np.array([False for _ in first_labels])
    else:
        try:
            first_labels = cluster_static(distances, metric=metric,
                                        allow_single_cluster=allow_single_cluster)
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