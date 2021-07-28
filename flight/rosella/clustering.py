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
                             binning_method='eom',
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


    def pairwise_cluster(self, tids, max_bin_id, plots,
                           x_min=20, x_max=20, y_min=20, y_max=20, n=0,
                           delete_unbinned = False,
                           debug=False):
        """
        Functions much like reembed but does not perform any extra UMAP and only clusters on the pairwise
        distance matrix which is based on the Aggregate score
        """

        precomputed = False # Whether the precomputed clustering was the best result
        tids = list(np.sort(tids))
        contigs, log_lengths, tnfs = self.extract_contigs(tids)
        original_size = contigs['contigLen'].sum()

        if len(set(tids)) > 1:
            unbinned_array = self.large_contigs[~self.disconnected][~self.disconnected_intersected]['tid'].isin(tids)
            unbinned_embeddings = self.embeddings[unbinned_array]

            self.min_cluster_size = 2
            try:

                # Try out precomputed method, validity metric does not work here
                # so we just set it to 1 and hope it ain't shit. Method for this is
                # not accept a clustering result with noise. Not great, but

                if self.n_samples > 0:
                    n_samples = self.n_samples
                    sample_distances = self.short_sample_distance
                else:
                    n_samples = self.long_samples
                    sample_distances = self.long_sample_distance

                distances = metrics.distance_matrix(np.concatenate((contigs.iloc[:, 3:].values,
                                                        log_lengths.values[:, None],
                                                        tnfs.iloc[:, 2:].values), axis=1),
                                                        n_samples,
                                                        sample_distances)

                labels_precomputed = self.iterative_clustering(distances, metric="precomputed")
                validity_precom, _ = self._validity(labels_precomputed, unbinned_embeddings)

                # Calculate silhouette scores, will fail if only one label
                # Silhouette scores don't work too well with HDBSCAN though since it
                # usually requires pretty uniform clusters to generate a value of use
                try:
                    silho_precom = sk_metrics.silhouette_score(distances, labels_precomputed)
                except ValueError:
                    silho_precom = -1

                self.labels = labels_precomputed
                precomputed = True
                max_validity = max(validity_precom, silho_precom)

                if debug:
                    print('precom cluster validity: ', max_validity)

                set_labels = set(self.labels)

                if debug:
                    print("No. of Clusters:", len(set_labels), set_labels)

            except IndexError:
                # Index error occurs when doing recluster after adding disconnected TNF
                # contigs. Since the embedding array does not contain the missing contigs
                # as such, new embeddings have to be calculated
                max_validity = 0
            else:
                max_validity = -1

            set_labels = set(self.labels)

            if debug:
                print("No. of Clusters:", len(set_labels), set_labels)
                print("Max validity: ", max_validity)

            plots = self.add_plot(plots, unbinned_embeddings, contigs,
                                  n, x_min, x_max, y_min, y_max, max_validity, precomputed)

            if delete_unbinned:
                self.unbinned_tids = []


            new_bins = {}
            unbinned = []

            for (idx, label) in enumerate(self.labels):
                if label != -1:
                    bin_key = max_bin_id + label.item() + 1
                    if isinstance(bin_key, np.int64):
                        bin_key = bin_key.item()
                    try:
                        new_bins[bin_key].append(tids[idx])  # inputs values as tid
                    except KeyError:
                        new_bins[bin_key] = [tids[idx]]
                else:
                    unbinned.append(tids[idx])
            if debug:
                print("No. of new bins:", new_bins.keys())
                print("No. unbinned: ", len(unbinned))

            for bin, new_tids in new_bins.items():
                new_tids = list(np.sort(new_tids))
                contigs, log_lengths, tnfs = self.extract_contigs(new_tids)
                bin_size = contigs['contigLen'].sum()
                if bin_size >= self.min_bin_size:
                    #  Keep this bin
                    if debug:
                        print("Removing original bin, keeping bin: ", bin)
                        print("Length: ", bin_size)
                    self.bins[bin] = new_tids
                    if bin_size >= 14e6:
                        self.overclustered = True
                else:
                    # put into unbinned
                    if debug:
                        print("Not adding new bin: ", bin, bin_size)
                    unbinned = unbinned + new_tids


            if len(unbinned) != len(tids):
                logging.debug("New bin(s) added... Total bins: ", len(self.bins.keys()))
                contigs, log_lengths, tnfs = self.extract_contigs(unbinned)
                bin_size = contigs['contigLen'].sum()
                if self.n_samples > 0:
                    n_samples = self.n_samples
                    sample_distances = self.short_sample_distance
                else:
                    n_samples = self.long_samples
                    sample_distances = self.long_sample_distance
                try:
                    _, \
                    _, \
                    _, \
                    mean_agg, \
                    per_contig_avg = \
                        metrics.get_averages(np.concatenate((contigs.iloc[:, 3:].values,
                                                                log_lengths.values[:, None],
                                                                tnfs.iloc[:, 2:].values), axis=1),
                                                                n_samples,
                                                                sample_distances)
                except ZeroDivisionError:
                    mean_agg = 0

                bin_id = max(self.bins.keys()) + 1
                if bin_size >= 2e6 and mean_agg < 0.25: # just treat it as a bin
                    if debug:
                        print("Unbinned contigs are bin: %d of size: %d" % (bin_id, bin_size))
                    self.bins[bin_id] = unbinned
                else:
                    for contig in contigs.itertuples():
                        self.unbinned_tids.append(self.assembly[contig.contigName])


        return plots